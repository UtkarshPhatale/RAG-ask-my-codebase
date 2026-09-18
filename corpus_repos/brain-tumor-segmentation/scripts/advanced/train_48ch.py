#!/usr/bin/env python3
"""
train_48ch.py
Training script for 48-Channel Attention U-Net
Save to: scripts/advanced/train_48ch.py

Key differences vs train_augmented.py:
  1. Uses AttentionUNet48Ch (base_channels=48, ~51M params)
  2. Dynamic class-balanced loss weighting
  3. CosineAnnealingWarmRestarts LR scheduler (T_0=30, T_mult=2)
  4. LR 2e-4 (slightly lower for larger model)
  5. Checkpoints -> models/checkpoints/v3/attention_48ch_v1/
  6. Results    -> results/v3/attention_48ch_v1/
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------ Paths
BASE = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts' / 'advanced'))

from attention_unet_48ch import AttentionUNet48Ch
from create_dataset_augmented import BraTSDatasetAugmented
from advanced_losses import ComboLoss

# ------------------------------------------------------------------ Config
DATA    = BASE / "data/raw/brats_gli/training_data1_v2"
SPLITS  = BASE / "data/splits/data_splits.json"
CKPT    = BASE / "models/checkpoints/v3/attention_48ch_v1"
LOGDIR  = BASE / "models/logs/v3"

CONFIG = {
    'base_channels':    48,
    'num_classes':      4,
    'in_channels':      4,
    'dropout_p':        0.15,
    'deep_supervision': True,
    'batch_size':       2,
    'num_epochs':       120,
    'base_lr':          2e-4,
    'weight_decay':     1e-5,
    'grad_clip':        1.0,
    'crop_size':        (128, 128, 128),
    'num_workers':      4,
    'save_every':       5,
    # CosineAnnealingWarmRestarts: restarts at ep 30, 90
    'scheduler_T0':     30,
    'scheduler_Tmult':  2,
    'scheduler_eta_min': 1e-6,
    # Deep supervision loss weights
    'ds_weights':       [1.0, 0.4, 0.2, 0.1],
}


# ------------------------------------------------------------------ Logging
import logging
def setup_logging(job_id):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGDIR / f'train_48ch_{job_id}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


# ------------------------------------------------------------------ BraTS Metrics
def brats_metrics(pred, target, smooth=1e-5):
    """
    pred, target: [B, H, W, D] integer tensors (CPU)
    Returns dict: WT, TC, ET mean Dice across batch
    """
    def dice_binary(p, t):
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float()
        if union == 0:
            return 1.0
        return ((2.0 * inter + smooth) / (union + smooth)).item()

    wt = dice_binary(pred >= 1,            target >= 1)
    tc = dice_binary((pred==1)|(pred==3),  (target==1)|(target==3))
    et = dice_binary(pred == 3,            target == 3)
    return {'WT': wt, 'TC': tc, 'ET': et, 'Mean': (wt+tc+et)/3.0}


# ------------------------------------------------------------------ Deep Supervision Loss
def ds_loss(outputs, target, criterion, weights):
    """Weighted loss across deep supervision heads."""
    total = 0.0
    for pred, w in zip(outputs, weights):
        if pred.shape[2:] != target.shape[1:]:
            pred = F.interpolate(pred, size=target.shape[1:],
                                 mode='trilinear', align_corners=False)
        total += w * criterion(pred, target)
    return total / sum(weights)


# ------------------------------------------------------------------ Train Epoch
def train_epoch(model, loader, optimizer, criterion, scaler, device, logger):
    model.train()
    total_loss = 0.0
    all_wt, all_tc, all_et = [], [], []

    for i, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outs = model(imgs)
            if isinstance(outs, (list, tuple)):
                loss = ds_loss(outs, lbls, criterion, CONFIG['ds_weights'])
                main = outs[0]
            else:
                loss = criterion(outs, lbls)
                main = outs

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            pred = torch.argmax(main, dim=1).cpu()
            m = brats_metrics(pred, lbls.cpu())
            all_wt.append(m['WT'])
            all_tc.append(m['TC'])
            all_et.append(m['ET'])

        if (i + 1) % 50 == 0:
            logger.info(f"  Batch {i+1}/{len(loader)} loss={loss.item():.4f}")

    return {
        'loss': total_loss / len(loader),
        'WT':   np.mean(all_wt),
        'TC':   np.mean(all_tc),
        'ET':   np.mean(all_et),
        'Mean': np.mean([np.mean(all_wt), np.mean(all_tc), np.mean(all_et)]),
    }


# ------------------------------------------------------------------ Val Epoch
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_wt, all_tc, all_et = [], [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        with autocast():
            outs = model(imgs)
            main = outs[0] if isinstance(outs, (list, tuple)) else outs
            loss = criterion(main, lbls)

        total_loss += loss.item()

        pred = torch.argmax(main, dim=1).cpu()
        m = brats_metrics(pred, lbls.cpu())
        all_wt.append(m['WT'])
        all_tc.append(m['TC'])
        all_et.append(m['ET'])

    return {
        'loss': total_loss / len(loader),
        'WT':   np.mean(all_wt),
        'TC':   np.mean(all_tc),
        'ET':   np.mean(all_et),
        'Mean': np.mean([np.mean(all_wt), np.mean(all_tc), np.mean(all_et)]),
    }


# ------------------------------------------------------------------ Main
def main():
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    CKPT.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(job_id)
    logger.info("=" * 60)
    logger.info("48-Channel Attention U-Net Training")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Config: {CONFIG}")
    logger.info("=" * 60)

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs: {num_gpus}")
    for i in range(num_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ---- Data ----
    train_ds = BraTSDatasetAugmented(
        str(DATA), str(SPLITS),
        split='train', augment=True
    )
    val_ds = BraTSDatasetAugmented(
        str(DATA), str(SPLITS),
        split='val', augment=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'],
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True
    )
    logger.info(f"Train: {len(train_ds)} patients | Val: {len(val_ds)} patients")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = AttentionUNet48Ch(
        in_channels=CONFIG['in_channels'],
        num_classes=CONFIG['num_classes'],
        base_channels=CONFIG['base_channels'],
        deep_supervision=CONFIG['deep_supervision'],
        dropout_p=CONFIG['dropout_p'],
    )
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel across {num_gpus} GPUs")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params/1e6:.1f}M")

    # ---- Loss / Optimizer / Scheduler ----
    criterion = ComboLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG['scheduler_T0'],
        T_mult=CONFIG['scheduler_Tmult'],
        eta_min=CONFIG['scheduler_eta_min'],
    )
    scaler = GradScaler()

    # ---- Training State ----
    best_wt        = 0.0
    best_mean      = 0.0
    best_wt_epoch  = 0
    best_mean_epoch = 0
    history        = []

    logger.info(f"\nStarting training — {CONFIG['num_epochs']} epochs")

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        t0  = time.time()
        lr  = optimizer.param_groups[0]['lr']

        train_m = train_epoch(model, train_loader, optimizer, criterion, scaler, device, logger)
        val_m   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{CONFIG['num_epochs']} | "
            f"LR={lr:.2e} | "
            f"TrainLoss={train_m['loss']:.4f} | "
            f"Val WT={val_m['WT']:.4f} TC={val_m['TC']:.4f} "
            f"ET={val_m['ET']:.4f} Mean={val_m['Mean']:.4f} | "
            f"Time={elapsed:.0f}s"
        )

        history.append({
            'epoch': epoch, 'lr': lr,
            'train_loss': train_m['loss'],
            'val_WT': val_m['WT'], 'val_TC': val_m['TC'],
            'val_ET': val_m['ET'], 'val_Mean': val_m['Mean'],
        })

        # Save best WT checkpoint
        if val_m['WT'] > best_wt:
            best_wt       = val_m['WT']
            best_wt_epoch = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_wt_dice': val_m['WT'],
                'val_tc_dice': val_m['TC'],
                'val_et_dice': val_m['ET'],
                'val_mean':    val_m['Mean'],
                'config':      CONFIG,
            }, CKPT / 'best_model.pth')
            logger.info(f"  -> New best WT={best_wt:.4f} at epoch {epoch}")

        # Save best Mean checkpoint
        if val_m['Mean'] > best_mean:
            best_mean       = val_m['Mean']
            best_mean_epoch = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_wt_dice': val_m['WT'],
                'val_tc_dice': val_m['TC'],
                'val_et_dice': val_m['ET'],
                'val_mean':    val_m['Mean'],
                'config':      CONFIG,
            }, CKPT / 'best_mean_model.pth')
            logger.info(f"  -> New best Mean={best_mean:.4f} at epoch {epoch}")

        # Periodic checkpoint
        if epoch % CONFIG['save_every'] == 0:
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_wt_dice': val_m['WT'],
                'val_tc_dice': val_m['TC'],
                'val_et_dice': val_m['ET'],
                'val_mean':    val_m['Mean'],
                'config':      CONFIG,
            }, CKPT / f'checkpoint_ep{epoch:03d}.pth')

        # Save history every epoch
        with open(CKPT / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # ---- Summary ----
    summary = {
        'job_id':           job_id,
        'config':           CONFIG,
        'best_wt':          best_wt,
        'best_wt_epoch':    best_wt_epoch,
        'best_mean':        best_mean,
        'best_mean_epoch':  best_mean_epoch,
        'total_epochs':     epoch,
        'total_params_M':   total_params / 1e6,
    }
    with open(CKPT / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best WT:   {best_wt:.4f} at epoch {best_wt_epoch}")
    logger.info(f"Best Mean: {best_mean:.4f} at epoch {best_mean_epoch}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
