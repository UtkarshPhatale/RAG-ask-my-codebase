#!/usr/bin/env python3
"""
Experiment: Percentile Normalization Retraining
================================================
Hypothesis: Scanner intensity variability causes 83 TC failures.
Fix: Replace z-score normalization with percentile (p1-p99) normalization.

Key difference from baseline:
- Dataset: BraTSDatasetPercentile (percentile norm) instead of original
- Everything else IDENTICAL to best model (ultimate_32ch_rerun)

Save locations:
- Checkpoints: models/checkpoints/v3/percentile_norm/
- Logs:        models/logs/v3/
- Results:     results/v3/percentile_norm/
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from attention_unet_v2_improved import ImprovedAttentionUNet3D
from advanced_losses import ComboLoss
from create_dataset_percentile import BraTSDatasetPercentile

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA_DIR    = BASE_DIR / "data/raw/brats_gli/training_data1_v2"
SPLITS_FILE = BASE_DIR / "data/splits/data_splits.json"
CKPT_DIR    = BASE_DIR / "models/checkpoints/v3/percentile_norm"
LOG_DIR     = BASE_DIR / "models/logs/v3"
RESULTS_DIR = BASE_DIR / "results/v3/percentile_norm"

for d in [CKPT_DIR, LOG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    'base_channels':    32,
    'num_classes':       4,
    'in_channels':       4,
    'batch_size':        2,
    'num_epochs':      100,
    'base_lr':        5e-4,
    'warmup_epochs':     5,
    'weight_decay':   1e-4,
    'grad_clip':       1.0,
    'crop_size':  (128, 128, 128),
    'num_workers':       4,
    'deep_supervision': True,
    'experiment':  'percentile_norm',
    'note': 'Percentile normalization to fix scanner variability in 83 failure cases'
}

# ── DICE METRIC ──────────────────────────────────────────────────────────────
def compute_dice(pred, target, num_classes=4, smooth=1e-5):
    """Compute per-class Dice scores."""
    dice_scores = {}
    pred_oh   = torch.zeros(pred.shape[0], num_classes, *pred.shape[1:],
                            device=pred.device)
    pred_oh.scatter_(1, pred.unsqueeze(1), 1)
    target_oh = torch.zeros_like(pred_oh)
    target_oh.scatter_(1, target.unsqueeze(1), 1)

    for c in range(1, num_classes):  # skip background
        inter = (pred_oh[:, c] * target_oh[:, c]).sum()
        union = pred_oh[:, c].sum() + target_oh[:, c].sum()
        dice_scores[c] = ((2 * inter + smooth) / (union + smooth)).item()

    # BraTS composite scores
    # WT = NCR(1) + ED(2) + ET(3)
    wt_pred = (pred >= 1).float()
    wt_tgt  = (target >= 1).float()
    inter = (wt_pred * wt_tgt).sum()
    union = wt_pred.sum() + wt_tgt.sum()
    dice_scores['WT'] = ((2 * inter + smooth) / (union + smooth)).item()

    # TC = NCR(1) + ET(3)
    tc_pred = ((pred == 1) | (pred == 3)).float()
    tc_tgt  = ((target == 1) | (target == 3)).float()
    inter = (tc_pred * tc_tgt).sum()
    union = tc_pred.sum() + tc_tgt.sum()
    dice_scores['TC'] = ((2 * inter + smooth) / (union + smooth)).item()

    # ET = ET(3) only
    et_pred = (pred == 3).float()
    et_tgt  = (target == 3).float()
    inter = (et_pred * et_tgt).sum()
    union = et_pred.sum() + et_tgt.sum()
    dice_scores['ET'] = ((2 * inter + smooth) / (union + smooth)).item()

    return dice_scores


# ── LR SCHEDULER ─────────────────────────────────────────────────────────────
def get_lr(optimizer, epoch, config):
    """Warmup + cosine annealing."""
    if epoch < config['warmup_epochs']:
        lr = config['base_lr'] * (epoch + 1) / config['warmup_epochs']
    else:
        progress = (epoch - config['warmup_epochs']) / \
                   (config['num_epochs'] - config['warmup_epochs'])
        lr = config['base_lr'] * 0.5 * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss, total_dice_wt = 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            # Handle deep supervision (list of outputs)
            if isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
                loss = criterion(main_out, labels)
                # Add auxiliary losses with decreasing weights
                for i, aux_out in enumerate(outputs[1:], 1):
                    weight = 0.5 ** i
                    loss += weight * criterion(aux_out, labels)
            else:
                main_out = outputs
                loss = criterion(main_out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = main_out.argmax(dim=1)
            dice = compute_dice(pred, labels)
            total_dice_wt += dice['WT']

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"WT Dice: {dice['WT']:.4f}")

    n = len(loader)
    return total_loss / n, total_dice_wt / n


# ── VALIDATION ────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_wt, dice_tc, dice_et = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                main_out = outputs[0] if isinstance(outputs, (list, tuple)) \
                           else outputs
                loss = criterion(main_out, labels)

            pred = main_out.argmax(dim=1)
            dice = compute_dice(pred, labels)

            total_loss += loss.item()
            dice_wt    += dice['WT']
            dice_tc    += dice['TC']
            dice_et    += dice['ET']

    n = len(loader)
    return {
        'loss':    total_loss / n,
        'wt_dice': dice_wt / n,
        'tc_dice': dice_tc / n,
        'et_dice': dice_et / n,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("EXPERIMENT: Percentile Normalization")
    print("=" * 65)
    print(f"Checkpoint dir : {CKPT_DIR}")
    print(f"Results dir    : {RESULTS_DIR}")
    print(f"Config         : {json.dumps(CONFIG, indent=2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Datasets ────────────────────────────────────────────────
    train_ds = BraTSDatasetPercentile(
        DATA_DIR, SPLITS_FILE, split='train', 
        crop_size=CONFIG['crop_size'], augment=True
    )
    val_ds = BraTSDatasetPercentile(
        DATA_DIR, SPLITS_FILE, split='val',
        crop_size=CONFIG['crop_size'], augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'],
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Model ───────────────────────────────────────────────────
    model = ImprovedAttentionUNet3D(
        in_channels=CONFIG['in_channels'],
        num_classes=CONFIG["num_classes"],
        base_channels=CONFIG["base_channels"],
        deep_supervision=CONFIG['deep_supervision']
    )

    # Use both GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters    : {total_params/1e6:.1f}M")

    # ── Loss, optimizer, scaler ──────────────────────────────────
    criterion = ComboLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scaler = GradScaler()

    # ── Training ────────────────────────────────────────────────
    history = []
    best_val_wt = 0.0
    best_epoch  = 0
    start_time  = time.time()

    print("\n" + "=" * 65)
    print("Starting training...")
    print("=" * 65)

    for epoch in range(CONFIG['num_epochs']):
        lr = get_lr(optimizer, epoch, CONFIG)

        # Train
        train_loss, train_wt = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = (time.time() - start_time) / 3600
        print(f"\nEpoch {epoch+1:3d}/{CONFIG['num_epochs']} | "
              f"LR: {lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | Train WT: {train_wt:.4f} | "
              f"Val WT: {val_metrics['wt_dice']:.4f} | "
              f"Val TC: {val_metrics['tc_dice']:.4f} | "
              f"Val ET: {val_metrics['et_dice']:.4f} | "
              f"Elapsed: {elapsed:.1f}h")

        # Save history
        history.append({
            'epoch':      epoch + 1,
            'lr':         lr,
            'train_loss': train_loss,
            'train_wt':   train_wt,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        # Save best model
        if val_metrics['wt_dice'] > best_val_wt:
            best_val_wt = val_metrics['wt_dice']
            best_epoch  = epoch + 1
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_wt_dice': best_val_wt,
                'config':      CONFIG,
            }, CKPT_DIR / 'best_model.pth')
            print(f"  ✅ New best! Val WT: {best_val_wt:.4f} (epoch {best_epoch})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'config':      CONFIG,
            }, CKPT_DIR / f'epoch_{epoch+1:03d}.pth')

        # Save history every epoch (so you can monitor mid-training)
        with open(CKPT_DIR / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # ── Save final summary ───────────────────────────────────────
    total_time = (time.time() - start_time) / 3600
    summary = {
        'experiment':        CONFIG['experiment'],
        'best_val_wt_dice':  best_val_wt,
        'best_epoch':        best_epoch,
        'total_epochs':      CONFIG['num_epochs'],
        'total_time_hours':  round(total_time, 2),
        'config':            CONFIG,
        'baseline_val_wt':   0.8084,
        'improvement':       round(best_val_wt - 0.8084, 4),
    }
    with open(CKPT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"Best Val WT Dice : {best_val_wt:.4f} (epoch {best_epoch})")
    print(f"Baseline Val WT  : 0.8084")
    print(f"Improvement      : {best_val_wt - 0.8084:+.4f}")
    print(f"Total time       : {total_time:.1f}h")
    print(f"Checkpoint       : {CKPT_DIR}/best_model.pth")
    print("=" * 65)


if __name__ == '__main__':
    main()
