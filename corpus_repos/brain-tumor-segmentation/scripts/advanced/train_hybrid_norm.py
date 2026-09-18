#!/usr/bin/env python3
"""
Experiment: Hybrid Normalization
=================================
Building on percentile_norm results (WT=0.8439, TC=0.7493, ET=0.6551)

Problem: Percentile norm hurt ET (0.7878 -> 0.6551) by clipping T1C bright signal
Fix: T1C uses z-score (preserves ET brightness), T1N/T2W/T2F keep percentile

Save locations:
- Checkpoints: models/checkpoints/v3/hybrid_norm/
- Logs:        models/logs/v3/
- Results:     results/v3/hybrid_norm/
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

sys.path.insert(0, str(Path(__file__).parent))

from attention_unet_v2_improved import ImprovedAttentionUNet3D
from advanced_losses import ComboLoss
from create_dataset_hybrid_norm import BraTSDatasetHybridNorm

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA_DIR    = BASE_DIR / "data/raw/brats_gli/training_data1_v2"
SPLITS_FILE = BASE_DIR / "data/splits/data_splits.json"
CKPT_DIR    = BASE_DIR / "models/checkpoints/v3/hybrid_norm"
LOG_DIR     = BASE_DIR / "models/logs/v3"
RESULTS_DIR = BASE_DIR / "results/v3/hybrid_norm"

for d in [CKPT_DIR, LOG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    'base_channels':    32,
    'num_classes':       4,
    'in_channels':       4,
    'batch_size':        2,
    'num_epochs':       80,   # Stop earlier - we know overfitting starts ~epoch 60
    'base_lr':        5e-4,
    'warmup_epochs':     5,
    'weight_decay':   1e-4,
    'grad_clip':       1.0,
    'crop_size':  (128, 128, 128),
    'num_workers':       4,
    'deep_supervision': True,
    'experiment':  'hybrid_norm',
    'note': 'T1C=zscore for ET, T1N/T2W/T2F=percentile for NCR/ED. Stop at 80 epochs to avoid overfitting.'
}

# ── DICE METRIC ───────────────────────────────────────────────────────────────
def compute_dice(pred, target, smooth=1e-5):
    scores = {}
    wt_pred = (pred >= 1).float()
    wt_tgt  = (target >= 1).float()
    inter = (wt_pred * wt_tgt).sum()
    union = wt_pred.sum() + wt_tgt.sum()
    scores['WT'] = ((2*inter+smooth)/(union+smooth)).item()

    tc_pred = ((pred==1)|(pred==3)).float()
    tc_tgt  = ((target==1)|(target==3)).float()
    inter = (tc_pred*tc_tgt).sum()
    union = tc_pred.sum()+tc_tgt.sum()
    scores['TC'] = ((2*inter+smooth)/(union+smooth)).item()

    et_pred = (pred==3).float()
    et_tgt  = (target==3).float()
    inter = (et_pred*et_tgt).sum()
    union = et_pred.sum()+et_tgt.sum()
    scores['ET'] = ((2*inter+smooth)/(union+smooth)).item()

    return scores

# ── LR SCHEDULER ──────────────────────────────────────────────────────────────
def get_lr(optimizer, epoch, config):
    if epoch < config['warmup_epochs']:
        lr = config['base_lr'] * (epoch+1) / config['warmup_epochs']
    else:
        progress = (epoch-config['warmup_epochs']) / \
                   (config['num_epochs']-config['warmup_epochs'])
        lr = config['base_lr'] * 0.5 * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

# ── TRAIN ONE EPOCH ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, total_wt = 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
                loss = criterion(main_out, labels)
                for i, aux in enumerate(outputs[1:], 1):
                    loss += (0.5**i) * criterion(aux, labels)
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
            d = compute_dice(pred, labels)
            total_wt += d['WT']
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | WT: {d['WT']:.4f}")

    n = len(loader)
    return total_loss/n, total_wt/n

# ── VALIDATE ──────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    wt, tc, et = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(images)
                main_out = outputs[0] if isinstance(outputs,(list,tuple)) else outputs
                loss = criterion(main_out, labels)
            pred = main_out.argmax(dim=1)
            d = compute_dice(pred, labels)
            total_loss += loss.item()
            wt += d['WT']
            tc += d['TC']
            et += d['ET']

    n = len(loader)
    return {'loss': total_loss/n, 'wt_dice': wt/n,
            'tc_dice': tc/n, 'et_dice': et/n}

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("EXPERIMENT: Hybrid Normalization")
    print("=" * 65)
    print(f"Checkpoint dir : {CKPT_DIR}")
    print(f"Config         : {json.dumps(CONFIG, indent=2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Datasets ──────────────────────────────────────────────────
    train_ds = BraTSDatasetHybridNorm(
        DATA_DIR, SPLITS_FILE, split='train',
        crop_size=CONFIG['crop_size'], augment=True
    )
    val_ds = BraTSDatasetHybridNorm(
        DATA_DIR, SPLITS_FILE, split='val',
        crop_size=CONFIG['crop_size'], augment=False
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
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────
    model = ImprovedAttentionUNet3D(
        in_channels=CONFIG['in_channels'],
        num_classes=CONFIG['num_classes'],
        base_channels=CONFIG['base_channels'],
        deep_supervision=CONFIG['deep_supervision']
    )
    if torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── Loss / optimizer / scaler ──────────────────────────────────
    criterion = ComboLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scaler = GradScaler()

    # ── Training loop ──────────────────────────────────────────────
    history   = []
    best_val  = 0.0
    best_epoch = 0
    # Track best combined score (WT + TC + ET) / 3
    best_mean  = 0.0
    start_time = time.time()

    print("\n" + "=" * 65)
    print("Starting training...")
    print("=" * 65)

    for epoch in range(CONFIG['num_epochs']):
        lr = get_lr(optimizer, epoch, CONFIG)

        train_loss, train_wt = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = (time.time() - start_time) / 3600
        mean_dice = (val_metrics['wt_dice'] +
                     val_metrics['tc_dice'] +
                     val_metrics['et_dice']) / 3

        print(f"\nEpoch {epoch+1:3d}/{CONFIG['num_epochs']} | "
              f"LR: {lr:.2e} | "
              f"Train: {train_loss:.4f}/{train_wt:.4f} | "
              f"Val WT: {val_metrics['wt_dice']:.4f} | "
              f"TC: {val_metrics['tc_dice']:.4f} | "
              f"ET: {val_metrics['et_dice']:.4f} | "
              f"Mean: {mean_dice:.4f} | "
              f"Time: {elapsed:.1f}h")

        history.append({
            'epoch':      epoch+1,
            'lr':         lr,
            'train_loss': train_loss,
            'train_wt':   train_wt,
            'val_mean':   mean_dice,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        # Save best by WT Dice (primary metric)
        if val_metrics['wt_dice'] > best_val:
            best_val   = val_metrics['wt_dice']
            best_epoch = epoch+1
            torch.save({
                'epoch':       epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_wt_dice': best_val,
                'val_tc_dice': val_metrics['tc_dice'],
                'val_et_dice': val_metrics['et_dice'],
                'config':      CONFIG,
            }, CKPT_DIR / 'best_model.pth')
            print(f"  ✅ New best WT! {best_val:.4f} "
                  f"(TC={val_metrics['tc_dice']:.4f}, "
                  f"ET={val_metrics['et_dice']:.4f})")

        # Also save best by mean dice (catches balanced improvements)
        if mean_dice > best_mean:
            best_mean = mean_dice
            torch.save({
                'epoch':       epoch+1,
                'model_state': model.state_dict(),
                'val_wt_dice': val_metrics['wt_dice'],
                'val_tc_dice': val_metrics['tc_dice'],
                'val_et_dice': val_metrics['et_dice'],
                'val_mean':    mean_dice,
                'config':      CONFIG,
            }, CKPT_DIR / 'best_mean_model.pth')
            print(f"  ✅ New best Mean! {mean_dice:.4f}")

        # Checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch':       epoch+1,
                'model_state': model.state_dict(),
                'config':      CONFIG,
            }, CKPT_DIR / f'epoch_{epoch+1:03d}.pth')

        # Save history every epoch for live monitoring
        with open(CKPT_DIR / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────
    total_time = (time.time() - start_time) / 3600
    summary = {
        'experiment':       'hybrid_norm',
        'best_val_wt':      best_val,
        'best_epoch':       best_epoch,
        'best_mean_dice':   best_mean,
        'total_epochs':     CONFIG['num_epochs'],
        'total_time_hours': round(total_time, 2),
        'config':           CONFIG,
        'previous_percentile_norm': {
            'wt': 0.8439, 'tc': 0.7493, 'et': 0.6551
        },
        'baseline': {
            'wt': 0.8242, 'tc': 0.5931, 'et': 0.7878
        }
    }
    with open(CKPT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"Best Val WT      : {best_val:.4f} (epoch {best_epoch})")
    print(f"Best Mean Dice   : {best_mean:.4f}")
    print(f"Total time       : {total_time:.1f}h")
    print(f"Checkpoint saved : {CKPT_DIR}")
    print("=" * 65)

if __name__ == '__main__':
    main()
