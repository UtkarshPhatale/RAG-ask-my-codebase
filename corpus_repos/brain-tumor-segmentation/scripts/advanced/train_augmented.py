#!/usr/bin/env python3
"""
Experiment: Strong Augmentation + Spatial Dropout
===================================================
Builds on best setup so far (hybrid normalization).
Directly targets the overfitting observed in previous runs.

Changes vs hybrid_norm:
  1. Strong augmentation (rotation, elastic, noise, gamma, zoom)
  2. Spatial dropout p=0.1 at bottleneck
  3. Lower LR 3e-4 (augmentation acts as implicit regularization)
  4. 120 epochs (expect peak at epoch 70-90, not 36-56)
  5. Save every 5 epochs for finer tracking

Previous best: Equal ensemble WT=0.8488, TC=0.7744, ET=0.6778
Target:        WT>0.86,  TC>0.78,  ET>0.72
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from attention_unet_augmented_FEB20th import get_augmented_model
from advanced_losses import ComboLoss
from create_dataset_augmented import BraTSDatasetAugmented

# ── Paths ──────────────────────────────────────────────────────────
BASE    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA    = BASE / "data/raw/brats_gli/training_data1_v2"
SPLITS  = BASE / "data/splits/data_splits.json"
CKPT    = BASE / "models/checkpoints/v3/augmented"
LOGDIR  = BASE / "models/logs/v3"
RESULTS = BASE / "results/v3/augmented"

for d in [CKPT, LOGDIR, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
CONFIG = {
    'base_channels':    32,
    'num_classes':       4,
    'in_channels':       4,
    'dropout_p':       0.1,
    'batch_size':        2,
    'num_epochs':      120,
    'base_lr':        3e-4,   # lower than before — augmentation regularizes
    'warmup_epochs':     5,
    'weight_decay':   1e-4,
    'grad_clip':       1.0,
    'crop_size':  (128,128,128),
    'num_workers':       4,
    'deep_supervision': True,
    'save_every':        5,   # finer checkpoint tracking
    'experiment':  'augmented',
    'note': 'Strong aug + spatial dropout. Hybrid norm base. Target: overfitting delayed to epoch 70-90.'
}

# ── Dice metric ────────────────────────────────────────────────────
def compute_dice(pred, target, smooth=1e-5):
    scores = {}
    for name, p, t in [
        ('WT', pred>=1,          target>=1),
        ('TC', (pred==1)|(pred==3), (target==1)|(target==3)),
        ('ET', pred==3,          target==3),
    ]:
        p, t = p.float(), t.float()
        inter = (p*t).sum()
        scores[name] = ((2*inter+smooth)/(p.sum()+t.sum()+smooth)).item()
    return scores

# ── LR schedule ────────────────────────────────────────────────────
def set_lr(optimizer, epoch, config):
    if epoch < config['warmup_epochs']:
        lr = config['base_lr'] * (epoch+1) / config['warmup_epochs']
    else:
        prog = (epoch - config['warmup_epochs']) / \
               (config['num_epochs'] - config['warmup_epochs'])
        lr = config['base_lr'] * 0.5 * (1 + np.cos(np.pi * prog))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

# ── Train one epoch ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = total_wt = 0.0

    for i, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outs = model(imgs)
            main = outs[0] if isinstance(outs,(list,tuple)) else outs
            loss = criterion(main, lbls)
            if isinstance(outs,(list,tuple)):
                for j, aux in enumerate(outs[1:], 1):
                    loss += (0.5**j) * criterion(aux, lbls)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            d = compute_dice(main.argmax(dim=1), lbls)
            total_wt += d['WT']
        total_loss += loss.item()

        if i % 50 == 0:
            print(f"  Batch {i}/{len(loader)} | "
                  f"Loss:{loss.item():.4f} WT:{d['WT']:.4f}")

    n = len(loader)
    return total_loss/n, total_wt/n

# ── Validate ───────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = wt = tc = et = 0.0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with autocast():
                outs = model(imgs)
                main = outs[0] if isinstance(outs,(list,tuple)) else outs
                loss_sum += criterion(main, lbls).item()
            d = compute_dice(main.argmax(dim=1), lbls)
            wt += d['WT']; tc += d['TC']; et += d['ET']

    n = len(loader)
    return {'loss': loss_sum/n, 'wt_dice': wt/n,
            'tc_dice': tc/n, 'et_dice': et/n}

# ── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("EXPERIMENT: Strong Augmentation + Spatial Dropout")
    print("=" * 65)
    print(json.dumps(CONFIG, indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Datasets
    train_ds = BraTSDatasetAugmented(
        DATA, SPLITS, split='train',
        crop_size=CONFIG['crop_size'], augment=True
    )
    val_ds = BraTSDatasetAugmented(
        DATA, SPLITS, split='val',
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
    print(f"\nTrain: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # Model
    model = get_augmented_model(
        in_channels=CONFIG['in_channels'],
        num_classes=CONFIG['num_classes'],
        base_channels=CONFIG['base_channels'],
        deep_supervision=CONFIG['deep_supervision'],
        dropout_p=CONFIG['dropout_p']
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = ComboLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scaler = GradScaler()

    history   = []
    best_wt   = 0.0
    best_mean = 0.0
    best_wt_epoch   = 0
    best_mean_epoch = 0
    t0 = time.time()

    print("\n" + "=" * 65)
    print("Training started...")
    print("=" * 65)

    for epoch in range(CONFIG['num_epochs']):
        lr = set_lr(optimizer, epoch, CONFIG)

        train_loss, train_wt = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val = validate(model, val_loader, criterion, device)

        elapsed = (time.time() - t0) / 3600
        mean_dice = (val['wt_dice'] + val['tc_dice'] + val['et_dice']) / 3

        print(f"\nEpoch {epoch+1:3d}/{CONFIG['num_epochs']} | "
              f"LR:{lr:.2e} | "
              f"Train:{train_loss:.4f}/{train_wt:.4f} | "
              f"Val WT:{val['wt_dice']:.4f} "
              f"TC:{val['tc_dice']:.4f} "
              f"ET:{val['et_dice']:.4f} "
              f"Mean:{mean_dice:.4f} | "
              f"Time:{elapsed:.1f}h")

        history.append({
            'epoch': epoch+1, 'lr': lr,
            'train_loss': train_loss, 'train_wt': train_wt,
            'val_wt': val['wt_dice'], 'val_tc': val['tc_dice'],
            'val_et': val['et_dice'], 'val_mean': mean_dice,
        })

        # Save best WT checkpoint
        if val['wt_dice'] > best_wt:
            best_wt = val['wt_dice']
            best_wt_epoch = epoch+1
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'val_wt_dice': val['wt_dice'],
                'val_tc_dice': val['tc_dice'],
                'val_et_dice': val['et_dice'],
                'config': CONFIG,
            }, CKPT / 'best_model.pth')
            print(f"  ✅ New best WT! {best_wt:.4f} "
                  f"(TC={val['tc_dice']:.4f}, ET={val['et_dice']:.4f})")

        # Save best mean checkpoint
        if mean_dice > best_mean:
            best_mean = mean_dice
            best_mean_epoch = epoch+1
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'val_wt_dice': val['wt_dice'],
                'val_tc_dice': val['tc_dice'],
                'val_et_dice': val['et_dice'],
                'val_mean':    mean_dice,
                'config': CONFIG,
            }, CKPT / 'best_mean_model.pth')
            print(f"  ✅ New best Mean! {mean_dice:.4f}")

        # Periodic checkpoint every N epochs
        if (epoch+1) % CONFIG['save_every'] == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'val_wt_dice': val['wt_dice'],
                'config': CONFIG,
            }, CKPT / f'epoch_{epoch+1:03d}.pth')

        # Save history every epoch for live monitoring
        with open(CKPT / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping hint (don't stop, just flag)
        if epoch >= 30 and all(
            h['val_wt'] < best_wt - 0.02
            for h in history[-15:]
        ):
            print(f"  ⚠️  No WT improvement in 15 epochs. "
                  f"Best was epoch {best_wt_epoch}. Continuing...")

    total_h = (time.time() - t0) / 3600
    summary = {
        'experiment': 'augmented',
        'best_val_wt': best_wt, 'best_wt_epoch': best_wt_epoch,
        'best_mean':   best_mean, 'best_mean_epoch': best_mean_epoch,
        'total_epochs': CONFIG['num_epochs'],
        'total_hours':  round(total_h, 2),
        'config': CONFIG,
        'previous_results': {
            'ensemble_equal': {'wt':0.8488,'tc':0.7744,'et':0.6778},
            'hybrid_mean':    {'wt':0.8420,'tc':0.7737,'et':0.6769},
            'percentile':     {'wt':0.8439,'tc':0.7493,'et':0.6551},
            'baseline':       {'wt':0.8242,'tc':0.5931,'et':0.7878},
        }
    }
    with open(CKPT / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"Best Val WT   : {best_wt:.4f} at epoch {best_wt_epoch}")
    print(f"Best Mean     : {best_mean:.4f} at epoch {best_mean_epoch}")
    print(f"Total time    : {total_h:.1f}h")
    print("=" * 65)

if __name__ == '__main__':
    main()
