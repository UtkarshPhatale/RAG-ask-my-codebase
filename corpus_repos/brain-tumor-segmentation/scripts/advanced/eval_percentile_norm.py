#!/usr/bin/env python3
"""
Evaluation: Percentile Normalization Model
Tests the best checkpoint against the 204-patient test set
Saves results to results/v3/percentile_norm/
"""

import sys
import json
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/advanced')

from attention_unet_v2_improved import ImprovedAttentionUNet3D
from create_dataset_percentile import BraTSDatasetPercentile

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA_DIR    = BASE_DIR / "data/raw/brats_gli/training_data1_v2"
SPLITS_FILE = BASE_DIR / "data/splits/data_splits.json"
CKPT_PATH   = BASE_DIR / "models/checkpoints/v3/percentile_norm/best_model.pth"
RESULTS_DIR = BASE_DIR / "results/v3/percentile_norm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_dice_scores(pred, target, smooth=1e-5):
    """Compute WT, TC, ET dice for a single patient."""
    results = {}

    # WT = all tumor (labels 1,2,3)
    wt_pred = (pred >= 1).float()
    wt_tgt  = (target >= 1).float()
    inter = (wt_pred * wt_tgt).sum()
    union = wt_pred.sum() + wt_tgt.sum()
    results['wt_dice'] = ((2 * inter + smooth) / (union + smooth)).item()

    # TC = NCR + ET (labels 1,3)
    tc_pred = ((pred == 1) | (pred == 3)).float()
    tc_tgt  = ((target == 1) | (target == 3)).float()
    inter = (tc_pred * tc_tgt).sum()
    union = tc_pred.sum() + tc_tgt.sum()
    results['tc_dice'] = ((2 * inter + smooth) / (union + smooth)).item()

    # ET = enhancing tumor (label 3)
    et_pred = (pred == 3).float()
    et_tgt  = (target == 3).float()
    inter = (et_pred * et_tgt).sum()
    union = et_pred.sum() + et_tgt.sum()
    results['et_dice'] = ((2 * inter + smooth) / (union + smooth)).item()

    # Per-class dice
    for c, name in [(1,'ncr'), (2,'ed'), (3,'et_class')]:
        p = (pred == c).float()
        t = (target == c).float()
        i = (p * t).sum()
        u = p.sum() + t.sum()
        results[f'{name}_dice'] = ((2 * i + smooth) / (u + smooth)).item()

    return results

def main():
    print("=" * 60)
    print("EVALUATION: Percentile Normalization Model")
    print("=" * 60)
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Results:    {RESULTS_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model = ImprovedAttentionUNet3D(
        in_channels=4, num_classes=4,
        base_channels=32, deep_supervision=True
    )
    # Handle DataParallel checkpoint
    state_dict = checkpoint['model_state']
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace('module.', '')] = v
    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()

    saved_epoch = checkpoint.get('epoch', 'unknown')
    saved_val   = checkpoint.get('val_wt_dice', 'unknown')
    print(f"Loaded checkpoint: epoch {saved_epoch}, val WT={saved_val:.4f}")

    # ── Dataset ──────────────────────────────────────────────────
    test_ds = BraTSDatasetPercentile(
        DATA_DIR, SPLITS_FILE,
        split='test', augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=1,
        shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Test patients: {len(test_ds)}")

    # ── Evaluate ─────────────────────────────────────────────────
    all_metrics = []
    wt_list, tc_list, et_list = [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                main_out = outputs[0] if isinstance(outputs, (list, tuple)) \
                           else outputs

            pred = main_out.argmax(dim=1)
            scores = compute_dice_scores(pred[0], labels[0])

            patient_id = test_ds.patient_ids[i]
            scores['patient_id'] = patient_id
            all_metrics.append(scores)

            wt_list.append(scores['wt_dice'])
            tc_list.append(scores['tc_dice'])
            et_list.append(scores['et_dice'])

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(test_loader)} | "
                      f"WT={np.mean(wt_list):.4f} "
                      f"TC={np.mean(tc_list):.4f} "
                      f"ET={np.mean(et_list):.4f}")

    # ── Results ──────────────────────────────────────────────────
    wt_arr = np.array(wt_list)
    tc_arr = np.array(tc_list)
    et_arr = np.array(et_list)

    # TC failure analysis
    tc_failures  = (tc_arr < 0.1).sum()
    tc_excellent = (tc_arr >= 0.85).sum()

    summary = {
        'experiment': 'percentile_norm',
        'checkpoint_epoch': saved_epoch,
        'test_patients': len(test_ds),
        'wt_dice_mean': float(wt_arr.mean()),
        'wt_dice_std':  float(wt_arr.std()),
        'tc_dice_mean': float(tc_arr.mean()),
        'tc_dice_std':  float(tc_arr.std()),
        'et_dice_mean': float(et_arr.mean()),
        'et_dice_std':  float(et_arr.std()),
        'tc_failures_n':  int(tc_failures),
        'tc_excellent_n': int(tc_excellent),
        'baseline_wt':  0.8242,
        'baseline_tc':  0.5931,
        'baseline_et':  0.7878,
        'wt_improvement': float(wt_arr.mean() - 0.8242),
        'tc_improvement': float(tc_arr.mean() - 0.5931),
    }

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"WT Dice: {summary['wt_dice_mean']:.4f} ± {summary['wt_dice_std']:.4f}  "
          f"(baseline: 0.8242, change: {summary['wt_improvement']:+.4f})")
    print(f"TC Dice: {summary['tc_dice_mean']:.4f} ± {summary['tc_dice_std']:.4f}  "
          f"(baseline: 0.5931, change: {summary['tc_improvement']:+.4f})")
    print(f"ET Dice: {summary['et_dice_mean']:.4f} ± {summary['et_dice_std']:.4f}  "
          f"(baseline: 0.7878)")
    print(f"\nTC Failures (TC<0.1):    {tc_failures} "
          f"(baseline: 83, change: {tc_failures-83:+d})")
    print(f"TC Excellent (TC≥0.85):  {tc_excellent} "
          f"(baseline: 121)")

    if summary['wt_dice_mean'] > 0.8242:
        print(f"\n✅ IMPROVEMENT over baseline!")
    else:
        print(f"\n❌ Did not beat baseline")

    # Save CSV
    csv_path = RESULTS_DIR / 'test_metrics.csv'
    fieldnames = list(all_metrics[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved: {csv_path}")

    # Save summary JSON
    json_path = RESULTS_DIR / 'test_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
