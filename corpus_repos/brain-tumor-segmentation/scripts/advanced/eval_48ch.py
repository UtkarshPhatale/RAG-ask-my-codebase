#!/usr/bin/env python3
"""
eval_48ch.py
Evaluation script for 48-Channel Attention U-Net
Save to: scripts/advanced/eval_48ch.py
"""

import os, sys, json, csv
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# ------------------------------------------------------------------ Paths
BASE = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts' / 'advanced'))

from attention_unet_48ch import AttentionUNet48Ch
from create_dataset_augmented import BraTSDatasetAugmented

DATA    = BASE / "data/raw/brats_gli/training_data1_v2"
SPLITS  = BASE / "data/splits/data_splits.json"
CKPT    = BASE / "models/checkpoints/v3/attention_48ch_v1"
RESULTS = BASE / "results/v3/attention_48ch_v1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ Metrics
def brats_metrics_per_case(pred, target, smooth=1e-5):
    """pred, target: [H,W,D] integer tensors on CPU"""
    def dice_binary(p, t):
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float()
        if union == 0:
            return 1.0
        return ((2.0 * inter + smooth) / (union + smooth)).item()

    return {
        'WT': dice_binary(pred >= 1,           target >= 1),
        'TC': dice_binary((pred==1)|(pred==3), (target==1)|(target==3)),
        'ET': dice_binary(pred == 3,           target == 3),
    }


# ------------------------------------------------------------------ Evaluate one checkpoint
@torch.no_grad()
def evaluate_checkpoint(model, ckpt_path, test_loader, device, tag):
    logger.info(f"\nEvaluating: {ckpt_path} [{tag}]")

    ckpt       = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state']

    # Strip DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"  Epoch: {ckpt.get('epoch','?')} | "
                f"val_WT={ckpt.get('val_wt_dice',0):.4f} "
                f"val_TC={ckpt.get('val_tc_dice',0):.4f} "
                f"val_ET={ckpt.get('val_et_dice',0):.4f}")

    per_case = []
    tc_failures = 0

    for i, (imgs, lbls) in enumerate(test_loader):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.squeeze(0)          # [H,W,D]

        with autocast():
            outs = model(imgs)
            main = outs[0] if isinstance(outs, (list, tuple)) else outs

        pred    = torch.argmax(main.squeeze(0), dim=0).cpu()
        metrics = brats_metrics_per_case(pred, lbls.long())
        metrics['patient_id'] = f'case_{i:04d}'
        metrics['Mean']       = (metrics['WT'] + metrics['TC'] + metrics['ET']) / 3.0
        metrics['TC_failure'] = int(metrics['TC'] < 0.1)
        per_case.append(metrics)
        tc_failures += metrics['TC_failure']

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(test_loader)}")

    wt = np.mean([r['WT'] for r in per_case])
    tc = np.mean([r['TC'] for r in per_case])
    et = np.mean([r['ET'] for r in per_case])

    summary = {
        'tag':           tag,
        'epoch':         ckpt.get('epoch', '?'),
        'num_cases':     len(per_case),
        'WT_mean':       float(wt),
        'TC_mean':       float(tc),
        'ET_mean':       float(et),
        'Mean_dice':     float((wt + tc + et) / 3.0),
        'TC_failures':   tc_failures,
    }

    logger.info(f"\n  WT  = {wt:.4f}")
    logger.info(f"  TC  = {tc:.4f}")
    logger.info(f"  ET  = {et:.4f}")
    logger.info(f"  Mean= {(wt+tc+et)/3.0:.4f}")
    logger.info(f"  TC failures: {tc_failures}/{len(per_case)}")

    # Save CSV
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f'test_metrics_{tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id','WT','TC','ET','Mean','TC_failure'])
        writer.writeheader()
        writer.writerows(per_case)

    # Save JSON
    json_path = RESULTS / f'test_summary_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Saved: {csv_path}")
    logger.info(f"  Saved: {json_path}")
    return summary


# ------------------------------------------------------------------ Main
def main():
    logger.info("=" * 60)
    logger.info("48-Channel Attention U-Net — Test Set Evaluation")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Dataset
    test_ds = BraTSDatasetAugmented(
        str(DATA), str(SPLITS),
        split='test', augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )
    logger.info(f"Test cases: {len(test_ds)}")

    # Model — deep_supervision=True to match checkpoint
    model = AttentionUNet48Ch(
        in_channels=4, num_classes=4,
        base_channels=48, deep_supervision=True,
    ).to(device)

    # Evaluate both checkpoints
    checkpoints = [
        (CKPT / 'best_model.pth',      'best_model'),
        (CKPT / 'best_mean_model.pth', 'best_mean_model'),
    ]

    summaries = {}
    for ckpt_path, tag in checkpoints:
        if not ckpt_path.exists():
            logger.warning(f"Not found, skipping: {ckpt_path}")
            continue
        summaries[tag] = evaluate_checkpoint(model, ckpt_path, test_loader, device, tag)

    # Comparison table
    logger.info("\n" + "=" * 60)
    logger.info("48-CH MODEL — FINAL RESULTS")
    logger.info(f"{'Checkpoint':<20} {'WT':>8} {'TC':>8} {'ET':>8} {'Mean':>8} {'TC_fail':>8}")
    logger.info("-" * 60)
    for tag, s in summaries.items():
        logger.info(
            f"{tag:<20} {s['WT_mean']:>8.4f} {s['TC_mean']:>8.4f} "
            f"{s['ET_mean']:>8.4f} {s['Mean_dice']:>8.4f} {s['TC_failures']:>8d}"
        )
    logger.info("\nCurrent best (3-model ensemble):")
    logger.info("  WT=0.8629  TC=0.7833  ET=0.6858  Mean=0.7773  TC_fail=10")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
