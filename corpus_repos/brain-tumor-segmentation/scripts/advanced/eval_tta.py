"""
eval_tta.py
Test-Time Augmentation (TTA) Evaluation
Save to: scripts/advanced/eval_tta.py

Usage:
  python scripts/advanced/eval_tta.py --model augmented --checkpoint best_model
  python scripts/advanced/eval_tta.py --model percentile --checkpoint best_model
  python scripts/advanced/eval_tta.py --model hybrid --checkpoint best_mean_model
  python scripts/advanced/eval_tta.py --model 48ch --checkpoint best_model
  python scripts/advanced/eval_tta.py --model all --checkpoint best_model
"""

import os
import sys
import json
import csv
import argparse
import logging
import numpy as np
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# ------------------------------------------------------------------ Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'advanced'))

from attention_unet_augmented_FEB20th import get_augmented_model
from create_dataset_augmented import BraTSDatasetAugmented

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ Config
DATA_DIR   = str(PROJECT_ROOT / 'data' / 'raw' / 'brats_gli' / 'training_data1_v2')
SPLIT_FILE = str(PROJECT_ROOT / 'data' / 'splits' / 'data_splits.json')
USE_AMP    = True

MODEL_REGISTRY = {
    'augmented': {
        'checkpoint_dir': 'models/checkpoints/v3/augmented',
        'base_channels': 32,
        'dropout_p': 0.1,
        'type': '32ch',
    },
    'percentile': {
        'checkpoint_dir': 'models/checkpoints/v3/percentile_norm',
        'base_channels': 32,
        'dropout_p': 0.1,
        'type': '32ch',
    },
    'hybrid': {
        'checkpoint_dir': 'models/checkpoints/v3/hybrid_norm',
        'base_channels': 32,
        'dropout_p': 0.1,
        'type': '32ch',
    },
    '48ch': {
        'checkpoint_dir': 'models/checkpoints/v3/attention_48ch_v1',
        'base_channels': 48,
        'dropout_p': 0.15,
        'type': '48ch',
    },
}


# ------------------------------------------------------------------ TTA Transforms
def get_tta_transforms():
    """
    8 flip-based transforms: identity + 3 single-axis + 3 two-axis + 1 triple-axis.
    Each is (forward_fn, inverse_fn) — inverse is identical to forward for flips.
    Spatial dims for a [B, C, H, W, D] tensor are 2, 3, 4.
    """
    transforms = []

    # Identity
    transforms.append((lambda x: x, lambda x: x))

    # Single-axis flips
    for ax in [2, 3, 4]:
        transforms.append((
            lambda x, a=ax: torch.flip(x, [a]),
            lambda x, a=ax: torch.flip(x, [a]),
        ))

    # Two-axis flips
    for ax1, ax2 in combinations([2, 3, 4], 2):
        transforms.append((
            lambda x, a1=ax1, a2=ax2: torch.flip(x, [a1, a2]),
            lambda x, a1=ax1, a2=ax2: torch.flip(x, [a1, a2]),
        ))

    # Triple-axis flip
    transforms.append((
        lambda x: torch.flip(x, [2, 3, 4]),
        lambda x: torch.flip(x, [2, 3, 4]),
    ))

    return transforms  # 8 total


# ------------------------------------------------------------------ TTA Inference
@torch.no_grad()
def predict_with_tta(model, images, device):
    """
    Average softmax probabilities across 8 flip orientations.
    images: [1, 4, H, W, D]
    Returns: averaged probs [1, num_classes, H, W, D]
    """
    transforms = get_tta_transforms()
    accumulated_probs = None

    for fwd_fn, inv_fn in transforms:
        x_aug = fwd_fn(images)

        with autocast(enabled=USE_AMP):
            outputs = model(x_aug)
            # Model returns tuple (main, dsv1, dsv2, dsv3) or just main tensor
            if isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
            else:
                main_out = outputs

        probs     = F.softmax(main_out, dim=1)
        probs_inv = inv_fn(probs)

        if accumulated_probs is None:
            accumulated_probs = probs_inv
        else:
            accumulated_probs = accumulated_probs + probs_inv

    return accumulated_probs / len(transforms)


# ------------------------------------------------------------------ Metrics
def brats_metrics_per_case(pred, target, smooth=1e-5):
    """
    pred, target: [H, W, D] integer tensors on CPU.
    Returns dict with WT, TC, ET Dice.
    """
    def dice_binary(p, t):
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float()
        if union == 0:
            return 1.0   # Both empty = perfect (BraTS convention)
        return ((2.0 * inter + smooth) / (union + smooth)).item()

    pred_wt = pred >= 1
    tgt_wt  = target >= 1
    pred_tc = (pred == 1) | (pred == 3)
    tgt_tc  = (target == 1) | (target == 3)
    pred_et = pred == 3
    tgt_et  = target == 3

    return {
        'WT': dice_binary(pred_wt, tgt_wt),
        'TC': dice_binary(pred_tc, tgt_tc),
        'ET': dice_binary(pred_et, tgt_et),
    }


# ------------------------------------------------------------------ Load Model
def load_model(model_name, checkpoint_name, device):
    """
    Load model with deep_supervision=True (matches how checkpoint was saved),
    then set eval mode so dsv layers are present but forward() only returns main output.
    DataParallel 'module.' prefix is stripped automatically.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY.keys())}")

    cfg      = MODEL_REGISTRY[model_name]
    ckpt_path = os.path.join(PROJECT_ROOT, cfg['checkpoint_dir'], f'{checkpoint_name}.pth')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- Instantiate model ----
    # IMPORTANT: deep_supervision=True so dsv3/dsv2/dsv1 layers exist in state dict
    if cfg['type'] == '48ch':
        from attention_unet_48ch import AttentionUNet48Ch
        model = AttentionUNet48Ch(
            in_channels=4,
            num_classes=4,
            base_channels=48,
            deep_supervision=True,   # must match checkpoint
            dropout_p=cfg['dropout_p'],
        )
    else:
        model = get_augmented_model(
            in_channels=4,
            num_classes=4,
            base_channels=cfg['base_channels'],
            deep_supervision=True,   # must match checkpoint
            dropout_p=cfg['dropout_p'],
        )

    # ---- Load weights ----
    ckpt = torch.load(ckpt_path, map_location=device)

    # Try both key names
    state_dict = ckpt.get('model_state', ckpt.get('model_state_dict', None))
    if state_dict is None:
        raise KeyError(f"No model weights found. Keys in checkpoint: {list(ckpt.keys())}")

    # Strip DataParallel 'module.' prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        logger.info("  Stripping DataParallel 'module.' prefix from checkpoint")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    # ---- Eval mode ----
    # AugmentedAttentionUNet3D.forward() uses 'if not self.training: return super().forward(x)'
    # which returns only the main output — no dsv outputs. Perfect for TTA.
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val_wt = ckpt.get('val_wt_dice', ckpt.get('val_metrics', {}).get('WT', '?'))
    logger.info(f"  Loaded '{model_name}' | epoch={epoch} | val_WT={val_wt}")
    return model


# ------------------------------------------------------------------ Run Evaluation
@torch.no_grad()
def evaluate_with_tta(model, test_loader, device, results_dir, tag):
    os.makedirs(results_dir, exist_ok=True)

    per_case_results = []
    tc_failures      = 0

    for batch_idx, batch in enumerate(test_loader):
        # Handle dict or tuple batch format
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1]
        else:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label']

        # Squeeze label to [H, W, D]
        while labels.dim() > 3:
            labels = labels.squeeze(0)

        patient_id = f'case_{batch_idx:04d}'
        if isinstance(batch, dict) and 'patient_id' in batch:
            patient_id = batch['patient_id'][0]

        # TTA inference
        avg_probs = predict_with_tta(model, images, device)
        pred      = torch.argmax(avg_probs.squeeze(0), dim=0).cpu()  # [H, W, D]

        # Metrics
        metrics               = brats_metrics_per_case(pred, labels.long())
        metrics['patient_id'] = patient_id
        metrics['Mean']       = (metrics['WT'] + metrics['TC'] + metrics['ET']) / 3.0
        metrics['TC_failure'] = int(metrics['TC'] < 0.1)

        per_case_results.append(metrics)
        tc_failures += metrics['TC_failure']

        # Progress log every 25 cases
        if (batch_idx + 1) % 25 == 0:
            wt_now = np.mean([r['WT'] for r in per_case_results])
            tc_now = np.mean([r['TC'] for r in per_case_results])
            et_now = np.mean([r['ET'] for r in per_case_results])
            logger.info(
                f"  [{batch_idx+1:>3}/{len(test_loader)}] "
                f"Running avg  WT={wt_now:.4f}  TC={tc_now:.4f}  ET={et_now:.4f}"
            )

    # ---- Aggregate ----
    wt_scores = [r['WT'] for r in per_case_results]
    tc_scores = [r['TC'] for r in per_case_results]
    et_scores = [r['ET'] for r in per_case_results]

    summary = {
        'tag':           tag,
        'num_test_cases': len(per_case_results),
        'WT_mean':       float(np.mean(wt_scores)),
        'WT_std':        float(np.std(wt_scores)),
        'TC_mean':       float(np.mean(tc_scores)),
        'TC_std':        float(np.std(tc_scores)),
        'ET_mean':       float(np.mean(et_scores)),
        'ET_std':        float(np.std(et_scores)),
        'Mean_dice':     float(np.mean([np.mean(wt_scores), np.mean(tc_scores), np.mean(et_scores)])),
        'TC_failures':   tc_failures,
        'TTA_transforms': 8,
    }

    logger.info(f"\n  Results [{tag}]  (TTA — 8 flip orientations):")
    logger.info(f"  WT   = {summary['WT_mean']:.4f}  +/-  {summary['WT_std']:.4f}")
    logger.info(f"  TC   = {summary['TC_mean']:.4f}  +/-  {summary['TC_std']:.4f}")
    logger.info(f"  ET   = {summary['ET_mean']:.4f}  +/-  {summary['ET_std']:.4f}")
    logger.info(f"  Mean = {summary['Mean_dice']:.4f}")
    logger.info(f"  TC failures (TC<0.1): {tc_failures} / {len(per_case_results)}")

    # ---- Save CSV ----
    csv_path = os.path.join(results_dir, f'test_metrics_{tag}_tta.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'WT', 'TC', 'ET', 'Mean', 'TC_failure'])
        writer.writeheader()
        writer.writerows(per_case_results)

    # ---- Save JSON ----
    json_path = os.path.join(results_dir, f'test_summary_{tag}_tta.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  CSV  saved: {csv_path}")
    logger.info(f"  JSON saved: {json_path}")
    return summary


# ------------------------------------------------------------------ Main
def main():
    parser = argparse.ArgumentParser(description='TTA Evaluation')
    parser.add_argument('--model',      type=str, default='augmented',
                        choices=list(MODEL_REGISTRY.keys()) + ['all'],
                        help='Which model to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best_model',
                        choices=['best_model', 'best_mean_model'],
                        help='Which checkpoint file to load')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Test-Time Augmentation (TTA) Evaluation")
    logger.info(f"Model: {args.model} | Checkpoint: {args.checkpoint}")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ---- Dataset ----
    test_dataset = BraTSDatasetAugmented(
        data_dir=DATA_DIR,
        split_file=SPLIT_FILE,
        split='test',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,    # Reduced from 4 — cluster warning said max=1, 2 is safe
        pin_memory=True,
    )
    logger.info(f"Test cases: {len(test_dataset)}")

    # ---- Models to evaluate ----
    models_to_eval = list(MODEL_REGISTRY.keys()) if args.model == 'all' else [args.model]

    all_summaries = {}
    for model_name in models_to_eval:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating: {model_name} [{args.checkpoint}]")
        logger.info(f"{'='*40}")
        try:
            model   = load_model(model_name, args.checkpoint, device)
            res_dir = str(PROJECT_ROOT / 'results' / 'v3' / f'{model_name}_tta')
            tag     = f'{model_name}_{args.checkpoint}'
            summary = evaluate_with_tta(model, test_loader, device, res_dir, tag)
            all_summaries[tag] = summary
        except FileNotFoundError as e:
            logger.warning(f"Skipping {model_name}: {e}")
        except Exception as e:
            logger.error(f"Error on {model_name}: {e}")
            raise

    # ---- Final comparison table ----
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TTA RESULTS SUMMARY")
    logger.info(f"{'Model':<35} {'WT':>8} {'TC':>8} {'ET':>8} {'Mean':>8} {'TC_fail':>8}")
    logger.info("-" * 60)
    for tag, s in all_summaries.items():
        logger.info(
            f"{tag:<35} {s['WT_mean']:>8.4f} {s['TC_mean']:>8.4f} "
            f"{s['ET_mean']:>8.4f} {s['Mean_dice']:>8.4f} {s['TC_failures']:>8d}"
        )
    logger.info("\nCurrent best without TTA (3-model ensemble):")
    logger.info("  WT=0.8629  TC=0.7833  ET=0.6858  Mean=0.7773  TC_fail=10")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
