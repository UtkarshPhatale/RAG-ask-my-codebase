#!/usr/bin/env python3
"""
eval_sliding_window.py
Sliding Window Inference on full 182x218x182 volumes.
Save to: scripts/advanced/eval_sliding_window.py

Instead of center-cropping to 128^3 at test time, this script:
  1. Loads the full volume (182x218x182)
  2. Runs overlapping 128^3 patches across the full volume
  3. Averages softmax probabilities in overlap regions (Gaussian weighting)
  4. Produces a full-resolution prediction

This recovers tumor regions near volume edges that center crop misses.

Usage:
  # Evaluate single model with sliding window
  python scripts/advanced/eval_sliding_window.py --model augmented

  # Evaluate 3-model ensemble with sliding window (best expected result)
  python scripts/advanced/eval_sliding_window.py --model ensemble3

  # Evaluate all single models
  python scripts/advanced/eval_sliding_window.py --model all
"""

import os, sys, json, csv
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import nibabel as nib

BASE = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts' / 'advanced'))

from attention_unet_augmented_FEB20th import get_augmented_model
from advanced_losses import ComboLoss

DATA   = BASE / "data/raw/brats_gli/training_data1_v2"
SPLITS = BASE / "data/splits/data_splits.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ Model configs
# Matches exactly how each checkpoint was saved
MODEL_CONFIGS = {
    'augmented': {
        'ckpt': BASE / 'models/checkpoints/v3/augmented/best_model.pth',
        'base_channels': 32, 'dropout_p': 0.1,
    },
    'percentile': {
        'ckpt': BASE / 'models/checkpoints/v3/percentile_norm/best_model.pth',
        'base_channels': 32, 'dropout_p': 0.1,
    },
    'hybrid': {
        'ckpt': BASE / 'models/checkpoints/v3/hybrid_norm/best_mean_model.pth',
        'base_channels': 32, 'dropout_p': 0.1,
    },
    '48ch': {
        'ckpt': BASE / 'models/checkpoints/v3/attention_48ch_v1/best_mean_model.pth',
        'base_channels': 48, 'dropout_p': 0.15,
    },
}

# Ensemble configs — (model_name, weight) pairs
ENSEMBLE_CONFIGS = {
    'ensemble3': [
        ('percentile', 1/3),
        ('hybrid',     1/3),
        ('augmented',  1/3),
    ],
    'ensemble4': [
        ('percentile', 0.25),
        ('hybrid',     0.25),
        ('augmented',  0.25),
        ('48ch',       0.25),
    ],
}


# ------------------------------------------------------------------ Load model
def load_model(model_name, device):
    cfg = MODEL_CONFIGS[model_name]
    ckpt_path = cfg['ckpt']

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if cfg['base_channels'] == 48:
        from attention_unet_48ch import AttentionUNet48Ch
        model = AttentionUNet48Ch(
            in_channels=4, num_classes=4,
            base_channels=48, deep_supervision=True,
            dropout_p=cfg['dropout_p'],
        )
    else:
        model = get_augmented_model(
            in_channels=4, num_classes=4,
            base_channels=cfg['base_channels'],
            deep_supervision=True,
            dropout_p=cfg['dropout_p'],
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state']
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    epoch  = ckpt.get('epoch', '?')
    val_wt = ckpt.get('val_wt_dice', 0)
    logger.info(f"  Loaded '{model_name}' | epoch={epoch} | val_WT={val_wt:.4f}")
    return model


# ------------------------------------------------------------------ Normalization
# Must match exactly what each model was trained with
def percentile_normalize(volume):
    brain = volume[volume > 0]
    if len(brain) == 0:
        return volume.astype(np.float32)
    p1, p99 = np.percentile(brain, 1), np.percentile(brain, 99)
    clipped = np.clip(volume, p1, p99)
    r = p99 - p1
    out = np.zeros_like(clipped)
    if r > 0:
        mask = volume > 0
        out[mask] = (clipped[mask] - p1) / r
    return out.astype(np.float32)

def zscore_normalize(volume):
    brain = volume[volume > 0]
    if len(brain) == 0:
        return volume.astype(np.float32)
    mu, sigma = brain.mean(), brain.std()
    out = np.zeros_like(volume, dtype=np.float32)
    if sigma > 0:
        mask = volume > 0
        out[mask] = (volume[mask] - mu) / sigma
    return out

def normalize_hybrid(modalities):
    """
    Hybrid normalization matching train_augmented (create_dataset_augmented):
    All 4 modalities use percentile norm (augmented dataset uses percentile for all).
    Order: t1n, t1c, t2w, t2f
    """
    return [percentile_normalize(m) for m in modalities]


# ------------------------------------------------------------------ Load full volume
def load_patient_full_volume(patient_id):
    """
    Load full 182x218x182 volume without any cropping.
    Returns: images [4, D, H, W] tensor, label [D, H, W] numpy array
    """
    # Find patient directory
    patient_dir = None
    exact = DATA / patient_id
    if exact.exists():
        patient_dir = exact
    else:
        for d in DATA.iterdir():
            if patient_id in d.name:
                patient_dir = d
                break

    if patient_dir is None:
        raise ValueError(f"Cannot find directory for patient: {patient_id}")

    suffixes = ['t1n', 't1c', 't2w', 't2f']
    modalities = []
    for suffix in suffixes:
        vol = None
        for pattern in [f"*{suffix}*.nii.gz", f"*{suffix}*.nii"]:
            files = list(patient_dir.glob(pattern))
            if files:
                vol = nib.load(files[0]).get_fdata().astype(np.float32)
                break
        if vol is None:
            raise ValueError(f"Cannot find {suffix} for {patient_id}")
        modalities.append(vol)

    # Normalize
    modalities = normalize_hybrid(modalities)

    # Load segmentation
    seg = None
    for pattern in ["*seg*.nii.gz", "*seg*.nii", "*mask*.nii.gz"]:
        files = list(patient_dir.glob(pattern))
        if files:
            seg = nib.load(files[0]).get_fdata().astype(np.int32)
            break
    if seg is None:
        raise ValueError(f"Cannot find segmentation for {patient_id}")

    # Remap label 4 -> 3 (ET)
    label = np.zeros_like(seg)
    label[seg == 1] = 1
    label[seg == 2] = 2
    label[seg == 4] = 3

    # Stack modalities: [4, D, H, W]
    image_tensor = torch.tensor(np.stack(modalities, axis=0), dtype=torch.float32)
    return image_tensor, label


# ------------------------------------------------------------------ Gaussian weight map
def make_gaussian_weight(patch_size):
    """
    3D Gaussian weight map for a patch — center gets weight ~1, edges get lower weight.
    This smooths out artifacts at patch boundaries when averaging overlapping predictions.
    """
    sigma = 0.125   # Controls falloff — 0.125 means edges are ~exp(-2) ≈ 0.14
    coords = [np.linspace(-0.5, 0.5, s) for s in patch_size]
    grids  = np.meshgrid(*coords, indexing='ij')
    dist   = sum(g**2 for g in grids)
    weight = np.exp(-dist / (2 * sigma**2))
    return torch.tensor(weight, dtype=torch.float32)


# ------------------------------------------------------------------ Sliding window
@torch.no_grad()
def sliding_window_inference(models_and_weights, image, device,
                              patch_size=(128,128,128), overlap=0.5):
    """
    Run inference over full volume using overlapping patches.

    models_and_weights: list of (model, weight) tuples
    image: [4, D, H, W] tensor
    overlap: fraction of patch size to overlap (0.5 = 50% overlap)

    Returns: [num_classes, D, H, W] averaged softmax probabilities
    """
    C, D, H, W = image.shape
    pd, ph, pw = patch_size
    num_classes = 4

    # Stride = patch_size * (1 - overlap)
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    # Accumulated probabilities and weights
    prob_map    = torch.zeros(num_classes, D, H, W, dtype=torch.float32)
    weight_map  = torch.zeros(1, D, H, W, dtype=torch.float32)
    gauss_w     = make_gaussian_weight(patch_size)  # [pd, ph, pw]

    # Compute patch start positions — ensure we always cover the full volume
    def get_starts(size, patch, stride):
        starts = list(range(0, size - patch + 1, stride))
        if not starts or starts[-1] + patch < size:
            starts.append(max(0, size - patch))
        return starts

    d_starts = get_starts(D, pd, sd)
    h_starts = get_starts(H, ph, sh)
    w_starts = get_starts(W, pw, sw)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    logger.debug(f"  Sliding window: {total_patches} patches "
                 f"({len(d_starts)}×{len(h_starts)}×{len(w_starts)})")

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                d1, h1, w1 = d0+pd, h0+ph, w0+pw

                patch = image[:, d0:d1, h0:h1, w0:w1].unsqueeze(0).to(device)

                # Accumulate weighted predictions from all models
                patch_probs = torch.zeros(num_classes, pd, ph, pw, dtype=torch.float32)

                for model, weight in models_and_weights:
                    with autocast():
                        outs = model(patch)
                        main = outs[0] if isinstance(outs, (list, tuple)) else outs
                    probs = F.softmax(main.squeeze(0), dim=0).cpu()
                    patch_probs += weight * probs

                # Add gaussian-weighted patch to accumulator
                gw = gauss_w.unsqueeze(0)   # [1, pd, ph, pw]
                prob_map[:, d0:d1, h0:h1, w0:w1]  += patch_probs * gw
                weight_map[:, d0:d1, h0:h1, w0:w1] += gw

    # Normalize by accumulated weights
    weight_map = weight_map.clamp(min=1e-8)
    prob_map   = prob_map / weight_map

    return prob_map   # [num_classes, D, H, W]


# ------------------------------------------------------------------ Metrics
def brats_metrics_per_case(pred, target, smooth=1e-5):
    """pred, target: [D,H,W] numpy integer arrays"""
    def dice_binary(p, t):
        inter = np.logical_and(p, t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            return 1.0
        return float((2.0 * inter + smooth) / (union + smooth))

    return {
        'WT': dice_binary(pred >= 1,                    target >= 1),
        'TC': dice_binary((pred==1)|(pred==3),          (target==1)|(target==3)),
        'ET': dice_binary(pred == 3,                    target == 3),
    }


# ------------------------------------------------------------------ Evaluate
def evaluate(model_spec, device, overlap=0.5):
    """
    model_spec: either a single model name string, or an ensemble name string
    """
    # Resolve models
    if model_spec in ENSEMBLE_CONFIGS:
        spec_list = ENSEMBLE_CONFIGS[model_spec]
        logger.info(f"Ensemble: {model_spec}")
        models_and_weights = []
        for name, w in spec_list:
            logger.info(f"  Loading {name} (weight={w:.3f})")
            m = load_model(name, device)
            models_and_weights.append((m, w))
        tag = model_spec
    else:
        logger.info(f"Single model: {model_spec}")
        m = load_model(model_spec, device)
        models_and_weights = [(m, 1.0)]
        tag = model_spec

    # Results dir
    results_dir = BASE / f'results/v3/{tag}_sliding_window'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test split
    with open(SPLITS) as f:
        splits = json.load(f)
    test_ids = splits['test']
    logger.info(f"Test patients: {len(test_ids)}")
    logger.info(f"Overlap: {overlap*100:.0f}%")

    per_case    = []
    tc_failures = 0

    for i, patient_id in enumerate(test_ids):
        try:
            image, label = load_patient_full_volume(patient_id)
        except Exception as e:
            logger.warning(f"  Skipping {patient_id}: {e}")
            continue

        prob_map = sliding_window_inference(
            models_and_weights, image, device,
            patch_size=(128, 128, 128), overlap=overlap
        )
        pred = torch.argmax(prob_map, dim=0).numpy()   # [D, H, W]

        metrics               = brats_metrics_per_case(pred, label)
        metrics['patient_id'] = patient_id
        metrics['Mean']       = (metrics['WT'] + metrics['TC'] + metrics['ET']) / 3.0
        metrics['TC_failure'] = int(metrics['TC'] < 0.1)
        per_case.append(metrics)
        tc_failures += metrics['TC_failure']

        if (i + 1) % 25 == 0:
            wt = np.mean([r['WT'] for r in per_case])
            tc = np.mean([r['TC'] for r in per_case])
            et = np.mean([r['ET'] for r in per_case])
            logger.info(
                f"  [{i+1:>3}/{len(test_ids)}] "
                f"WT={wt:.4f}  TC={tc:.4f}  ET={et:.4f}"
            )

    wt = np.mean([r['WT'] for r in per_case])
    tc = np.mean([r['TC'] for r in per_case])
    et = np.mean([r['ET'] for r in per_case])

    summary = {
        'tag':          tag,
        'inference':    'sliding_window',
        'overlap':      overlap,
        'num_cases':    len(per_case),
        'WT_mean':      float(wt),
        'TC_mean':      float(tc),
        'ET_mean':      float(et),
        'Mean_dice':    float((wt+tc+et)/3.0),
        'TC_failures':  tc_failures,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"SLIDING WINDOW RESULTS — {tag}")
    logger.info(f"{'='*50}")
    logger.info(f"WT   = {wt:.4f}")
    logger.info(f"TC   = {tc:.4f}")
    logger.info(f"ET   = {et:.4f}")
    logger.info(f"Mean = {(wt+tc+et)/3.0:.4f}")
    logger.info(f"TC failures: {tc_failures}/{len(per_case)}")
    logger.info(f"\nCurrent best (center crop, ensemble3):")
    logger.info(f"WT=0.8629  TC=0.7833  ET=0.6858  Mean=0.7773")

    # Save
    csv_path = results_dir / f'test_metrics_{tag}_sw.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['patient_id','WT','TC','ET','Mean','TC_failure'])
        writer.writeheader()
        writer.writerows(per_case)

    json_path = results_dir / f'test_summary_{tag}_sw.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved: {csv_path}")
    logger.info(f"Saved: {json_path}")
    return summary


# ------------------------------------------------------------------ Main
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ensemble3',
                        choices=list(MODEL_CONFIGS.keys()) +
                                list(ENSEMBLE_CONFIGS.keys()) + ['all'],
                        help='Model or ensemble to evaluate')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Patch overlap fraction (default 0.5 = 50%%)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Sliding Window Inference Evaluation")
    logger.info(f"Model: {args.model} | Overlap: {args.overlap*100:.0f}%")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if args.model == 'all':
        # Run single models first, then ensembles
        targets = list(MODEL_CONFIGS.keys()) + list(ENSEMBLE_CONFIGS.keys())
    else:
        targets = [args.model]

    all_summaries = {}
    for target in targets:
        logger.info(f"\n{'='*60}")
        try:
            s = evaluate(target, device, overlap=args.overlap)
            all_summaries[target] = s
        except Exception as e:
            logger.error(f"Failed on {target}: {e}")
            raise

    # Final table
    if len(all_summaries) > 1:
        logger.info("\n" + "="*60)
        logger.info("FINAL COMPARISON TABLE")
        logger.info(f"{'Model':<20} {'WT':>8} {'TC':>8} {'ET':>8} "
                    f"{'Mean':>8} {'TC_fail':>8}")
        logger.info("-"*60)
        for tag, s in all_summaries.items():
            logger.info(
                f"{tag:<20} {s['WT_mean']:>8.4f} {s['TC_mean']:>8.4f} "
                f"{s['ET_mean']:>8.4f} {s['Mean_dice']:>8.4f} "
                f"{s['TC_failures']:>8d}"
            )
        logger.info("\nBaseline (center crop, ensemble3):")
        logger.info("  WT=0.8629  TC=0.7833  ET=0.6858  Mean=0.7773  TC_fail=10")


if __name__ == '__main__':
    main()
