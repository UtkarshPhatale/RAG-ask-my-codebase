#!/usr/bin/env python3
"""
Weighted Ensemble: Percentile Norm + Hybrid Norm (Best Mean)
=============================================================
Strategy: Average softmax probabilities from both models.
Each model was trained with different normalization, so they
make different errors — ensemble corrects for both.
"""

import sys, json, csv, torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/advanced')

from attention_unet_v2_improved import ImprovedAttentionUNet3D
from create_dataset_percentile import BraTSDatasetPercentile
from create_dataset_hybrid_norm import BraTSDatasetHybridNorm

BASE_DIR    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA_DIR    = BASE_DIR / "data/raw/brats_gli/training_data1_v2"
SPLITS_FILE = BASE_DIR / "data/splits/data_splits.json"
RESULTS_DIR = BASE_DIR / "results/v3/ensemble"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PCT    = BASE_DIR / "models/checkpoints/v3/percentile_norm/best_model.pth"
CKPT_HYB    = BASE_DIR / "models/checkpoints/v3/hybrid_norm/best_mean_model.pth"

BASELINE   = {'wt': 0.8242, 'tc': 0.5931, 'et': 0.7878}
PERCENTILE = {'wt': 0.8439, 'tc': 0.7493, 'et': 0.6551}
HYBRID     = {'wt': 0.8420, 'tc': 0.7737, 'et': 0.6769}


def compute_dice(pred, target, smooth=1e-5):
    out = {}
    for name, p, t in [
        ('wt', pred >= 1,           target >= 1),
        ('tc', (pred==1)|(pred==3), (target==1)|(target==3)),
        ('et', pred == 3,           target == 3),
    ]:
        p, t = p.float(), t.float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        out[f'{name}_dice'] = ((2*inter+smooth)/(union+smooth)).item()
    return out


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ImprovedAttentionUNet3D(
        in_channels=4, num_classes=4,
        base_channels=32, deep_supervision=True
    )
    state = {k.replace('module.',''): v
             for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    wt    = ckpt.get('val_wt_dice', 0)
    print(f"  Loaded epoch={epoch}, val_WT={wt:.4f}")
    return model


def run_ensemble(weight_pct, weight_hyb, device,
                 model_pct, model_hyb,
                 pct_loader, hyb_loader,
                 pct_ds, label):
    """Run inference with weighted average of softmax outputs."""
    print(f"\n--- {label}  (w_pct={weight_pct:.2f}, w_hyb={weight_hyb:.2f}) ---")

    wt_l, tc_l, et_l = [], [], []
    all_metrics = []

    pct_iter = iter(pct_loader)
    hyb_iter = iter(hyb_loader)

    with torch.no_grad():
        for i in range(204):
            pct_imgs, labels = next(pct_iter)
            hyb_imgs, _      = next(hyb_iter)

            pct_imgs = pct_imgs.to(device)
            hyb_imgs = hyb_imgs.to(device)
            labels   = labels.to(device)

            with autocast():
                out_pct = model_pct(pct_imgs)
                out_hyb = model_hyb(hyb_imgs)
                main_pct = out_pct[0] if isinstance(out_pct,(list,tuple)) else out_pct
                main_hyb = out_hyb[0] if isinstance(out_hyb,(list,tuple)) else out_hyb

            # Weighted average of softmax probabilities
            prob_pct = torch.softmax(main_pct, dim=1)
            prob_hyb = torch.softmax(main_hyb, dim=1)
            prob_ens = weight_pct * prob_pct + weight_hyb * prob_hyb
            pred = prob_ens.argmax(dim=1)

            s = compute_dice(pred[0], labels[0])
            s['patient_id'] = pct_ds.patient_ids[i]
            all_metrics.append(s)
            wt_l.append(s['wt_dice'])
            tc_l.append(s['tc_dice'])
            et_l.append(s['et_dice'])

            if (i+1) % 40 == 0:
                print(f"  {i+1}/204 | WT={np.mean(wt_l):.4f} "
                      f"TC={np.mean(tc_l):.4f} ET={np.mean(et_l):.4f}")

    wt = np.array(wt_l); tc = np.array(tc_l); et = np.array(et_l)
    mean_dice = (wt.mean() + tc.mean() + et.mean()) / 3
    tc_fail   = (tc < 0.1).sum()

    print(f"\nWT: {wt.mean():.4f}±{wt.std():.4f}  "
          f"(vs percentile: {wt.mean()-PERCENTILE['wt']:+.4f})")
    print(f"TC: {tc.mean():.4f}±{tc.std():.4f}  "
          f"(vs percentile: {tc.mean()-PERCENTILE['tc']:+.4f})")
    print(f"ET: {et.mean():.4f}±{et.std():.4f}  "
          f"(vs baseline:   {et.mean()-BASELINE['et']:+.4f})")
    print(f"Mean: {mean_dice:.4f}  TC failures: {tc_fail}")

    # Save
    tag = label.lower().replace('/','').replace(' ','_').replace('(','').replace(')','').replace('=','').replace('.','') #().replace(' ','_').replace('(','').replace(')','').replace('=','').replace('.','')
    csv_path = RESULTS_DIR / f'ensemble_{tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader(); writer.writerows(all_metrics)

    return {
        'label': label,
        'w_pct': weight_pct, 'w_hyb': weight_hyb,
        'wt': float(wt.mean()), 'tc': float(tc.mean()),
        'et': float(et.mean()), 'mean': float(mean_dice),
        'tc_fail': int(tc_fail),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading models...")
    print("Percentile model:")
    model_pct = load_model(CKPT_PCT, device)
    print("Hybrid mean model:")
    model_hyb = load_model(CKPT_HYB, device)

    # Two separate loaders — different normalization per model
    pct_ds = BraTSDatasetPercentile(
        DATA_DIR, SPLITS_FILE, split='test', augment=False
    )
    hyb_ds = BraTSDatasetHybridNorm(
        DATA_DIR, SPLITS_FILE, split='test', augment=False
    )
    pct_loader = DataLoader(pct_ds, batch_size=1, shuffle=False, num_workers=2)
    hyb_loader = DataLoader(hyb_ds, batch_size=1, shuffle=False, num_workers=2)

    # Test three weightings
    configs = [
        (0.5, 0.5,  "Equal (0.5/0.5)"),
        (0.6, 0.4,  "PCT-heavy (0.6/0.4)"),
        (0.4, 0.6,  "HYB-heavy (0.4/0.6)"),
    ]

    all_results = []
    for w_pct, w_hyb, label in configs:
        r = run_ensemble(w_pct, w_hyb, device,
                         model_pct, model_hyb,
                         pct_loader, hyb_loader,
                         pct_ds, label)
        all_results.append(r)

    # Final table
    print(f"\n{'='*70}")
    print("ENSEMBLE COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'WT':>7} {'TC':>7} {'ET':>7} {'Mean':>7} {'TCFail':>7}")
    print("-"*70)
    print(f"{'Baseline':<28} {BASELINE['wt']:>7.4f} {BASELINE['tc']:>7.4f} "
          f"{BASELINE['et']:>7.4f} {'---':>7} {'83':>7}")
    print(f"{'Percentile Norm':<28} {PERCENTILE['wt']:>7.4f} {PERCENTILE['tc']:>7.4f} "
          f"{PERCENTILE['et']:>7.4f} {'0.7161':>7} {'9':>7}")
    print(f"{'Hybrid Mean (ep73)':<28} {HYBRID['wt']:>7.4f} {HYBRID['tc']:>7.4f} "
          f"{HYBRID['et']:>7.4f} {'0.7642':>7} {'12':>7}")
    for r in all_results:
        print(f"{r['label']:<28} {r['wt']:>7.4f} {r['tc']:>7.4f} "
              f"{r['et']:>7.4f} {r['mean']:>7.4f} {r['tc_fail']:>7}")

    with open(RESULTS_DIR / 'ensemble_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/v3/ensemble/")


if __name__ == '__main__':
    main()
