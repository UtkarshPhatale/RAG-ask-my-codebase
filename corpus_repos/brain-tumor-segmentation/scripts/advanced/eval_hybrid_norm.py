#!/usr/bin/env python3
"""
Evaluate both hybrid norm checkpoints:
1. best_model.pth      - best Val WT (epoch 36)
2. best_mean_model.pth - best Val Mean (epoch 73)
"""

import sys, json, csv, torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/advanced')

from attention_unet_v2_improved import ImprovedAttentionUNet3D
from create_dataset_hybrid_norm import BraTSDatasetHybridNorm

BASE_DIR    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA_DIR    = BASE_DIR / "data/raw/brats_gli/training_data1_v2"
SPLITS_FILE = BASE_DIR / "data/splits/data_splits.json"
CKPT_DIR    = BASE_DIR / "models/checkpoints/v3/hybrid_norm"
RESULTS_DIR = BASE_DIR / "results/v3/hybrid_norm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline and previous best for comparison
BASELINE = {'wt': 0.8242, 'tc': 0.5931, 'et': 0.7878}
PERCENTILE = {'wt': 0.8439, 'tc': 0.7493, 'et': 0.6551}

def compute_dice(pred, target, smooth=1e-5):
    results = {}
    for name, p, t in [
        ('wt', pred >= 1,          target >= 1),
        ('tc', (pred==1)|(pred==3), (target==1)|(target==3)),
        ('et', pred == 3,           target == 3),
    ]:
        p, t = p.float(), t.float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        results[f'{name}_dice'] = ((2*inter+smooth)/(union+smooth)).item()

    for c, name in [(1,'ncr'),(2,'ed'),(3,'et_class')]:
        p = (pred==c).float(); t = (target==c).float()
        i = (p*t).sum(); u = p.sum()+t.sum()
        results[f'{name}_dice'] = ((2*i+smooth)/(u+smooth)).item()

    return results

def evaluate_checkpoint(ckpt_path, label, device, test_ds, test_loader):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Saved at epoch {ckpt.get('epoch','?')} | "
          f"Val WT={ckpt.get('val_wt_dice',0):.4f} | "
          f"Val TC={ckpt.get('val_tc_dice',0):.4f} | "
          f"Val ET={ckpt.get('val_et_dice',0):.4f}")

    model = ImprovedAttentionUNet3D(
        in_channels=4, num_classes=4,
        base_channels=32, deep_supervision=True
    )
    state = {k.replace('module.',''): v
             for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    all_metrics, wt_l, tc_l, et_l = [], [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                out = model(images)
                main = out[0] if isinstance(out,(list,tuple)) else out
            pred = main.argmax(dim=1)
            s = compute_dice(pred[0], labels[0])
            s['patient_id'] = test_ds.patient_ids[i]
            all_metrics.append(s)
            wt_l.append(s['wt_dice'])
            tc_l.append(s['tc_dice'])
            et_l.append(s['et_dice'])

            if (i+1) % 40 == 0:
                print(f"  {i+1}/204 | WT={np.mean(wt_l):.4f} "
                      f"TC={np.mean(tc_l):.4f} ET={np.mean(et_l):.4f}")

    wt = np.array(wt_l); tc = np.array(tc_l); et = np.array(et_l)
    tc_fail = (tc < 0.1).sum()
    mean_dice = (wt.mean() + tc.mean() + et.mean()) / 3

    print(f"\n--- TEST RESULTS: {label} ---")
    print(f"WT: {wt.mean():.4f}±{wt.std():.4f}  "
          f"(baseline:{BASELINE['wt']} pct:{PERCENTILE['wt']} "
          f"change:{wt.mean()-BASELINE['wt']:+.4f})")
    print(f"TC: {tc.mean():.4f}±{tc.std():.4f}  "
          f"(baseline:{BASELINE['tc']} pct:{PERCENTILE['tc']} "
          f"change:{tc.mean()-BASELINE['tc']:+.4f})")
    print(f"ET: {et.mean():.4f}±{et.std():.4f}  "
          f"(baseline:{BASELINE['et']} pct:{PERCENTILE['et']} "
          f"change:{et.mean()-BASELINE['et']:+.4f})")
    print(f"Mean: {mean_dice:.4f}")
    print(f"TC failures: {tc_fail} (baseline:83, percentile:9)")

    # Save CSV
    tag = ckpt_path.stem
    csv_path = RESULTS_DIR / f'test_metrics_{tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader(); writer.writerows(all_metrics)

    summary = {
        'label': label,
        'checkpoint': ckpt_path.name,
        'epoch': ckpt.get('epoch'),
        'wt_mean': float(wt.mean()), 'wt_std': float(wt.std()),
        'tc_mean': float(tc.mean()), 'tc_std': float(tc.std()),
        'et_mean': float(et.mean()), 'et_std': float(et.std()),
        'mean_dice': float(mean_dice),
        'tc_failures': int(tc_fail),
        'wt_vs_baseline': float(wt.mean()-BASELINE['wt']),
        'tc_vs_baseline': float(tc.mean()-BASELINE['tc']),
        'et_vs_baseline': float(et.mean()-BASELINE['et']),
    }
    json_path = RESULTS_DIR / f'test_summary_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {csv_path.name}, {json_path.name}")
    return summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_ds = BraTSDatasetHybridNorm(
        DATA_DIR, SPLITS_FILE, split='test', augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    results = []
    for ckpt_name, label in [
        ('best_model.pth',      'Best WT (epoch 36)'),
        ('best_mean_model.pth', 'Best Mean (epoch 73)'),
    ]:
        ckpt_path = CKPT_DIR / ckpt_name
        if ckpt_path.exists():
            r = evaluate_checkpoint(ckpt_path, label, device, test_ds, test_loader)
            results.append(r)
        else:
            print(f"WARNING: {ckpt_name} not found, skipping")

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'WT':>7} {'TC':>7} {'ET':>7} {'Mean':>7} {'TC Fail':>8}")
    print("-"*60)
    print(f"{'Baseline':25} {BASELINE['wt']:>7.4f} {BASELINE['tc']:>7.4f} "
          f"{BASELINE['et']:>7.4f} {'N/A':>7} {'83':>8}")
    print(f"{'Percentile Norm':25} {'0.8439':>7} {'0.7493':>7} "
          f"{'0.6551':>7} {'0.7161':>7} {'9':>8}")
    for r in results:
        print(f"{r['label']:<25} {r['wt_mean']:>7.4f} {r['tc_mean']:>7.4f} "
              f"{r['et_mean']:>7.4f} {r['mean_dice']:>7.4f} {r['tc_failures']:>8}")

if __name__ == '__main__':
    main()
