#!/usr/bin/env python3
"""
Three-Model Ensemble:
  Model A: Percentile Norm     (ep56)  - strong WT boundary
  Model B: Hybrid Mean         (ep73)  - balanced TC/ET
  Model C: Augmented           (ep105) - strongest overall

Tests several weighting strategies.
"""

import sys, json, csv, torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/advanced')

from attention_unet_v2_improved import ImprovedAttentionUNet3D
from attention_unet_augmented_FEB20th import get_augmented_model
from create_dataset_percentile import BraTSDatasetPercentile
from create_dataset_hybrid_norm import BraTSDatasetHybridNorm
from create_dataset_augmented import BraTSDatasetAugmented

BASE    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA    = BASE / "data/raw/brats_gli/training_data1_v2"
SPLITS  = BASE / "data/splits/data_splits.json"
RESULTS = BASE / "results/v3/ensemble_three"
RESULTS.mkdir(parents=True, exist_ok=True)

CKPT_PCT = BASE / "models/checkpoints/v3/percentile_norm/best_model.pth"
CKPT_HYB = BASE / "models/checkpoints/v3/hybrid_norm/best_mean_model.pth"
CKPT_AUG = BASE / "models/checkpoints/v3/augmented/best_model.pth"

PREV = {
    'baseline':       {'wt':0.8242,'tc':0.5931,'et':0.7878},
    'ensemble_equal': {'wt':0.8488,'tc':0.7744,'et':0.6778},
    'augmented':      {'wt':0.8613,'tc':0.7807,'et':0.6800},
}

def compute_dice(pred, target, smooth=1e-5):
    out = {}
    for name, p, t in [
        ('wt', pred>=1,            target>=1),
        ('tc', (pred==1)|(pred==3),(target==1)|(target==3)),
        ('et', pred==3,            target==3),
    ]:
        p,t = p.float(),t.float()
        i=(p*t).sum(); u=p.sum()+t.sum()
        out[f'{name}_dice'] = ((2*i+smooth)/(u+smooth)).item()
    return out

def load_standard_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ImprovedAttentionUNet3D(
        in_channels=4, num_classes=4,
        base_channels=32, deep_supervision=True
    )
    state = {k.replace('module.',''):v for k,v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded {ckpt_path.name}: epoch={ckpt.get('epoch','?')} "
          f"val_WT={ckpt.get('val_wt_dice',0):.4f}")
    return model

def load_augmented_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = get_augmented_model(
        in_channels=4, num_classes=4,
        base_channels=32, deep_supervision=True, dropout_p=0.1
    )
    state = {k.replace('module.',''):v for k,v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded {ckpt_path.name}: epoch={ckpt.get('epoch','?')} "
          f"val_WT={ckpt.get('val_wt_dice',0):.4f}")
    return model

def run_ensemble(wa, wb, wc, label,
                 model_pct, model_hyb, model_aug,
                 pct_loader, hyb_loader, aug_loader,
                 pct_ds, device):
    print(f"\n--- {label} (pct={wa:.2f} hyb={wb:.2f} aug={wc:.2f}) ---")
    wt_l,tc_l,et_l,all_metrics = [],[],[],[]

    iters = [iter(pct_loader), iter(hyb_loader), iter(aug_loader)]

    with torch.no_grad():
        for i in range(204):
            pct_imgs, labels = next(iters[0])
            hyb_imgs, _      = next(iters[1])
            aug_imgs, _      = next(iters[2])

            labels   = labels.to(device)
            pct_imgs = pct_imgs.to(device)
            hyb_imgs = hyb_imgs.to(device)
            aug_imgs = aug_imgs.to(device)

            with autocast():
                out_pct = model_pct(pct_imgs)
                out_hyb = model_hyb(hyb_imgs)
                out_aug = model_aug(aug_imgs)
                m_pct = out_pct[0] if isinstance(out_pct,(list,tuple)) else out_pct
                m_hyb = out_hyb[0] if isinstance(out_hyb,(list,tuple)) else out_hyb
                m_aug = out_aug[0] if isinstance(out_aug,(list,tuple)) else out_aug

            prob = (wa * torch.softmax(m_pct, dim=1) +
                    wb * torch.softmax(m_hyb, dim=1) +
                    wc * torch.softmax(m_aug, dim=1))
            pred = prob.argmax(dim=1)

            s = compute_dice(pred[0], labels[0])
            s['patient_id'] = pct_ds.patient_ids[i]
            all_metrics.append(s)
            wt_l.append(s['wt_dice'])
            tc_l.append(s['tc_dice'])
            et_l.append(s['et_dice'])

            if (i+1) % 50 == 0:
                print(f"  {i+1}/204 | WT={np.mean(wt_l):.4f} "
                      f"TC={np.mean(tc_l):.4f} ET={np.mean(et_l):.4f}")

    wt=np.array(wt_l); tc=np.array(tc_l); et=np.array(et_l)
    mean_dice = (wt.mean()+tc.mean()+et.mean())/3
    tc_fail   = (tc<0.1).sum()

    print(f"  WT:{wt.mean():.4f} TC:{tc.mean():.4f} ET:{et.mean():.4f} "
          f"Mean:{mean_dice:.4f} TCfail:{tc_fail}")

    tag = f"w{int(wa*10)}{int(wb*10)}{int(wc*10)}"
    with open(RESULTS/f'ensemble_{tag}.csv','w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader(); w.writerows(all_metrics)

    return {'label':label,'wa':wa,'wb':wb,'wc':wc,
            'wt':float(wt.mean()),'tc':float(tc.mean()),
            'et':float(et.mean()),'mean':float(mean_dice),
            'tc_fail':int(tc_fail)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("Loading models...")
    model_pct = load_standard_model(CKPT_PCT, device)
    model_hyb = load_standard_model(CKPT_HYB, device)
    model_aug = load_augmented_model(CKPT_AUG, device)

    # Three datasets with their respective normalizations
    pct_ds = BraTSDatasetPercentile(DATA, SPLITS, split='test', augment=False)
    hyb_ds = BraTSDatasetHybridNorm(DATA, SPLITS, split='test', augment=False)
    aug_ds = BraTSDatasetAugmented( DATA, SPLITS, split='test', augment=False)

    pct_loader = DataLoader(pct_ds, batch_size=1, shuffle=False, num_workers=2)
    hyb_loader = DataLoader(hyb_ds, batch_size=1, shuffle=False, num_workers=2)
    aug_loader = DataLoader(aug_ds, batch_size=1, shuffle=False, num_workers=2)

    # Weighting strategies to test
    configs = [
        (0.20, 0.20, 0.60, "Aug-dominant  (0.2/0.2/0.6)"),
        (0.25, 0.25, 0.50, "Aug-heavy     (0.25/0.25/0.5)"),
        (0.33, 0.33, 0.34, "Equal         (0.33/0.33/0.34)"),
        (0.20, 0.30, 0.50, "Hyb+Aug heavy (0.2/0.3/0.5)"),
        (0.15, 0.25, 0.60, "Aug+Hyb focus (0.15/0.25/0.6)"),
    ]

    results = []
    for wa,wb,wc,label in configs:
        r = run_ensemble(wa,wb,wc,label,
                         model_pct, model_hyb, model_aug,
                         pct_loader, hyb_loader, aug_loader,
                         pct_ds, device)
        results.append(r)

    print(f"\n{'='*72}")
    print("THREE-MODEL ENSEMBLE — FINAL COMPARISON")
    print(f"{'='*72}")
    print(f"{'Model':<30} {'WT':>7} {'TC':>7} {'ET':>7} {'Mean':>7} {'TCFail':>7}")
    print("-"*72)
    for name,v in PREV.items():
        mean=(v['wt']+v['tc']+v['et'])/3
        print(f"{name:<30} {v['wt']:>7.4f} {v['tc']:>7.4f} "
              f"{v['et']:>7.4f} {mean:>7.4f}")
    print("-"*72)
    for r in results:
        print(f"{r['label']:<30} {r['wt']:>7.4f} {r['tc']:>7.4f} "
              f"{r['et']:>7.4f} {r['mean']:>7.4f} {r['tc_fail']:>7}")

    best = max(results, key=lambda x: x['mean'])
    print(f"\n🏆 Best config: {best['label']}")
    print(f"   WT={best['wt']:.4f} TC={best['tc']:.4f} "
          f"ET={best['et']:.4f} Mean={best['mean']:.4f}")

    with open(RESULTS/'ensemble_three_summary.json','w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
