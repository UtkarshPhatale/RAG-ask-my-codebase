#!/usr/bin/env python3
"""
Evaluate augmented model — both best_model and best_mean_model checkpoints.
"""

import sys, json, csv, torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/advanced')

from attention_unet_augmented_FEB20th import get_augmented_model
from create_dataset_augmented import BraTSDatasetAugmented

BASE    = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA    = BASE / "data/raw/brats_gli/training_data1_v2"
SPLITS  = BASE / "data/splits/data_splits.json"
CKPT    = BASE / "models/checkpoints/v3/augmented"
RESULTS = BASE / "results/v3/augmented"
RESULTS.mkdir(parents=True, exist_ok=True)

PREV = {
    'baseline':        {'wt':0.8242,'tc':0.5931,'et':0.7878},
    'percentile':      {'wt':0.8439,'tc':0.7493,'et':0.6551},
    'hybrid_mean':     {'wt':0.8420,'tc':0.7737,'et':0.6769},
    'ensemble_equal':  {'wt':0.8488,'tc':0.7744,'et':0.6778},
}

def compute_dice(pred, target, smooth=1e-5):
    out = {}
    for name, p, t in [
        ('wt', pred>=1,           target>=1),
        ('tc', (pred==1)|(pred==3),(target==1)|(target==3)),
        ('et', pred==3,           target==3),
    ]:
        p,t = p.float(),t.float()
        i = (p*t).sum(); u = p.sum()+t.sum()
        out[f'{name}_dice'] = ((2*i+smooth)/(u+smooth)).item()
    return out

def evaluate(ckpt_path, label, device, ds, loader):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location=device)
    ep   = ckpt.get('epoch','?')
    vwt  = ckpt.get('val_wt_dice',0)
    vtc  = ckpt.get('val_tc_dice',0)
    vet  = ckpt.get('val_et_dice',0)
    print(f"Checkpoint: epoch={ep} | Val WT={vwt:.4f} TC={vtc:.4f} ET={vet:.4f}")

    model = get_augmented_model(
        in_channels=4, num_classes=4,
        base_channels=32, deep_supervision=True, dropout_p=0.1
    )
    state = {k.replace('module.',''):v for k,v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    all_metrics, wt_l, tc_l, et_l = [], [], [], []

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            imgs,lbls = imgs.to(device), lbls.to(device)
            with autocast():
                outs = model(imgs)
                main = outs[0] if isinstance(outs,(list,tuple)) else outs
            pred = main.argmax(dim=1)
            s = compute_dice(pred[0], lbls[0])
            s['patient_id'] = ds.patient_ids[i]
            all_metrics.append(s)
            wt_l.append(s['wt_dice'])
            tc_l.append(s['tc_dice'])
            et_l.append(s['et_dice'])

            if (i+1) % 40 == 0:
                print(f"  {i+1}/204 | WT={np.mean(wt_l):.4f} "
                      f"TC={np.mean(tc_l):.4f} ET={np.mean(et_l):.4f}")

    wt=np.array(wt_l); tc=np.array(tc_l); et=np.array(et_l)
    mean_dice = (wt.mean()+tc.mean()+et.mean())/3
    tc_fail   = (tc<0.1).sum()

    print(f"\n--- RESULTS: {label} ---")
    print(f"WT: {wt.mean():.4f}±{wt.std():.4f}  "
          f"(vs baseline: {wt.mean()-PREV['baseline']['wt']:+.4f}  "
          f"vs ensemble: {wt.mean()-PREV['ensemble_equal']['wt']:+.4f})")
    print(f"TC: {tc.mean():.4f}±{tc.std():.4f}  "
          f"(vs baseline: {tc.mean()-PREV['baseline']['tc']:+.4f}  "
          f"vs ensemble: {tc.mean()-PREV['ensemble_equal']['tc']:+.4f})")
    print(f"ET: {et.mean():.4f}±{et.std():.4f}  "
          f"(vs baseline: {et.mean()-PREV['baseline']['et']:+.4f}  "
          f"vs ensemble: {et.mean()-PREV['ensemble_equal']['et']:+.4f})")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"TC failures: {tc_fail}  (baseline:83, ensemble:8)")

    tag = ckpt_path.stem
    with open(RESULTS/f'test_metrics_{tag}.csv','w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader(); w.writerows(all_metrics)

    summary = {
        'label':label, 'epoch':ep,
        'wt':float(wt.mean()), 'wt_std':float(wt.std()),
        'tc':float(tc.mean()), 'tc_std':float(tc.std()),
        'et':float(et.mean()), 'et_std':float(et.std()),
        'mean':float(mean_dice), 'tc_failures':int(tc_fail),
    }
    with open(RESULTS/f'test_summary_{tag}.json','w') as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = BraTSDatasetAugmented(
        DATA, SPLITS, split='test', augment=False
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    results = []
    for ckpt_name, label in [
        ('best_model.pth',      'Best WT  (ep105)'),
        ('best_mean_model.pth', 'Best Mean(ep105)'),
    ]:
        p = CKPT / ckpt_name
        if p.exists():
            results.append(evaluate(p, label, device, ds, loader))

    print(f"\n{'='*70}")
    print("FULL EXPERIMENT COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'WT':>7} {'TC':>7} {'ET':>7} {'Mean':>7} {'TCFail':>7}")
    print("-"*70)
    for name, v in PREV.items():
        mean = (v['wt']+v['tc']+v['et'])/3
        print(f"{name:<28} {v['wt']:>7.4f} {v['tc']:>7.4f} "
              f"{v['et']:>7.4f} {mean:>7.4f}")
    for r in results:
        print(f"{r['label']:<28} {r['wt']:>7.4f} {r['tc']:>7.4f} "
              f"{r['et']:>7.4f} {r['mean']:>7.4f} {r['tc_failures']:>7}")

if __name__ == '__main__':
    main()
