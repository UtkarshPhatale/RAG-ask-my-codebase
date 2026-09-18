#!/usr/bin/env python3
"""Ensemble 48-ch and NCR-focused models"""

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

from create_dataset import BraTSSegmentationDataset
from attention_unet_v2_improved import ImprovedAttentionUNet3D
from metrics import SegmentationMetrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test split
    with open('../../data/splits/data_splits.json') as f:
        splits = json.load(f)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    test_dirs = [data_root / pid for pid in splits['test']]
    
    print(f"Evaluating ENSEMBLE on {len(test_dirs)} test patients")
    
    # Load 48-ch model
    print("Loading 48-channel model...")
    model_48ch = ImprovedAttentionUNet3D(4, 4, base_channels=48, deep_supervision=True)
    checkpoint_48 = torch.load('../../models/checkpoints/v2/attention_ultimate/best_model.pth')
    model_48ch.load_state_dict(checkpoint_48['model_state_dict'])
    model_48ch = model_48ch.to(device)
    model_48ch.eval()
    print(f"✓ Loaded 48-ch model (val WT: {checkpoint_48.get('best_wt_dice', 'N/A')})")
    
    # Load NCR-focused model
    print("Loading NCR-focused model...")
    model_ncr = ImprovedAttentionUNet3D(4, 4, base_channels=32, deep_supervision=True)
    checkpoint_ncr = torch.load('../../models/checkpoints/v2/ncr_focused/best_model.pth')
    model_ncr.load_state_dict(checkpoint_ncr['model_state_dict'])
    model_ncr = model_ncr.to(device)
    model_ncr.eval()
    print(f"✓ Loaded NCR-focused model (val WT: {checkpoint_ncr.get('best_wt', 'N/A')})")
    
    # Create dataset
    test_dataset = BraTSSegmentationDataset(test_dirs, transform=None, crop_size=(128,128,128))
    
    # Evaluate
    metrics_calc = SegmentationMetrics(4)
    results = []
    
    print("\nRunning ensemble inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            images, masks = test_dataset[idx]
            images_batch = images.unsqueeze(0).to(device)
            
            # Get predictions from both models
            out_48ch = model_48ch(images_batch)
            out_ncr = model_ncr(images_batch)
            
            if isinstance(out_48ch, tuple):
                out_48ch = out_48ch[0]
            if isinstance(out_ncr, tuple):
                out_ncr = out_ncr[0]
            
            # Average logits (soft ensemble)
            ensemble_logits = (out_48ch + out_ncr) / 2.0
            
            pred = torch.argmax(ensemble_logits, dim=1).cpu().numpy()[0]
            mask = masks.numpy()
            
            wt_dice = metrics_calc.whole_tumor_dice(pred, mask)
            tc_dice = metrics_calc.tumor_core_dice(pred, mask)
            et_dice = metrics_calc.enhancing_tumor_dice(pred, mask)
            
            results.append({
                'patient_id': test_dirs[idx].name,
                'wt_dice': wt_dice,
                'tc_dice': tc_dice,
                'et_dice': et_dice
            })
    
    # Save results
    df = pd.DataFrame(results)
    output_dir = Path('../../results/v2/model_ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'test_metrics.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE MODEL - TEST SET RESULTS")
    print("="*80)
    print(f"WT Dice: {df['wt_dice'].mean():.4f} ± {df['wt_dice'].std():.4f}")
    print(f"TC Dice: {df['tc_dice'].mean():.4f} ± {df['tc_dice'].std():.4f}")
    print(f"ET Dice: {df['et_dice'].mean():.4f} ± {df['et_dice'].std():.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    print(f"Baseline WT:  0.8281")
    print(f"Ensemble WT:  {df['wt_dice'].mean():.4f}")
    print(f"Improvement:  {(df['wt_dice'].mean() - 0.8281)*100:+.2f}%")
    
    if df['wt_dice'].mean() >= 0.84:
        print("\n🎉 TARGET ACHIEVED! WT Dice ≥ 0.84!")
    
    summary = {
        'wt_dice_mean': float(df['wt_dice'].mean()),
        'wt_dice_std': float(df['wt_dice'].std()),
        'tc_dice_mean': float(df['tc_dice'].mean()),
        'tc_dice_std': float(df['tc_dice'].std()),
        'et_dice_mean': float(df['et_dice'].mean()),
        'et_dice_std': float(df['et_dice'].std()),
        'baseline_wt': 0.8281,
        'improvement': float((df['wt_dice'].mean() - 0.8281) * 100)
    }
    
    with open(output_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()
