#!/usr/bin/env python3
"""Evaluate NCR-focused model on test set"""

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
    
    print(f"Evaluating on {len(test_dirs)} test patients")
    
    # Load model (32 channels for NCR-focused)
    model = ImprovedAttentionUNet3D(4, 4, base_channels=32, deep_supervision=True)
    checkpoint = torch.load('../../models/checkpoints/v2/ncr_focused/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded NCR model - Best val WT: {checkpoint.get('best_wt', 'N/A')}")
    
    # Create dataset
    test_dataset = BraTSSegmentationDataset(test_dirs, transform=None, crop_size=(128,128,128))
    
    # Evaluate
    metrics_calc = SegmentationMetrics(4)
    results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            images, masks = test_dataset[idx]
            images = images.unsqueeze(0).to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
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
    output_dir = Path('../../results/v2/model_ncr')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'test_metrics.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("NCR-FOCUSED MODEL - TEST SET RESULTS")
    print("="*80)
    print(f"WT Dice: {df['wt_dice'].mean():.4f} ± {df['wt_dice'].std():.4f}")
    print(f"TC Dice: {df['tc_dice'].mean():.4f} ± {df['tc_dice'].std():.4f}")
    print(f"ET Dice: {df['et_dice'].mean():.4f} ± {df['et_dice'].std():.4f}")
    
    summary = {
        'wt_dice_mean': float(df['wt_dice'].mean()),
        'wt_dice_std': float(df['wt_dice'].std()),
        'tc_dice_mean': float(df['tc_dice'].mean()),
        'tc_dice_std': float(df['tc_dice'].std()),
        'et_dice_mean': float(df['et_dice'].mean()),
        'et_dice_std': float(df['et_dice'].std())
    }
    
    with open(output_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()
