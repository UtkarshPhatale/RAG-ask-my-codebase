#!/usr/bin/env python3
"""
Evaluate Ultimate 32-Channel Model (Re-run) on Test Set
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime

from create_dataset import BraTSSegmentationDataset
from attention_unet_v2_improved import ImprovedAttentionUNet3D
from metrics import SegmentationMetrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("EVALUATING ULTIMATE 32-CHANNEL MODEL (RE-RUN)")
    print("="*80)
    
    # Load test split
    with open('../../data/splits/data_splits.json') as f:
        splits = json.load(f)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    test_dirs = [data_root / pid for pid in splits['test']]
    
    print(f"Test patients: {len(test_dirs)}")
    
    # Load model
    print("\nLoading model...")
    model = ImprovedAttentionUNet3D(
        in_channels=4,
        num_classes=4,
        base_channels=32,
        deep_supervision=True
    )
    
    checkpoint_path = '../../models/checkpoints/v2/ultimate_32ch_rerun/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Val WT Dice: {checkpoint['val_wt_dice']:.4f}")
    print(f"  Val TC Dice: {checkpoint['val_tc_dice']:.4f}")
    print(f"  Val ET Dice: {checkpoint['val_et_dice']:.4f}")
    
    # Create dataset
    test_dataset = BraTSSegmentationDataset(
        patient_dirs=test_dirs,
        transform=None,
        crop_size=(128, 128, 128)
    )
    
    # Evaluate
    print("\nRunning evaluation on test set...")
    metrics_calc = SegmentationMetrics(num_classes=4)
    results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
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
    output_dir = Path('../../results/v2/ultimate_32ch_rerun')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_dir / 'test_metrics.csv', index=False)
    
    # Calculate statistics
    summary = {
        'model_name': 'ultimate_32ch_rerun',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_test_cases': len(df),
        'wt_dice': {
            'mean': float(df['wt_dice'].mean()),
            'std': float(df['wt_dice'].std()),
            'median': float(df['wt_dice'].median()),
            'min': float(df['wt_dice'].min()),
            'max': float(df['wt_dice'].max())
        },
        'tc_dice': {
            'mean': float(df['tc_dice'].mean()),
            'std': float(df['tc_dice'].std()),
            'median': float(df['tc_dice'].median()),
            'min': float(df['tc_dice'].min()),
            'max': float(df['tc_dice'].max())
        },
        'et_dice': {
            'mean': float(df['et_dice'].mean()),
            'std': float(df['et_dice'].std()),
            'median': float(df['et_dice'].median()),
            'min': float(df['et_dice'].min()),
            'max': float(df['et_dice'].max())
        }
    }
    
    # Save summary
    with open(output_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("TEST SET RESULTS - ULTIMATE 32-CHANNEL MODEL (RE-RUN)")
    print("="*80)
    print(f"\nWT Dice: {summary['wt_dice']['mean']:.4f} ± {summary['wt_dice']['std']:.4f}")
    print(f"TC Dice: {summary['tc_dice']['mean']:.4f} ± {summary['tc_dice']['std']:.4f}")
    print(f"ET Dice: {summary['et_dice']['mean']:.4f} ± {summary['et_dice']['std']:.4f}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - test_metrics.csv")
    print(f"  - test_summary.json")
    print("="*80)

if __name__ == '__main__':
    main()

