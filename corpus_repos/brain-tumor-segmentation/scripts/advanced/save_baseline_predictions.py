#!/usr/bin/env python3
"""
Save baseline predictions for post-processing
Saves both predictions and original images for analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from create_dataset import BraTSSegmentationDataset
from attention_unet_v2_improved import ImprovedAttentionUNet3D

def save_predictions():
    """Save all test set predictions and metadata"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path('../../results/v2/predictions/baseline_32ch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SAVING BASELINE PREDICTIONS FOR POST-PROCESSING")
    print("="*80)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = ImprovedAttentionUNet3D(
        in_channels=4,
        num_classes=4,
        base_channels=32,
        deep_supervision=True
    ).to(device)
    
    checkpoint_path = Path('../../models/checkpoints/v2/ultimate_32ch_rerun/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Loaded from epoch {checkpoint['epoch']}")
    print(f"  ✓ Val WT Dice: {checkpoint.get('val_wt_dice', 'N/A')}")
    
    # Load test data
    print("\n[2/4] Loading test data...")
    with open('../../data/splits/data_splits.json') as f:
        splits = json.load(f)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    test_dirs = [data_root / pid for pid in splits['test']]
    
    print(f"  ✓ Test patients: {len(test_dirs)}")
    
    # Create dataset WITHOUT cropping for full predictions
    test_dataset = BraTSSegmentationDataset(
        patient_dirs=test_dirs,
        transform=None,
        crop_size=(128, 128, 128)  # Full size
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Process one at a time
        shuffle=False,
        num_workers=0
    )
    
    # Save predictions
    print("\n[3/4] Generating and saving predictions...")
    
    metadata = {
        'model': 'baseline_32ch',
        'checkpoint': str(checkpoint_path),
        'num_patients': len(test_dirs),
        'predictions': {}
    }
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Saving")):
            patient_id = splits['test'][idx]
            
            images = images.to(device)
            
            # Get prediction
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            mask = masks.cpu().numpy()[0]
            
            # Save prediction as .npy (lightweight)
            pred_file = output_dir / f'{patient_id}_pred.npy'
            np.save(pred_file, pred)
            
            # Save ground truth
            mask_file = output_dir / f'{patient_id}_mask.npy'
            np.save(mask_file, mask)
            
            # Calculate basic metrics for this case
            wt_pred = (pred > 0).astype(float)
            wt_true = (mask > 0).astype(float)
            wt_dice = 2 * (wt_pred * wt_true).sum() / (wt_pred.sum() + wt_true.sum() + 1e-8)
            
            tc_pred = ((pred == 1) | (pred == 3)).astype(float)
            tc_true = ((mask == 1) | (mask == 3)).astype(float)
            tc_dice = 2 * (tc_pred * tc_true).sum() / (tc_pred.sum() + tc_true.sum() + 1e-8)
            
            et_pred = (pred == 3).astype(float)
            et_true = (mask == 3).astype(float)
            et_dice = 2 * (et_pred * et_true).sum() / (et_pred.sum() + et_true.sum() + 1e-8)
            
            # Store metadata
            metadata['predictions'][patient_id] = {
                'pred_file': str(pred_file),
                'mask_file': str(mask_file),
                'wt_dice': float(wt_dice),
                'tc_dice': float(tc_dice),
                'et_dice': float(et_dice),
                'tc_failure': bool(tc_dice == 0),
                'shape': list(pred.shape)
            }
    
    # Save metadata
    print("\n[4/4] Saving metadata...")
    metadata_file = output_dir / 'prediction_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    tc_failures = sum(1 for p in metadata['predictions'].values() if p['tc_failure'])
    avg_wt = np.mean([p['wt_dice'] for p in metadata['predictions'].values()])
    avg_tc = np.mean([p['tc_dice'] for p in metadata['predictions'].values()])
    
    print("\n" + "="*80)
    print("PREDICTIONS SAVED!")
    print("="*80)
    print(f"  Output dir:    {output_dir}")
    print(f"  Total cases:   {len(metadata['predictions'])}")
    print(f"  TC failures:   {tc_failures} ({tc_failures/len(metadata['predictions'])*100:.1f}%)")
    print(f"  Avg WT Dice:   {avg_wt:.4f}")
    print(f"  Avg TC Dice:   {avg_tc:.4f}")
    print(f"  Metadata:      {metadata_file}")
    print("="*80)

if __name__ == '__main__':
    save_predictions()
