"""
Evaluate Augmented Model on Test Set
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D
from torch.utils.data import DataLoader


def calculate_metrics(pred, target):
    """Calculate comprehensive metrics"""
    
    metrics = {}
    
    # Per-class Dice
    for class_id, class_name in [(1, 'NCR'), (2, 'ED'), (3, 'ET')]:
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum().float() + target_mask.sum().float()
        
        if union > 0:
            metrics[f'dice_{class_name}'] = (2. * intersection / union).item()
        else:
            metrics[f'dice_{class_name}'] = 1.0 if pred_mask.sum() == 0 else 0.0
    
    # Composite regions (BraTS standard)
    # Whole Tumor (WT)
    wt_pred = (pred > 0)
    wt_target = (target > 0)
    wt_int = (wt_pred & wt_target).sum().float()
    wt_union = wt_pred.sum().float() + wt_target.sum().float()
    metrics['dice_WT'] = (2. * wt_int / wt_union).item() if wt_union > 0 else 1.0
    
    # Tumor Core (TC)
    tc_pred = ((pred == 1) | (pred == 3))
    tc_target = ((target == 1) | (target == 3))
    tc_int = (tc_pred & tc_target).sum().float()
    tc_union = tc_pred.sum().float() + tc_target.sum().float()
    metrics['dice_TC'] = (2. * tc_int / tc_union).item() if tc_union > 0 else 1.0
    
    # Enhancing Tumor (ET) = class 3
    
    return metrics


def evaluate_model():
    """Evaluate augmented model on test set"""
    
    # Setup
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    model_path = base_dir / "models/checkpoints/augmented_best_model.pth"
    data_path = base_dir / "data/raw/brats_gli/training_data1_v2"
    splits_file = base_dir / "data/splits/data_splits.json"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load checkpoint
    print("📂 Loading augmented model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = UNet3D(in_channels=4, num_classes=4, base_channels=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (Epoch {checkpoint['epoch']})")
    print(f"   Best Val Dice: {checkpoint['val_dice']:.4f}")
    
    # Load test data
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    all_patient_dirs = {d.name: d for d in data_path.iterdir() 
                       if d.is_dir() and d.name.startswith("BraTS-GLI-")}
    
    test_dirs = [all_patient_dirs[pid] for pid in splits['test']]
    print(f"   Test set: {len(test_dirs)} patients")
    
    test_dataset = BraTSSegmentationDataset(test_dirs)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Evaluate
    print(f"\n🧪 Evaluating on test set...")
    all_metrics = []
    
    with torch.no_grad():
        for volumes, segmentations in tqdm(test_loader, desc="Testing"):
            volumes = volumes.to(device)
            segmentations = segmentations.to(device)
            
            outputs = model(volumes)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(predictions.shape[0]):
                metrics = calculate_metrics(predictions[i], segmentations[i])
                all_metrics.append(metrics)
    
    # Calculate statistics
    print("\n" + "="*70)
    print("AUGMENTED MODEL - TEST SET RESULTS")
    print("="*70)
    
    print("\n📊 BraTS Standard Metrics:")
    print("-" * 70)
    
    for metric in ['dice_WT', 'dice_TC', 'dice_ET']:
        values = [m[metric] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        median = np.median(values)
        
        metric_name = metric.replace('dice_', '')
        print(f"{metric_name:<10} Mean: {mean:.4f} ± {std:.4f}  |  Median: {median:.4f}")
    
    print("\n📊 Per-Class Metrics:")
    print("-" * 70)
    
    for metric in ['dice_NCR', 'dice_ED', 'dice_ET']:
        values = [m[metric] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        
        class_name = metric.replace('dice_', '')
        print(f"{class_name:<10} Mean: {mean:.4f} ± {std:.4f}")
    
    print("\n" + "="*70)
    
    # Save results
    summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    results_dir = base_dir / "results"
    results_file = results_dir / "augmented_test_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return summary


if __name__ == "__main__":
    results = evaluate_model()
