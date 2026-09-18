"""
Comprehensive Model Evaluation on Test Set
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D
from torch.utils.data import DataLoader


def calculate_detailed_metrics(pred, target):
    """Calculate comprehensive segmentation metrics"""
    
    metrics = {}
    
    # Per-class Dice scores
    for class_id, class_name in [(1, 'NCR'), (2, 'ED'), (3, 'ET')]:
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum().float() + target_mask.sum().float()
        
        if union > 0:
            dice = (2. * intersection) / union
            metrics[f'dice_{class_name}'] = dice.item()
        else:
            metrics[f'dice_{class_name}'] = 1.0 if pred_mask.sum() == 0 else 0.0
    
    # Composite regions (BraTS standard)
    # Whole Tumor (WT) = NCR + ED + ET = labels 1,2,3
    wt_pred = (pred > 0)
    wt_target = (target > 0)
    wt_intersection = (wt_pred & wt_target).sum().float()
    wt_union = wt_pred.sum().float() + wt_target.sum().float()
    metrics['dice_WT'] = (2. * wt_intersection / wt_union).item() if wt_union > 0 else 1.0
    
    # Tumor Core (TC) = NCR + ET = labels 1,3
    tc_pred = ((pred == 1) | (pred == 3))
    tc_target = ((target == 1) | (target == 3))
    tc_intersection = (tc_pred & tc_target).sum().float()
    tc_union = tc_pred.sum().float() + tc_target.sum().float()
    metrics['dice_TC'] = (2. * tc_intersection / tc_union).item() if tc_union > 0 else 1.0
    
    # Enhancing Tumor (ET) = label 3
    # Already calculated as dice_ET above
    
    return metrics


def evaluate_model(model_path, test_loader, device):
    """Evaluate model on test set"""
    
    print("📂 Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = UNet3D(in_channels=4, num_classes=4, base_channels=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"   Best Val Dice: {checkpoint['val_dice']:.4f}")
    
    all_metrics = []
    
    print(f"\n🧪 Evaluating on {len(test_loader)} test samples...")
    
    with torch.no_grad():
        for volumes, segmentations in tqdm(test_loader, desc="Testing"):
            volumes = volumes.to(device)
            segmentations = segmentations.to(device)
            
            # Forward pass
            outputs = model(volumes)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for each sample in batch
            for i in range(predictions.shape[0]):
                metrics = calculate_detailed_metrics(
                    predictions[i], 
                    segmentations[i]
                )
                all_metrics.append(metrics)
    
    return all_metrics


def summarize_results(all_metrics):
    """Calculate summary statistics"""
    
    # Calculate mean and std for each metric
    summary = {}
    
    metric_names = all_metrics[0].keys()
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        summary[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return summary


def print_results(summary):
    """Print formatted results"""
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS")
    print("="*70)
    
    print("\n📊 BraTS Standard Metrics (Mean ± Std):")
    print("-" * 70)
    
    # Main BraTS metrics
    print(f"{'Metric':<25} {'Mean':<15} {'Median':<15} {'Range':<15}")
    print("-" * 70)
    
    for metric in ['dice_WT', 'dice_TC', 'dice_ET']:
        if metric in summary:
            mean = summary[metric]['mean']
            std = summary[metric]['std']
            median = summary[metric]['median']
            min_val = summary[metric]['min']
            max_val = summary[metric]['max']
            
            metric_name = metric.replace('dice_', '')
            print(f"{metric_name:<25} {mean:.4f} ± {std:.4f}   {median:.4f}        [{min_val:.4f}, {max_val:.4f}]")
    
    print("\n📊 Per-Class Metrics (Mean ± Std):")
    print("-" * 70)
    
    for metric in ['dice_NCR', 'dice_ED', 'dice_ET']:
        if metric in summary:
            mean = summary[metric]['mean']
            std = summary[metric]['std']
            median = summary[metric]['median']
            
            class_name = metric.replace('dice_', '')
            print(f"{class_name:<25} {mean:.4f} ± {std:.4f}   {median:.4f}")
    
    print("\n" + "="*70)
    
    # Performance assessment
    wt_mean = summary['dice_WT']['mean']
    tc_mean = summary['dice_TC']['mean']
    et_mean = summary['dice_ET']['mean']
    
    print("\n🎯 Performance Assessment:")
    print("-" * 70)
    
    targets = {'WT': 0.88, 'TC': 0.80, 'ET': 0.75}
    results = {'WT': wt_mean, 'TC': tc_mean, 'ET': et_mean}
    
    for region, target in targets.items():
        result = results[region]
        status = "✅" if result >= target else "⚠️"
        gap = result - target
        print(f"{status} {region}: {result:.4f} (Target: {target:.4f}, Gap: {gap:+.4f})")
    
    print("="*70)


def save_results(summary, all_metrics, output_dir):
    """Save results to files"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics
    summary_file = output_dir / 'test_results_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Save individual results
    results_file = output_dir / 'test_results_individual.json'
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"💾 Individual results saved to: {results_file}")
    
    # Save formatted report
    report_file = output_dir / 'test_results_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BRATS SEGMENTATION - TEST SET RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("BraTS Standard Metrics:\n")
        f.write("-" * 70 + "\n")
        for metric in ['dice_WT', 'dice_TC', 'dice_ET']:
            mean = summary[metric]['mean']
            std = summary[metric]['std']
            f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"📄 Report saved to: {report_file}")


def main():
    # Setup
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    model_path = base_dir / "models/checkpoints/best_model.pth"
    data_path = base_dir / "data/raw/brats_gli/training_data1_v2"
    splits_file = base_dir / "data/splits/data_splits.json"
    results_dir = base_dir / "results"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load test split
    print("📂 Loading test data...")
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    all_patient_dirs = {d.name: d for d in data_path.iterdir() 
                       if d.is_dir() and d.name.startswith("BraTS-GLI-")}
    
    test_dirs = [all_patient_dirs[pid] for pid in splits['test']]
    print(f"   Test set size: {len(test_dirs)} patients")
    
    # Create test dataset
    test_dataset = BraTSSegmentationDataset(test_dirs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    all_metrics = evaluate_model(model_path, test_loader, device)
    
    # Summarize
    summary = summarize_results(all_metrics)
    
    # Print results
    print_results(summary)
    
    # Save results
    save_results(summary, all_metrics, results_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
