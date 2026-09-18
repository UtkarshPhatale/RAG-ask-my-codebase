"""
Generate Final Comprehensive Report
"""

import json
from pathlib import Path
from datetime import datetime


def generate_report():
    """Generate comprehensive final report"""
    
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    
    # Load all results
    with open(base_dir / "results/test_results_summary.json", 'r') as f:
        baseline = json.load(f)
    
    with open(base_dir / "results/augmented_test_results.json", 'r') as f:
        augmented = json.load(f)
    
    with open(base_dir / "models/checkpoints/augmented_training_history.json", 'r') as f:
        history = json.load(f)
    
    # Generate report
    report_path = base_dir / "results/FINAL_THESIS_REPORT.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MASTER'S THESIS - FINAL RESULTS REPORT\n")
        f.write("Brain Tumor Segmentation Using Deep Learning\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Models Trained: 2 (Baseline + Augmented)\n")
        f.write(f"Dataset: BraTS-GLI 2024 (1,350 patients)\n")
        f.write(f"Best Model: Augmented (WT Dice: {augmented['dice_WT']['mean']:.4f})\n")
        f.write(f"Improvement: {augmented['dice_WT']['mean'] - baseline['dice_WT']['mean']:+.4f}\n\n")
        
        # Baseline results
        f.write("1. BASELINE MODEL\n")
        f.write("-" * 80 + "\n")
        f.write("Configuration:\n")
        f.write("  - Architecture: 3D U-Net (32 base channels)\n")
        f.write("  - Parameters: 22.6M\n")
        f.write("  - Training: 50 epochs, no augmentation\n")
        f.write("  - Time: ~25 hours\n\n")
        f.write("Test Set Performance:\n")
        f.write(f"  - Whole Tumor (WT): {baseline['dice_WT']['mean']:.4f} ± {baseline['dice_WT']['std']:.4f}\n")
        f.write(f"  - Tumor Core (TC):  {baseline['dice_TC']['mean']:.4f} ± {baseline['dice_TC']['std']:.4f}\n")
        f.write(f"  - Enhancing (ET):   {baseline['dice_ET']['mean']:.4f} ± {baseline['dice_ET']['std']:.4f}\n\n")
        
        # Augmented results
        f.write("2. AUGMENTED MODEL\n")
        f.write("-" * 80 + "\n")
        f.write("Configuration:\n")
        f.write("  - Architecture: Same as baseline\n")
        f.write("  - Augmentation: Safe (flip + rotation ±5° + intensity ±5%)\n")
        f.write("  - Training: 75 epochs\n")
        f.write("  - Time: ~37.5 hours\n\n")
        f.write("Test Set Performance:\n")
        f.write(f"  - Whole Tumor (WT): {augmented['dice_WT']['mean']:.4f} ± {augmented['dice_WT']['std']:.4f}\n")
        f.write(f"  - Tumor Core (TC):  {augmented['dice_TC']['mean']:.4f} ± {augmented['dice_TC']['std']:.4f}\n")
        f.write(f"  - Enhancing (ET):   {augmented['dice_ET']['mean']:.4f} ± {augmented['dice_ET']['std']:.4f}\n\n")
        
        # Training history
        f.write("3. TRAINING HISTORY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Epochs: {len(history['val_dice'])}\n")
        f.write(f"Best Val Dice: {max(history['val_dice']):.4f}\n")
        f.write(f"Final Val Dice: {history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Train Dice: {history['train_dice'][-1]:.4f}\n\n")
        
        # Comparison
        f.write("4. IMPROVEMENT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for metric in ['dice_WT', 'dice_TC', 'dice_ET']:
            imp = augmented[metric]['mean'] - baseline[metric]['mean']
            imp_pct = (imp / baseline[metric]['mean']) * 100
            f.write(f"{metric.replace('dice_', '')}: {imp:+.4f} ({imp_pct:+.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("THESIS STATUS: READY FOR WRITING & DEFENSE\n")
        f.write("="*80 + "\n")
    
    print(f"\n📄 Final report saved to: {report_path}")
    print("\n✅ ALL EVALUATIONS COMPLETE!")


if __name__ == "__main__":
    generate_report()
