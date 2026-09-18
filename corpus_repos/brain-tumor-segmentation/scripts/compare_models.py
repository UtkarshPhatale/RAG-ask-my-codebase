"""
Compare Baseline vs Augmented Model Performance
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load both baseline and augmented results"""
    
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project/results"
    
    # Load baseline
    with open(base_dir / "test_results_summary.json", 'r') as f:
        baseline = json.load(f)
    
    # Load augmented
    with open(base_dir / "augmented_test_results.json", 'r') as f:
        augmented = json.load(f)
    
    return baseline, augmented


def create_comparison():
    """Create comparison visualization and report"""
    
    baseline, augmented = load_results()
    
    print("\n" + "="*80)
    print("MODEL COMPARISON: BASELINE vs AUGMENTED")
    print("="*80)
    
    # Table comparison
    print("\n📊 Performance Comparison (Test Set):")
    print("-" * 80)
    print(f"{'Metric':<15} {'Baseline':<20} {'Augmented':<20} {'Improvement':<15}")
    print("-" * 80)
    
    improvements = {}
    
    for metric in ['dice_WT', 'dice_TC', 'dice_ET']:
        base_mean = baseline[metric]['mean']
        base_std = baseline[metric]['std']
        aug_mean = augmented[metric]['mean']
        aug_std = augmented[metric]['std']
        
        improvement = aug_mean - base_mean
        improvement_pct = (improvement / base_mean) * 100
        
        improvements[metric] = improvement
        
        metric_name = metric.replace('dice_', '')
        print(f"{metric_name:<15} {base_mean:.4f} ± {base_std:.4f}   {aug_mean:.4f} ± {aug_std:.4f}   {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    print("-" * 80)
    
    # Average improvement
    avg_improvement = np.mean(list(improvements.values()))
    print(f"\n📈 Average Improvement: {avg_improvement:+.4f} ({(avg_improvement/0.75)*100:+.1f}%)")
    
    # Training comparison
    print("\n⏱️  Training Comparison:")
    print("-" * 80)
    print(f"{'Aspect':<30} {'Baseline':<25} {'Augmented':<25}")
    print("-" * 80)
    print(f"{'Epochs':<30} {'50':<25} {'75':<25}")
    print(f"{'Augmentation':<30} {'None':<25} {'Safe':<25}")
    print(f"{'Training Time':<30} {'~25 hours':<25} {'~37.5 hours':<25}")
    print(f"{'Parameters':<30} {'22.6M':<25} {'22.6M':<25}")
    
    # Separate variable for complex f-string
    baseline_val = "0.6615"
    augmented_val = f"{augmented['dice_WT']['mean']:.4f}"
    print(f"{'Best Val Dice':<30} {baseline_val:<25} {augmented_val:<25}")
    
    print("\n" + "="*80)
    
    # Create visualization
    create_comparison_plot(baseline, augmented, improvements)
    
    # Save comparison report
    save_comparison_report(baseline, augmented, improvements)


def create_comparison_plot(baseline, augmented, improvements):
    """Create comparison bar chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Performance comparison
    regions = ['WT', 'TC', 'ET']
    base_scores = [baseline[f'dice_{r}']['mean'] for r in regions]
    aug_scores = [augmented[f'dice_{r}']['mean'] for r in regions]
    
    x = np.arange(len(regions))
    width = 0.35
    
    ax1.bar(x - width/2, base_scores, width, label='Baseline', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, aug_scores, width, label='Augmented', color='forestgreen', alpha=0.8)
    
    ax1.set_xlabel('Tumor Region', fontsize=12)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('Test Set Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Whole Tumor', 'Tumor Core', 'Enhancing'])
    ax1.legend(fontsize=11)
    ax1.set_ylim([0.6, 0.9])
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, (b, a) in enumerate(zip(base_scores, aug_scores)):
        ax1.text(i - width/2, b + 0.01, f'{b:.3f}', ha='center', fontsize=9)
        ax1.text(i + width/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
    
    # Plot 2: Improvement bars
    imp_values = [improvements[f'dice_{r}'] for r in regions]
    imp_pct = [(imp / baseline[f'dice_{r}']['mean']) * 100 for imp, r in zip(imp_values, regions)]
    
    colors = ['green' if x > 0 else 'red' for x in imp_values]
    ax2.bar(x, imp_pct, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Tumor Region', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Percentage Improvement with Augmentation', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Whole Tumor', 'Tumor Core', 'Enhancing'])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, pct in enumerate(imp_pct):
        ax2.text(i, pct + 0.5, f'{pct:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path.home() / "brain_tumor_thesis/segmentation_project/results"
    fig_path = results_dir / "model_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    print(f"\n📊 Comparison plot saved to: {fig_path}")
    plt.close()


def save_comparison_report(baseline, augmented, improvements):
    """Save detailed comparison report"""
    
    results_dir = Path.home() / "brain_tumor_thesis/segmentation_project/results"
    report_path = results_dir / "comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE vs AUGMENTED MODEL - COMPREHENSIVE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. TEST SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['dice_WT', 'dice_TC', 'dice_ET']:
            f.write(f"\n{metric.replace('dice_', '')}:\n")
            f.write(f"  Baseline:  {baseline[metric]['mean']:.4f} ± {baseline[metric]['std']:.4f}\n")
            f.write(f"  Augmented: {augmented[metric]['mean']:.4f} ± {augmented[metric]['std']:.4f}\n")
            f.write(f"  Improvement: {improvements[metric]:+.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📄 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    create_comparison()
