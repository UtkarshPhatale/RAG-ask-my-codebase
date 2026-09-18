"""
Generate All Thesis Figures
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_all_figures():
    """Generate publication-quality figures for thesis"""
    
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    
    # Load data
    with open(base_dir / "results/test_results_summary.json", 'r') as f:
        baseline = json.load(f)
    
    with open(base_dir / "results/augmented_test_results.json", 'r') as f:
        augmented = json.load(f)
    
    with open(base_dir / "models/checkpoints/training_history.json", 'r') as f:
        baseline_history = json.load(f)
    
    with open(base_dir / "models/checkpoints/augmented_training_history.json", 'r') as f:
        aug_history = json.load(f)
    
    figures_dir = base_dir / "thesis_figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Figure 1: Training curves comparison
    create_training_curves(baseline_history, aug_history, figures_dir)
    
    # Figure 2: Test results comparison
    create_results_comparison(baseline, augmented, figures_dir)
    
    # Figure 3: Improvement breakdown
    create_improvement_breakdown(baseline, augmented, figures_dir)
    
    print(f"\n✅ All figures saved to: {figures_dir}")


def create_training_curves(baseline_hist, aug_hist, output_dir):
    """Training curves comparison"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline training
    ax = axes[0]
    epochs_base = range(1, len(baseline_hist['val_dice']) + 1)
    ax.plot(epochs_base, baseline_hist['train_dice'], 'b-', label='Train', linewidth=2, alpha=0.7)
    ax.plot(epochs_base, baseline_hist['val_dice'], 'r-', label='Validation', linewidth=2)
    ax.axhline(y=0.6615, color='g', linestyle='--', alpha=0.5, label='Best (0.66)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Baseline Model (No Augmentation)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.2, 0.8])
    
    # Augmented training
    ax = axes[1]
    epochs_aug = range(1, len(aug_hist['val_dice']) + 1)
    ax.plot(epochs_aug, aug_hist['train_dice'], 'b-', label='Train', linewidth=2, alpha=0.7)
    ax.plot(epochs_aug, aug_hist['val_dice'], 'r-', label='Validation', linewidth=2)
    ax.axhline(y=0.70, color='g', linestyle='--', alpha=0.5, label='Best (0.70)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Augmented Model (Safe Augmentation)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.2, 0.8])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 1: Training curves")


def create_results_comparison(baseline, augmented, output_dir):
    """Test results bar chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regions = ['WT', 'TC', 'ET']
    base_scores = [baseline[f'dice_{r}']['mean'] for r in regions]
    aug_scores = [augmented[f'dice_{r}']['mean'] for r in regions]
    targets = [0.88, 0.80, 0.75]
    
    x = np.arange(len(regions))
    width = 0.25
    
    ax.bar(x - width, base_scores, width, label='Baseline', color='steelblue', alpha=0.8)
    ax.bar(x, aug_scores, width, label='Augmented', color='forestgreen', alpha=0.8)
    ax.bar(x + width, targets, width, label='Literature Target', color='coral', alpha=0.6)
    
    ax.set_xlabel('Tumor Region', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Test Set Performance: Baseline vs Augmented vs Literature', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Whole Tumor', 'Tumor Core', 'Enhancing Tumor'])
    ax.legend(fontsize=11)
    ax.set_ylim([0.6, 0.95])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, (b, a, t) in enumerate(zip(base_scores, aug_scores, targets)):
        ax.text(i - width, b + 0.01, f'{b:.3f}', ha='center', fontsize=9)
        ax.text(i, a + 0.01, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width, t + 0.01, f'{t:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 2: Results comparison")


def create_improvement_breakdown(baseline, augmented, output_dir):
    """Improvement breakdown"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regions = ['WT', 'TC', 'ET']
    improvements = []
    improvement_pcts = []
    
    for r in regions:
        imp = augmented[f'dice_{r}']['mean'] - baseline[f'dice_{r}']['mean']
        imp_pct = (imp / baseline[f'dice_{r}']['mean']) * 100
        improvements.append(imp)
        improvement_pcts.append(imp_pct)
    
    x = np.arange(len(regions))
    colors = ['green' if p > 0 else 'red' for p in improvement_pcts]
    
    bars = ax.bar(x, improvement_pcts, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Tumor Region', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Performance Improvement with Safe Augmentation', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Whole Tumor\n(+1.7%)', 'Tumor Core\n(+13.0%)', 'Enhancing\n(+6.0%)'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, pct, imp) in enumerate(zip(bars, improvement_pcts, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'+{pct:.1f}%\n(+{imp:.4f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_imp = np.mean(improvement_pcts)
    ax.axhline(y=avg_imp, color='blue', linestyle='--', linewidth=2, 
               label=f'Average: +{avg_imp:.1f}%', alpha=0.7)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_improvement_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 3: Improvement breakdown")


if __name__ == "__main__":
    create_all_figures()
