"""
Comprehensive Individual Model Analysis
Generates detailed visualizations and metrics for both models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def load_all_data():
    """Load all training and test data"""
    
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    
    # Baseline
    with open(base_dir / "results/test_results_summary.json", 'r') as f:
        baseline_test = json.load(f)
    
    with open(base_dir / "models/checkpoints/training_history.json", 'r') as f:
        baseline_history = json.load(f)
    
    # Augmented
    with open(base_dir / "results/augmented_test_results.json", 'r') as f:
        augmented_test = json.load(f)
    
    with open(base_dir / "models/checkpoints/augmented_training_history.json", 'r') as f:
        augmented_history = json.load(f)
    
    return baseline_test, baseline_history, augmented_test, augmented_history


def create_baseline_analysis(baseline_test, baseline_history, output_dir):
    """Create comprehensive baseline model analysis"""
    
    print("\n📊 Creating Baseline Model Analysis...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Training Curves (Loss and Dice)
    ax1 = plt.subplot(3, 3, 1)
    epochs = range(1, len(baseline_history['train_loss']) + 1)
    ax1.plot(epochs, baseline_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, baseline_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dice Score Progress
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, baseline_history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    ax2.plot(epochs, baseline_history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    best_epoch = np.argmax(baseline_history['val_dice']) + 1
    best_dice = max(baseline_history['val_dice'])
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (E{best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Dice Score Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(epochs, baseline_history['learning_rates'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Train/Val Gap
    ax4 = plt.subplot(3, 3, 4)
    gap = np.array(baseline_history['train_dice']) - np.array(baseline_history['val_dice'])
    ax4.plot(epochs, gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train - Val Dice')
    ax4.set_title('Overfitting Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Test Set Performance by Region
    ax5 = plt.subplot(3, 3, 5)
    regions = ['WT', 'TC', 'ET']
    means = [baseline_test[f'dice_{r}']['mean'] for r in regions]
    stds = [baseline_test[f'dice_{r}']['std'] for r in regions]
    x = np.arange(len(regions))
    bars = ax5.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'forestgreen', 'coral'], alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Whole\nTumor', 'Tumor\nCore', 'Enhancing\nTumor'])
    ax5.set_ylabel('Dice Score')
    ax5.set_title('Test Set Performance by Region')
    ax5.set_ylim([0, 1])
    ax5.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax5.text(bar.get_x() + bar.get_width()/2., mean + std + 0.02,
                f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 6. Per-Class Performance
    ax6 = plt.subplot(3, 3, 6)
    classes = ['NCR', 'ED', 'ET']
    class_means = [baseline_test[f'dice_{c}']['mean'] for c in classes]
    class_stds = [baseline_test[f'dice_{c}']['std'] for c in classes]
    x = np.arange(len(classes))
    bars = ax6.bar(x, class_means, yerr=class_stds, capsize=5, 
                   color=['indianred', 'gold', 'lightblue'], alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Necrotic\nCore', 'Edema', 'Enhancing\nTumor'])
    ax6.set_ylabel('Dice Score')
    ax6.set_title('Per-Class Performance')
    ax6.set_ylim([0, 1])
    ax6.grid(True, axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, class_means, class_stds)):
        ax6.text(bar.get_x() + bar.get_width()/2., mean + std + 0.02,
                f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 7. Convergence Analysis
    ax7 = plt.subplot(3, 3, 7)
    window = 5
    val_smooth = np.convolve(baseline_history['val_dice'], np.ones(window)/window, mode='valid')
    ax7.plot(range(window, len(baseline_history['val_dice'])+1), val_smooth, 'b-', linewidth=2)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Validation Dice (Smoothed)')
    ax7.set_title('Convergence Pattern (5-epoch moving average)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Distribution (Box plot)
    ax8 = plt.subplot(3, 3, 8)
    data = [[baseline_test[f'dice_{r}']['mean']] for r in ['WT', 'TC', 'ET']]
    bp = ax8.boxplot(data, labels=['WT', 'TC', 'ET'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'forestgreen', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax8.set_ylabel('Dice Score')
    ax8.set_title('Performance Distribution')
    ax8.grid(True, axis='y', alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    BASELINE MODEL SUMMARY
    
    Training:
    • Epochs: {len(baseline_history['train_dice'])}
    • Best Epoch: {best_epoch}
    • Best Val Dice: {best_dice:.4f}
    • Final Train: {baseline_history['train_dice'][-1]:.4f}
    • Final Val: {baseline_history['val_dice'][-1]:.4f}
    
    Test Performance:
    • WT: {baseline_test['dice_WT']['mean']:.4f} ± {baseline_test['dice_WT']['std']:.4f}
    • TC: {baseline_test['dice_TC']['mean']:.4f} ± {baseline_test['dice_TC']['std']:.4f}
    • ET: {baseline_test['dice_ET']['mean']:.4f} ± {baseline_test['dice_ET']['std']:.4f}
    
    Configuration:
    • Architecture: 3D U-Net
    • Parameters: 22.6M
    • Augmentation: None
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('Baseline Model - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Baseline analysis saved")


def create_augmented_analysis(augmented_test, augmented_history, output_dir):
    """Create comprehensive augmented model analysis"""
    
    print("\n📊 Creating Augmented Model Analysis...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Training Curves (Loss and Dice)
    ax1 = plt.subplot(3, 3, 1)
    epochs = range(1, len(augmented_history['train_loss']) + 1)
    ax1.plot(epochs, augmented_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, augmented_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dice Score Progress
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, augmented_history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    ax2.plot(epochs, augmented_history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    best_epoch = np.argmax(augmented_history['val_dice']) + 1
    best_dice = max(augmented_history['val_dice'])
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (E{best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Dice Score Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(epochs, augmented_history['learning_rates'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Train/Val Gap
    ax4 = plt.subplot(3, 3, 4)
    gap = np.array(augmented_history['train_dice']) - np.array(augmented_history['val_dice'])
    ax4.plot(epochs, gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    ax4.fill_between(epochs, 0, gap, alpha=0.3, color='purple')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train - Val Dice')
    ax4.set_title('Overfitting Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Test Set Performance by Region
    ax5 = plt.subplot(3, 3, 5)
    regions = ['WT', 'TC', 'ET']
    means = [augmented_test[f'dice_{r}']['mean'] for r in regions]
    stds = [augmented_test[f'dice_{r}']['std'] for r in regions]
    medians = [augmented_test[f'dice_{r}']['median'] for r in regions]
    
    x = np.arange(len(regions))
    width = 0.35
    bars1 = ax5.bar(x - width/2, means, width, yerr=stds, capsize=5, 
                    label='Mean', color=['steelblue', 'forestgreen', 'coral'], alpha=0.7)
    bars2 = ax5.bar(x + width/2, medians, width, 
                    label='Median', color=['navy', 'darkgreen', 'orangered'], alpha=0.7)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Whole\nTumor', 'Tumor\nCore', 'Enhancing\nTumor'])
    ax5.set_ylabel('Dice Score')
    ax5.set_title('Test Set: Mean vs Median Performance')
    ax5.legend()
    ax5.set_ylim([0, 1])
    ax5.grid(True, axis='y', alpha=0.3)
    
    # 6. Per-Class Performance
    ax6 = plt.subplot(3, 3, 6)
    classes = ['NCR', 'ED', 'ET']
    class_means = [augmented_test[f'dice_{c}']['mean'] for c in classes]
    class_stds = [augmented_test[f'dice_{c}']['std'] for c in classes]
    x = np.arange(len(classes))
    bars = ax6.bar(x, class_means, yerr=class_stds, capsize=5, 
                   color=['indianred', 'gold', 'lightblue'], alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Necrotic\nCore', 'Edema', 'Enhancing\nTumor'])
    ax6.set_ylabel('Dice Score')
    ax6.set_title('Per-Class Performance')
    ax6.set_ylim([0, 1])
    ax6.grid(True, axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, class_means, class_stds)):
        ax6.text(bar.get_x() + bar.get_width()/2., mean + std + 0.02,
                f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 7. Learning Progress (First vs Last 10 epochs)
    ax7 = plt.subplot(3, 3, 7)
    first_10 = np.mean(augmented_history['val_dice'][:10])
    last_10 = np.mean(augmented_history['val_dice'][-10:])
    improvement = last_10 - first_10
    
    bars = ax7.bar(['First\n10 Epochs', 'Last\n10 Epochs'], [first_10, last_10], 
                   color=['lightcoral', 'lightgreen'], alpha=0.7, edgecolor='black', linewidth=2)
    ax7.set_ylabel('Average Dice Score')
    ax7.set_title('Learning Progress')
    ax7.set_ylim([0, 0.8])
    ax7.grid(True, axis='y', alpha=0.3)
    
    # Add values and arrow
    for bar, val in zip(bars, [first_10, last_10]):
        ax7.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    ax7.annotate('', xy=(1, last_10), xytext=(0, first_10),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax7.text(0.5, (first_10 + last_10)/2, f'+{improvement:.3f}',
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    # 8. Performance Variability
    ax8 = plt.subplot(3, 3, 8)
    regions_data = []
    for r in ['WT', 'TC', 'ET']:
        mean = augmented_test[f'dice_{r}']['mean']
        std = augmented_test[f'dice_{r}']['std']
        cv = std / mean  # Coefficient of variation
        regions_data.append(cv)
    
    bars = ax8.bar(range(3), regions_data, color=['steelblue', 'forestgreen', 'coral'], alpha=0.7)
    ax8.set_xticks(range(3))
    ax8.set_xticklabels(['WT', 'TC', 'ET'])
    ax8.set_ylabel('Coefficient of Variation (Std/Mean)')
    ax8.set_title('Performance Consistency')
    ax8.grid(True, axis='y', alpha=0.3)
    
    for bar, cv in zip(bars, regions_data):
        ax8.text(bar.get_x() + bar.get_width()/2., cv + 0.01,
                f'{cv:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    AUGMENTED MODEL SUMMARY
    
    Training:
    • Epochs: {len(augmented_history['train_dice'])}
    • Best Epoch: {best_epoch}
    • Best Val Dice: {best_dice:.4f}
    • Final Train: {augmented_history['train_dice'][-1]:.4f}
    • Final Val: {augmented_history['val_dice'][-1]:.4f}
    
    Test Performance:
    • WT: {augmented_test['dice_WT']['mean']:.4f} ± {augmented_test['dice_WT']['std']:.4f}
    • TC: {augmented_test['dice_TC']['mean']:.4f} ± {augmented_test['dice_TC']['std']:.4f}
    • ET: {augmented_test['dice_ET']['mean']:.4f} ± {augmented_test['dice_ET']['std']:.4f}
    
    Configuration:
    • Architecture: 3D U-Net
    • Parameters: 22.6M
    • Augmentation: Safe (flip+rotate+scale)
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('Augmented Model - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'augmented_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Augmented analysis saved")


def create_detailed_metrics_report(baseline_test, baseline_history, 
                                   augmented_test, augmented_history, output_dir):
    """Create detailed text report with all metrics"""
    
    print("\n📄 Creating Detailed Metrics Report...")
    
    report_path = output_dir / 'detailed_metrics_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE METRICS REPORT\n")
        f.write("="*100 + "\n\n")
        
        # BASELINE MODEL
        f.write("="*100 + "\n")
        f.write("BASELINE MODEL - DETAILED METRICS\n")
        f.write("="*100 + "\n\n")
        
        f.write("TRAINING METRICS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Epochs: {len(baseline_history['train_dice'])}\n")
        f.write(f"Best Validation Epoch: {np.argmax(baseline_history['val_dice']) + 1}\n")
        f.write(f"Best Validation Dice: {max(baseline_history['val_dice']):.4f}\n")
        f.write(f"Final Training Dice: {baseline_history['train_dice'][-1]:.4f}\n")
        f.write(f"Final Validation Dice: {baseline_history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Training Loss: {baseline_history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {baseline_history['val_loss'][-1]:.4f}\n")
        f.write(f"Train/Val Gap (final): {baseline_history['train_dice'][-1] - baseline_history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Learning Rate: {baseline_history['learning_rates'][-1]:.6f}\n\n")
        
        f.write("TEST SET METRICS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Region':<20} {'Mean':<15} {'Std':<15} {'Median':<15} {'Min':<15} {'Max':<15}\n")
        f.write("-" * 100 + "\n")
        
        for metric in ['dice_WT', 'dice_TC', 'dice_ET', 'dice_NCR', 'dice_ED']:
            name = metric.replace('dice_', '')
            mean = baseline_test[metric]['mean']
            std = baseline_test[metric]['std']
            median = baseline_test[metric]['median']
            min_val = baseline_test[metric]['min']
            max_val = baseline_test[metric]['max']
            f.write(f"{name:<20} {mean:<15.4f} {std:<15.4f} {median:<15.4f} {min_val:<15.4f} {max_val:<15.4f}\n")
        
        f.write("\n\n")
        
        # AUGMENTED MODEL
        f.write("="*100 + "\n")
        f.write("AUGMENTED MODEL - DETAILED METRICS\n")
        f.write("="*100 + "\n\n")
        
        f.write("TRAINING METRICS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Epochs: {len(augmented_history['train_dice'])}\n")
        f.write(f"Best Validation Epoch: {np.argmax(augmented_history['val_dice']) + 1}\n")
        f.write(f"Best Validation Dice: {max(augmented_history['val_dice']):.4f}\n")
        f.write(f"Final Training Dice: {augmented_history['train_dice'][-1]:.4f}\n")
        f.write(f"Final Validation Dice: {augmented_history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Training Loss: {augmented_history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {augmented_history['val_loss'][-1]:.4f}\n")
        f.write(f"Train/Val Gap (final): {augmented_history['train_dice'][-1] - augmented_history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Learning Rate: {augmented_history['learning_rates'][-1]:.6f}\n\n")
        
        f.write("TEST SET METRICS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Region':<20} {'Mean':<15} {'Std':<15} {'Median':<15} {'Min':<15} {'Max':<15}\n")
        f.write("-" * 100 + "\n")
        
        for metric in ['dice_WT', 'dice_TC', 'dice_ET', 'dice_NCR', 'dice_ED']:
            name = metric.replace('dice_', '')
            mean = augmented_test[metric]['mean']
            std = augmented_test[metric]['std']
            median = augmented_test[metric]['median']
            min_val = augmented_test[metric]['min']
            max_val = augmented_test[metric]['max']
            f.write(f"{name:<20} {mean:<15.4f} {std:<15.4f} {median:<15.4f} {min_val:<15.4f} {max_val:<15.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"✅ Detailed report saved to: {report_path}")


def main():
    """Generate all comprehensive analyses"""
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE INDIVIDUAL MODEL ANALYSIS")
    print("="*80)
    
    # Load data
    baseline_test, baseline_history, augmented_test, augmented_history = load_all_data()
    
    # Create output directory
    output_dir = Path.home() / "brain_tumor_thesis/segmentation_project/individual_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Generate analyses
    create_baseline_analysis(baseline_test, baseline_history, output_dir)
    create_augmented_analysis(augmented_test, augmented_history, output_dir)
    create_detailed_metrics_report(baseline_test, baseline_history, 
                                   augmented_test, augmented_history, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {output_dir}")
    print("  • baseline_comprehensive_analysis.png")
    print("  • augmented_comprehensive_analysis.png")
    print("  • detailed_metrics_report.txt")
    print("\n🎓 Ready for thesis and presentation!")


if __name__ == "__main__":
    main()
