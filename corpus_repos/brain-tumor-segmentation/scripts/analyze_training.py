"""
Comprehensive Training Analysis and Visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def analyze_training_history():
    """Analyze complete training history"""
    
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    history_file = base_dir / "models/checkpoints/training_history.json"
    checkpoint_file = base_dir / "models/checkpoints/best_model.pth"
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Load checkpoint for additional info
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    epochs = range(1, len(history['train_dice']) + 1)
    
    # Calculate statistics
    stats = {
        'total_epochs': len(history['train_dice']),
        'best_val_dice': max(history['val_dice']),
        'best_val_epoch': np.argmax(history['val_dice']) + 1,
        'final_val_dice': history['val_dice'][-1],
        'final_train_dice': history['train_dice'][-1],
        'train_val_gap': history['train_dice'][-1] - history['val_dice'][-1],
        'peak_lr': max(history['learning_rates']),
        'final_lr': history['learning_rates'][-1],
    }
    
    print("="*60)
    print("TRAINING ANALYSIS REPORT")
    print("="*60)
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Epochs: {stats['total_epochs']}")
    print(f"   Best Val Dice: {stats['best_val_dice']:.4f} (Epoch {stats['best_val_epoch']})")
    print(f"   Final Val Dice: {stats['final_val_dice']:.4f}")
    print(f"   Final Train Dice: {stats['final_train_dice']:.4f}")
    print(f"   Train/Val Gap: {stats['train_val_gap']:.4f}")
    
    print(f"\n📈 Learning Rate Schedule:")
    print(f"   Initial LR: {stats['peak_lr']:.6f}")
    print(f"   Final LR: {stats['final_lr']:.6f}")
    
    # Find improvement phases
    val_dice = np.array(history['val_dice'])
    improvements = np.diff(val_dice)
    
    print(f"\n📉 Learning Phases:")
    rapid_phase = np.where(improvements > 0.05)[0]
    if len(rapid_phase) > 0:
        print(f"   Rapid improvement: Epochs {rapid_phase[0]+1}-{rapid_phase[-1]+2}")
    
    plateau_start = next((i for i in range(20, len(val_dice)) 
                         if all(abs(val_dice[i:i+5] - val_dice[i]) < 0.02)), None)
    if plateau_start:
        print(f"   Plateau started: ~Epoch {plateau_start}")
    
    print(f"\n🎯 Performance Assessment:")
    if stats['best_val_dice'] >= 0.75:
        print(f"   ✅ EXCELLENT - Meets target (≥0.75)")
    elif stats['best_val_dice'] >= 0.70:
        print(f"   ✅ GOOD - Close to target")
    elif stats['best_val_dice'] >= 0.65:
        print(f"   ⚠️  ACCEPTABLE - Below target but reasonable")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT - Significantly below target")
    
    # Overfitting check
    print(f"\n🔍 Overfitting Analysis:")
    if stats['train_val_gap'] < 0.05:
        print(f"   ✅ No overfitting (gap < 0.05)")
    elif stats['train_val_gap'] < 0.10:
        print(f"   ✅ Minimal overfitting (gap < 0.10)")
    else:
        print(f"   ⚠️  Some overfitting detected (gap = {stats['train_val_gap']:.4f})")
    
    return history, stats


def create_visualizations(history, stats):
    """Create comprehensive training visualizations"""
    
    epochs = range(1, len(history['train_dice']) + 1)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Dice Score
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    ax1.plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    ax1.axhline(y=0.75, color='g', linestyle='--', label='Target (0.75)', alpha=0.5)
    ax1.axvline(x=stats['best_val_epoch'], color='orange', linestyle='--', 
                label=f'Best (Epoch {stats["best_val_epoch"]})', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Train/Val Gap
    ax4 = plt.subplot(2, 3, 4)
    gap = np.array(history['train_dice']) - np.array(history['val_dice'])
    ax4.plot(epochs, gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train - Val Dice')
    ax4.set_title('Overfitting Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Improvement Rate
    ax5 = plt.subplot(2, 3, 5)
    val_improvement = np.diff(history['val_dice'])
    smoothed_improvement = np.convolve(val_improvement, np.ones(5)/5, mode='valid')
    ax5.plot(range(3, len(smoothed_improvement)+3), smoothed_improvement, 'r-', linewidth=2)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Validation Dice Improvement (smoothed)')
    ax5.set_title('Learning Progress')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    
    Final Performance:
    • Val Dice: {stats['final_val_dice']:.4f}
    • Train Dice: {stats['final_train_dice']:.4f}
    
    Best Performance:
    • Best Val Dice: {stats['best_val_dice']:.4f}
    • Achieved at Epoch: {stats['best_val_epoch']}
    
    Model Behavior:
    • Train/Val Gap: {stats['train_val_gap']:.4f}
    • Final LR: {stats['final_lr']:.6f}
    
    Target: 0.75-0.80
    Status: {'✅ Target Met' if stats['best_val_dice'] >= 0.75 else '⚠️ Below Target'}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path.home() / "brain_tumor_thesis/segmentation_project/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / 'training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    # Also save high-level summary
    summary_path = results_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BRATS SEGMENTATION TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Epochs: {stats['total_epochs']}\n")
        f.write(f"Best Validation Dice: {stats['best_val_dice']:.4f}\n")
        f.write(f"Final Validation Dice: {stats['final_val_dice']:.4f}\n")
        f.write(f"Final Training Dice: {stats['final_train_dice']:.4f}\n")
        f.write(f"\nTarget Performance: 0.75-0.80\n")
        f.write(f"Target Met: {'Yes' if stats['best_val_dice'] >= 0.75 else 'No'}\n")
    
    print(f"📄 Summary saved to: {summary_path}")


def suggest_improvements(stats):
    """Suggest improvements based on training results"""
    
    print(f"\n" + "="*60)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*60)
    
    if stats['best_val_dice'] < 0.75:
        print("\n🔧 To Improve Performance:")
        
        print("\n1. **Increase Model Capacity**")
        print("   - Current: base_channels=32")
        print("   - Suggestion: Try base_channels=48 or 64")
        print("   - Expected gain: +3-5% Dice")
        
        print("\n2. **Longer Training**")
        print("   - Current: 50 epochs")
        print("   - Suggestion: Train for 100 epochs")
        print("   - Expected gain: +1-3% Dice")
        
        print("\n3. **Advanced Data Augmentation**")
        print("   - Add: Random rotation, elastic deformation")
        print("   - Add: Intensity augmentation")
        print("   - Expected gain: +2-4% Dice")
        
        print("\n4. **Multi-Scale Training**")
        print("   - Current: 128×128×128")
        print("   - Suggestion: Try 160×160×160")
        print("   - Expected gain: +2-3% Dice")
        
        print("\n5. **Deep Supervision**")
        print("   - Add intermediate supervision layers")
        print("   - Expected gain: +1-2% Dice")
    
    if stats['train_val_gap'] > 0.10:
        print("\n⚠️  Overfitting Detected:")
        print("   - Add more dropout (current: none explicit)")
        print("   - Increase data augmentation")
        print("   - Add weight decay regularization")
    
    print("\n" + "="*60)


def main():
    print("\n🔍 Analyzing training results...\n")
    
    # Analyze training
    history, stats = analyze_training_history()
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_visualizations(history, stats)
    
    # Suggest improvements
    suggest_improvements(stats)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
