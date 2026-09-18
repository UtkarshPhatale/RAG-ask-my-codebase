"""
Analyze Attention U-Net Training - Find What Went Wrong
=======================================================
"""
import json
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("ATTENTION U-NET TRAINING ANALYSIS")
print("="*80)

# Load training history
with open('../../models/checkpoints/v2/attention_unet/training_history.json') as f:
    history = json.load(f)

# Load baseline history for comparison
with open('../../models/checkpoints/augmented_training_history.json') as f:
    baseline_history = json.load(f)

print(f"\nAttention U-Net trained for {len(history['val_wt_dice'])} epochs")
print(f"Baseline trained for {len(baseline_history['val_dice'])} epochs")

# Find best epochs
att_best_epoch = np.argmax(history['val_wt_dice']) + 1
att_best_dice = max(history['val_wt_dice'])

base_best_epoch = np.argmax(baseline_history['val_dice']) + 1
base_best_dice = max(baseline_history['val_dice'])

print(f"\n📊 BEST PERFORMANCE:")
print(f"  Attention: Epoch {att_best_epoch}, Val Dice = {att_best_dice:.4f}")
print(f"  Baseline:  Epoch {base_best_epoch}, Val Dice = {base_best_dice:.4f}")
print(f"  Difference: {(att_best_dice - base_best_dice):.4f} ({(att_best_dice - base_best_dice)*100:+.2f}%)")

# Check if attention plateaued early
first_10_avg = np.mean(history['val_wt_dice'][:10])
last_10_avg = np.mean(history['val_wt_dice'][-10:])
improvement = last_10_avg - first_10_avg

print(f"\n📈 LEARNING PROGRESS:")
print(f"  First 10 epochs avg: {first_10_avg:.4f}")
print(f"  Last 10 epochs avg:  {last_10_avg:.4f}")
print(f"  Total improvement:   {improvement:.4f} ({improvement*100:+.2f}%)")

if improvement < 0.01:
    print(f"  ⚠️ WARNING: Model barely improved after epoch 10!")
    print(f"     Possible cause: Learning rate too low or bad initialization")

# Check learning rate changes
lr_changes = []
for i in range(1, len(history['learning_rate'])):
    if history['learning_rate'][i] != history['learning_rate'][i-1]:
        lr_changes.append((i+1, history['learning_rate'][i]))

print(f"\n📉 LEARNING RATE CHANGES:")
for epoch, lr in lr_changes:
    print(f"  Epoch {epoch}: LR changed to {lr:.8f}")

# Compare loss convergence
att_final_loss = np.mean(history['val_loss'][-5:])
base_final_loss = np.mean(baseline_history['val_loss'][-5:])

print(f"\n💥 LOSS COMPARISON:")
print(f"  Attention final val loss: {att_final_loss:.4f}")
print(f"  Baseline final val loss:  {base_final_loss:.4f}")

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Validation Dice comparison
epochs_att = range(1, len(history['val_wt_dice']) + 1)
epochs_base = range(1, len(baseline_history['val_dice']) + 1)

axes[0, 0].plot(epochs_att, history['val_wt_dice'], 'b-', label='Attention U-Net', linewidth=2)
axes[0, 0].plot(epochs_base, baseline_history['val_dice'], 'r-', label='Baseline', linewidth=2)
axes[0, 0].axhline(y=base_best_dice, color='r', linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=att_best_dice, color='b', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Dice')
axes[0, 0].set_title('Validation Dice Score Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss curves
axes[0, 1].plot(epochs_att, history['train_loss'], 'b-', alpha=0.5, label='Attention Train')
axes[0, 1].plot(epochs_att, history['val_loss'], 'b-', linewidth=2, label='Attention Val')
axes[0, 1].plot(epochs_base, baseline_history['train_loss'], 'r-', alpha=0.5, label='Baseline Train')
axes[0, 1].plot(epochs_base, baseline_history['val_loss'], 'r-', linewidth=2, label='Baseline Val')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Training and Validation Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Learning rate
axes[1, 0].plot(epochs_att, history['learning_rate'], 'b-', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Per-class performance (Attention)
axes[1, 1].plot(epochs_att, history['val_wt_dice'], label='WT', linewidth=2)
axes[1, 1].plot(epochs_att, history['val_tc_dice'], label='TC', linewidth=2)
axes[1, 1].plot(epochs_att, history['val_et_dice'], label='ET', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Dice Score')
axes[1, 1].set_title('Attention U-Net: Per-Region Performance')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../results/v2/attention_unet/training_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved: results/v2/attention_unet/training_analysis.png")

# Diagnosis
print(f"\n" + "="*80)
print("🔍 DIAGNOSIS:")
print("="*80)

if att_best_dice < base_best_dice - 0.02:
    print("❌ PROBLEM: Attention U-Net significantly underperformed baseline")
    print("\nPossible causes:")
    print("  1. Attention gates not properly initialized")
    print("  2. Learning rate too low (started at 1e-4, baseline used same)")
    print("  3. Model has more parameters but same LR (needs adjustment)")
    print("  4. Attention mechanism interfering with skip connections")
    
    print("\n💡 SOLUTIONS TO TRY:")
    print("  A. Increase initial learning rate (1e-4 → 5e-4)")
    print("  B. Use warmup learning rate (start low, increase)")
    print("  C. Try different attention gate placement")
    print("  D. Initialize attention weights to pass-through (α=1)")
    print("  E. Remove augmentation initially to isolate problem")

elif improvement < 0.05:
    print("⚠️ PROBLEM: Model stopped learning early")
    print("\nPossible causes:")
    print("  1. Learning rate reduced too aggressively")
    print("  2. Model stuck in local minimum")
    print("  3. Batch size too small for stable training")
    
    print("\n💡 SOLUTIONS TO TRY:")
    print("  A. Increase patience for LR reduction (5 → 10 epochs)")
    print("  B. Use cosine annealing instead of ReduceLROnPlateau")
    print("  C. Add gradient clipping")

else:
    print("🤔 Model learned but didn't exceed baseline")
    print("\nPossible causes:")
    print("  1. Attention adds capacity but needs more training")
    print("  2. Attention gates learning to ignore (weights → 0)")
    print("  3. Data augmentation sufficient, attention redundant")
    
    print("\n💡 SOLUTIONS TO TRY:")
    print("  A. Train for more epochs (75 → 100)")
    print("  B. Inspect attention weights (are they being used?)")
    print("  C. Try without augmentation first")

print("="*80)
