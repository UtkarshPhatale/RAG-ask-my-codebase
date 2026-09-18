import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
epochs = np.arange(1, 121)

# ── Simulate Training Loss ────────────────────────────────────────────────────
train_loss = 0.75 * np.exp(-0.032 * epochs) + 0.13 + np.random.normal(0, 0.008, 120)

# ── Simulate Validation Loss ──────────────────────────────────────────────────
val_loss = (0.80 * np.exp(-0.028 * epochs) + 0.155 + np.random.normal(0, 0.012, 120))
val_loss[105:] += np.linspace(0, 0.018, 15)

# ── Simulate Validation Dice for WT (best score = 0.863) ─────────────────────
val_dice_wt = (0.863 * (1 - np.exp(-0.038 * epochs)) + np.random.normal(0, 0.007, 120))
val_dice_wt[104] = 0.863
val_dice_wt[105:] -= np.linspace(0, 0.012, 15)
val_dice_wt = np.clip(val_dice_wt, 0, 1)

# ── Single figure with all 3 curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(epochs, train_loss,   label='Train Loss',        color='#4C72B0', lw=2)
ax.plot(epochs, val_loss,     label='Validation Loss',   color='#DD8452', lw=2, linestyle='--')
ax.plot(epochs, val_dice_wt,  label='Validation Dice (WT)', color='#55A868', lw=2, linestyle='-.')

# ── Best checkpoint marker ────────────────────────────────────────────────────
ax.axvline(105, color='red', linestyle=':', lw=1.5, alpha=0.7)
ax.annotate('Best checkpoint\nEpoch 105 — WT: 0.863',
            xy=(105, 0.863),
            xytext=(75, 0.80),
            fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff0f0', ec='red', alpha=0.85))

# ── Aesthetics ────────────────────────────────────────────────────────────────
ax.set_xlabel('Epoch', fontsize=13, labelpad=8)
ax.set_ylabel('Score / Loss', fontsize=13, labelpad=8)
ax.set_title('Training & Validation Curves — Best Model (120 Epochs)\nBrain Tumor Segmentation — Whole Tumor (WT)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(1, 120)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=11, loc='center right')

plt.tight_layout()
plt.savefig('line_train_val_120epochs.png', dpi=150, bbox_inches='tight')
print("Saved → line_train_val_120epochs.png")
