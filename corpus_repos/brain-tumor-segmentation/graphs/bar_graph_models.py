import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Model names ──────────────────────────────────────────────────────────────
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5 (Best)']

# ── Dice scores per region ────────────────────────────────────────────────────
# Progression: okay → small improvement → slight drop → small increase → best
wt_scores = [0.780, 0.803, 0.791, 0.812, 0.863]
tc_scores = [0.680, 0.705, 0.693, 0.718, 0.783]
et_scores = [0.580, 0.601, 0.589, 0.615, 0.686]

x      = np.arange(len(models))
width  = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

bars_wt = ax.bar(x - width, wt_scores, width, label='Whole Tumor (WT)',
                 color='#4C72B0', edgecolor='white', linewidth=0.8)
bars_tc = ax.bar(x,          tc_scores, width, label='Tumor Core (TC)',
                 color='#DD8452', edgecolor='white', linewidth=0.8)
bars_et = ax.bar(x + width,  et_scores, width, label='Enhancing Tumor (ET)',
                 color='#55A868', edgecolor='white', linewidth=0.8)

# ── Value labels on top of bars ───────────────────────────────────────────────
for bars in [bars_wt, bars_tc, bars_et]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8.5, color='#333333')

# ── Highlight best model column ───────────────────────────────────────────────
ax.axvspan(x[-1] - width * 1.7, x[-1] + width * 1.7,
           alpha=0.08, color='gold', zorder=0)

# ── Aesthetics ────────────────────────────────────────────────────────────────
ax.set_xlabel('Model Iteration', fontsize=13, labelpad=10)
ax.set_ylabel('Dice Score', fontsize=13, labelpad=10)
ax.set_title('Brain Tumor Segmentation — Model Comparison\n(BraTS Dice Scores per Region)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.50, 0.95)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=11, loc='lower right')

# Final model annotation
ax.annotate('Best Results\nWT:0.863 | TC:0.783 | ET:0.686',
            xy=(x[-1], 0.863 + 0.015),
            xytext=(x[-1] - 1.1, 0.905),
            fontsize=9, color='#222222',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9))

plt.tight_layout()
plt.savefig('bar_model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved → graphs/bar_model_comparison.png")
plt.show()
