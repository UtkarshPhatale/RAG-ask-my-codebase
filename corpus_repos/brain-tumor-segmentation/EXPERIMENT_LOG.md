# Brain Tumor Segmentation - Experiment Log

## Current Best Result
- **Model:** 32-channel Attention U-Net (Ultimate V2)
- **Test WT Dice:** 0.8242 ± 0.2032
- **Test TC Dice:** 0.5931 ± 0.4925
- **Test ET Dice:** 0.7878 ± 0.3228
- **Checkpoint:** `models/checkpoints/v2/ultimate_32ch_rerun/best_model.pth`
- **Results:** `results/v2/ultimate_32ch_rerun/`
- **Date:** February 13, 2026

---

## Experiment History

### Experiment 1: Baseline 32-Channel (BEST)
- **Status:** ✅ Complete
- **Date:** Jan 29 - Feb 1, 2026
- **Model:** Attention U-Net, 32 channels, deep supervision
- **Val WT Dice:** 0.8084 (epoch 98)
- **Test WT Dice:** 0.8242
- **Checkpoint:** `models/checkpoints/v2/ultimate_32ch_rerun/best_model.pth`
- **Notes:** This is our baseline to beat

### Experiment 2: 48-Channel Scale-Up
- **Status:** ❌ Failed (worse than baseline)
- **Date:** Jan 30-31, 2026
- **Model:** Attention U-Net, 48 channels
- **Test WT Dice:** 0.8234 (-0.47% vs baseline)
- **Checkpoint:** `models/checkpoints/v2/attention_ultimate_48ch/best_model.pth`
- **Notes:** Capacity scaling didn't help

### Experiment 3: NCR Aggressive Weighting (5×)
- **Status:** ⚠️ Partial (improved TC, hurt WT)
- **Date:** Jan 31, 2026
- **Model:** Attention U-Net, 32ch, NCR weight 5×
- **Test WT Dice:** 0.7905 (-3.76%)
- **Test TC Dice:** 0.7754 (+30.8%)
- **Checkpoint:** `models/checkpoints/v2/ncr_focused/best_model.pth`
- **Notes:** Proved NCR weighting works for TC but creates WT trade-off

### Experiment 4: NCR Balanced Weighting (2.5×)
- **Status:** ❌ Failed (worse than baseline)
- **Date:** Feb 11-13, 2026
- **Model:** Attention U-Net, 32ch, NCR weight 2.5×
- **Val WT Dice:** 0.7803 (-5.33%)
- **Val TC Dice:** 0.8107 (+36.69%)
- **Checkpoint:** `models/checkpoints/v2/ncr_balanced/best_model.pth`
- **Notes:** Even balanced weighting hurts WT too much

### Experiment 5: Ensemble (48ch + NCR)
- **Status:** ❌ Failed
- **Test WT Dice:** 0.8180 (-1.01%)
- **Notes:** Simple averaging didn't help

---

## Next Experiment: nnU-Net

### Experiment 6: nnU-Net SOTA (PLANNED)
- **Status:** 🚧 In Progress
- **Start Date:** February 13, 2026
- **Model:** nnU-Net (established SOTA for medical imaging)
- **Expected WT Dice:** 0.85-0.87
- **Target:** Beat 0.84 threshold
- **Timeline:** 10-12 days
- **Checkpoint:** `models/checkpoints/nnunet/` (TBD)
- **Notes:** 
  - Using official nnU-Net framework
  - Automatic hyperparameter optimization
  - Expected to significantly outperform custom architecture

---

## Key Findings So Far

1. **NCR prediction is the bottleneck** - 40.7% of cases have TC=0 due to NCR=0
2. **Class weighting creates fundamental trade-offs** - Cannot improve both WT and TC simultaneously
3. **Capacity scaling doesn't help** - 48 channels performed worse than 32
4. **Bimodal distribution suggests data-level issues** - Not just architecture
5. **Custom architecture plateaued at 0.8242** - Need SOTA approach

---

## Updated: February 13, 2026

## Feb 18, 2026 - Intensity Analysis + Preprocessing Experiment
- Ran intensity distribution analysis on 83 failure vs 121 success cases
- CONFIRMED: T1C median 31.5% lower in failures (p=0.0143)
- CONFIRMED: T1N mean 28.2% lower in failures (p=0.0293)
- Hypothesis: Scanner/protocol variability causes NCR prediction failures
- Plan: Retrain with percentile normalization (p1-p99 per modality)
- New experiment folder: v3/percentile_norm

## Feb 19, 2026 - Percentile Normalization Results (BREAKTHROUGH)
- Experiment: v3/percentile_norm
- Job ID: 20376
- Best checkpoint: epoch 56, val WT=0.8193
- TEST RESULTS:
  - WT Dice: 0.8439 (+0.0197 vs baseline) ✅ TARGET EXCEEDED
  - TC Dice: 0.7493 (+0.1562 vs baseline) ✅ HUGE improvement
  - ET Dice: 0.6551 (-0.1327 vs baseline) ⚠️ Regression
  - TC Failures: 9 (was 83) - 74 cases recovered
- Conclusion: Hypothesis CONFIRMED. Scanner variability was root cause.
- Issue: ET regression needs to be addressed
- Next: Hybrid normalization (percentile for T1N/T2W, z-score for T1C/ET)

## Feb 19, 2026 - Ensemble Results
- Models: percentile_norm (ep56) + hybrid_norm best_mean (ep73)
- Best config: Equal weighting (0.5/0.5)
- TEST: WT=0.8488, TC=0.7744, ET=0.6778, Mean=0.7670, TC_fail=8
- ET still below baseline (0.7878) - target for next experiment
- Next: Strong augmentation + spatial dropout on hybrid norm base

## Feb 20, 2026 - Strong Augmentation + Spatial Dropout (Job 20389)
- Experiment: v3/augmented
- Base: Hybrid normalization (T1C=zscore, rest=percentile)
- New: Rotation ±15°, elastic deformation, gaussian noise,
       gamma correction, random zoom + spatial dropout p=0.1
- LR: 3e-4 (lower than before), 120 epochs
- Target: WT>0.86, TC>0.78, ET>0.72
- Previous best: Ensemble WT=0.8488, TC=0.7744, ET=0.6778

## Feb 22, 2026 - Augmented Model Results (Job 20389)
- TEST: WT=0.8613, TC=0.7807, ET=0.6800, Mean=0.7740
- Best WT achieved so far (+0.0371 vs baseline, +0.0125 vs ensemble)
- Augmentation delayed overfitting from epoch 36 to epoch 105 ✅
- ET gap vs baseline persists (0.6800 vs 0.7878)
- Next: Three-model ensemble to push further

## Feb 22, 2026 - Three-Model Ensemble (FINAL BEST RESULT)
- Models: percentile(ep56) + hybrid_mean(ep73) + augmented(ep105)
- Best config: Equal weighting (0.33/0.33/0.34)
- TEST: WT=0.8629, TC=0.7833, ET=0.6858, Mean=0.7773, TC_fail=10
- vs baseline: WT+0.0387, TC+0.1902, ET-0.1020, TC_fail 83→10
- Augmentation successfully delayed overfitting to epoch 105
- ET gap vs baseline persists — structural limitation of preprocessing
- DECISION: This is the final result for thesis writeup
