# Brain Tumor Segmentation — MSc Thesis Project
**3D Glioma Segmentation on BraTS 2024 GLI Dataset**
California State University, Los Angeles | Curie HPC Cluster

---

## Project Overview

This project develops a 3D brain tumor segmentation system using the BraTS 2024
GLI dataset. The core contribution is identifying and solving a specific failure
mode — scanner intensity variability causing 40.7% complete TC segmentation
failures — through systematic preprocessing experiments, ensemble learning, and
test-time augmentation.

**Final Best Result (3-Model Ensemble):**

| Region | Baseline | Final | Change |
|---|---|---|---|
| WT (Whole Tumor) | 0.8242 | 0.8629 | +0.0387 |
| TC (Tumor Core) | 0.5931 | 0.7833 | +0.1902 |
| ET (Enhancing Tumor) | 0.7878 | 0.6858 | -0.1020 |
| Mean Dice | 0.7350 | 0.7773 | +0.0423 |
| TC Failures (TC<0.1) | 83/204 | 10/204 | -88% |

---

## Hardware & Environment

| Item | Details |
|---|---|
| Cluster | Curie HPC — California State University LA |
| GPUs | 2× NVIDIA A30 (24GB VRAM each) |
| CUDA | 11.6 |
| PyTorch | 1.13.1 |
| Python | 3.9 |
| Conda env | `brain_tumor_thesis` |

```bash
# Connect
ssh uphatal@curie.calstatela.edu
srun --pty -p gpu --gpus-per-node=2 -t 72:00:00 /bin/bash
module load mambaforge
conda activate brain_tumor_thesis
cd ~/brain_tumor_thesis/segmentation_project
```

---

## Dataset

**BraTS 2024 GLI (Glioma) Dataset**

| Split | Patients |
|---|---|
| Train | 944 |
| Validation | 202 |
| Test | 204 |
| **Total** | **1,350** |

**Input:** 4 MRI modalities per patient — T1N, T1C, T2W, T2F
(Each volume: 182×218×182 voxels, NIfTI format)

**Segmentation targets (BraTS standard):**

| Label | Region | Code |
|---|---|---|
| 0 | Background | — |
| 1 | NCR — Necrotic Core | — |
| 2 | ED — Edema | — |
| 3 | ET — Enhancing Tumor | (raw label 4 remapped to 3) |

**Evaluation regions:**
- **WT** = Whole Tumor = all labels ≥ 1
- **TC** = Tumor Core = labels 1 + 3 (NCR + ET)
- **ET** = Enhancing Tumor = label 3 only

Data path on cluster:

data/raw/brats_gli/training_data1_v2/
data/splits/data_splits.json


---

## Model Architecture

**ImprovedAttentionUNet3D** — 3D Attention U-Net with Deep Supervision

Input [4, 128, 128, 128]
│
├── Encoder Level 1: ConvBlock → 32ch
├── Encoder Level 2: ConvBlock → 64ch
├── Encoder Level 3: ConvBlock → 128ch
├── Encoder Level 4: ConvBlock → 256ch
│
├── Bottleneck: ConvBlock → 512ch [+ Spatial Dropout p=0.1]
│
├── Decoder Level 4: AttentionGate + UpBlock → 256ch
├── Decoder Level 3: AttentionGate + UpBlock → 128ch
├── Decoder Level 2: AttentionGate + UpBlock → 64ch
├── Decoder Level 1: AttentionGate + UpBlock → 32ch
│
└── Output heads (deep supervision):
├── Main output [4, 128, 128, 128]
├── Aux head 1 (upsampled from decoder level 2)
├── Aux head 2 (upsampled from decoder level 3)
└── Aux head 3 (upsampled from decoder level 4)


- **Base model:** 22.7M parameters (32 base channels)
- **Extended model:** 51.1M parameters (48 base channels)
- **Loss:** ComboLoss = Dice Loss + Focal Loss (α=0.5)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Training crop:** 128×128×128 random crop
- **Inference crop:** 128×128×128 center crop

**Key files:**

scripts/advanced/attention_unet_v2_improved.py # Base architecture
scripts/advanced/attention_unet_augmented_FEB20th.py # + Spatial Dropout
scripts/advanced/attention_unet_48ch.py # 48-channel variant
scripts/advanced/advanced_losses.py # ComboLoss


---

## Experiment History

### Phase 1 — Baseline & Architecture Search (Pre Feb 18)

| Experiment | WT | TC | ET | TC_fail | Notes |
|---|---|---|---|---|---|
| Baseline z-score 32ch | 0.8242 | 0.5931 | 0.7878 | 83 | Best WT |
| 48-channel scale-up | 0.8234 | 0.5931 | 0.7689 | — | No gain |
| NCR loss weight 5× | 0.7905 | 0.7754 | 0.7126 | — | TC↑ but WT↓ |
| NCR loss weight 2.5× | 0.7803 | 0.8107 | — | — | Worse WT |
| Ensemble (48ch + NCR) | 0.8180 | 0.6293 | 0.7643 | — | No gain |
| TTA on baseline | 0.8269 | — | — | — | Minimal gain |

**Root cause identified:** Statistical analysis (two-sample t-test, p=0.0143)
confirmed T1C intensities 31.5% lower in the 83 failure cases. Scanner intensity
variability — not architecture — was causing NCR prediction failures.

---

### Phase 2 — Preprocessing Experiments (Feb 18–19)

**Experiment 1: Percentile Normalization (Job 20376)**
- Normalize all 4 modalities to 1st–99th percentile range
- Trained 100 epochs, best checkpoint epoch 56
- **Result:** TC failures 83 → 9 ✅ | ET 0.7878 → 0.6551 ❌
- **Finding:** Percentile clipping suppressed bright ET signal on T1C

**Experiment 2: Hybrid Normalization (Job 20384)**
- T1N, T2W, T2F → percentile norm | T1C → z-score (preserves ET brightness)
- Trained 80 epochs, evaluated epoch 36 (best WT) and epoch 73 (best Mean)
- **Result:** ET partially recovered to 0.6769, TC failures 7–12

**2-Model Ensemble (Feb 19)**
- Percentile (ep56) + Hybrid mean (ep73), equal weights (0.5/0.5)
- **Result:** WT=0.8488, TC=0.7744, ET=0.6778, Mean=0.7670, TC_fail=8

---

### Phase 3 — Augmentation & Regularization (Feb 20–22, Job 20389)

**Strong Augmentation + Spatial Dropout (Experiment 4)**

Augmentations applied:
- Random flip (all axes, p=0.5 each)
- Random rotation ±15° (p=0.4)
- Elastic deformation α=200–400, σ=10–15 (p=0.3)
- Gaussian noise std=0.01–0.05 (p=0.4)
- Gamma correction 0.7–1.5 (p=0.35)
- Random zoom 0.85–1.15 (p=0.3)
- Spatial Dropout p=0.1 at bottleneck

- **Result:** Model peaked at epoch 105 vs epoch 36 without augmentation
- **Best single model:** WT=0.8613, TC=0.7807, ET=0.6800, Mean=0.7740

**3-Model Ensemble (Feb 22)**
- Percentile(ep56) + Hybrid_mean(ep73) + Augmented(ep105)
- Equal weights (1/3 each)
- **🏆 Best result:** WT=0.8629, TC=0.7833, ET=0.6858, Mean=0.7773, TC_fail=10

---

### Phase 4 — 48-Channel Extended Model (Job 20484, Mar 2026)

**48-Channel Attention U-Net with 3 simultaneous modifications:**
1. 48 base channels (51.1M params vs 22.7M)
2. Dynamic class-balanced loss (inverse frequency weighting)
3. CosineAnnealingWarmRestarts LR (T_0=30, T_mult=2)

| Checkpoint | WT | TC | ET | Mean | TC_fail |
|---|---|---|---|---|---|
| best_model (ep81) | 0.8431 | 0.7624 | 0.6368 | 0.7474 | 12 |
| best_mean (ep120) | 0.8416 | 0.7788 | 0.6681 | 0.7629 | 8 |

**Finding:** Larger capacity did not improve results. 944 training patients
is near the capacity sweet spot for 22.7M params. The 48ch model has TC_fail=8
(best ever) making it a useful ensemble candidate.

---

### Complete Results Table

| Model | WT | TC | ET | Mean | TC_fail |
|---|---|---|---|---|---|
| Baseline (z-score) | 0.8242 | 0.5931 | 0.7878 | 0.7350 | 83 |
| Percentile norm | 0.8439 | 0.7493 | 0.6551 | 0.7494 | 9 |
| Hybrid norm (ep73) | 0.8420 | 0.7737 | 0.6769 | 0.7642 | 12 |
| 2-Model ensemble | 0.8488 | 0.7744 | 0.6778 | 0.7670 | 8 |
| Augmented (ep105) | 0.8613 | 0.7807 | 0.6800 | 0.7740 | 12 |
| **3-Model ensemble** | **0.8629** | **0.7833** | **0.6858** | **0.7773** | **10** |
| 48ch best_model | 0.8431 | 0.7624 | 0.6368 | 0.7474 | 12 |
| 48ch best_mean | 0.8416 | 0.7788 | 0.6681 | 0.7629 | 8 |

---

## SOTA Comparison

| Method | WT | TC | ET |
|---|---|---|---|
| nnU-Net | 0.88–0.91 | 0.84–0.87 | 0.82–0.85 |
| TransBTS | 0.88–0.90 | 0.84–0.86 | 0.82–0.84 |
| Swin UNETR | 0.89–0.91 | 0.85–0.87 | 0.82–0.85 |
| **This work** | **0.8629** | **0.7833** | **0.6858** |

WT is within ~0.03 of SOTA. TC improved 31.8% from baseline.
ET gap remains the primary open challenge (structural preprocessing conflict).

---

## Repository Contents vs. What Stays on the Cluster

This repo tracks **code, configs, small results, and documentation only**.
Large artifacts remain on Curie and are *not* pushed to GitHub:

| Folder | Size (measured Aug 2026) | In repo? |
|---|---|---|
| `models/` | ~42 GB | ❌ No (checkpoints too large) |
| `data/` | ~36 GB | ❌ No (raw BraTS data — download separately) |
| `backups/` | ~7.4 GB | ❌ No (under review — likely checkpoint backups) |
| `results/` | ~7.0 GB | ⚠️ Under review — likely contains large prediction volumes; small CSV/JSON summaries will be kept, bulky files excluded |
| `all_experiments/` | ~260 MB | ⚠️ Under review before inclusion |
| `individual_analysis/` | ~3.2 MB | ✅ Yes |
| `scripts/` | ~909 KB | ✅ Yes |
| `thesis_figures/` | ~821 KB | ✅ Yes |
| `graphs/` | ~272 KB | ✅ Yes |
| `docs/` | ~4.5 KB | ✅ Yes |
| `nnunet_workspace/`, `notebooks/`, `experiments/` | ~0 | ✅ Yes (empty/minimal) |

Raw data (`data/raw/`) is NOT included in this repository.
Download BraTS 2024 GLI from: https://www.synapse.org/brats2024

---

## Key Scientific Findings

1. **Root cause was scanner intensity variability** — confirmed statistically
   (two-sample t-test, p=0.0143). T1C intensities 31.5% lower in failure cases.
   The model never predicted NCR in 83/204 (40.7%) test cases.

2. **Percentile normalization fixed 74/83 TC failures** — direct proof of
   hypothesis. TC failures 83 → 9 in one experiment.

3. **ET-TC normalization conflict is structural** — percentile clipping helps
   NCR (dark on T1N) but hurts ET (bright on T1C). Hybrid normalization
   partially mitigates this trade-off. The full gap cannot be closed without
   a fundamentally different preprocessing approach.

4. **Augmentation delayed overfitting from epoch 36 to epoch 105** — 3× longer
   productive training. Strong augmentation (6 operations) + Spatial Dropout
   at the bottleneck was the most effective regularization combination.

5. **Equal-weight ensembling consistently outperforms single models** — different
   normalization strategies create genuinely complementary error patterns.
   Diversity in preprocessing is a more effective source of ensemble diversity
   than architecture diversity (48ch vs 32ch).

6. **More parameters ≠ better results at this data scale** — 51.1M parameter
   48ch model underperformed 22.7M parameter 32ch model. 944 training patients
   is near the capacity sweet spot for this architecture family.

---

## How to Reproduce

### 1. Setup
```bash
conda create -n brain_tumor_thesis python=3.9
conda activate brain_tumor_thesis
pip install torch==1.13.1+cu116 torchvision --extra-index-url \
    https://download.pytorch.org/whl/cu116
pip install nibabel numpy scipy scikit-learn
```

### 2. Prepare data splits
```bash
# data_splits.json already included in repo
# Place BraTS 2024 GLI data at:
# data/raw/brats_gli/training_data1_v2/
```

### 3. Reproduce best single model (augmented, Job 20389)
```bash
sbatch scripts/advanced/submit_augmented.sh
# Training: ~48-60 hours on 2× A30
# Best checkpoint: models/checkpoints/v3/augmented/best_model.pth (epoch 105)
```

### 4. Reproduce best result (3-model ensemble)
```bash
python scripts/advanced/ensemble_three.py
# Results saved to: results/v3/ensemble_three/
```

### 5. Run TTA evaluation
```bash
python scripts/advanced/eval_tta.py --model augmented --checkpoint best_model
```

### 6. Run sliding window inference
```bash
python scripts/advanced/eval_sliding_window.py --model ensemble3
```

---

## SLURM Job History

| Job ID | Experiment | Epochs | Cluster Node | Result |
|---|---|---|---|---|
| 20376 | Percentile norm | 100 (best ep56) | gpu — | WT=0.8439 |
| 20384 | Hybrid norm | 80 (best ep73) | gpu — | WT=0.8420 |
| 20389 | Augmented + Dropout | 120 (best ep105) | gpu02 | WT=0.8613 ★ |
| 20483 | 48ch (failed — import error) | — | gpu05 | Failed |
| 20484 | 48ch (fixed) | 120 (best ep81) | gpu05 | WT=0.8431 |

---

## Citation

If you use this work, please cite:

@mastersthesis{uphatal2026braintumor,
title = {3D Brain Tumor Segmentation with Attention U-Net:
Addressing Scanner Intensity Variability in BraTS 2024},
author = {Uphatal},
school = {California State University, Los Angeles},
year = {2026},
note = {BraTS 2024 GLI Dataset}
}
