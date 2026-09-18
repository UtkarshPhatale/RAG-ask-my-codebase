#!/bin/bash
#SBATCH --job-name=brats_48ch
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=models/logs/v3/train_48ch_%j.out
#SBATCH --error=models/logs/v3/train_48ch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uphatal@calstatela.edu

# ============================================================
# submit_48ch.sh
# SLURM submission script for 48-Channel Attention U-Net
# Save to: scripts/advanced/submit_48ch.sh
#
# Submit with:
#   cd ~/brain_tumor_thesis/segmentation_project
#   sbatch scripts/advanced/submit_48ch.sh
# ============================================================

echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time:    $(date)"
echo "============================================================"

# ---- Environment ----
module load mambaforge
conda activate brain_tumor_thesis

# ---- Working directory ----
cd ~/brain_tumor_thesis/segmentation_project
echo "Working dir: $(pwd)"

# ---- Verify GPUs ----
echo ""
echo "GPU check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)')
"

# ---- STEP 1: Feasibility check ----
# Run the feasibility check first to confirm batch_size=2 fits
echo ""
echo "Step 1: Running feasibility check..."
python scripts/advanced/attention_unet_48ch.py
FEASIBILITY_EXIT=$?

if [ $FEASIBILITY_EXIT -ne 0 ]; then
    echo "ERROR: Feasibility check failed. Check GPU memory."
    exit 1
fi

echo ""
echo "Feasibility check passed. Starting training..."

# ---- STEP 2: Training ----
echo ""
echo "Step 2: Training 48-Channel Attention U-Net..."
python scripts/advanced/train_48ch.py
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo ""
echo "Training complete."

# ---- STEP 3: Evaluation ----
echo ""
echo "Step 3: Running evaluation on test set..."
python scripts/advanced/eval_48ch.py
EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT"
    exit $EVAL_EXIT
fi

# ---- Done ----
echo ""
echo "============================================================"
echo "All steps completed successfully"
echo "End time: $(date)"
echo ""
echo "Checkpoint dir: models/checkpoints/v3/attention_48ch_v1/"
echo "Results dir:    results/v3/attention_48ch_v1/"
echo ""
echo "Next step: Review results then run ensemble_four.py"
echo "============================================================"
