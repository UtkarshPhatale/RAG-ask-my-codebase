#!/bin/bash
#SBATCH --job-name=ultimate_v2_complete
#SBATCH --output=../../models/logs/v2/attention_ultimate_%j.log
#SBATCH --error=../../models/logs/v2/attention_ultimate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1

# Print job info
echo "========================================================================"
echo "ULTIMATE ATTENTION U-NET V2 - COMPLETE TRAINING"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo ""
echo "Configuration:"
echo "  - Epochs: 100"
echo "  - Deep Supervision: Yes"
echo "  - Combo Loss: Dice + Focal"
echo "  - Learning Rate: 5e-4 with warmup"
echo "  - Mixed Precision: Enabled"
echo "  - Time Limit: 72 hours"
echo "========================================================================"
echo ""

# Change to working directory
cd ~/brain_tumor_thesis/segmentation_project/scripts/advanced

# Activate conda environment (adjust if needed)
source ~/.bashrc
module load mambaforge
mamba activate brain_tumor_thesis

# Verify GPU is available
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run training
echo "Starting training at $(date)"
echo "========================================================================"
python3 train_attention_ultimate.py

# Print completion info
echo ""
echo "========================================================================"
echo "Training completed at $(date)"
echo "========================================================================"

# Check if best model was saved
if [ -f "../../models/checkpoints/v2/attention_ultimate/best_model.pth" ]; then
    echo "✓ Best model saved successfully"
    python3 -c "
import torch
checkpoint = torch.load('../../models/checkpoints/v2/attention_ultimate/best_model.pth')
print(f'  Best epoch: {checkpoint[\"epoch\"]}')
print(f'  Best WT Dice: {checkpoint[\"best_wt_dice\"]:.4f}')
"
else
    echo "⚠ Warning: Best model not found!"
fi

echo "========================================================================"
