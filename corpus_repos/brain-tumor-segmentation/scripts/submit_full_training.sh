#!/bin/bash
#SBATCH --job-name=brats_seg_full
#SBATCH --output=../models/logs/training_%j.log
#SBATCH --error=../models/logs/training_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Job info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate brain_tumor_thesis

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Navigate to scripts
cd ~/brain_tumor_thesis/segmentation_project/scripts

# Run training
echo "Starting training for 50 epochs..."
echo ""

python3 train_segmentation.py

# Completion info
echo ""
echo "=================================================="
echo "Training completed at: $(date)"
echo "=================================================="

# Show final results
echo ""
echo "Best model location:"
ls -lh ../models/checkpoints/best_model.pth

echo ""
echo "Training history:"
cat ../models/checkpoints/training_history.json | grep "val_dice" | tail -5
