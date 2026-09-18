#!/bin/bash
#SBATCH --job-name=brats_aug
#SBATCH --output=../models/logs/augmented_%j.log
#SBATCH --error=../models/logs/augmented_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

echo "=================================================="
echo "BraTS Training with Safe Augmentation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================================="

# Activate environment
source ~/.bashrc
conda activate brain_tumor_thesis

# GPU info
nvidia-smi --query-gpu=name,memory.free --format=csv

# Run training
cd ~/brain_tumor_thesis/segmentation_project/scripts
python3 train_with_augmentation.py

echo "Complete: $(date)"

# Show final results
echo ""
echo "Best model location:"
ls -lh ../models/checkpoints/train_aug_best_model.pth

echo ""
echo "Training history:"
cat ../models/checkpoints/training_aug_history.json | grep "val_dice" | tail -5
