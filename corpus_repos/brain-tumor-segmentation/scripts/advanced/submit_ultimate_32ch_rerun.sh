#!/bin/bash
#SBATCH --job-name=ultimate_32ch_rerun
#SBATCH --output=../../models/logs/v2/ultimate_32ch_rerun/training_%j.log
#SBATCH --error=../../models/logs/v2/ultimate_32ch_rerun/training_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "=========================================="
echo "Ultimate 32-Channel Model - Clean Re-run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Run training
python train_ultimate_32ch_rerun.py

echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
