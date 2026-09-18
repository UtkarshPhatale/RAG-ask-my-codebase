#!/bin/bash
#SBATCH --job-name=ncr_bal_2.5x
#SBATCH --output=../../models/logs/v2/ncr_balanced_%j.log
#SBATCH --error=../../models/logs/v2/ncr_balanced_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

echo "=========================================="
echo "BALANCED NCR EXPERIMENT (2.5× weight)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

source ~/miniconda3/bin/activate brain_tumor_thesis

cd ~/brain_tumor_thesis/segmentation_project/scripts/advanced
python train_ncr_balanced.py

echo "=========================================="
echo "Complete: $(date)"
echo "=========================================="
