#!/bin/bash
#SBATCH --job-name=attention_unet_w1
#SBATCH --output=../../models/logs/v2/attention_unet_%j.log
#SBATCH --error=../../models/logs/v2/attention_unet_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1

echo "========================================"
echo "Attention U-Net Training - Week 1"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

cd ~/brain_tumor_thesis/segmentation_project/scripts/advanced
python3 train_attention_unet.py

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
