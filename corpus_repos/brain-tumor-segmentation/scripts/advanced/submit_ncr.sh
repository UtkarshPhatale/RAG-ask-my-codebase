#!/bin/bash
#SBATCH --job-name=brats_ncr
#SBATCH --output=../../models/logs/v2/ncr_training_%j.log
#SBATCH --error=../../models/logs/v2/ncr_training_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

python train_ncr_focused.py
