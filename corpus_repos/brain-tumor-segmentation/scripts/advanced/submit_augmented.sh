#!/bin/bash
#SBATCH --job-name=aug_v3
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/uphatal/brain_tumor_thesis/segmentation_project/models/logs/v3/augmented_%j.log
#SBATCH --error=/home/uphatal/brain_tumor_thesis/segmentation_project/models/logs/v3/augmented_%j.err

echo "=========================================="
echo "JOB: Strong Augmentation + Spatial Dropout"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $SLURM_NODELIST"
echo "START: $(date)"
echo "=========================================="

module load mambaforge
conda activate brain_tumor_thesis
cd /home/uphatal/brain_tumor_thesis/segmentation_project

python -c "
import torch
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
"

python scripts/advanced/train_augmented.py

echo "=========================================="
echo "END: $(date)"
echo "=========================================="
