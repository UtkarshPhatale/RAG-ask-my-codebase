#!/bin/bash
#SBATCH --job-name=pct_norm_v3
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/uphatal/brain_tumor_thesis/segmentation_project/models/logs/v3/percentile_norm_%j.log
#SBATCH --error=/home/uphatal/brain_tumor_thesis/segmentation_project/models/logs/v3/percentile_norm_%j.err

echo "=========================================="
echo "JOB: Percentile Normalization Experiment"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $SLURM_NODELIST"
echo "START: $(date)"
echo "=========================================="

# Environment
module load mambaforge
conda activate brain_tumor_thesis

# Go to project root
cd /home/uphatal/brain_tumor_thesis/segmentation_project

# Confirm GPU
python -c "import torch; print('GPUs:', torch.cuda.device_count()); \
           [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) \
            for i in range(torch.cuda.device_count())]"

# Run training
python scripts/advanced/train_percentile_norm.py

echo "=========================================="
echo "END: $(date)"
echo "=========================================="
