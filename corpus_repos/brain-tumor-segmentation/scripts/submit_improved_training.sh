#!/bin/bash
#SBATCH --job-name=brats_improved
#SBATCH --output=../models/logs/improved_training_%j.log
#SBATCH --error=../models/logs/improved_training_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

echo "=================================================="
echo "IMPROVED BraTS Segmentation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Find available CUDA module
echo ""
echo "Loading CUDA module..."
module avail cuda 2>&1 | head -20

# Try different CUDA versions (adjust based on what's available)
if module load cuda/11.3 2>/dev/null; then
    echo "✅ Loaded cuda/11.3"
elif module load cuda/11.7 2>/dev/null; then
    echo "✅ Loaded cuda/11.7"
elif module load cuda 2>/dev/null; then
    echo "✅ Loaded default cuda"
else
    echo "⚠️ No CUDA module found, continuing anyway..."
fi

# Initialize conda
echo ""
echo "Initializing conda..."
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Try to activate conda environment
echo ""
echo "Activating environment..."
if command -v conda &> /dev/null; then
    conda activate brain_tumor_thesis
    echo "✅ Conda environment activated"
else
    echo "⚠️ Conda not found, using system Python"
fi

# Verify Python and PyTorch
echo ""
echo "Python environment:"
which python3
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1 || echo "⚠️ PyTorch not available"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Navigate to scripts
cd ~/brain_tumor_thesis/segmentation_project/scripts

# Run training
echo ""
echo "Starting improved training (75 epochs, base_channels=64, augmentation)..."
echo ""

python3 train_improved.py

echo ""
echo "=================================================="
echo "Training completed at: $(date)"
echo "=================================================="

# Show results if they exist
if [ -f ../models/checkpoints/improved_best_model.pth ]; then
    echo ""
    echo "Best model:"
    ls -lh ../models/checkpoints/improved_best_model.pth
fi

if [ -f ../models/checkpoints/improved_training_history.json ]; then
    echo ""
    echo "Training history (last 20 lines):"
    tail -20 ../models/checkpoints/improved_training_history.json
fi
