#!/bin/bash
# Backup script - never lose results again

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="../../results/archive"

echo "=========================================="
echo "BACKING UP RESULTS"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Function to backup a model
backup_model() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    
    if [ -f "$MODEL_PATH" ]; then
        BACKUP_NAME="${MODEL_NAME}_${TIMESTAMP}"
        mkdir -p "${ARCHIVE_DIR}/models/${BACKUP_NAME}"
        
        # Copy model checkpoint
        cp "$MODEL_PATH" "${ARCHIVE_DIR}/models/${BACKUP_NAME}/"
        
        # Copy associated results if they exist
        RESULTS_DIR=$(dirname $(dirname "$MODEL_PATH"))/../../results/v2/$(basename $(dirname "$MODEL_PATH"))
        if [ -d "$RESULTS_DIR" ]; then
            cp -r "$RESULTS_DIR" "${ARCHIVE_DIR}/evaluations/${BACKUP_NAME}/"
        fi
        
        echo "✓ Backed up: $BACKUP_NAME"
    else
        echo "✗ Not found: $MODEL_PATH"
    fi
}

# Backup all current models
backup_model "baseline_32ch" "../../models/checkpoints/v2/ultimate_32ch_rerun/best_model.pth"
backup_model "ncr_balanced_2.5x" "../../models/checkpoints/v2/ncr_balanced/best_model.pth"
backup_model "ncr_aggressive_5x" "../../models/checkpoints/v2/ncr_focused/best_model.pth"
backup_model "model_48ch" "../../models/checkpoints/v2/attention_ultimate_48ch/best_model.pth"

echo "=========================================="
echo "BACKUP COMPLETE"
echo "Location: $ARCHIVE_DIR"
echo "=========================================="
