"""
Experiment 1: Increased Model Capacity
Changes from baseline:
- base_channels: 32 → 64
- Expected improvement: +3-5% Dice
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import time

from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D
from train_segmentation import DiceLoss, SegmentationTrainer, load_data_splits, create_dataloaders
from experiment_tracker import ExperimentTracker


def train_improved_v1():
    """Train model with increased capacity"""
    
    # Initialize experiment tracker
    tracker = ExperimentTracker("improved_capacity_64ch")
    
    # Configuration - INCREASED CAPACITY
    config = {
        'experiment': 'Improved V1 - Increased Capacity',
        'in_channels': 4,
        'num_classes': 4,
        'base_channels': 64,  # ← DOUBLED from 32
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 1,  # Reduced due to larger model
        'num_epochs': 60,  # Slightly longer
        'num_workers': 0,
        'model_params': '~90M',  # 4x more than baseline
        'changes': 'Increased base_channels from 32 to 64'
    }
    
    tracker.save_config(config)
    tracker.add_note("Hypothesis: Larger capacity will better capture tumor heterogeneity")
    
    # Setup paths
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    data_path = base_dir / "data/raw/brats_gli/training_data1_v2"
    splits_file = base_dir / "data/splits/data_splits.json"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📂 Loading data splits...")
    splits = load_data_splits(splits_file)
    
    print("🔄 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        splits, 
        data_path, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Initialize LARGER model
    print("🧠 Initializing model with 64 base channels...")
    model = UNet3D(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_channels=config['base_channels']  # 64 instead of 32
    ).to(device)
    
    total_params, trainable_params = model.count_parameters()
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    tracker.add_note(f"Model size: {total_params:,} parameters")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    criterion = DiceLoss()
    
    # Training setup
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'learning_rates': []
    }
    
    best_val_dice = 0.0
    
    print("\n" + "="*60)
    print("🚀 TRAINING IMPROVED MODEL V1")
    print("="*60)
    print(f"Model: 3D U-Net with {config['base_channels']} base channels")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("="*60 + "\n")
    
    # Training loop (reuse trainer logic)
    trainer = SegmentationTrainer(config)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.criterion = criterion
    trainer.history = history
    trainer.best_val_dice = best_val_dice
    trainer.device = device
    trainer.config['checkpoint_dir'] = tracker.checkpoints_dir
    
    # Train
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Training
        train_loss, train_dice = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_loss, val_dice = trainer.validate(val_loader, epoch)
        
        # Update scheduler
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['learning_rates'].append(current_lr)
        
        # Check if best
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            trainer.best_val_dice = best_val_dice
            tracker.save_checkpoint(epoch, model, optimizer, 
                                  {'val_dice': val_dice, 'train_dice': train_dice}, 
                                  is_best=True)
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            tracker.save_checkpoint(epoch, model, optimizer, 
                                  {'val_dice': val_dice, 'train_dice': train_dice})
        
        # Print summary
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        print(f"Best Val Dice: {best_val_dice:.4f}\n")
    
    training_time = time.time() - start_time
    
    print("\n🎉 Training complete!")
    print(f"Time: {training_time/3600:.2f} hours")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    
    # Save training history
    tracker.save_training_history(history)
    tracker.add_note(f"Training completed. Best Val Dice: {best_val_dice:.4f}")
    tracker.add_note(f"Training time: {training_time/3600:.2f} hours")
    
    print(f"\n📁 Experiment saved: {tracker.exp_dir}")
    
    return tracker, best_val_dice


def submit_job():
    """Create SLURM job script"""
    
    job_script = """#!/bin/bash
#SBATCH --job-name=improved_v1
#SBATCH --output=../all_experiments/improved_v1_%j.log
#SBATCH --error=../all_experiments/improved_v1_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

module load cuda/11.8
source ~/.bashrc
conda activate brain_tumor_thesis

cd ~/brain_tumor_thesis/segmentation_project/scripts

echo "Starting Improved Model V1 Training..."
python3 train_improved_v1.py

echo "Training complete!"
"""
    
    script_path = Path("submit_improved_v1.sh")
    with open(script_path, 'w') as f:
        f.write(job_script)
    
    script_path.chmod(0o755)
    print(f"✅ Job script created: {script_path}")
    print("\nTo submit: sbatch submit_improved_v1.sh")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-job":
        submit_job()
    else:
        train_improved_v1()
