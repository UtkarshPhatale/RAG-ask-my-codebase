"""
Resume Training from Checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time

from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D
from train_segmentation import DiceLoss, SegmentationTrainer, load_data_splits, create_dataloaders


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return all saved states"""
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"✅ Checkpoint loaded successfully")
    print(f"   - Saved at epoch: {checkpoint['epoch']}")
    print(f"   - Best Val Dice: {checkpoint['val_dice']:.4f}")
    
    return checkpoint


def resume_training(checkpoint_path, num_additional_epochs=26):
    """Resume training from checkpoint"""
    
    # Configuration (must match original training)
    config = {
        'in_channels': 4,
        'num_classes': 4,
        'base_channels': 32,
        'learning_rate': 1e-4,  # Will be overridden by checkpoint
        'weight_decay': 1e-5,
        'batch_size': 2,
        'num_workers': 0,
    }
    
    # Setup paths
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    data_path = base_dir / "data/raw/brats_gli/training_data1_v2"
    splits_file = base_dir / "data/splits/data_splits.json"
    checkpoint_dir = base_dir / "models/checkpoints"
    
    config['checkpoint_dir'] = checkpoint_dir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Determine starting epoch
    start_epoch = checkpoint['epoch'] + 1
    total_epochs = start_epoch + num_additional_epochs - 1
    
    print(f"\n📊 Resume Configuration:")
    print(f"   - Resume from epoch: {start_epoch}")
    print(f"   - Train until epoch: {total_epochs}")
    print(f"   - Additional epochs: {num_additional_epochs}")
    
    # Load data
    print("\n📂 Loading data splits...")
    splits = load_data_splits(splits_file)
    
    print("🔄 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        splits, 
        data_path, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Initialize model
    model = UNet3D(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_channels=config['base_channels']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✅ Optimizer state loaded")
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"✅ Current learning rate: {current_lr:.6f}")
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Load training history
    history = checkpoint.get('history', {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'learning_rates': []
    })
    
    best_val_dice = checkpoint['val_dice']
    
    print(f"\n✅ Loaded training history:")
    print(f"   - Epochs completed: {len(history['train_loss'])}")
    print(f"   - Best Val Dice so far: {best_val_dice:.4f}")
    
    # Initialize criterion
    criterion = DiceLoss()
    
    # Create trainer instance (reuse training logic)
    trainer = SegmentationTrainer(config)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.criterion = criterion
    trainer.history = history
    trainer.best_val_dice = best_val_dice
    trainer.device = device
    
    print("\n" + "="*60)
    print("🚀 RESUMING TRAINING")
    print("="*60)
    print(f"Starting from epoch: {start_epoch}")
    print(f"Training until epoch: {total_epochs}")
    print(f"Best Val Dice so far: {best_val_dice:.4f}")
    print("="*60 + "\n")
    
    # Resume training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_loss, train_dice = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_loss, val_dice = trainer.validate(val_loader, epoch)
        
        # Update learning rate
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['learning_rates'].append(current_lr)
        
        # Check if best model
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            trainer.best_val_dice = best_val_dice
        
        # Save checkpoint
        if epoch % 5 == 0 or is_best:
            trainer.save_checkpoint(epoch, val_dice, is_best)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{total_epochs} - {epoch_time:.1f}s")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Best Val Dice: {best_val_dice:.4f}")
        print(f"{'='*60}\n")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Epochs {start_epoch}-{total_epochs} completed")
    print("="*60 + "\n")
    
    # Save final history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training complete!")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"📊 Best validation Dice: {best_val_dice:.4f}")
    
    return history


def main():
    # Path to the best checkpoint
    checkpoint_path = Path.home() / "brain_tumor_thesis/segmentation_project/models/checkpoints/best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = checkpoint_path.parent
        for ckpt in sorted(checkpoint_dir.glob("*.pth")):
            print(f"  - {ckpt.name}")
        return
    
    # Resume training for remaining 26 epochs (24 done, need 26 more to reach 50)
    history = resume_training(checkpoint_path, num_additional_epochs=26)


if __name__ == "__main__":
    main()
