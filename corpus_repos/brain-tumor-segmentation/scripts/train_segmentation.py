"""
Training Pipeline for Brain Tumor Segmentation using 3D U-Net
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
from datetime import datetime

# Import our custom modules
from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D


class DiceLoss(nn.Module):
    """Dice Loss for segmentation (better than CrossEntropy for medical imaging)"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, ignore_index=None):
        """
        Args:
            predictions: (B, C, H, W, D) logits
            targets: (B, H, W, D) class indices
        """
        # Convert logits to probabilities
        predictions = torch.softmax(predictions, dim=1)
        
        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate Dice coefficient per class
        dice_scores = []
        for i in range(num_classes):
            if ignore_index is not None and i == ignore_index:
                continue
            
            pred_i = predictions[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Return 1 - mean Dice (loss should be minimized)
        return 1.0 - torch.mean(torch.stack(dice_scores))


class SegmentationTrainer:
    """Complete training pipeline for brain tumor segmentation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize model
        self.model = UNet3D(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ).to(self.device)
        
        # Loss function
        self.criterion = DiceLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        self.best_val_dice = 0.0
    
    def calculate_dice_score(self, predictions, targets):
        """Calculate Dice score for evaluation"""
        # Convert logits to class predictions
        pred_classes = torch.argmax(predictions, dim=1)
        
        dice_scores = []
        
        # Calculate Dice for each class (excluding background)
        for class_id in [1, 2, 4]:  # NCR, ED, ET
            pred_mask = (pred_classes == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum().float() + target_mask.sum().float()
            
            if union > 0:
                dice = (2. * intersection) / union
                dice_scores.append(dice.item())
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (volumes, segmentations) in enumerate(progress_bar):
            volumes = volumes.to(self.device)
            segmentations = segmentations.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(volumes)
            
            # Calculate loss
            loss = self.criterion(outputs, segmentations)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            dice_score = self.calculate_dice_score(outputs, segmentations)
            total_dice += dice_score
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_score:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for volumes, segmentations in tqdm(val_loader, desc=f"Validation {epoch}"):
                volumes = volumes.to(self.device)
                segmentations = segmentations.to(self.device)
                
                # Forward pass
                outputs = self.model(volumes)
                
                # Calculate loss
                loss = self.criterion(outputs, segmentations)
                
                # Calculate metrics
                total_loss += loss.item()
                dice_score = self.calculate_dice_score(outputs, segmentations)
                total_dice += dice_score
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_dice': val_dice,
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config['checkpoint_dir'] / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config['checkpoint_dir'] / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with Dice: {val_dice:.4f}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"Total epochs: {num_epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_dice = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_dice = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rates'].append(current_lr)
            
            # Check if best model
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_dice, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs} - {epoch_time:.1f}s")
            print(f"{'='*60}")
            print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Best Val Dice: {self.best_val_dice:.4f}")
            print(f"{'='*60}\n")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE")
        print("="*60)
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print("="*60 + "\n")
        
        # Save final history
        history_path = self.config['checkpoint_dir'] / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def load_data_splits(splits_file):
    """Load train/val/test patient splits"""
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    return splits


def create_dataloaders(splits, data_path, batch_size=2, num_workers=4):
    """Create train and validation dataloaders"""
    
    # Get patient directories
    data_path = Path(data_path)
    all_patient_dirs = {d.name: d for d in data_path.iterdir() 
                       if d.is_dir() and d.name.startswith("BraTS-GLI-")}
    
    # Create datasets
    train_dirs = [all_patient_dirs[pid] for pid in splits['train']]
    val_dirs = [all_patient_dirs[pid] for pid in splits['val']]
    
    train_dataset = BraTSSegmentationDataset(train_dirs)
    val_dataset = BraTSSegmentationDataset(val_dirs)
    
    print(f"📊 Dataset sizes:")
    print(f"   Training: {len(train_dataset)} patients")
    print(f"   Validation: {len(val_dataset)} patients")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    # Configuration
    config = {
        'in_channels': 4,          # T1, T1CE, T2, FLAIR
        'num_classes': 4,          # Background, NCR, ED, ET
        'base_channels': 32,       # Base feature channels
        'learning_rate': 1e-4,     # Initial learning rate
        'weight_decay': 1e-5,      # L2 regularization
        'batch_size': 2,           # Batch size (reduce if GPU memory issues)
        'num_epochs': 50,          # Number of training epochs
        'num_workers': 0,          # DataLoader workers
    }
    
    # Setup paths
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    data_path = base_dir / "data/raw/brats_gli/training_data1_v2"
    splits_file = base_dir / "data/splits/data_splits.json"
    checkpoint_dir = base_dir / "models/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config['checkpoint_dir'] = checkpoint_dir
    
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
    
    # Initialize trainer
    trainer = SegmentationTrainer(config)
    
    # Start training
    history = trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print(f"\n✅ Training complete!")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"📊 Best validation Dice: {trainer.best_val_dice:.4f}")


if __name__ == "__main__":
    main()
