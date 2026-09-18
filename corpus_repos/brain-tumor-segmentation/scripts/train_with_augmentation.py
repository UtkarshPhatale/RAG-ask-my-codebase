"""
Complete Training Script with Safe Augmentation
Uses baseline model (32 channels) with conservative augmentation
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
from augmentation_safe import SafeMedicalAugmentation
from train_segmentation import DiceLoss, load_data_splits


class AugmentedDataset(BraTSSegmentationDataset):
    """Dataset with optional augmentation"""
    
    def __init__(self, patient_dirs, use_augmentation=False, crop_size=(128, 128, 128)):
        super().__init__(patient_dirs, crop_size=crop_size)
        
        if use_augmentation:
            self.augmentation = SafeMedicalAugmentation()
        else:
            self.augmentation = None
    
    def __getitem__(self, idx):
        volume, segmentation = super().__getitem__(idx)
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            volume, segmentation = self.augmentation(volume, segmentation)
        
        return volume, segmentation


def create_dataloaders(splits, data_path, config):
    """Create train and validation dataloaders"""
    
    data_path = Path(data_path)
    all_patient_dirs = {d.name: d for d in data_path.iterdir() 
                       if d.is_dir() and d.name.startswith("BraTS-GLI-")}
    
    # Training dataset WITH augmentation
    train_dirs = [all_patient_dirs[pid] for pid in splits['train']]
    train_dataset = AugmentedDataset(
        train_dirs,
        use_augmentation=config['use_augmentation']
    )
    
    # Validation dataset WITHOUT augmentation
    val_dirs = [all_patient_dirs[pid] for pid in splits['val']]
    val_dataset = AugmentedDataset(
        val_dirs,
        use_augmentation=False
    )
    
    print(f"📊 Dataset sizes:")
    print(f"   Training: {len(train_dataset)} patients")
    print(f"   Augmentation: {'✅ Enabled' if config['use_augmentation'] else '❌ Disabled'}")
    print(f"   Validation: {len(val_dataset)} patients")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


class Trainer:
    """Training pipeline"""
    
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
        
        total_params, _ = self.model.count_parameters()
        print(f"Model parameters: {total_params:,}")
        
        # Loss and optimizer
        self.criterion = DiceLoss()
        
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
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        self.best_val_dice = 0.0
    
    def calculate_dice_score(self, predictions, targets):
        """Calculate Dice score"""
        pred_classes = torch.argmax(predictions, dim=1)
        
        dice_scores = []
        for class_id in [1, 2, 3]:
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
        
        for volumes, segmentations in progress_bar:
            volumes = volumes.to(self.device)
            segmentations = segmentations.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(volumes)
            
            # Calculate loss
            loss = self.criterion(outputs, segmentations)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            dice_score = self.calculate_dice_score(outputs, segmentations)
            total_dice += dice_score
            
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
                
                outputs = self.model(volumes)
                loss = self.criterion(outputs, segmentations)
                
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
        
        checkpoint_dir = self.config['checkpoint_dir']
        
        # Save every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f'augmented_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'augmented_best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with Dice: {val_dice:.4f}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        
        aug_status = "WITH" if self.config['use_augmentation'] else "WITHOUT"
        
        print("\n" + "="*60)
        print(f"🚀 TRAINING {aug_status} AUGMENTATION")
        print("="*60)
        print(f"Model: 3D U-Net (base_channels={self.config['base_channels']})")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
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
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        
        if self.config['use_augmentation']:
            improvement = self.best_val_dice - 0.6615  # vs baseline
            print(f"Improvement over baseline: {improvement:+.4f}")
        
        print("="*60 + "\n")
        
        # Save final history
        history_path = self.config['checkpoint_dir'] / 'augmented_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'in_channels': 4,
        'num_classes': 4,
        'base_channels': 32,            # Baseline model size
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 2,
        'num_epochs': 75,               # CHANGE THIS FOR TESTING
        'num_workers': 0,
        'use_augmentation': True,       # CHANGE THIS TO TEST
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
    train_loader, val_loader = create_dataloaders(splits, data_path, config)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    history = trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print(f"\n✅ Training complete!")
    print(f"📁 Checkpoints: {checkpoint_dir}")
    print(f"📊 Best Val Dice: {trainer.best_val_dice:.4f}")


if __name__ == "__main__":
    main()
