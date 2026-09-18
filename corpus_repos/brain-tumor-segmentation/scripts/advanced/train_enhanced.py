"""
Enhanced Training Script with Comprehensive Checkpointing
==========================================================

Features:
- Saves multiple checkpoint types (best overall, best per-metric, periodic)
- Tracks comprehensive metrics during training
- Enables loading checkpoints for analysis
- Generates training curves and logs

Usage:
    python train_enhanced.py --config config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from metrics import SegmentationMetrics, MetricsTracker


class CheckpointManager:
    """Manages saving and loading of model checkpoints"""
    
    def __init__(self, save_dir, model_name="unet3d"):
        """
        Args:
            save_dir: Directory to save checkpoints
            model_name: Name prefix for checkpoint files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # Track best metrics
        self.best_metrics = {
            'best_overall_dice': 0.0,
            'best_wt_dice': 0.0,
            'best_tc_dice': 0.0,
            'best_et_dice': 0.0,
            'best_hausdorff': float('inf'),
        }
        
        # Track checkpoint paths
        self.checkpoint_paths = {
            'best_overall': None,
            'best_wt': None,
            'best_tc': None,
            'best_et': None,
            'latest': None,
        }
    
    def save_checkpoint(self, epoch, model, optimizer, metrics, checkpoint_type='periodic'):
        """
        Save a checkpoint
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            metrics: Dictionary of current metrics
            checkpoint_type: Type of checkpoint ('best_overall', 'best_wt', 'periodic', etc.)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Determine filename
        if checkpoint_type == 'periodic':
            filename = f"{self.model_name}_epoch_{epoch:03d}.pth"
        elif checkpoint_type == 'latest':
            filename = f"{self.model_name}_latest.pth"
        else:
            filename = f"{self.model_name}_{checkpoint_type}.pth"
        
        filepath = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Update checkpoint paths
        if checkpoint_type in self.checkpoint_paths:
            self.checkpoint_paths[checkpoint_type] = str(filepath)
        
        print(f"Checkpoint saved: {filepath}")
        
        return str(filepath)
    
    def update_best_checkpoints(self, epoch, model, optimizer, metrics):
        """Check if current metrics are best and save accordingly"""
        saved_checkpoints = []
        
        # Check overall Dice (mean of WT, TC, ET)
        current_overall = (metrics.get('WT_Dice', 0) + 
                          metrics.get('TC_Dice', 0) + 
                          metrics.get('ET_Dice', 0)) / 3.0
        
        if current_overall > self.best_metrics['best_overall_dice']:
            self.best_metrics['best_overall_dice'] = current_overall
            path = self.save_checkpoint(epoch, model, optimizer, metrics, 'best_overall')
            saved_checkpoints.append(('best_overall', current_overall))
        
        # Check best WT Dice
        if metrics.get('WT_Dice', 0) > self.best_metrics['best_wt_dice']:
            self.best_metrics['best_wt_dice'] = metrics['WT_Dice']
            path = self.save_checkpoint(epoch, model, optimizer, metrics, 'best_wt')
            saved_checkpoints.append(('best_wt', metrics['WT_Dice']))
        
        # Check best TC Dice
        if metrics.get('TC_Dice', 0) > self.best_metrics['best_tc_dice']:
            self.best_metrics['best_tc_dice'] = metrics['TC_Dice']
            path = self.save_checkpoint(epoch, model, optimizer, metrics, 'best_tc')
            saved_checkpoints.append(('best_tc', metrics['TC_Dice']))
        
        # Check best ET Dice
        if metrics.get('ET_Dice', 0) > self.best_metrics['best_et_dice']:
            self.best_metrics['best_et_dice'] = metrics['ET_Dice']
            path = self.save_checkpoint(epoch, model, optimizer, metrics, 'best_et')
            saved_checkpoints.append(('best_et', metrics['ET_Dice']))
        
        # Always save latest
        self.save_checkpoint(epoch, model, optimizer, metrics, 'latest')
        
        return saved_checkpoints
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def get_best_checkpoint_path(self, metric_type='best_overall'):
        """Get path to best checkpoint"""
        return self.checkpoint_paths.get(metric_type)
    
    def save_training_history(self, history, filename='training_history.json'):
        """Save training history to JSON"""
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {filepath}")


class TrainingLogger:
    """Log training progress and generate visualizations"""
    
    def __init__(self, log_dir):
        """
        Args:
            log_dir: Directory for logs and plots
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'val_wt_dice': [],
            'val_tc_dice': [],
            'val_et_dice': [],
            'learning_rate': [],
            'epoch': [],
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, train_dice, val_metrics, lr):
        """Log metrics for one epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_dice'].append(train_dice)
        self.history['val_dice'].append(val_metrics.get('Mean_Dice', 0))
        self.history['val_wt_dice'].append(val_metrics.get('WT_Dice', 0))
        self.history['val_tc_dice'].append(val_metrics.get('TC_Dice', 0))
        self.history['val_et_dice'].append(val_metrics.get('ET_Dice', 0))
        self.history['learning_rate'].append(lr)
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], 
                        label='Train Loss', marker='o')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], 
                        label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice score curves
        axes[0, 1].plot(self.history['epoch'], self.history['train_dice'], 
                        label='Train Dice', marker='o')
        axes[0, 1].plot(self.history['epoch'], self.history['val_dice'], 
                        label='Val Dice', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # BraTS composite metrics
        axes[1, 0].plot(self.history['epoch'], self.history['val_wt_dice'], 
                        label='WT Dice', marker='o')
        axes[1, 0].plot(self.history['epoch'], self.history['val_tc_dice'], 
                        label='TC Dice', marker='s')
        axes[1, 0].plot(self.history['epoch'], self.history['val_et_dice'], 
                        label='ET Dice', marker='^')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].set_title('BraTS Composite Metrics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(self.history['epoch'], self.history['learning_rate'], 
                        marker='o', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
        
        plt.close()
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")


class EnhancedTrainer:
    """Enhanced trainer with comprehensive checkpointing and metrics"""
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        self.criterion = nn.CrossEntropyLoss()  # Or Dice Loss
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Setup checkpoint manager and logger
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.get('checkpoint_dir', './checkpoints'),
            model_name=config.get('model_name', 'unet3d')
        )
        self.logger = TrainingLogger(
            log_dir=config.get('log_dir', './logs')
        )
        
        # Setup metrics calculator
        self.metrics_calc = SegmentationMetrics(num_classes=4)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate Dice score
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                dice = self.metrics_calc.whole_tumor_dice(
                    preds.cpu().numpy(), 
                    masks.cpu().numpy()
                )
                total_dice += dice
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"Loss = {loss.item():.4f}, Dice = {dice:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Metrics tracker
        tracker = MetricsTracker(self.metrics_calc)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate comprehensive metrics
                preds = torch.argmax(outputs, dim=1)
                tracker.add_sample(
                    preds.cpu().numpy(), 
                    masks.cpu().numpy()
                )
        
        avg_loss = total_loss / num_batches
        summary = tracker.get_summary_statistics()
        
        return avg_loss, summary['mean']
    
    def train(self, num_epochs, save_every=5):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save periodic checkpoint every N epochs
        """
        print("="*80)
        print("STARTING ENHANCED TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*80 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)
            
            # Training
            train_loss, train_dice = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            
            # Validation
            val_loss, val_metrics = self.validate_epoch()
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val WT Dice: {val_metrics['WT_Dice']:.4f}")
            print(f"Val TC Dice: {val_metrics['TC_Dice']:.4f}")
            print(f"Val ET Dice: {val_metrics['ET_Dice']:.4f}")
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['WT_Dice'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != current_lr:
                print(f"Learning rate reduced: {current_lr:.6f} → {new_lr:.6f}")
            
            # Log metrics
            self.logger.log_epoch(
                epoch, train_loss, val_loss, train_dice, val_metrics, current_lr
            )
            
            # Save best checkpoints
            saved = self.checkpoint_manager.update_best_checkpoints(
                epoch, self.model, self.optimizer, val_metrics
            )
            
            if saved:
                print("New best model(s) saved:")
                for metric_type, value in saved:
                    print(f"  - {metric_type}: {value:.4f}")
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.checkpoint_manager.save_checkpoint(
                    epoch, self.model, self.optimizer, val_metrics, 'periodic'
                )
            
            print("-" * 80)
        
        # Save final results
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        self.logger.plot_training_curves()
        self.logger.save_history()
        self.checkpoint_manager.save_training_history(
            self.logger.history, 'training_history.json'
        )
        
        # Print best metrics
        print("\nBest Metrics Achieved:")
        print(f"  Best Overall Dice: {self.checkpoint_manager.best_metrics['best_overall_dice']:.4f}")
        print(f"  Best WT Dice: {self.checkpoint_manager.best_metrics['best_wt_dice']:.4f}")
        print(f"  Best TC Dice: {self.checkpoint_manager.best_metrics['best_tc_dice']:.4f}")
        print(f"  Best ET Dice: {self.checkpoint_manager.best_metrics['best_et_dice']:.4f}")
        print("\nCheckpoint paths:")
        for key, path in self.checkpoint_manager.checkpoint_paths.items():
            if path:
                print(f"  {key}: {path}")
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    print("Enhanced Training Script with Comprehensive Checkpointing")
    print("=" * 60)
    print("\nThis script provides:")
    print("  ✓ Multiple checkpoint types (best overall, best per-metric)")
    print("  ✓ Comprehensive metrics tracking")
    print("  ✓ Training curve visualization")
    print("  ✓ Easy checkpoint loading for analysis")
    print("\nUsage example:")
    print("""
    config = {
        'learning_rate': 0.001,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'model_name': 'unet3d_attention'
    }
    
    trainer = EnhancedTrainer(model, train_loader, val_loader, config)
    trainer.train(num_epochs=75, save_every=5)
    """)
