"""
Automatic Experiment Tracking System
Saves all results, visualizations, and reports for thesis
"""
import matplotlib
matplotlib.use('Agg')

import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil


class ExperimentTracker:
    """Track all experiments with automatic documentation"""
    
    def __init__(self, experiment_name, base_dir=None):
        if base_dir is None:
            base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
        
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.exp_dir = self.base_dir / "all_experiments" / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.results_dir = self.exp_dir / "results"
        self.figures_dir = self.exp_dir / "figures"
        
        for d in [self.checkpoints_dir, self.logs_dir, self.results_dir, self.figures_dir]:
            d.mkdir(exist_ok=True)
        
        print(f"📁 Experiment directory created: {self.exp_dir}")
        
        # Initialize experiment log
        self.log = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'status': 'initialized',
            'config': {},
            'training_history': {},
            'test_results': {},
            'notes': []
        }
        
        self.save_log()
    
    def save_config(self, config):
        """Save experiment configuration"""
        self.log['config'] = config
        
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved")
        self.save_log()
    
    def add_note(self, note):
        """Add a note to the experiment"""
        self.log['notes'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'note': note
        })
        print(f"📝 Note added: {note}")
        self.save_log()
    
    def save_checkpoint(self, epoch, model, optimizer, metrics, is_best=False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.log['config'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Regular checkpoint
        ckpt_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved (epoch {epoch}, metrics: {metrics})")
        
        self.save_log()
    
    def save_training_history(self, history):
        """Save complete training history"""
        self.log['training_history'] = history
        
        history_file = self.results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create training curves
        self.plot_training_curves(history)
        
        print(f"✅ Training history saved")
        self.save_log()
    
    def save_test_results(self, results):
        """Save test set evaluation results"""
        self.log['test_results'] = results
        
        results_file = self.results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create results visualization
        self.plot_test_results(results)
        
        print(f"✅ Test results saved")
        self.save_log()
    
    def plot_training_curves(self, history):
        """Create publication-quality training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(history['train_dice']) + 1)
        
        # Dice scores
        axes[0, 0].plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, history['val_dice'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Dice Score', fontsize=11)
        axes[0, 0].set_title('Dice Score Progression', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Losses
        axes[0, 1].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Loss', fontsize=11)
        axes[0, 1].set_title('Loss Progression', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Train-Val gap
        gap = np.array(history['train_dice']) - np.array(history['val_dice'])
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Train - Val Dice', fontsize=11)
        axes[1, 1].set_title('Generalization Gap', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_path = self.figures_dir / "training_curves.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved: {fig_path}")
    
    def plot_test_results(self, results):
        """Create test results visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart comparing with targets
        regions = ['WT', 'TC', 'ET']
        means = [results[f'dice_{r}']['mean'] for r in regions]
        stds = [results[f'dice_{r}']['std'] for r in regions]
        targets = [0.88, 0.80, 0.75]
        
        x = np.arange(len(regions))
        width = 0.35
        
        axes[0].bar(x - width/2, means, width, label='Our Model', 
                   yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        axes[0].bar(x + width/2, targets, width, label='Target',
                   color='lightcoral', alpha=0.8)
        
        axes[0].set_xlabel('Tumor Region', fontsize=11)
        axes[0].set_ylabel('Dice Score', fontsize=11)
        axes[0].set_title('Performance vs Targets', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Whole Tumor', 'Tumor Core', 'Enhancing'])
        axes[0].legend(fontsize=10)
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(True, axis='y', alpha=0.3)
        
        # Per-class performance
        classes = ['NCR', 'ED', 'ET']
        class_means = [results[f'dice_{c}']['mean'] for c in classes]
        class_stds = [results[f'dice_{c}']['std'] for c in classes]
        
        axes[1].bar(classes, class_means, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        axes[1].errorbar(classes, class_means, yerr=class_stds, fmt='none', 
                        color='black', capsize=5)
        
        axes[1].set_xlabel('Tumor Class', fontsize=11)
        axes[1].set_ylabel('Dice Score', fontsize=11)
        axes[1].set_title('Per-Class Performance', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (m, s) in enumerate(zip(class_means, class_stds)):
            axes[1].text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        fig_path = self.figures_dir / "test_results.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Test results visualization saved: {fig_path}")
    
    def save_log(self):
        """Save experiment log"""
        log_file = self.exp_dir / "experiment_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def finalize(self):
        """Finalize experiment and create summary"""
        self.log['status'] = 'completed'
        self.log['completion_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create text summary
        summary_file = self.exp_dir / "SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPERIMENT: {self.experiment_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-"*80 + "\n")
            for key, value in self.log['config'].items():
                f.write(f"  {key}: {value}\n")
            
            if self.log['test_results']:
                f.write("\n\nTEST RESULTS:\n")
                f.write("-"*80 + "\n")
                results = self.log['test_results']
                f.write(f"  WT Dice: {results['dice_WT']['mean']:.4f} ± {results['dice_WT']['std']:.4f}\n")
                f.write(f"  TC Dice: {results['dice_TC']['mean']:.4f} ± {results['dice_TC']['std']:.4f}\n")
                f.write(f"  ET Dice: {results['dice_ET']['mean']:.4f} ± {results['dice_ET']['std']:.4f}\n")
            
            if self.log['notes']:
                f.write("\n\nNOTES:\n")
                f.write("-"*80 + "\n")
                for note in self.log['notes']:
                    f.write(f"  [{note['timestamp']}] {note['note']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        self.save_log()
        print(f"\n✅ Experiment finalized: {self.exp_dir}")
        print(f"📄 Summary: {summary_file}")


def get_all_experiments():
    """List all tracked experiments"""
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project/all_experiments"
    
    if not base_dir.exists():
        return []
    
    experiments = []
    for exp_dir in sorted(base_dir.iterdir()):
        if exp_dir.is_dir():
            log_file = exp_dir / "experiment_log.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log = json.load(f)
                experiments.append({
                    'name': log['experiment_name'],
                    'path': exp_dir,
                    'timestamp': log['timestamp'],
                    'status': log.get('status', 'unknown')
                })
    
    return experiments


def compare_experiments(experiment_names):
    """Compare multiple experiments"""
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project/all_experiments"
    
    comparison = {}
    
    for exp_name in experiment_names:
        # Find most recent experiment with this name
        exp_dirs = sorted([d for d in base_dir.glob(f"{exp_name}_*")])
        if not exp_dirs:
            continue
        
        latest_exp = exp_dirs[-1]
        log_file = latest_exp / "experiment_log.json"
        
        with open(log_file, 'r') as f:
            log = json.load(f)
        
        comparison[exp_name] = {
            'config': log['config'],
            'test_results': log.get('test_results', {})
        }
    
    # Create comparison table
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100)
    print(f"{'Experiment':<20} {'WT Dice':<12} {'TC Dice':<12} {'ET Dice':<12} {'Notes':<30}")
    print("-"*100)
    
    for exp_name, data in comparison.items():
        if data['test_results']:
            wt = data['test_results']['dice_WT']['mean']
            tc = data['test_results']['dice_TC']['mean']
            et = data['test_results']['dice_ET']['mean']
            config = data['config']
            notes = f"ch={config.get('base_channels', 'N/A')}, ep={config.get('num_epochs', 'N/A')}"
            print(f"{exp_name:<20} {wt:<12.4f} {tc:<12.4f} {et:<12.4f} {notes:<30}")
    
    print("="*100)
    
    return comparison


# Example usage functions
def track_baseline():
    """Track the baseline experiment retroactively"""
    tracker = ExperimentTracker("baseline_unet")
    
    config = {
        'model': '3D U-Net',
        'base_channels': 32,
        'input_size': 128,
        'batch_size': 2,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'optimizer': 'Adam',
        'loss': 'Dice Loss',
        'augmentation': 'None'
    }
    
    tracker.save_config(config)
    tracker.add_note("Baseline model with standard U-Net architecture")
    
    # Copy existing results
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    
    # Copy checkpoint
    src_checkpoint = base_dir / "models/checkpoints/best_model.pth"
    if src_checkpoint.exists():
        shutil.copy(src_checkpoint, tracker.checkpoints_dir / "best_model.pth")
    
    # Load and save history
    history_file = base_dir / "models/checkpoints/training_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        tracker.save_training_history(history)
    
    # Load and save test results
    results_file = base_dir / "results/test_results_summary.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        tracker.save_test_results(results)
    
    tracker.finalize()
    
    return tracker


if __name__ == "__main__":
    # Track baseline experiment
    print("📁 Tracking baseline experiment...")
    track_baseline()
    
    print("\n✅ Baseline experiment tracked!")
    print("\nAll future experiments will be automatically tracked.")
    print("Use ExperimentTracker class in your training scripts.")
