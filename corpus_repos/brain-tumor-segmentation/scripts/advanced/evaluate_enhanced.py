"""
Comprehensive Model Evaluation Script
=====================================

Load saved checkpoints and perform detailed analysis:
- Calculate all metrics on test set
- Generate visualizations
- Compare predictions with ground truth
- Create publication-ready figures

Usage:
    python evaluate_model.py --checkpoint checkpoints/unet3d_best_overall.pth --test_data test_loader.pkl
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pandas as pd
from scipy import stats

from metrics import SegmentationMetrics, MetricsTracker


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations"""
    
    def __init__(self, model, checkpoint_path, device='cuda'):
        """
        Args:
            model: Model architecture (without weights)
            checkpoint_path: Path to saved checkpoint
            device: Device to run evaluation on
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Setup metrics
        self.metrics_calc = SegmentationMetrics(num_classes=4)
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Evaluation device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"\nCheckpoint Information:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Metrics: {checkpoint.get('metrics', 'N/A')}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    def evaluate_dataset(self, data_loader, save_predictions=False, output_dir=None):
        """
        Evaluate model on entire dataset
        
        Args:
            data_loader: DataLoader for test set
            save_predictions: Whether to save predictions as .npy files
            output_dir: Directory to save predictions
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print("\n" + "="*80)
        print("EVALUATING MODEL ON DATASET")
        print("="*80)
        
        self.model.eval()
        tracker = MetricsTracker(self.metrics_calc)
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(data_loader):
                images = images.to(self.device)
                masks_np = masks.numpy()
                
                # Forward pass
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                preds_np = preds.cpu().numpy()
                
                # Calculate metrics for each sample in batch
                for i in range(preds_np.shape[0]):
                    tracker.add_sample(
                        preds_np[i],
                        masks_np[i],
                        patient_id=f'patient_{batch_idx * data_loader.batch_size + i}'
                    )
                
                all_predictions.append(preds_np)
                all_ground_truths.append(masks_np)
                
                # Progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Get summary statistics
        summary = tracker.get_summary_statistics()
        
        # Print summary
        tracker.print_summary()
        
        # Save detailed results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save per-sample metrics
            tracker.save_results(output_dir / 'per_sample_metrics.csv')
            
            # Save summary statistics
            self._save_summary_json(summary, output_dir / 'summary_statistics.json')
            
            # Save predictions if requested
            if save_predictions:
                np.save(output_dir / 'predictions.npy', np.concatenate(all_predictions))
                np.save(output_dir / 'ground_truths.npy', np.concatenate(all_ground_truths))
        
        return summary, tracker.all_metrics
    
    def _save_summary_json(self, summary, filepath):
        """Save summary statistics to JSON"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {filepath}")
    
    def visualize_predictions(self, data_loader, num_samples=10, output_dir='./visualizations'):
        """
        Create visualizations comparing predictions with ground truth
        
        Args:
            data_loader: DataLoader for test set
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations for {num_samples} samples...")
        
        self.model.eval()
        samples_visualized = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(data_loader):
                if samples_visualized >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Visualize each sample in batch
                for i in range(images.shape[0]):
                    if samples_visualized >= num_samples:
                        break
                    
                    self._visualize_sample(
                        images[i].cpu().numpy(),
                        masks[i].numpy(),
                        preds[i].cpu().numpy(),
                        sample_id=f'sample_{samples_visualized:03d}',
                        save_dir=output_dir
                    )
                    
                    samples_visualized += 1
        
        print(f"Visualizations saved to: {output_dir}")
    
    def _visualize_sample(self, image, ground_truth, prediction, sample_id, save_dir):
        """Create visualization for one sample"""
        # Select middle slices for visualization
        slice_idx = image.shape[-1] // 2  # Middle axial slice
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Image (use first channel - T1CE typically)
        axes[0, 0].imshow(image[0, :, :, slice_idx], cmap='gray')
        axes[0, 0].set_title('Input Image (T1CE)')
        axes[0, 0].axis('off')
        
        # Ground truth segmentation
        axes[0, 1].imshow(image[0, :, :, slice_idx], cmap='gray')
        axes[0, 1].imshow(ground_truth[:, :, slice_idx], cmap='jet', alpha=0.5)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].imshow(image[0, :, :, slice_idx], cmap='gray')
        axes[0, 2].imshow(prediction[:, :, slice_idx], cmap='jet', alpha=0.5)
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
        
        # Per-class comparison
        class_names = ['NCR', 'ED', 'ET']
        class_labels = [1, 2, 3]  # After remapping
        
        for idx, (name, label) in enumerate(zip(class_names, class_labels)):
            gt_mask = (ground_truth == label)
            pred_mask = (prediction == label)
            
            # Show comparison
            comparison = np.zeros_like(gt_mask, dtype=float)
            comparison[gt_mask & pred_mask] = 1.0  # True positive (green)
            comparison[gt_mask & ~pred_mask] = 0.5  # False negative (yellow)
            comparison[~gt_mask & pred_mask] = 0.3  # False positive (red)
            
            axes[1, idx].imshow(image[0, :, :, slice_idx], cmap='gray')
            axes[1, idx].imshow(comparison[:, :, slice_idx], cmap='RdYlGn', alpha=0.6, vmin=0, vmax=1)
            axes[1, idx].set_title(f'{name}: TP (green), FN (yellow), FP (red)')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{sample_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_performance_plots(self, metrics_list, output_dir='./plots'):
        """
        Generate comprehensive performance analysis plots
        
        Args:
            metrics_list: List of per-sample metrics from evaluation
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating performance analysis plots...")
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # 1. Distribution plots for main metrics
        self._plot_metric_distributions(df, output_dir)
        
        # 2. Correlation analysis
        self._plot_metric_correlations(df, output_dir)
        
        # 3. Per-class performance comparison
        self._plot_class_comparison(df, output_dir)
        
        # 4. Box plots showing variability
        self._plot_performance_boxplots(df, output_dir)
        
        print(f"Performance plots saved to: {output_dir}")
    
    def _plot_metric_distributions(self, df, output_dir):
        """Plot distribution of Dice scores"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['WT_Dice', 'TC_Dice', 'ET_Dice']
        colors = ['blue', 'green', 'red']
        
        for ax, metric, color in zip(axes, metrics, colors):
            values = df[metric].values
            
            # Histogram
            ax.hist(values, bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
            ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
            
            ax.set_xlabel('Dice Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dice_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_correlations(self, df, output_dir):
        """Plot correlation between different metrics"""
        metrics = ['WT_Dice', 'TC_Dice', 'ET_Dice', 'NCR_Dice', 'ED_Dice']
        
        # Calculate correlation matrix
        corr_matrix = df[metrics].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(metrics)
        
        # Values in cells
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color='black')
        
        ax.set_title('Metric Correlation Matrix')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / 'metric_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_comparison(self, df, output_dir):
        """Compare performance across tumor classes"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = ['NCR', 'ED', 'ET']
        metrics_to_plot = ['Dice', 'Sensitivity', 'Specificity', 'Precision']
        
        x = np.arange(len(classes))
        width = 0.2
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [df[f'{cls}_{metric}'].mean() for cls in classes]
            offset = (idx - len(metrics_to_plot)/2) * width
            ax.bar(x + offset, values, width, label=metric)
        
        ax.set_xlabel('Tumor Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_boxplots(self, df, output_dir):
        """Create box plots showing performance variability"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # BraTS composite metrics
        brats_data = [df['WT_Dice'], df['TC_Dice'], df['ET_Dice']]
        axes[0].boxplot(brats_data, labels=['WT', 'TC', 'ET'])
        axes[0].set_ylabel('Dice Score')
        axes[0].set_title('BraTS Composite Metrics Variability')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Per-class metrics
        class_data = [df['NCR_Dice'], df['ED_Dice'], df['ET_Dice']]
        axes[1].boxplot(class_data, labels=['NCR', 'ED', 'ET'])
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Per-Class Dice Score Variability')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_with_baseline(self, baseline_metrics, current_metrics, output_dir='./comparison'):
        """
        Compare current model with baseline
        
        Args:
            baseline_metrics: Metrics from baseline model
            current_metrics: Metrics from current model
            output_dir: Directory to save comparison plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating comparison with baseline...")
        
        # Convert to DataFrames
        baseline_df = pd.DataFrame(baseline_metrics)
        current_df = pd.DataFrame(current_metrics)
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['WT_Dice', 'TC_Dice', 'ET_Dice']
        colors = ['blue', 'green', 'red']
        
        for ax, metric, color in zip(axes, metrics, colors):
            baseline_mean = baseline_df[metric].mean()
            current_mean = current_df[metric].mean()
            
            improvement = ((current_mean - baseline_mean) / baseline_mean) * 100
            
            # Bar chart
            x = ['Baseline', 'Current']
            values = [baseline_mean, current_mean]
            bars = ax.bar(x, values, color=[color, 'orange'], alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom')
            
            ax.set_ylabel('Dice Score')
            ax.set_title(f'{metric}\nImprovement: {improvement:+.2f}%')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print improvement summary
        print("\nImprovement Summary:")
        print("-" * 60)
        for metric in metrics:
            baseline_mean = baseline_df[metric].mean()
            current_mean = current_df[metric].mean()
            improvement = ((current_mean - baseline_mean) / baseline_mean) * 100
            print(f"{metric:12} | Baseline: {baseline_mean:.4f} | Current: {current_mean:.4f} | "
                  f"Improvement: {improvement:+.2f}%")
        print("-" * 60)


# Command-line interface
def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_loader', type=str, required=True, help='Path to test data loader')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Directory for outputs')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Load model and evaluator
    # Note: You need to provide your model architecture here
    # from your_model import UNet3D
    # model = UNet3D(in_channels=4, out_channels=4)
    
    # evaluator = ModelEvaluator(model, args.checkpoint)
    
    # Load test data
    # test_loader = torch.load(args.test_loader)
    
    # Evaluate
    # summary, metrics_list = evaluator.evaluate_dataset(
    #     test_loader,
    #     save_predictions=True,
    #     output_dir=args.output_dir
    # )
    
    # Generate plots
    # evaluator.generate_performance_plots(metrics_list, output_dir=args.output_dir + '/plots')
    
    # if args.visualize:
    #     evaluator.visualize_predictions(test_loader, num_samples=args.num_vis,
    #                                    output_dir=args.output_dir + '/visualizations')
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    print("Comprehensive Model Evaluation Script")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Load and evaluate any saved checkpoint")
    print("  ✓ Calculate comprehensive metrics")
    print("  ✓ Generate publication-quality visualizations")
    print("  ✓ Compare with baseline models")
    print("\nUsage:")
    print("  python evaluate_model.py --checkpoint model.pth --test_loader data.pkl")
