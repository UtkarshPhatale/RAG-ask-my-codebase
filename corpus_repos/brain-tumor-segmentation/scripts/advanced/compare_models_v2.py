"""
Model Comparison Script
=======================

Compare multiple trained models to track improvement progress
and generate publication-ready comparison tables and figures.

Usage:
    python compare_models.py --models baseline attention_unet ensemble --output comparison_results/
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

from metrics import SegmentationMetrics, MetricsTracker


class ModelComparator:
    """Compare multiple models and generate comprehensive comparison"""
    
    def __init__(self, output_dir='./model_comparison'):
        """
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.metrics_calc = SegmentationMetrics(num_classes=4)
    
    def add_model(self, model_name, checkpoint_path, model_architecture, description=""):
        """
        Add a model for comparison
        
        Args:
            model_name: Identifier for the model
            checkpoint_path: Path to saved checkpoint
            model_architecture: Model class
            description: Description of model/technique
        """
        print(f"\nLoading model: {model_name}")
        
        # Load model
        model = model_architecture
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store model and metadata
        self.models[model_name] = {
            'model': model,
            'checkpoint': checkpoint,
            'description': description,
            'checkpoint_path': checkpoint_path
        }
        
        print(f"  ✓ Loaded from: {checkpoint_path}")
        print(f"  ✓ Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  ✓ Description: {description}")
    
    def evaluate_all_models(self, test_loader, device='cuda'):
        """
        Evaluate all added models on test set
        
        Args:
            test_loader: DataLoader for test set
            device: Device to run evaluation on
        """
        print("\n" + "="*80)
        print("EVALUATING ALL MODELS")
        print("="*80)
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        for model_name, model_data in self.models.items():
            print(f"\n--- Evaluating: {model_name} ---")
            
            model = model_data['model'].to(device)
            model.eval()
            
            tracker = MetricsTracker(self.metrics_calc)
            
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(test_loader):
                    images = images.to(device)
                    masks_np = masks.numpy()
                    
                    # Forward pass
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    preds_np = preds.cpu().numpy()
                    
                    # Calculate metrics
                    for i in range(preds_np.shape[0]):
                        tracker.add_sample(preds_np[i], masks_np[i])
                    
                    if (batch_idx + 1) % 20 == 0:
                        print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
            
            # Store results
            summary = tracker.get_summary_statistics()
            self.results[model_name] = {
                'summary': summary,
                'per_sample': tracker.all_metrics,
                'description': model_data['description']
            }
            
            # Print summary
            print(f"\n  Results for {model_name}:")
            print(f"    WT Dice: {summary['mean']['WT_Dice']:.4f} ± {summary['std']['WT_Dice']:.4f}")
            print(f"    TC Dice: {summary['mean']['TC_Dice']:.4f} ± {summary['std']['TC_Dice']:.4f}")
            print(f"    ET Dice: {summary['mean']['ET_Dice']:.4f} ± {summary['std']['ET_Dice']:.4f}")
        
        print("\n" + "="*80)
        print("ALL MODELS EVALUATED")
        print("="*80)
    
    def generate_comparison_table(self):
        """Generate comprehensive comparison table"""
        print("\nGenerating comparison table...")
        
        # Prepare data for table
        table_data = []
        
        for model_name, result in self.results.items():
            summary = result['summary']
            
            row = {
                'Model': model_name,
                'Description': result['description'],
                'WT_Dice_Mean': summary['mean']['WT_Dice'],
                'WT_Dice_Std': summary['std']['WT_Dice'],
                'WT_Dice_Median': summary['median']['WT_Dice'],
                'TC_Dice_Mean': summary['mean']['TC_Dice'],
                'TC_Dice_Std': summary['std']['TC_Dice'],
                'TC_Dice_Median': summary['median']['TC_Dice'],
                'ET_Dice_Mean': summary['mean']['ET_Dice'],
                'ET_Dice_Std': summary['std']['ET_Dice'],
                'ET_Dice_Median': summary['median']['ET_Dice'],
                'NCR_Dice': summary['mean']['NCR_Dice'],
                'ED_Dice': summary['mean']['ED_Dice'],
                'Mean_Dice': summary['mean']['Mean_Dice']
            }
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Sort by WT Dice (descending)
        df = df.sort_values('WT_Dice_Mean', ascending=False)
        
        # Save to CSV
        csv_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  ✓ Table saved to: {csv_path}")
        
        # Create formatted table for thesis
        self._create_thesis_table(df)
        
        return df
    
    def _create_thesis_table(self, df):
        """Create LaTeX-formatted table for thesis"""
        latex_path = self.output_dir / 'comparison_table.tex'
        
        # Select key columns
        thesis_cols = ['Model', 'WT_Dice_Mean', 'WT_Dice_Std', 
                      'TC_Dice_Mean', 'TC_Dice_Std',
                      'ET_Dice_Mean', 'ET_Dice_Std']
        
        thesis_df = df[thesis_cols].copy()
        
        # Format for LaTeX
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison of Segmentation Models on BraTS Test Set}\n")
            f.write("\\label{tab:model_comparison}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\n")
            f.write("Model & WT Dice & TC Dice & ET Dice \\\\\n")
            f.write("\\hline\n")
            
            for _, row in thesis_df.iterrows():
                f.write(f"{row['Model']} & "
                       f"{row['WT_Dice_Mean']:.3f} $\\pm$ {row['WT_Dice_Std']:.3f} & "
                       f"{row['TC_Dice_Mean']:.3f} $\\pm$ {row['TC_Dice_Std']:.3f} & "
                       f"{row['ET_Dice_Mean']:.3f} $\\pm$ {row['ET_Dice_Std']:.3f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"  ✓ LaTeX table saved to: {latex_path}")
    
    def plot_comparison_bars(self):
        """Create bar chart comparing models"""
        print("\nGenerating comparison bar charts...")
        
        models = list(self.results.keys())
        n_models = len(models)
        
        # Extract metrics
        wt_scores = [self.results[m]['summary']['mean']['WT_Dice'] for m in models]
        tc_scores = [self.results[m]['summary']['mean']['TC_Dice'] for m in models]
        et_scores = [self.results[m]['summary']['mean']['ET_Dice'] for m in models]
        
        wt_stds = [self.results[m]['summary']['std']['WT_Dice'] for m in models]
        tc_stds = [self.results[m]['summary']['std']['TC_Dice'] for m in models]
        et_stds = [self.results[m]['summary']['std']['ET_Dice'] for m in models]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        x = np.arange(n_models)
        width = 0.6
        
        # WT Dice
        bars1 = axes[0].bar(x, wt_scores, width, yerr=wt_stds, capsize=5,
                           color='skyblue', edgecolor='black', alpha=0.8)
        axes[0].set_ylabel('Dice Score', fontsize=12)
        axes[0].set_title('Whole Tumor (WT) Dice', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylim([0.6, 1.0])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(y=0.88, color='red', linestyle='--', linewidth=2, 
                       label='Literature Target (0.88)')
        axes[0].legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
        
        # TC Dice
        bars2 = axes[1].bar(x, tc_scores, width, yerr=tc_stds, capsize=5,
                           color='lightgreen', edgecolor='black', alpha=0.8)
        axes[1].set_ylabel('Dice Score', fontsize=12)
        axes[1].set_title('Tumor Core (TC) Dice', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylim([0.5, 1.0])
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=0.85, color='red', linestyle='--', linewidth=2,
                       label='Literature Target (0.85)')
        axes[1].legend()
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
        
        # ET Dice
        bars3 = axes[2].bar(x, et_scores, width, yerr=et_stds, capsize=5,
                           color='salmon', edgecolor='black', alpha=0.8)
        axes[2].set_ylabel('Dice Score', fontsize=12)
        axes[2].set_title('Enhancing Tumor (ET) Dice', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylim([0.5, 1.0])
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].axhline(y=0.80, color='red', linestyle='--', linewidth=2,
                       label='Literature Target (0.80)')
        axes[2].legend()
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        bar_path = self.output_dir / 'comparison_bars.png'
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Bar chart saved to: {bar_path}")
    
    def plot_improvement_trajectory(self, baseline_name):
        """Plot improvement from baseline to each model"""
        print("\nGenerating improvement trajectory...")
        
        if baseline_name not in self.results:
            print(f"  ✗ Baseline '{baseline_name}' not found")
            return
        
        baseline_wt = self.results[baseline_name]['summary']['mean']['WT_Dice']
        baseline_tc = self.results[baseline_name]['summary']['mean']['TC_Dice']
        baseline_et = self.results[baseline_name]['summary']['mean']['ET_Dice']
        
        # Calculate improvements
        improvements = []
        for model_name in self.results.keys():
            if model_name == baseline_name:
                continue
            
            wt = self.results[model_name]['summary']['mean']['WT_Dice']
            tc = self.results[model_name]['summary']['mean']['TC_Dice']
            et = self.results[model_name]['summary']['mean']['ET_Dice']
            
            improvements.append({
                'model': model_name,
                'wt_improvement': (wt - baseline_wt) * 100,
                'tc_improvement': (tc - baseline_tc) * 100,
                'et_improvement': (et - baseline_et) * 100
            })
        
        # Sort by WT improvement
        improvements.sort(key=lambda x: x['wt_improvement'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = [imp['model'] for imp in improvements]
        wt_imps = [imp['wt_improvement'] for imp in improvements]
        tc_imps = [imp['tc_improvement'] for imp in improvements]
        et_imps = [imp['et_improvement'] for imp in improvements]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.barh(x - width, wt_imps, width, label='WT Improvement', color='skyblue')
        ax.barh(x, tc_imps, width, label='TC Improvement', color='lightgreen')
        ax.barh(x + width, et_imps, width, label='ET Improvement', color='salmon')
        
        ax.set_yticks(x)
        ax.set_yticklabels(models)
        ax.set_xlabel('Improvement over Baseline (%)', fontsize=12)
        ax.set_title(f'Model Improvements Compared to {baseline_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linewidth=1.5)
        
        plt.tight_layout()
        
        # Save
        traj_path = self.output_dir / 'improvement_trajectory.png'
        plt.savefig(traj_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Trajectory plot saved to: {traj_path}")
    
    def plot_distribution_comparison(self):
        """Compare Dice score distributions across models"""
        print("\nGenerating distribution comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for model_name, result in self.results.items():
            per_sample = result['per_sample']
            df = pd.DataFrame(per_sample)
            
            # WT distribution
            axes[0].hist(df['WT_Dice'], bins=30, alpha=0.5, label=model_name, edgecolor='black')
            
            # TC distribution
            axes[1].hist(df['TC_Dice'], bins=30, alpha=0.5, label=model_name, edgecolor='black')
            
            # ET distribution
            axes[2].hist(df['ET_Dice'], bins=30, alpha=0.5, label=model_name, edgecolor='black')
        
        # WT
        axes[0].set_xlabel('Dice Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('WT Dice Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # TC
        axes[1].set_xlabel('Dice Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('TC Dice Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # ET
        axes[2].set_xlabel('Dice Score')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('ET Dice Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        dist_path = self.output_dir / 'distribution_comparison.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Distribution plot saved to: {dist_path}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\nGenerating summary report...")
        
        report_path = self.output_dir / 'comparison_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Find best model
            best_wt_model = max(self.results.items(), 
                              key=lambda x: x[1]['summary']['mean']['WT_Dice'])
            best_tc_model = max(self.results.items(),
                              key=lambda x: x[1]['summary']['mean']['TC_Dice'])
            best_et_model = max(self.results.items(),
                              key=lambda x: x[1]['summary']['mean']['ET_Dice'])
            
            f.write("BEST PERFORMING MODELS\n")
            f.write("-"*80 + "\n")
            f.write(f"Best WT Dice: {best_wt_model[0]} "
                   f"({best_wt_model[1]['summary']['mean']['WT_Dice']:.4f})\n")
            f.write(f"Best TC Dice: {best_tc_model[0]} "
                   f"({best_tc_model[1]['summary']['mean']['TC_Dice']:.4f})\n")
            f.write(f"Best ET Dice: {best_et_model[0]} "
                   f"({best_et_model[1]['summary']['mean']['ET_Dice']:.4f})\n")
            f.write("\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            for model_name, result in sorted(self.results.items(),
                                            key=lambda x: x[1]['summary']['mean']['WT_Dice'],
                                            reverse=True):
                summary = result['summary']
                
                f.write(f"Model: {model_name}\n")
                f.write(f"Description: {result['description']}\n")
                f.write(f"\nBraTS Composite Metrics:\n")
                f.write(f"  WT Dice: {summary['mean']['WT_Dice']:.4f} ± {summary['std']['WT_Dice']:.4f} "
                       f"(median: {summary['median']['WT_Dice']:.4f})\n")
                f.write(f"  TC Dice: {summary['mean']['TC_Dice']:.4f} ± {summary['std']['TC_Dice']:.4f} "
                       f"(median: {summary['median']['TC_Dice']:.4f})\n")
                f.write(f"  ET Dice: {summary['mean']['ET_Dice']:.4f} ± {summary['std']['ET_Dice']:.4f} "
                       f"(median: {summary['median']['ET_Dice']:.4f})\n")
                f.write(f"\nPer-Class Metrics:\n")
                f.write(f"  NCR Dice: {summary['mean']['NCR_Dice']:.4f} ± {summary['std']['NCR_Dice']:.4f}\n")
                f.write(f"  ED Dice:  {summary['mean']['ED_Dice']:.4f} ± {summary['std']['ED_Dice']:.4f}\n")
                f.write(f"  ET Dice:  {summary['mean']['ET_Dice']:.4f} ± {summary['std']['ET_Dice']:.4f}\n")
                f.write("\n" + "-"*80 + "\n\n")
        
        print(f"  ✓ Summary report saved to: {report_path}")
    
    def save_all_results(self):
        """Save all results to JSON"""
        results_path = self.output_dir / 'all_results.json'
        
        # Prepare serializable version
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'description': result['description'],
                'summary': result['summary']
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n  ✓ All results saved to: {results_path}")


# Example usage
def main():
    """Example usage with command-line interface"""
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names to compare')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='List of checkpoint paths (same order as models)')
    parser.add_argument('--baseline', type=str, default='baseline',
                       help='Name of baseline model for improvement calculation')
    parser.add_argument('--test_loader', type=str, required=True,
                       help='Path to test data loader')
    parser.add_argument('--output_dir', type=str, default='./model_comparison',
                       help='Directory for comparison outputs')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(output_dir=args.output_dir)
    
    # Add models
    # Note: You need to provide model architectures here
    # for model_name, checkpoint_path in zip(args.models, args.checkpoints):
    #     comparator.add_model(model_name, checkpoint_path, YourModelClass(), 
    #                         description=f"Model {model_name}")
    
    # Load test data
    # test_loader = torch.load(args.test_loader)
    
    # Evaluate all
    # comparator.evaluate_all_models(test_loader)
    
    # Generate comparisons
    # comparator.generate_comparison_table()
    # comparator.plot_comparison_bars()
    # comparator.plot_improvement_trajectory(args.baseline)
    # comparator.plot_distribution_comparison()
    # comparator.generate_summary_report()
    # comparator.save_all_results()
    
    print("\nComparison complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    print("Model Comparison Script")
    print("=" * 60)
    print("\nCompare multiple trained models:")
    print("  ✓ Generate comparison tables")
    print("  ✓ Create bar charts")
    print("  ✓ Plot improvement trajectories")
    print("  ✓ Compare distributions")
    print("\nUsage:")
    print("  python compare_models.py --models baseline attention_unet --checkpoints model1.pth model2.pth --test_loader test.pkl")
