"""
Comprehensive Medical Segmentation Metrics
============================================

This script implements all major metrics for medical image segmentation:
- Dice Score (per class and composite)
- Hausdorff Distance (95th percentile)
- Sensitivity, Specificity, Precision
- F1 Score, IoU
- Surface Dice
- Volumetric metrics

Usage:
    from metrics import SegmentationMetrics
    
    metrics = SegmentationMetrics(num_classes=4)
    scores = metrics.calculate_all_metrics(prediction, ground_truth)
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
import warnings

class SegmentationMetrics:
    """Comprehensive metrics for 3D medical image segmentation"""
    
    def __init__(self, num_classes=4, class_names=None):
        """
        Args:
            num_classes: Number of segmentation classes (default: 4 for BraTS)
            class_names: Names of classes (default: ['Background', 'NCR', 'ED', 'ET'])
        """
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = ['Background', 'NCR', 'ED', 'ET']
        else:
            self.class_names = class_names
    
    def calculate_all_metrics(self, pred, target, spacing=(1.0, 1.0, 1.0)):
        """
        Calculate all metrics for a prediction
        
        Args:
            pred: Predicted segmentation (numpy array or torch tensor)
            target: Ground truth segmentation (numpy array or torch tensor)
            spacing: Voxel spacing for distance metrics (x, y, z)
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Convert to numpy if needed
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        metrics = {}
        
        # Per-class metrics
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            
            # Dice Score
            dice = self.dice_score(pred, target, class_idx)
            metrics[f'{class_name}_Dice'] = dice
            
            # IoU (Jaccard Index)
            iou = self.iou_score(pred, target, class_idx)
            metrics[f'{class_name}_IoU'] = iou
            
            # Sensitivity (Recall/TPR)
            sensitivity = self.sensitivity(pred, target, class_idx)
            metrics[f'{class_name}_Sensitivity'] = sensitivity
            
            # Specificity (TNR)
            specificity = self.specificity(pred, target, class_idx)
            metrics[f'{class_name}_Specificity'] = specificity
            
            # Precision (PPV)
            precision = self.precision(pred, target, class_idx)
            metrics[f'{class_name}_Precision'] = precision
            
            # F1 Score
            f1 = self.f1_score(pred, target, class_idx)
            metrics[f'{class_name}_F1'] = f1
            
            # Hausdorff Distance (skip background)
            if class_idx > 0:
                hd95 = self.hausdorff_distance_95(pred, target, class_idx, spacing)
                metrics[f'{class_name}_Hausdorff95'] = hd95
            
            # Surface Dice (skip background)
            if class_idx > 0:
                surf_dice = self.surface_dice(pred, target, class_idx, tolerance=2.0)
                metrics[f'{class_name}_SurfaceDice'] = surf_dice
        
        # BraTS composite metrics
        metrics['WT_Dice'] = self.whole_tumor_dice(pred, target)
        metrics['TC_Dice'] = self.tumor_core_dice(pred, target)
        metrics['ET_Dice'] = self.enhancing_tumor_dice(pred, target)
        
        # Overall metrics
        metrics['Mean_Dice'] = np.mean([metrics[f'{name}_Dice'] 
                                        for name in self.class_names[1:]])  # Exclude background
        
        return metrics
    
    def dice_score(self, pred, target, class_idx):
        """Calculate Dice score for a specific class"""
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return float(dice)
    
    def iou_score(self, pred, target, class_idx):
        """Calculate IoU (Jaccard Index) for a specific class"""
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return float(iou)
    
    def sensitivity(self, pred, target, class_idx):
        """Calculate Sensitivity (Recall/TPR) for a specific class"""
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        true_positive = np.logical_and(pred_mask, target_mask).sum()
        false_negative = np.logical_and(~pred_mask, target_mask).sum()
        
        if (true_positive + false_negative) == 0:
            return 1.0 if true_positive == 0 else 0.0
        
        sensitivity = true_positive / (true_positive + false_negative)
        return float(sensitivity)
    
    def specificity(self, pred, target, class_idx):
        """Calculate Specificity (TNR) for a specific class"""
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        true_negative = np.logical_and(~pred_mask, ~target_mask).sum()
        false_positive = np.logical_and(pred_mask, ~target_mask).sum()
        
        if (true_negative + false_positive) == 0:
            return 1.0 if true_negative == 0 else 0.0
        
        specificity = true_negative / (true_negative + false_positive)
        return float(specificity)
    
    def precision(self, pred, target, class_idx):
        """Calculate Precision (PPV) for a specific class"""
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        true_positive = np.logical_and(pred_mask, target_mask).sum()
        false_positive = np.logical_and(pred_mask, ~target_mask).sum()
        
        if (true_positive + false_positive) == 0:
            return 1.0 if true_positive == 0 else 0.0
        
        precision = true_positive / (true_positive + false_positive)
        return float(precision)
    
    def f1_score(self, pred, target, class_idx):
        """Calculate F1 Score for a specific class"""
        prec = self.precision(pred, target, class_idx)
        sens = self.sensitivity(pred, target, class_idx)
        
        if (prec + sens) == 0:
            return 0.0
        
        f1 = 2 * (prec * sens) / (prec + sens)
        return float(f1)
    
    def hausdorff_distance_95(self, pred, target, class_idx, spacing=(1.0, 1.0, 1.0)):
        """
        Calculate 95th percentile Hausdorff Distance
        
        This is more robust than max Hausdorff distance as it's less sensitive to outliers.
        Lower is better (0 = perfect boundary match)
        """
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        # Check if either mask is empty
        if not pred_mask.any() or not target_mask.any():
            if not pred_mask.any() and not target_mask.any():
                return 0.0
            else:
                return 373.0  # Maximum possible distance for 128³ volume
        
        # Get surface points
        pred_border = self._get_boundary(pred_mask)
        target_border = self._get_boundary(target_mask)
        
        if not pred_border.any() or not target_border.any():
            return 0.0
        
        # Get coordinates of border pixels
        pred_coords = np.argwhere(pred_border) * spacing
        target_coords = np.argwhere(target_border) * spacing
        
        # Calculate distances
        distances_pred_to_target = np.min(
            np.sqrt(((pred_coords[:, None, :] - target_coords[None, :, :]) ** 2).sum(axis=2)),
            axis=1
        )
        distances_target_to_pred = np.min(
            np.sqrt(((target_coords[:, None, :] - pred_coords[None, :, :]) ** 2).sum(axis=2)),
            axis=1
        )
        
        # Get 95th percentile
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        hd95 = np.percentile(all_distances, 95)
        
        return float(hd95)
    
    def surface_dice(self, pred, target, class_idx, tolerance=2.0):
        """
        Calculate Surface Dice Score
        
        Measures boundary agreement within a tolerance distance.
        Args:
            tolerance: Distance threshold (in voxels) for considering surfaces as matching
        """
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        # Check if either mask is empty
        if not pred_mask.any() or not target_mask.any():
            if not pred_mask.any() and not target_mask.any():
                return 1.0
            else:
                return 0.0
        
        # Get boundaries
        pred_border = self._get_boundary(pred_mask)
        target_border = self._get_boundary(target_mask)
        
        if not pred_border.any() or not target_border.any():
            return 0.0
        
        # Distance transforms
        pred_dist = distance_transform_edt(~pred_border)
        target_dist = distance_transform_edt(~target_border)
        
        # Count boundary pixels within tolerance
        pred_border_in_tol = (target_dist[pred_border] <= tolerance).sum()
        target_border_in_tol = (pred_dist[target_border] <= tolerance).sum()
        
        # Calculate surface Dice
        surf_dice = (pred_border_in_tol + target_border_in_tol) / \
                    (pred_border.sum() + target_border.sum())
        
        return float(surf_dice)
    
    def _get_boundary(self, mask):
        """Extract boundary of a binary mask using erosion"""
        from scipy.ndimage import binary_erosion
        
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        return boundary
    
    # BraTS-specific composite metrics
    def whole_tumor_dice(self, pred, target):
        """Whole Tumor = NCR + ED + ET (all tumor classes)"""
        pred_wt = (pred > 0)  # Any non-background class
        target_wt = (target > 0)
        
        intersection = np.logical_and(pred_wt, target_wt).sum()
        union = pred_wt.sum() + target_wt.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float((2.0 * intersection) / union)
    
    def tumor_core_dice(self, pred, target):
        """Tumor Core = NCR + ET (labels 1 and 4)"""
        pred_tc = np.logical_or(pred == 1, pred == 4)
        target_tc = np.logical_or(target == 1, target == 4)
        
        intersection = np.logical_and(pred_tc, target_tc).sum()
        union = pred_tc.sum() + target_tc.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float((2.0 * intersection) / union)
    
    def enhancing_tumor_dice(self, pred, target):
        """Enhancing Tumor = ET only (label 4)"""
        return self.dice_score(pred, target, class_idx=3)  # Index 3 corresponds to label 4


class MetricsTracker:
    """Track and aggregate metrics across multiple samples"""
    
    def __init__(self, metrics_calculator):
        """
        Args:
            metrics_calculator: Instance of SegmentationMetrics
        """
        self.metrics_calculator = metrics_calculator
        self.all_metrics = []
    
    def add_sample(self, pred, target, patient_id=None):
        """Add metrics for one sample"""
        metrics = self.metrics_calculator.calculate_all_metrics(pred, target)
        if patient_id is not None:
            metrics['patient_id'] = patient_id
        self.all_metrics.append(metrics)
    
    def get_summary_statistics(self):
        """Calculate mean, std, median, min, max for all metrics"""
        if not self.all_metrics:
            return {}
        
        import pandas as pd
        
        df = pd.DataFrame(self.all_metrics)
        
        # Remove patient_id if present
        if 'patient_id' in df.columns:
            df = df.drop('patient_id', axis=1)
        
        summary = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'median': df.median().to_dict(),
            'min': df.min().to_dict(),
            'max': df.max().to_dict(),
            'count': len(self.all_metrics)
        }
        
        return summary
    
    def save_results(self, filepath):
        """Save all metrics to CSV"""
        import pandas as pd
        
        df = pd.DataFrame(self.all_metrics)
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to {filepath}")
    
    def print_summary(self):
        """Print formatted summary statistics"""
        summary = self.get_summary_statistics()
        
        print("\n" + "="*80)
        print("SEGMENTATION METRICS SUMMARY")
        print("="*80)
        print(f"Number of samples: {summary['count']}")
        print("\n" + "-"*80)
        print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'Median':<10}")
        print("-"*80)
        
        # BraTS primary metrics first
        for metric in ['WT_Dice', 'TC_Dice', 'ET_Dice']:
            if metric in summary['mean']:
                print(f"{metric:<30} {summary['mean'][metric]:.4f}    "
                      f"{summary['std'][metric]:.4f}    {summary['median'][metric]:.4f}")
        
        print("-"*80)
        
        # Per-class Dice scores
        for class_name in ['NCR', 'ED', 'ET']:
            metric = f'{class_name}_Dice'
            if metric in summary['mean']:
                print(f"{metric:<30} {summary['mean'][metric]:.4f}    "
                      f"{summary['std'][metric]:.4f}    {summary['median'][metric]:.4f}")
        
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    print("Comprehensive Segmentation Metrics Module")
    print("=" * 50)
    print("\nUsage Example:")
    print("""
    from metrics import SegmentationMetrics, MetricsTracker
    
    # Initialize
    metrics_calc = SegmentationMetrics(num_classes=4)
    tracker = MetricsTracker(metrics_calc)
    
    # For each patient
    for pred, target in test_loader:
        tracker.add_sample(pred, target, patient_id='BraTS-001')
    
    # Get results
    tracker.print_summary()
    tracker.save_results('test_metrics.csv')
    summary = tracker.get_summary_statistics()
    """)
