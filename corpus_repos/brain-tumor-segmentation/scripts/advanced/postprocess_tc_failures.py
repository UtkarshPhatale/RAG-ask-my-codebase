#!/usr/bin/env python3
"""
Post-Processing for TC=0 Failures
Strategy: Use T1CE intensity to recover NCR in failed cases
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology
from pathlib import Path
import json
from tqdm import tqdm

class TCFailurePostProcessor:
    """
    Post-process TC=0 cases using T1CE intensity information
    
    Key insight: NCR (necrotic core) appears DARK on T1CE
                 ET (enhancing tumor) appears BRIGHT on T1CE
    """
    
    def __init__(self, 
                 ncr_percentile=30,  # Lower percentile = darker = NCR
                 et_percentile=75,   # Higher percentile = brighter = ET
                 min_size=50):       # Minimum voxels for valid region
        self.ncr_percentile = ncr_percentile
        self.et_percentile = et_percentile
        self.min_size = min_size
    
    def process(self, prediction, t1ce_image, patient_id=None):
        """
        Post-process a single prediction
        
        Args:
            prediction: (H,W,D) array with labels [0,1,2,3]
            t1ce_image: (H,W,D) T1CE modality
            patient_id: for debugging
        
        Returns:
            refined_prediction: (H,W,D) refined labels
            was_modified: bool indicating if changes were made
        """
        refined = prediction.copy()
        
        # Check for TC failure
        has_ncr = np.any(prediction == 1)
        has_et = np.any(prediction == 3)
        tc_failed = (not has_ncr) and (not has_et)
        
        if not tc_failed:
            return refined, False
        
        # TC failed - try to recover using T1CE
        wt_mask = prediction > 0  # WT should be predicted correctly
        
        if not np.any(wt_mask):
            # Can't recover if WT also failed
            return refined, False
        
        # Get intensity distribution within WT
        wt_intensities = t1ce_image[wt_mask]
        
        # Define thresholds
        ncr_threshold = np.percentile(wt_intensities, self.ncr_percentile)
        et_threshold = np.percentile(wt_intensities, self.et_percentile)
        
        # Find potential NCR (dark regions)
        potential_ncr = (t1ce_image < ncr_threshold) & wt_mask
        potential_ncr = morphology.remove_small_objects(
            potential_ncr, 
            min_size=self.min_size
        )
        
        # Find potential ET (bright regions)
        potential_et = (t1ce_image > et_threshold) & wt_mask & ~potential_ncr
        potential_et = morphology.remove_small_objects(
            potential_et,
            min_size=self.min_size
        )
        
        # Apply conservatively - only if we found substantial regions
        modified = False
        
        if np.sum(potential_ncr) > self.min_size:
            refined[potential_ncr] = 1  # NCR label
            modified = True
        
        if np.sum(potential_et) > self.min_size:
            refined[potential_et] = 3  # ET label
            modified = True
        
        return refined, modified


def run_postprocessing():
    """Run post-processing on all TC failure cases"""
    
    print("="*80)
    print("POST-PROCESSING TC FAILURES")
    print("="*80)
    
    # Paths
    pred_dir = Path('../../results/v2/predictions/baseline_32ch')
    output_dir = Path('../../results/v2/predictions/postprocessed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    
    # Load metadata
    with open(pred_dir / 'prediction_metadata.json') as f:
        metadata = json.load(f)
    
    print(f"\n[1/3] Loaded {len(metadata['predictions'])} predictions")
    
    # Initialize post-processor
    processor = TCFailurePostProcessor(
        ncr_percentile=30,
        et_percentile=75,
        min_size=50
    )
    
    # Process each case
    print("\n[2/3] Processing...")
    
    results = {
        'config': {
            'ncr_percentile': processor.ncr_percentile,
            'et_percentile': processor.et_percentile,
            'min_size': processor.min_size
        },
        'cases': {},
        'summary': {
            'total_cases': 0,
            'tc_failures': 0,
            'modified': 0,
            'improvements': 0
        }
    }
    
    for patient_id, info in tqdm(metadata['predictions'].items(), desc="Post-processing"):
        # Load prediction and mask
        pred = np.load(info['pred_file'])
        mask = np.load(info['mask_file'])
        
        # Load T1CE image
        t1ce_path = data_root / patient_id / f'{patient_id}_t1c.nii.gz'
        if not t1ce_path.exists():
            print(f"Warning: T1CE not found for {patient_id}")
            continue
        
        t1ce_img = nib.load(t1ce_path).get_fdata()
        
        # Apply post-processing
        refined, was_modified = processor.process(pred, t1ce_img, patient_id)
        
        # Calculate metrics before and after
        def calc_dice(pred, true, classes):
            pred_mask = np.isin(pred, classes)
            true_mask = np.isin(true, classes)
            intersection = (pred_mask & true_mask).sum()
            return 2 * intersection / (pred_mask.sum() + true_mask.sum() + 1e-8)
        
        # Before
        wt_before = calc_dice(pred, mask, [1,2,3])
        tc_before = calc_dice(pred, mask, [1,3])
        
        # After
        wt_after = calc_dice(refined, mask, [1,2,3])
        tc_after = calc_dice(refined, mask, [1,3])
        
        # Save refined prediction
        refined_file = output_dir / f'{patient_id}_pred.npy'
        np.save(refined_file, refined)
        
        # Track results
        improved = (wt_after > wt_before) or (tc_after > tc_before)
        
        results['cases'][patient_id] = {
            'was_tc_failure': info['tc_failure'],
            'was_modified': was_modified,
            'wt_before': float(wt_before),
            'wt_after': float(wt_after),
            'tc_before': float(tc_before),
            'tc_after': float(tc_after),
            'wt_change': float(wt_after - wt_before),
            'tc_change': float(tc_after - tc_before),
            'improved': improved
        }
        
        results['summary']['total_cases'] += 1
        if info['tc_failure']:
            results['summary']['tc_failures'] += 1
        if was_modified:
            results['summary']['modified'] += 1
        if improved:
            results['summary']['improvements'] += 1
    
    # Calculate overall metrics
    print("\n[3/3] Computing final metrics...")
    
    wt_dices = [c['wt_after'] for c in results['cases'].values()]
    tc_dices = [c['tc_after'] for c in results['cases'].values()]
    et_dices = []  # Calculate if needed
    
    results['summary']['final_metrics'] = {
        'wt_dice_mean': float(np.mean(wt_dices)),
        'wt_dice_std': float(np.std(wt_dices)),
        'tc_dice_mean': float(np.mean(tc_dices)),
        'tc_dice_std': float(np.std(tc_dices)),
        'baseline_wt': 0.8242,
        'improvement_wt': float(np.mean(wt_dices) - 0.8242)
    }
    
    # Save results
    results_file = output_dir / 'postprocessing_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("POST-PROCESSING COMPLETE!")
    print("="*80)
    print(f"  Total cases:      {results['summary']['total_cases']}")
    print(f"  TC failures:      {results['summary']['tc_failures']}")
    print(f"  Modified:         {results['summary']['modified']}")
    print(f"  Improvements:     {results['summary']['improvements']}")
    print(f"\n  Baseline WT:      {results['summary']['final_metrics']['baseline_wt']:.4f}")
    print(f"  Post-proc WT:     {results['summary']['final_metrics']['wt_dice_mean']:.4f}")
    print(f"  Improvement:      {results['summary']['final_metrics']['improvement_wt']:+.4f}")
    print(f"\n  Results saved:    {results_file}")
    print("="*80)

if __name__ == '__main__':
    run_postprocessing()
