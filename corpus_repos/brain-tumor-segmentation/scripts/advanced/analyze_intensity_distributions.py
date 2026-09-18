"""
Intensity Distribution Analysis
Goal: Check if failure cases have different intensity distributions than success cases
Runtime: ~10-15 minutes (CPU only)
"""

import numpy as np
import json
import os
import nibabel as nib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/uphatal/brain_tumor_thesis/segmentation_project")
DATA_DIR = BASE_DIR / "data/raw/brats_gli/training_data1_v2"
SPLITS_FILE = BASE_DIR / "data/splits/data_splits.json"
RESULTS_FILE = BASE_DIR / "results/v2/ultimate_32ch_rerun/test_metrics.csv"
OUTPUT_DIR = BASE_DIR / "results/v2/intensity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = ['t1n', 't1c', 't2w', 't2f']  # adjust if your naming differs
TC_FAILURE_THRESHOLD = 0.1  # cases below this are "failures"
# ────────────────────────────────────────────────────────────────────────────

def find_patient_dir(patient_id, data_dir):
    """Find patient directory regardless of exact naming."""
    data_dir = Path(data_dir)
    # Try exact match first
    exact = data_dir / patient_id
    if exact.exists():
        return exact
    # Try partial match
    for d in data_dir.iterdir():
        if patient_id in d.name:
            return d
    return None

def load_nifti_data(patient_dir, modality):
    """Load a NIfTI file for a given modality."""
    patient_dir = Path(patient_dir)
    # Try common naming patterns
    patterns = [
        f"*{modality}*.nii.gz",
        f"*{modality}*.nii",
        f"*{modality.upper()}*.nii.gz",
    ]
    for pattern in patterns:
        files = list(patient_dir.glob(pattern))
        if files:
            return nib.load(files[0]).get_fdata().astype(np.float32)
    return None

def compute_intensity_stats(data, mask=None):
    """Compute comprehensive intensity statistics."""
    if mask is not None:
        values = data[mask > 0]
    else:
        # Use non-zero voxels (brain mask approximation)
        values = data[data > 0]
    
    if len(values) == 0:
        return None
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p1': float(np.percentile(values, 1)),
        'p5': float(np.percentile(values, 5)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
        'cv': float(np.std(values) / (np.mean(values) + 1e-8)),  # coefficient of variation
        'n_voxels': int(len(values))
    }

def main():
    print("=" * 60)
    print("INTENSITY DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # ── Load splits and results ──────────────────────────────────
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_patients = splits['test']
    print(f"Test patients: {len(test_patients)}")
    
    # Load test metrics to identify failures vs successes
    import csv
    metrics = {}
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get('patient_id') or row.get('case_id') or row.get('id')
            if pid:
                metrics[pid.strip()] = {
                    'wt_dice': float(row.get('wt_dice', row.get('WT_Dice', 0))),
                    'tc_dice': float(row.get('tc_dice', row.get('TC_Dice', 0))),
                    'et_dice': float(row.get('et_dice', row.get('ET_Dice', 0))),
                }
    
    print(f"Loaded metrics for {len(metrics)} patients")
    
    # Categorize into failures and successes
    failures = [pid for pid, m in metrics.items() if m['tc_dice'] < TC_FAILURE_THRESHOLD]
    successes = [pid for pid, m in metrics.items() if m['tc_dice'] >= 0.85]
    print(f"\nFailures (TC < {TC_FAILURE_THRESHOLD}): {len(failures)}")
    print(f"Successes (TC >= 0.85): {len(successes)}")
    
    # ── Collect intensity stats for each group ───────────────────
    print("\nCollecting intensity statistics...")
    print("(This takes ~10 minutes - processing NIfTI files)")
    
    results = {'failures': {}, 'successes': {}}
    
    for group_name, patient_list in [('failures', failures[:40]), ('successes', successes[:40])]:
        print(f"\nProcessing {group_name} ({len(patient_list)} patients)...")
        group_stats = {mod: [] for mod in MODALITIES}
        
        for i, pid in enumerate(patient_list):
            if i % 10 == 0:
                print(f"  {i}/{len(patient_list)}: {pid}")
            
            patient_dir = find_patient_dir(pid, DATA_DIR)
            if patient_dir is None:
                print(f"  WARNING: Cannot find directory for {pid}")
                continue
            
            for modality in MODALITIES:
                data = load_nifti_data(patient_dir, modality)
                if data is None:
                    continue
                stats_dict = compute_intensity_stats(data)
                if stats_dict:
                    group_stats[modality].append(stats_dict)
        
        results[group_name] = group_stats
    
    # ── Statistical comparison ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON: FAILURES vs SUCCESSES")
    print("=" * 60)
    
    findings = {}
    for modality in MODALITIES:
        f_stats = results['failures'][modality]
        s_stats = results['successes'][modality]
        
        if not f_stats or not s_stats:
            continue
        
        print(f"\n── {modality.upper()} ──")
        findings[modality] = {}
        
        for metric in ['mean', 'std', 'median', 'p95', 'cv']:
            f_vals = [s[metric] for s in f_stats]
            s_vals = [s[metric] for s in s_stats]
            
            # Mann-Whitney U test (non-parametric, robust)
            stat, p_val = stats.mannwhitneyu(f_vals, s_vals, alternative='two-sided')
            significant = p_val < 0.05
            
            print(f"  {metric:8s}: failures={np.mean(f_vals):.3f}±{np.std(f_vals):.3f}  "
                  f"successes={np.mean(s_vals):.3f}±{np.std(s_vals):.3f}  "
                  f"p={p_val:.4f} {'⚠️ SIGNIFICANT' if significant else ''}")
            
            findings[modality][metric] = {
                'failure_mean': float(np.mean(f_vals)),
                'success_mean': float(np.mean(s_vals)),
                'p_value': float(p_val),
                'significant': significant
            }
    
    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 60)
    
    significant_diffs = []
    for mod, mod_findings in findings.items():
        for metric, result in mod_findings.items():
            if result['significant']:
                diff_pct = abs(result['failure_mean'] - result['success_mean']) / (abs(result['success_mean']) + 1e-8) * 100
                significant_diffs.append((mod, metric, diff_pct, result['p_value']))
    
    if significant_diffs:
        print("\n✅ HYPOTHESIS CONFIRMED: Significant intensity differences found!")
        print("These differences likely explain failure cases:")
        for mod, metric, diff_pct, p in sorted(significant_diffs, key=lambda x: x[2], reverse=True):
            print(f"  {mod.upper()} {metric}: {diff_pct:.1f}% difference (p={p:.4f})")
        print("\n→ RECOMMENDATION: Proceed with normalization (preprocessing will help)")
    else:
        print("\n❌ HYPOTHESIS NOT CONFIRMED: No significant intensity differences")
        print("→ RECOMMENDATION: Skip preprocessing, try two-stage training instead")
    
    # ── Visualization ────────────────────────────────────────────
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Intensity Distributions: Failures vs Successes', fontsize=14, fontweight='bold')
    
    for col, modality in enumerate(MODALITIES):
        f_stats = results['failures'][modality]
        s_stats = results['successes'][modality]
        if not f_stats or not s_stats:
            continue
        
        for row, metric in enumerate(['mean', 'cv']):
            ax = axes[row][col]
            f_vals = [s[metric] for s in f_stats]
            s_vals = [s[metric] for s in s_stats]
            
            ax.hist(f_vals, bins=20, alpha=0.6, color='red', label='Failures', density=True)
            ax.hist(s_vals, bins=20, alpha=0.6, color='green', label='Successes', density=True)
            ax.set_title(f'{modality.upper()} - {metric}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            
            # Add p-value
            if modality in findings and metric in findings[modality]:
                p = findings[modality][metric]['p_value']
                ax.text(0.05, 0.95, f'p={p:.3f}', transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       color='red' if p < 0.05 else 'gray')
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "intensity_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    # Save findings JSON
    findings_path = OUTPUT_DIR / "intensity_findings.json"
    with open(findings_path, 'w') as f:
        json.dump({'findings': findings, 'significant_diffs': significant_diffs,
                   'n_failures': len(failures), 'n_successes': len(successes)}, f, indent=2)
    print(f"Saved: {findings_path}")
    
    print("\n✅ Analysis complete!")
    print(f"Results in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
