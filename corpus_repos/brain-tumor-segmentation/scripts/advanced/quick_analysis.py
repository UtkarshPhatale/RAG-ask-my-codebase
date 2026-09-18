#!/usr/bin/env python3
"""
Quick Failure Analysis - Uses existing test_metrics.csv without needing predictions
Perfect for rapid analysis before full visualization pipeline

Author: Brain Tumor Thesis Project
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def quick_analysis(results_csv, output_dir):
    """
    Perform rapid analysis on existing test results
    
    Args:
        results_csv: Path to test_metrics.csv
        output_dir: Where to save analysis outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    print(f"Columns found: {list(df.columns)}")
    print(f"\n{'='*80}")
    print(f"QUICK FAILURE ANALYSIS - {len(df)} Test Cases")
    print(f"{'='*80}\n")
    
    # === SECTION 1: OVERALL STATISTICS ===
    print("="*80)
    print("1. OVERALL PERFORMANCE STATISTICS")
    print("="*80)
    
    for metric in ['wt_dice', 'tc_dice', 'et_dice']:
        if metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
            median = df[metric].median()
            q25 = df[metric].quantile(0.25)
            q75 = df[metric].quantile(0.75)
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            print(f"\n{metric.upper()}:")
            print(f"  Mean ± Std:  {mean:.4f} ± {std:.4f}")
            print(f"  Median:      {median:.4f}")
            print(f"  Range:       [{min_val:.4f}, {max_val:.4f}]")
            print(f"  IQR:         [{q25:.4f}, {q75:.4f}]")
            
            # Flag high variance
            if std > 0.4:
                print(f"  ⚠️  WARNING: Very high variance detected!")
    
    # === SECTION 2: FAILURE CATEGORIZATION ===
    print(f"\n{'='*80}")
    print("2. FAILURE CATEGORIZATION (TC Dice)")
    print("="*80)
    
    severe = df[df['tc_dice'] < 0.3]
    major = df[(df['tc_dice'] >= 0.3) & (df['tc_dice'] < 0.5)]
    moderate = df[(df['tc_dice'] >= 0.5) & (df['tc_dice'] < 0.7)]
    good = df[(df['tc_dice'] >= 0.7) & (df['tc_dice'] < 0.85)]
    excellent = df[df['tc_dice'] >= 0.85]
    
    print(f"\nSevere Failures (TC < 0.3):       {len(severe):3d} ({100*len(severe)/len(df):5.1f}%)")
    print(f"Major Failures (0.3 ≤ TC < 0.5):  {len(major):3d} ({100*len(major)/len(df):5.1f}%)")
    print(f"Moderate (0.5 ≤ TC < 0.7):        {len(moderate):3d} ({100*len(moderate)/len(df):5.1f}%)")
    print(f"Good (0.7 ≤ TC < 0.85):           {len(good):3d} ({100*len(good)/len(df):5.1f}%)")
    print(f"Excellent (TC ≥ 0.85):            {len(excellent):3d} ({100*len(excellent)/len(df):5.1f}%)")
    
    # Save worst cases
    worst_30 = df.nsmallest(30, 'tc_dice')
    worst_30.to_csv(output_dir / 'worst_30_cases.csv', index=False)
    print(f"\n✅ Saved worst 30 cases to: {output_dir / 'worst_30_cases.csv'}")
    
    # === SECTION 3: PATTERN DETECTION ===
    print(f"\n{'='*80}")
    print("3. PATTERN DETECTION")
    print("="*80)
    
    patterns_found = {}
    
    # Pattern 1: Small tumor analysis
    if 'tc_volume' in df.columns:
        print("\n--- Pattern 1: Small Tumor Analysis ---")
        q25 = df['tc_volume'].quantile(0.25)
        q75 = df['tc_volume'].quantile(0.75)
        
        small = df[df['tc_volume'] < q25]
        large = df[df['tc_volume'] > q75]
        
        small_mean = small['tc_dice'].mean()
        large_mean = large['tc_dice'].mean()
        
        t_stat, p_value = stats.ttest_ind(small['tc_dice'], large['tc_dice'])
        
        print(f"Small tumors (< {q25:.1f} voxels): TC Dice = {small_mean:.4f}")
        print(f"Large tumors (> {q75:.1f} voxels): TC Dice = {large_mean:.4f}")
        print(f"Performance gap: {large_mean - small_mean:.4f} (p={p_value:.4e})")
        
        if p_value < 0.05 and (large_mean - small_mean) > 0.1:
            print("✅ SIGNIFICANT PATTERN: Small tumors perform worse!")
            print("   → Recommended: Size-weighted loss, multi-scale features")
            patterns_found['small_tumor'] = True
        else:
            print("❌ No significant small tumor pattern")
            patterns_found['small_tumor'] = False
    
    # Pattern 2: Boundary confusion
    print("\n--- Pattern 2: Boundary Confusion ---")
    boundary_confusion = df[(df['wt_dice'] > 0.8) & (df['tc_dice'] < 0.6)]
    print(f"Cases with good WT but poor TC: {len(boundary_confusion)} ({100*len(boundary_confusion)/len(df):.1f}%)")
    
    if len(boundary_confusion) > 0.15 * len(df):
        print("✅ SIGNIFICANT PATTERN: Boundary confusion detected!")
        print("   → Recommended: Boundary-aware loss, edge-preserving attention")
        patterns_found['boundary_confusion'] = True
    else:
        print("❌ No significant boundary confusion")
        patterns_found['boundary_confusion'] = False
    
    # Pattern 3: Class imbalance
    if 'tc_volume' in df.columns and 'wt_volume' in df.columns:
        print("\n--- Pattern 3: Class Imbalance ---")
        df['tc_wt_ratio'] = df['tc_volume'] / (df['wt_volume'] + 1e-6)
        q10 = df['tc_wt_ratio'].quantile(0.1)
        
        extreme_imbalance = df[df['tc_wt_ratio'] < q10]
        normal = df[df['tc_wt_ratio'] > df['tc_wt_ratio'].median()]
        
        if len(extreme_imbalance) > 0 and len(normal) > 0:
            extreme_mean = extreme_imbalance['tc_dice'].mean()
            normal_mean = normal['tc_dice'].mean()
            
            t_stat, p_value = stats.ttest_ind(extreme_imbalance['tc_dice'], normal['tc_dice'])
            
            print(f"Extreme imbalance (TC/WT < {q10:.3f}): TC Dice = {extreme_mean:.4f}")
            print(f"Normal balance: TC Dice = {normal_mean:.4f}")
            print(f"Performance gap: {normal_mean - extreme_mean:.4f} (p={p_value:.4e})")
            
            if p_value < 0.05 and (normal_mean - extreme_mean) > 0.1:
                print("✅ SIGNIFICANT PATTERN: Class imbalance affects performance!")
                print("   → Recommended: Generalized Dice Loss, dynamic weighting")
                patterns_found['class_imbalance'] = True
            else:
                print("❌ No significant class imbalance pattern")
                patterns_found['class_imbalance'] = False
    
    # Pattern 4: High variance
    print("\n--- Pattern 4: High Variance Analysis ---")
    tc_std = df['tc_dice'].std()
    print(f"TC Dice standard deviation: {tc_std:.4f}")
    
    if tc_std > 0.35:
        print("✅ SIGNIFICANT PATTERN: Very high variance detected!")
        print("   Model lacks consistency across cases")
        print("   → Recommended: Increase capacity (48-ch), ensemble, regularization")
        patterns_found['high_variance'] = True
    else:
        print("❌ Variance within acceptable range")
        patterns_found['high_variance'] = False
    
    # === SECTION 4: KEY FINDINGS & RECOMMENDATIONS ===
    print(f"\n{'='*80}")
    print("4. KEY FINDINGS & PRIORITIZED RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Rank by importance
    if patterns_found.get('high_variance'):
        recommendations.append({
            'priority': 1,
            'pattern': 'High Variance / Poor Robustness',
            'solution': 'Scale up to 48-channel model OR implement ensemble',
            'expected_gain': '+2-3%',
            'difficulty': 'Medium'
        })
    
    if patterns_found.get('small_tumor'):
        recommendations.append({
            'priority': 2,
            'pattern': 'Small Tumor Failures',
            'solution': 'Implement size-weighted Focal Loss + multi-scale features',
            'expected_gain': '+3-5%',
            'difficulty': 'Medium'
        })
    
    if patterns_found.get('boundary_confusion'):
        recommendations.append({
            'priority': 3,
            'pattern': 'Boundary Confusion',
            'solution': 'Add boundary-aware loss + edge-preserving attention',
            'expected_gain': '+2-4%',
            'difficulty': 'High'
        })
    
    if patterns_found.get('class_imbalance'):
        recommendations.append({
            'priority': 4,
            'pattern': 'Class Imbalance',
            'solution': 'Use Generalized Dice Loss + dynamic class weighting',
            'expected_gain': '+2-3%',
            'difficulty': 'Low'
        })
    
    if not recommendations:
        recommendations.append({
            'priority': 1,
            'pattern': 'No clear pattern',
            'solution': 'Scale up model capacity (48 channels)',
            'expected_gain': '+2-3%',
            'difficulty': 'Low'
        })
    
    print("\n📋 PRIORITIZED RECOMMENDATIONS:\n")
    for rec in sorted(recommendations, key=lambda x: x['priority']):
        print(f"{rec['priority']}. {rec['pattern']}")
        print(f"   Solution:     {rec['solution']}")
        print(f"   Expected:     {rec['expected_gain']} improvement")
        print(f"   Difficulty:   {rec['difficulty']}")
        print()
    
    # Save recommendations
    with open(output_dir / 'quick_recommendations.json', 'w') as f:
        json.dump({
            'patterns_detected': patterns_found,
            'recommendations': recommendations,
            'statistics': {
                'total_cases': len(df),
                'severe_failures': len(severe),
                'tc_mean': float(df['tc_dice'].mean()),
                'tc_std': float(df['tc_dice'].std())
            }
        }, f, indent=2)
    
    # === SECTION 5: CREATE QUICK VISUALIZATIONS ===
    print(f"{'='*80}")
    print("5. CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Distribution
    axes[0, 0].hist(df['tc_dice'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['tc_dice'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["tc_dice"].mean():.3f}')
    axes[0, 0].axvline(0.3, color='orange', linestyle='--', alpha=0.7, label='Severe threshold')
    axes[0, 0].axvline(0.5, color='yellow', linestyle='--', alpha=0.7, label='Major threshold')
    axes[0, 0].set_xlabel('TC Dice', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
    axes[0, 0].set_title('TC Dice Distribution', fontweight='bold', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Box plot comparison
    dice_data = [df['wt_dice'], df['tc_dice'], df['et_dice']]
    bp = axes[0, 1].boxplot(dice_data, labels=['WT', 'TC', 'ET'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Dice Score', fontweight='bold', fontsize=12)
    axes[0, 1].set_title('Performance Comparison', fontweight='bold', fontsize=14)
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Plot 3: WT vs TC scatter
    axes[1, 0].scatter(df['wt_dice'], df['tc_dice'], alpha=0.5, s=30, c=df['et_dice'], cmap='viridis')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 0].axhline(0.6, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0.8, color='green', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('WT Dice', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('TC Dice', fontweight='bold', fontsize=12)
    axes[1, 0].set_title('WT vs TC Performance', fontweight='bold', fontsize=14)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Category distribution
    categories = ['Severe\n(<0.3)', 'Major\n(0.3-0.5)', 'Moderate\n(0.5-0.7)', 'Good\n(0.7-0.85)', 'Excellent\n(≥0.85)']
    counts = [len(severe), len(major), len(moderate), len(good), len(excellent)]
    colors = ['#d62728', '#ff7f0e', '#ffff00', '#2ca02c', '#1f77b4']
    
    axes[1, 1].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Number of Cases', fontweight='bold', fontsize=12)
    axes[1, 1].set_title('Failure Category Distribution', fontweight='bold', fontsize=14)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quick_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to: {output_dir / 'quick_analysis_summary.png'}")
    
    # === SECTION 6: SUMMARY ===
    print(f"\n{'='*80}")
    print("✅ QUICK ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  1. {output_dir / 'worst_30_cases.csv'}")
    print(f"  2. {output_dir / 'quick_recommendations.json'}")
    print(f"  3. {output_dir / 'quick_analysis_summary.png'}")
    print(f"\nNext steps:")
    print(f"  1. Review quick_recommendations.json")
    print(f"  2. Look at worst_30_cases.csv to identify patient IDs for detailed inspection")
    print(f"  3. For full visualization pipeline, run: bash run_complete_analysis.sh")
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Quick failure analysis using existing metrics')
    parser.add_argument('--results_csv', type=str, required=True,
                       help='Path to test_metrics.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save quick analysis results')
    
    args = parser.parse_args()
    
    quick_analysis(args.results_csv, args.output_dir)


if __name__ == '__main__':
    main()
