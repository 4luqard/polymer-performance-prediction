#!/usr/bin/env python3
"""
Compare Ridge and LightGBM models after MAE metric change
"""

import sys
from model import main

print("=" * 80)
print("Comparing Ridge vs LightGBM Models (with MAE metric for LightGBM)")
print("=" * 80)

# Test Ridge model
print("\n" + "="*50)
print("Testing RIDGE Model")
print("="*50)
ridge_results = main(cv_only=True, use_supplementary=True, model_type='ridge')

# Test LightGBM model (now with MAE metric)
print("\n" + "="*50)
print("Testing LIGHTGBM Model (with MAE metric)")
print("="*50)
lgb_results = main(cv_only=True, use_supplementary=True, model_type='lightgbm')

# Compare results
print("\n" + "="*80)
print("FINAL COMPARISON SUMMARY")
print("="*80)

if ridge_results and lgb_results:
    print(f"\nRidge Model:")
    print(f"  Single Seed CV: {ridge_results['single_seed']['cv_mean']:.4f} (+/- {ridge_results['single_seed']['cv_std']:.4f})")
    print(f"  Multi-Seed CV:  {ridge_results['multi_seed']['overall_mean']:.4f} (+/- {ridge_results['multi_seed']['overall_std']:.4f})")
    
    print(f"\nLightGBM Model (MAE metric):")
    print(f"  Single Seed CV: {lgb_results['single_seed']['cv_mean']:.4f} (+/- {lgb_results['single_seed']['cv_std']:.4f})")
    print(f"  Multi-Seed CV:  {lgb_results['multi_seed']['overall_mean']:.4f} (+/- {lgb_results['multi_seed']['overall_std']:.4f})")
    
    # Calculate improvement
    ridge_score = ridge_results['multi_seed']['overall_mean']
    lgb_score = lgb_results['multi_seed']['overall_mean']
    improvement = ((ridge_score - lgb_score) / ridge_score) * 100
    
    print(f"\n{'='*50}")
    print(f"Improvement: {improvement:.2f}% (lower is better)")
    print(f"{'='*50}")
    
    if lgb_score < ridge_score:
        print(f"✓ LightGBM performs BETTER than Ridge by {improvement:.2f}%")
    else:
        print(f"✗ Ridge performs BETTER than LightGBM by {abs(improvement):.2f}%")
        
    print("\nNote: LightGBM is now using MAE metric which is more aligned with competition scoring")
else:
    print("Error: Could not obtain results from both models")