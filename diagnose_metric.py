#!/usr/bin/env python3
"""
Diagnose the difference between CV and LB scores
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.competition_metric import neurips_polymer_metric

# Load data
train_df = pd.read_csv('data/raw/train.csv')
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

print("=== Data Statistics ===")
print(f"Total training samples: {len(train_df)}")

# Check missing values
print("\nMissing values per target:")
for col in target_columns:
    missing = train_df[col].isna().sum()
    print(f"{col}: {missing} ({missing/len(train_df)*100:.1f}%)")

# Check value ranges
print("\nValue ranges in training data:")
for col in target_columns:
    if train_df[col].notna().sum() > 0:
        print(f"{col}: [{train_df[col].min():.4f}, {train_df[col].max():.4f}]")

# Compare with competition constants
print("\n=== Competition Metric Constants ===")
from src.competition_metric import MINMAX_DICT
for prop, (min_val, max_val) in MINMAX_DICT.items():
    print(f"{prop}: [{min_val:.4f}, {max_val:.4f}]")

# Check if our data is within the expected ranges
print("\n=== Data vs Competition Range Check ===")
for col in target_columns:
    if col in MINMAX_DICT and train_df[col].notna().sum() > 0:
        comp_min, comp_max = MINMAX_DICT[col]
        data_min = train_df[col].min()
        data_max = train_df[col].max()
        
        if data_min < comp_min or data_max > comp_max:
            print(f"WARNING - {col}: Data range [{data_min:.4f}, {data_max:.4f}] exceeds competition range [{comp_min:.4f}, {comp_max:.4f}]")
        else:
            print(f"OK - {col}: Data within competition range")

# Test metric calculation with dummy predictions
print("\n=== Testing Metric Calculation ===")

# Create dummy perfect predictions (same as actual)
subset = train_df[target_columns].dropna()
if len(subset) > 0:
    y_true = subset
    y_pred = subset.copy()
    
    score, individual = neurips_polymer_metric(y_true, y_pred)
    print(f"Perfect prediction score (should be ~0): {score:.6f}")
    
    # Add small noise
    noise_level = 0.01
    y_pred_noisy = y_pred + np.random.randn(*y_pred.shape) * noise_level
    score_noisy, _ = neurips_polymer_metric(y_true, y_pred_noisy)
    print(f"Prediction with {noise_level} noise: {score_noisy:.6f}")
    
    # Add larger noise
    noise_level = 0.1
    y_pred_noisy = y_pred + np.random.randn(*y_pred.shape) * noise_level
    score_noisy, _ = neurips_polymer_metric(y_true, y_pred_noisy)
    print(f"Prediction with {noise_level} noise: {score_noisy:.6f}")

# Test with constant predictions (median)
print("\n=== Baseline Predictions ===")
for col in target_columns:
    non_null = train_df[col].notna()
    if non_null.sum() > 100:  # Only test if we have enough data
        y_true_col = pd.DataFrame({col: train_df.loc[non_null, col]})
        
        # Median prediction
        median_val = train_df[col].median()
        y_pred_median = pd.DataFrame({col: [median_val] * len(y_true_col)})
        score_median, _ = neurips_polymer_metric(y_true_col, y_pred_median, [col])
        
        # Mean prediction  
        mean_val = train_df[col].mean()
        y_pred_mean = pd.DataFrame({col: [mean_val] * len(y_true_col)})
        score_mean, _ = neurips_polymer_metric(y_true_col, y_pred_mean, [col])
        
        print(f"{col}: Median baseline={score_median:.4f}, Mean baseline={score_mean:.4f}")

print("\n=== Analysis Summary ===")
print("1. Check if the metric implementation matches the competition exactly")
print("2. Verify that test data characteristics match training data") 
print("3. Consider that public LB might use a different subset of test data")
print("4. Our CV score of 0.0120 vs LB of 0.158 suggests a ~13x difference!")