"""Test to confirm the NaN median issue in cv.py"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_median_with_nan():
    """Test what happens when we take median of a column with all NaN values"""
    
    # Create a DataFrame similar to y_fold_train
    n_samples = 100
    y_fold_train = pd.DataFrame({
        'Tg': np.random.randn(n_samples),
        'FFV': np.random.randn(n_samples),
        'Tc': np.full(n_samples, np.nan),  # All NaN (common in real data)
        'Density': np.random.randn(n_samples),
        'Rg': np.random.randn(n_samples)
    })
    
    # Introduce some NaN values to other columns
    y_fold_train.loc[0:50, 'Tg'] = np.nan
    y_fold_train.loc[30:80, 'FFV'] = np.nan
    
    print("Testing median with NaN values:")
    print("=" * 60)
    
    # Check median for each column
    for col in y_fold_train.columns:
        median_val = y_fold_train[col].median()
        non_nan_count = y_fold_train[col].notna().sum()
        print(f"\n{col}:")
        print(f"  Non-NaN count: {non_nan_count}")
        print(f"  Median: {median_val}")
        print(f"  Is median NaN: {pd.isna(median_val)}")
        
        # Simulate what happens in cv.py
        y_pred_train = pd.DataFrame(index=range(n_samples), columns=[col])
        y_pred_train[col] = y_fold_train[col].median()
        
        # Check how many NaN in predictions
        nan_in_pred = y_pred_train[col].isna().sum()
        print(f"  NaN count in predictions: {nan_in_pred}")
        
        if pd.isna(median_val):
            print(f"  ⚠️  WARNING: All predictions will be NaN for {col}!")


def test_proper_train_predictions():
    """Test a better way to generate train predictions"""
    
    n_samples = 100
    y_fold_train = pd.DataFrame({
        'Tg': np.random.randn(n_samples),
        'FFV': np.random.randn(n_samples),
        'Tc': np.full(n_samples, np.nan),  # All NaN
        'Density': np.random.randn(n_samples),
        'Rg': np.random.randn(n_samples)
    })
    
    # Add some NaN values
    y_fold_train.loc[0:50, 'Tg'] = np.nan
    
    print("\n" + "=" * 60)
    print("Testing better train prediction generation:")
    print("=" * 60)
    
    # Better approach: use mean of non-NaN values or a default
    y_pred_train = pd.DataFrame(index=range(n_samples), columns=y_fold_train.columns)
    
    for col in y_fold_train.columns:
        # Calculate mean of non-NaN values
        non_nan_mean = y_fold_train[col].mean()
        
        if pd.isna(non_nan_mean):
            # If all values are NaN, use a default (e.g., 0)
            print(f"\n{col}: All NaN, using default value 0")
            y_pred_train[col] = 0
        else:
            print(f"\n{col}: Using mean = {non_nan_mean:.4f}")
            y_pred_train[col] = non_nan_mean
        
        # Check results
        nan_in_pred = y_pred_train[col].isna().sum()
        print(f"  NaN count in predictions: {nan_in_pred}")


def test_actual_model_predictions():
    """Test using actual model predictions for training data"""
    
    print("\n" + "=" * 60)
    print("Proper approach: Use actual model predictions")
    print("=" * 60)
    
    print("\nFor training residuals, we should:")
    print("1. Store model predictions during training")
    print("2. Use actual fitted model to predict on training data")
    print("3. Only calculate residuals for non-NaN samples")
    print("\nThis requires modifying the CV loop to properly generate train predictions")


if __name__ == "__main__":
    test_median_with_nan()
    test_proper_train_predictions()
    test_actual_model_predictions()