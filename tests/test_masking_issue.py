"""Test to diagnose the masking issue with pandas Series"""

import numpy as np
import pandas as pd


def test_pandas_series_masking_issue():
    """Test how pandas Series behaves with boolean masking"""
    
    print("Testing pandas Series masking behavior:")
    print("=" * 60)
    
    # Create y_true as dict with numpy arrays (like the actual data)
    n_samples = 10
    y_true = {
        'Tg': np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, np.nan])
    }
    
    # Create y_pred as dict with pandas Series (from cv.py)
    y_pred = {
        'Tg': pd.Series([2.0] * n_samples, index=range(n_samples))
    }
    
    print("y_true['Tg'] (numpy array):")
    print(y_true['Tg'])
    print(f"  Type: {type(y_true['Tg'])}")
    
    print("\ny_pred['Tg'] (pandas Series):")
    print(y_pred['Tg'].values)
    print(f"  Type: {type(y_pred['Tg'])}")
    print(f"  Index: {list(y_pred['Tg'].index)}")
    
    # Create mask for non-NaN values
    mask = ~np.isnan(y_true['Tg'])
    print(f"\nMask (non-NaN positions): {mask}")
    print(f"  Mask type: {type(mask)}")
    print(f"  Mask shape: {mask.shape}")
    
    # Try to apply mask (this is what happens in get_residuals_dataframe)
    print("\nApplying mask to y_pred['Tg']:")
    try:
        y_pred_masked = y_pred['Tg'][mask]
        print(f"  Result: {y_pred_masked.values}")
        print(f"  Shape: {y_pred_masked.shape}")
        print(f"  Has NaN: {y_pred_masked.isna().any()}")
        
        # Check if indices match
        print(f"  Result indices: {list(y_pred_masked.index)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Now test the actual calculation
    print("\nCalculating residuals:")
    y_true_masked = y_true['Tg'][mask]
    y_pred_masked = y_pred['Tg'][mask]
    
    print(f"  y_true[mask]: {y_true_masked}")
    print(f"  y_pred[mask]: {y_pred_masked.values if hasattr(y_pred_masked, 'values') else y_pred_masked}")
    
    residuals = y_pred_masked - y_true_masked
    print(f"  Residuals: {residuals if not hasattr(residuals, 'values') else residuals.values}")
    
    if hasattr(residuals, 'isna'):
        print(f"  Has NaN in residuals: {residuals.isna().any()}")


def test_correct_approach():
    """Test the correct way to handle pandas Series in residual calculation"""
    
    print("\n" + "=" * 60)
    print("Testing correct approach:")
    print("=" * 60)
    
    n_samples = 10
    y_true = {
        'Tg': np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, np.nan])
    }
    
    # y_pred as pandas Series
    y_pred = {
        'Tg': pd.Series([2.0] * n_samples, index=range(n_samples))
    }
    
    mask = ~np.isnan(y_true['Tg'])
    
    # Correct approach: convert Series to numpy array first
    print("\nApproach 1: Convert to numpy array before masking")
    y_pred_array = y_pred['Tg'].values if hasattr(y_pred['Tg'], 'values') else y_pred['Tg']
    y_pred_masked = y_pred_array[mask]
    y_true_masked = y_true['Tg'][mask]
    
    residuals = y_pred_masked - y_true_masked
    print(f"  y_pred (converted): {y_pred_array}")
    print(f"  y_pred[mask]: {y_pred_masked}")
    print(f"  y_true[mask]: {y_true_masked}")
    print(f"  Residuals: {residuals}")
    print(f"  Has NaN: {np.isnan(residuals).any()}")
    
    # Alternative approach: use iloc for integer position based indexing
    print("\nApproach 2: Use numpy where to get integer positions")
    positions = np.where(mask)[0]
    print(f"  Non-NaN positions: {positions}")
    
    if hasattr(y_pred['Tg'], 'iloc'):
        y_pred_masked = y_pred['Tg'].iloc[positions]
        print(f"  y_pred.iloc[positions]: {y_pred_masked.values}")
    else:
        y_pred_masked = y_pred['Tg'][positions]
        print(f"  y_pred[positions]: {y_pred_masked}")


if __name__ == "__main__":
    test_pandas_series_masking_issue()
    test_correct_approach()