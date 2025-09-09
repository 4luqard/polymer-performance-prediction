"""Test the issue with DataFrame to dict conversion in integration.py"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dataframe_dict_conversion_issue():
    """Test what happens when y_pred_train is a DataFrame"""
    
    print("Testing DataFrame to dict conversion issue:")
    print("=" * 60)
    
    # Create y_pred_train as DataFrame (like in cv.py)
    train_idx = list(range(10))
    y_pred_train = pd.DataFrame(index=train_idx, columns=['Tg', 'FFV'])
    y_pred_train['Tg'] = 1.0
    y_pred_train['FFV'] = 2.0
    
    print("y_pred_train (DataFrame):")
    print(y_pred_train)
    print(f"  Type: {type(y_pred_train)}")
    
    # Simulate what happens in integration.py
    if isinstance(y_pred_train, np.ndarray) and y_pred_train.ndim == 2:
        print("\n  Branch 1: Converting np.ndarray")
        targets = ['Tg', 'FFV']
        y_pred_train_dict = {targets[i]: y_pred_train[:, i] for i in range(y_pred_train.shape[1])}
    else:
        print("\n  Branch 2: Passing through as-is")
        y_pred_train_dict = y_pred_train
    
    print(f"\ny_pred_train_dict type: {type(y_pred_train_dict)}")
    
    if isinstance(y_pred_train_dict, pd.DataFrame):
        print("  Still a DataFrame! This is the problem.")
        print("\n  When used in get_residuals_dataframe:")
        
        # Simulate what happens in get_residuals_dataframe
        target = 'Tg'
        if target in y_pred_train_dict:  # This works for DataFrame
            print(f"    '{target}' in y_pred_train_dict: True")
            print(f"    y_pred_train_dict['{target}'] type: {type(y_pred_train_dict[target])}")
            print(f"    y_pred_train_dict['{target}'] is Series: {isinstance(y_pred_train_dict[target], pd.Series)}")
            
            # Create a mask
            mask = np.array([True, False, True, False, True, True, False, True, True, False])
            
            print(f"\n    Applying mask {mask}:")
            masked = y_pred_train_dict[target][mask]
            print(f"    Result type: {type(masked)}")
            print(f"    Result: {masked.values if hasattr(masked, 'values') else masked}")
            
            # Check indices
            if hasattr(masked, 'index'):
                print(f"    Result indices: {list(masked.index)}")


def test_correct_conversion():
    """Test the correct way to convert DataFrame to dict"""
    
    print("\n" + "=" * 60)
    print("Testing correct DataFrame to dict conversion:")
    print("=" * 60)
    
    # Create y_pred_train as DataFrame
    train_idx = list(range(10))
    y_pred_train = pd.DataFrame(index=train_idx, columns=['Tg', 'FFV'])
    y_pred_train['Tg'] = 1.0
    y_pred_train['FFV'] = 2.0
    
    print("Original y_pred_train (DataFrame):")
    print(f"  Shape: {y_pred_train.shape}")
    print(f"  Columns: {list(y_pred_train.columns)}")
    
    # Correct conversion (should be done in integration.py)
    if isinstance(y_pred_train, pd.DataFrame):
        y_pred_train_dict = {col: y_pred_train[col].values for col in y_pred_train.columns}
        print("\nConverted to dict of numpy arrays:")
    elif isinstance(y_pred_train, np.ndarray) and y_pred_train.ndim == 2:
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        y_pred_train_dict = {targets[i]: y_pred_train[:, i] for i in range(y_pred_train.shape[1])}
        print("\nConverted from 2D array to dict:")
    else:
        y_pred_train_dict = y_pred_train
        print("\nPassed through as-is:")
    
    for key, val in y_pred_train_dict.items():
        print(f"  {key}: type={type(val)}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}")
    
    # Test masking
    mask = np.array([True, False, True, False, True, True, False, True, True, False])
    target = 'Tg'
    
    masked = y_pred_train_dict[target][mask]
    print(f"\nAfter masking with {mask.sum()} True values:")
    print(f"  Result: {masked}")
    print(f"  Type: {type(masked)}")
    print(f"  Shape: {masked.shape}")


if __name__ == "__main__":
    test_dataframe_dict_conversion_issue()
    test_correct_conversion()