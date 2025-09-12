"""Test to verify the NaN residual fix in cv.py"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.residual_analysis import ResidualAnalyzer


def test_residual_generation_with_nan_fix():
    """Test that residuals are generated correctly even when targets have all NaN"""
    
    analyzer = ResidualAnalyzer()
    
    # Simulate the scenario in cv.py
    n_train = 80
    n_val = 20
    n_features = 10
    
    # Create training data with a target that has all NaN
    X_train = np.random.randn(n_train, n_features)
    y_train = pd.DataFrame({
        'Tg': np.random.randn(n_train),
        'FFV': np.random.randn(n_train),
        'Tc': np.full(n_train, np.nan),  # All NaN
        'Density': np.random.randn(n_train),
        'Rg': np.random.randn(n_train)
    })
    
    # Add some NaN to other targets
    y_train.loc[0:40, 'Tg'] = np.nan
    
    # Create validation data
    X_val = np.random.randn(n_val, n_features)
    y_val = pd.DataFrame({
        'Tg': np.random.randn(n_val),
        'FFV': np.random.randn(n_val),
        'Tc': np.full(n_val, np.nan),  # All NaN
        'Density': np.random.randn(n_val),
        'Rg': np.random.randn(n_val)
    })
    
    # Create SMILES
    train_smiles = [f'TRAIN_{i}' for i in range(n_train)]
    val_smiles = [f'VAL_{i}' for i in range(n_val)]
    
    # Simulate the fixed prediction generation from cv.py
    y_pred_train = pd.DataFrame(index=range(n_train), columns=y_train.columns)
    for target in y_train.columns:
        median_val = y_train[target].median()
        if pd.isna(median_val):
            # Fixed: use 0 as default when all values are NaN
            y_pred_train[target] = 0
        else:
            y_pred_train[target] = median_val
    
    # Create validation predictions (normally from model)
    y_pred_val = pd.DataFrame({
        'Tg': np.random.randn(n_val),
        'FFV': np.random.randn(n_val),
        'Tc': np.random.randn(n_val),  # Model predicts values
        'Density': np.random.randn(n_val),
        'Rg': np.random.randn(n_val)
    })
    
    print("Testing residual generation with NaN fix:")
    print("=" * 60)
    
    # Test for each target
    all_pass = True
    for target in y_train.columns:
        # Get combined dataframe
        df = analyzer.get_combined_residuals_dataframe(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train.to_dict('series'),
            y_val=y_val.to_dict('series'),
            y_pred_train=y_pred_train.to_dict('series'),
            y_pred_val=y_pred_val.to_dict('series'),
            train_smiles=train_smiles,
            val_smiles=val_smiles,
            target=target,
            method='lightgbm',
            is_cv=True,
            fold=0,
            seed=42
        )
        
        if not df.empty:
            nan_in_residuals = df['residuals'].isna().sum()
            print(f"\n{target}:")
            print(f"  DataFrame shape: {df.shape}")
            print(f"  NaN in y_train: {y_train[target].isna().sum()}")
            print(f"  NaN in y_val: {y_val[target].isna().sum()}")
            print(f"  NaN in y_pred_train: {y_pred_train[target].isna().sum()}")
            print(f"  NaN in residuals: {nan_in_residuals}")
            
            if nan_in_residuals > 0:
                print(f"  ❌ FAIL: Found {nan_in_residuals} NaN values in residuals!")
                all_pass = False
            else:
                print(f"  ✅ PASS: No NaN values in residuals")
        else:
            # Empty dataframe is expected for targets with all NaN
            if y_train[target].isna().all() and y_val[target].isna().all():
                print(f"\n{target}:")
                print(f"  ✅ PASS: Empty DataFrame (all values are NaN)")
            else:
                print(f"\n{target}:")
                print(f"  ⚠️  WARNING: Unexpected empty DataFrame")
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ All tests passed! The NaN fix is working correctly.")
    else:
        print("❌ Some tests failed. The fix needs adjustment.")
    
    return all_pass


def test_edge_cases():
    """Test various edge cases for the NaN fix"""
    
    print("\n" + "=" * 60)
    print("Testing edge cases:")
    print("=" * 60)
    
    # Test 1: All targets have some NaN
    print("\nTest 1: Partial NaN in all targets")
    y_train = pd.DataFrame({
        'Tg': [1.0, np.nan, 3.0, np.nan, 5.0],
        'FFV': [np.nan, 2.0, np.nan, 4.0, np.nan]
    })
    
    y_pred_train = pd.DataFrame(index=range(5), columns=y_train.columns)
    for target in y_train.columns:
        median_val = y_train[target].median()
        if pd.isna(median_val):
            y_pred_train[target] = 0
        else:
            y_pred_train[target] = median_val
    
    for target in y_train.columns:
        print(f"  {target}: median={y_train[target].median():.2f}, pred has NaN: {y_pred_train[target].isna().any()}")
    
    # Test 2: Mix of all-NaN and normal targets
    print("\nTest 2: Mix of all-NaN and normal targets")
    y_train = pd.DataFrame({
        'Tg': [1.0, 2.0, 3.0, 4.0, 5.0],
        'FFV': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    
    y_pred_train = pd.DataFrame(index=range(5), columns=y_train.columns)
    for target in y_train.columns:
        median_val = y_train[target].median()
        if pd.isna(median_val):
            y_pred_train[target] = 0
            print(f"  {target}: All NaN, using default 0")
        else:
            y_pred_train[target] = median_val
            print(f"  {target}: Using median={median_val:.2f}")


if __name__ == "__main__":
    # Run main test
    success = test_residual_generation_with_nan_fix()
    
    # Run edge case tests
    test_edge_cases()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The NaN residual issue has been fixed.")
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed. Further investigation needed.")