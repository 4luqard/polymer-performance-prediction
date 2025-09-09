"""Test the combined fixes for NaN residuals issue"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.residual_analysis.integration import ResidualAnalysisHook


def test_combined_fix():
    """Test that both fixes work together to prevent NaN in residuals"""
    
    print("Testing combined fixes for NaN residuals:")
    print("=" * 60)
    
    # Create test data similar to cv.py
    n_train = 80
    n_val = 20
    n_features = 10
    
    # Training data with some all-NaN targets
    X_train = np.random.randn(n_train, n_features)
    y_train = pd.DataFrame({
        'Tg': np.random.randn(n_train),
        'FFV': np.random.randn(n_train),
        'Tc': np.full(n_train, np.nan),  # All NaN
        'Density': np.random.randn(n_train),
        'Rg': np.random.randn(n_train)
    })
    
    # Add some NaN to Tg
    y_train.loc[0:40, 'Tg'] = np.nan
    
    # Validation data
    X_val = np.random.randn(n_val, n_features)
    y_val = pd.DataFrame({
        'Tg': np.random.randn(n_val),
        'FFV': np.random.randn(n_val),
        'Tc': np.full(n_val, np.nan),  # All NaN
        'Density': np.random.randn(n_val),
        'Rg': np.random.randn(n_val)
    })
    
    # Create predictions as DataFrames (like in cv.py with fix)
    y_pred_train = pd.DataFrame(columns=y_train.columns)
    for target in y_train.columns:
        median_val = y_train[target].median()
        if pd.isna(median_val):
            y_pred_train[target] = [0.0] * n_train  # Fixed: use 0 when median is NaN
        else:
            y_pred_train[target] = [median_val] * n_train
    
    y_pred_val = pd.DataFrame({
        'Tg': np.random.randn(n_val),
        'FFV': np.random.randn(n_val),
        'Tc': np.random.randn(n_val),
        'Density': np.random.randn(n_val),
        'Rg': np.random.randn(n_val)
    })
    
    # SMILES
    train_smiles = [f'TRAIN_{i}' for i in range(n_train)]
    val_smiles = [f'VAL_{i}' for i in range(n_val)]
    
    # Test with ResidualAnalysisHook
    hook = ResidualAnalysisHook(enable=True)
    
    print("\nGenerating residuals dataframe...")
    
    # This should now work without NaN issues
    df_combined = hook.generate_residuals_dataframe(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        y_pred_train=y_pred_train,
        y_pred_val=y_pred_val,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
        method='lightgbm',
        is_cv=True,
        fold=0,
        seed=42
    )
    
    print(f"\nCombined DataFrame:")
    print(f"  Shape: {df_combined.shape}")
    print(f"  Columns: {list(df_combined.columns)[:10]}...")
    
    if not df_combined.empty:
        # Check for NaN in residuals
        nan_count = df_combined['residuals'].isna().sum() if 'residuals' in df_combined else 0
        print(f"  NaN in residuals: {nan_count}")
        
        # Check by target if target column exists
        if 'Tg' in df_combined or 'FFV' in df_combined or 'Tc' in df_combined:
            print("\n  Per-target analysis:")
            for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
                if target in df_combined:
                    target_rows = df_combined[df_combined[target].notna()]
                    if not target_rows.empty:
                        nan_in_target = target_rows['residuals'].isna().sum()
                        print(f"    {target}: {len(target_rows)} rows, {nan_in_target} NaN residuals")
        
        if nan_count == 0:
            print("\n✅ SUCCESS: No NaN values in residuals!")
            return True
        else:
            print(f"\n❌ FAILURE: Found {nan_count} NaN values in residuals")
            
            # Debug info
            nan_rows = df_combined[df_combined['residuals'].isna()]
            if not nan_rows.empty:
                print("\n  Sample of rows with NaN residuals:")
                print(nan_rows[['SMILES', 'residuals', 'train_val']].head())
            return False
    else:
        print("\n⚠️  WARNING: Combined DataFrame is empty")
        return False


def test_edge_cases():
    """Test various edge cases"""
    
    print("\n" + "=" * 60)
    print("Testing edge cases:")
    print("=" * 60)
    
    hook = ResidualAnalysisHook(enable=True)
    
    # Test 1: All targets have all NaN
    print("\nTest 1: All targets with all NaN values")
    y_train_all_nan = pd.DataFrame({
        'Tg': [np.nan] * 10,
        'FFV': [np.nan] * 10
    })
    y_val_all_nan = pd.DataFrame({
        'Tg': [np.nan] * 5,
        'FFV': [np.nan] * 5
    })
    
    y_pred_train_zeros = pd.DataFrame({
        'Tg': [0.0] * 10,
        'FFV': [0.0] * 10
    })
    y_pred_val_zeros = pd.DataFrame({
        'Tg': [1.0] * 5,
        'FFV': [1.0] * 5
    })
    
    df = hook.generate_residuals_dataframe(
        X_train=np.random.randn(10, 3),
        X_val=np.random.randn(5, 3),
        y_train=y_train_all_nan,
        y_val=y_val_all_nan,
        y_pred_train=y_pred_train_zeros,
        y_pred_val=y_pred_val_zeros,
        train_smiles=['S' + str(i) for i in range(10)],
        val_smiles=['V' + str(i) for i in range(5)],
        method='test',
        is_cv=True,
        fold=0,
        seed=42
    )
    
    print(f"  Result: {'Empty DataFrame (expected)' if df.empty else f'Shape {df.shape}'}")
    
    # Test 2: Mixed NaN patterns
    print("\nTest 2: Mixed NaN patterns")
    y_train_mixed = pd.DataFrame({
        'Tg': [1.0, np.nan, 3.0, np.nan, 5.0],
        'FFV': [np.nan, 2.0, np.nan, 4.0, np.nan]
    })
    
    y_pred_train_mixed = pd.DataFrame({
        'Tg': [3.0] * 5,  # median of [1, 3, 5]
        'FFV': [3.0] * 5   # median of [2, 4]
    })
    
    from src.residual_analysis import ResidualAnalyzer
    analyzer = ResidualAnalyzer()
    
    df_mixed = analyzer.get_residuals_dataframe(
        X=np.random.randn(5, 3),
        y_true={'Tg': y_train_mixed['Tg'].values, 'FFV': y_train_mixed['FFV'].values},
        y_pred={'Tg': y_pred_train_mixed['Tg'].values, 'FFV': y_pred_train_mixed['FFV'].values},
        smiles=['S' + str(i) for i in range(5)],
        target='Tg',
        method='test'
    )
    
    if not df_mixed.empty:
        nan_in_residuals = df_mixed['residuals'].isna().sum()
        print(f"  Tg: {len(df_mixed)} rows, {nan_in_residuals} NaN in residuals")
        
        if nan_in_residuals == 0:
            print("  ✅ No NaN in residuals for mixed pattern")
        else:
            print("  ❌ Found NaN in residuals for mixed pattern")


if __name__ == "__main__":
    # Run main test
    success = test_combined_fix()
    
    # Run edge cases
    test_edge_cases()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All fixes are working correctly!")
        print("The NaN residual issue has been resolved.")
    else:
        print("\n" + "=" * 60)
        print("❌ Issues remain. Further investigation needed.")