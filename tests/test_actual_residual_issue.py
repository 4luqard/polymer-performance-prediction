"""Test to reproduce the actual NaN issue in residuals"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.residual_analysis import ResidualAnalyzer


def test_exact_scenario():
    """Reproduce the exact scenario that causes NaN in residuals"""
    
    print("Reproducing exact scenario from cv.py:")
    print("=" * 60)
    
    # Create training data similar to actual data
    n_train = 80
    train_idx = list(range(n_train))
    
    # Create y_fold_train as DataFrame (like in cv.py)
    y_fold_train = pd.DataFrame({
        'Tg': np.random.randn(n_train),
        'FFV': np.random.randn(n_train),
        'Tc': np.full(n_train, np.nan),  # All NaN
        'Density': np.random.randn(n_train),
        'Rg': np.random.randn(n_train)
    }, index=train_idx)
    
    # Add NaN values to Tg (like real data)
    y_fold_train.loc[0:40, 'Tg'] = np.nan
    
    print("y_fold_train info:")
    print(f"  Shape: {y_fold_train.shape}")
    print(f"  Index: {y_fold_train.index[:5].tolist()}...{y_fold_train.index[-5:].tolist()}")
    
    # Create y_pred_train EXACTLY as in cv.py (with train_idx as index)
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    y_pred_train = pd.DataFrame(index=train_idx, columns=target_columns)
    
    for target in target_columns:
        median_val = y_fold_train[target].median()
        if pd.isna(median_val):
            y_pred_train[target] = 0
        else:
            y_pred_train[target] = median_val
    
    print("\ny_pred_train info:")
    print(f"  Shape: {y_pred_train.shape}")
    print(f"  Index: {y_pred_train.index[:5].tolist()}...{y_pred_train.index[-5:].tolist()}")
    print(f"  Data types: {y_pred_train.dtypes.to_dict()}")
    
    # Check for NaN in predictions
    for target in target_columns:
        nan_count = y_pred_train[target].isna().sum()
        median_val = y_fold_train[target].median()
        median_str = f"{median_val:.4f}" if not pd.isna(median_val) else "NaN"
        print(f"  {target}: NaN count = {nan_count}, median = {median_str}")
    
    # Now test with ResidualAnalyzer
    analyzer = ResidualAnalyzer()
    
    # Create features and SMILES
    X_train = np.random.randn(n_train, 10)
    train_smiles = [f'SMILES_{i}' for i in range(n_train)]
    
    # Convert to dictionary format (as done in integration.py)
    y_train_dict = {col: y_fold_train[col].values for col in y_fold_train.columns}
    y_pred_train_dict = {col: y_pred_train[col].values for col in y_pred_train.columns}
    
    print("\nAfter conversion to dict:")
    print(f"  y_train_dict['Tg'] type: {type(y_train_dict['Tg'])}")
    print(f"  y_pred_train_dict['Tg'] type: {type(y_pred_train_dict['Tg'])}")
    print(f"  y_pred_train_dict['Tg'][:5]: {y_pred_train_dict['Tg'][:5]}")
    
    # Test residual calculation for Tg
    target = 'Tg'
    df = analyzer.get_residuals_dataframe(
        X=X_train,
        y_true=y_train_dict,
        y_pred=y_pred_train_dict,
        smiles=train_smiles,
        target=target,
        method='lightgbm',
        is_cv=True,
        fold=0,
        seed=42
    )
    
    print(f"\nResidual DataFrame for {target}:")
    print(f"  Shape: {df.shape}")
    if not df.empty:
        print(f"  NaN in residuals: {df['residuals'].isna().sum()}")
        if df['residuals'].isna().any():
            # Find where NaN occurs
            nan_indices = df[df['residuals'].isna()].index.tolist()
            print(f"  NaN at indices: {nan_indices[:10]}...")
            
            # Check corresponding values
            mask = ~np.isnan(y_train_dict[target])
            print(f"\n  Debugging NaN residuals:")
            print(f"    Mask sum (non-NaN in y_true): {mask.sum()}")
            print(f"    y_true[mask][:5]: {y_train_dict[target][mask][:5]}")
            print(f"    y_pred[mask][:5]: {y_pred_train_dict[target][mask][:5]}")
            
            # Calculate residuals manually
            residuals_manual = y_pred_train_dict[target][mask] - y_train_dict[target][mask]
            print(f"    Manual residuals[:5]: {residuals_manual[:5]}")
            print(f"    Manual residuals NaN count: {np.isnan(residuals_manual).sum()}")


def test_series_to_dict_conversion():
    """Test if the issue is in Series to dict conversion"""
    
    print("\n" + "=" * 60)
    print("Testing Series to dict conversion:")
    print("=" * 60)
    
    # Create a Series with some values
    s = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    s_dict = {'col': s.values}
    
    print(f"Original Series: {s.tolist()}")
    print(f"Series.values: {s.values}")
    print(f"Dict value: {s_dict['col']}")
    print(f"Types match: {type(s.values) == type(s_dict['col'])}")
    
    # Test with NaN median scenario
    s2 = pd.Series(index=[0, 1, 2], dtype=float)
    s2[:] = np.nan  # This should create NaN values
    
    print(f"\nSeries with NaN assignment:")
    print(f"  Values: {s2.values}")
    print(f"  All NaN: {np.isnan(s2.values).all()}")


if __name__ == "__main__":
    test_exact_scenario()
    test_series_to_dict_conversion()