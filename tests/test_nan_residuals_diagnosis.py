"""Test to diagnose NaN values in residuals dataframes"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.residual_analysis import ResidualAnalysis, ResidualAnalyzer


class TestNaNResidualsDiagnosis:
    """Test suite to diagnose why residuals have NaN values"""
    
    def test_residuals_with_missing_targets(self):
        """Test residual calculation when targets have missing values"""
        analyzer = ResidualAnalyzer()
        
        # Create test data with missing values
        n_samples = 100
        n_features = 10
        
        # Create features
        X = np.random.randn(n_samples, n_features)
        
        # Create targets with missing values (like real data)
        y_true = {
            'Tg': np.random.randn(n_samples),
            'FFV': np.random.randn(n_samples),
            'Tc': np.full(n_samples, np.nan),  # All missing
            'Density': np.random.randn(n_samples),
            'Rg': np.random.randn(n_samples)
        }
        
        # Introduce more missing values randomly
        for target in ['Tg', 'FFV', 'Density', 'Rg']:
            missing_mask = np.random.random(n_samples) < 0.3  # 30% missing
            y_true[target][missing_mask] = np.nan
        
        # Create predictions (should not have NaN)
        y_pred = {
            'Tg': np.random.randn(n_samples),
            'FFV': np.random.randn(n_samples),
            'Tc': np.random.randn(n_samples),
            'Density': np.random.randn(n_samples),
            'Rg': np.random.randn(n_samples)
        }
        
        # Create SMILES
        smiles = [f'SMILES_{i}' for i in range(n_samples)]
        
        # Test get_residuals_dataframe for each target
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        for target in targets:
            df = analyzer.get_residuals_dataframe(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                smiles=smiles,
                target=target,
                method='test',
                is_cv=True,
                fold=0,
                seed=42
            )
            
            if not df.empty:
                # Check for NaN in residuals
                nan_count = df['residuals'].isna().sum()
                print(f"\nTarget: {target}")
                print(f"  Original NaN count in y_true: {np.isnan(y_true[target]).sum()}")
                print(f"  DataFrame shape: {df.shape}")
                print(f"  NaN count in residuals: {nan_count}")
                
                # Residuals should NOT have NaN values after masking
                assert nan_count == 0, f"Residuals for {target} should not have NaN values"
                
                # Verify the number of residuals matches non-NaN samples
                non_nan_count = (~np.isnan(y_true[target])).sum()
                assert len(df) == non_nan_count, f"DataFrame size mismatch for {target}"
    
    def test_prediction_alignment_issue(self):
        """Test if predictions are properly aligned with true values"""
        analyzer = ResidualAnalyzer()
        
        n_samples = 50
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        smiles = [f'SMILES_{i}' for i in range(n_samples)]
        
        # Create y_true with specific pattern of missing values
        y_true = {
            'Tg': np.array([1.0 if i % 2 == 0 else np.nan for i in range(n_samples)]),
            'FFV': np.array([2.0 if i % 3 == 0 else np.nan for i in range(n_samples)])
        }
        
        # Create y_pred as if model predicted for all samples
        y_pred = {
            'Tg': np.array([1.5] * n_samples),  # All predictions
            'FFV': np.array([2.5] * n_samples)   # All predictions
        }
        
        # Test residual calculation
        for target in ['Tg', 'FFV']:
            df = analyzer.get_residuals_dataframe(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                smiles=smiles,
                target=target,
                method='test',
                is_cv=True,
                fold=0,
                seed=42
            )
            
            if not df.empty:
                # Check residuals are correctly calculated
                expected_residual = 0.5  # y_pred - y_true = 1.5 - 1.0 or 2.5 - 2.0
                actual_residuals = df['residuals'].values
                
                print(f"\nTarget: {target}")
                print(f"  Expected residual: {expected_residual}")
                print(f"  Actual residuals (first 5): {actual_residuals[:5]}")
                print(f"  All residuals equal to expected: {np.allclose(actual_residuals, expected_residual)}")
                
                # All residuals should be 0.5
                assert np.allclose(actual_residuals, expected_residual), \
                    f"Residuals not correctly calculated for {target}"
    
    def test_dataframe_structure_validation(self):
        """Test if the dataframe structure is correct"""
        analyzer = ResidualAnalyzer()
        
        n_samples = 30
        n_features = 3
        
        X = np.random.randn(n_samples, n_features)
        smiles = [f'SMILES_{i}' for i in range(n_samples)]
        
        # Create simple test data
        y_true = {'Tg': np.random.randn(n_samples)}
        y_true['Tg'][10:20] = np.nan  # Some missing values
        
        y_pred = {'Tg': np.random.randn(n_samples)}
        
        df = analyzer.get_residuals_dataframe(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            smiles=smiles,
            target='Tg',
            method='lightgbm',
            is_cv=True,
            fold=1,
            seed=123
        )
        
        # Check required columns exist
        required_cols = ['SMILES', 'Tg', 'residuals', 'method', 'cv', 'fold', 'seed']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check feature columns
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        for col in feature_cols:
            assert col in df.columns, f"Missing feature column: {col}"
        
        # Validate data types and values
        assert df['method'].unique()[0] == 'lightgbm'
        assert df['cv'].unique()[0] == True
        assert df['fold'].unique()[0] == 1
        assert df['seed'].unique()[0] == 123
        
        # Check no NaN in residuals
        assert df['residuals'].isna().sum() == 0, "Residuals should not have NaN"
        
        print("\nDataFrame structure validation:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  No NaN in residuals: {df['residuals'].isna().sum() == 0}")
    
    def test_combined_dataframe_train_val_split(self):
        """Test if combined dataframe correctly handles train/val split"""
        analyzer = ResidualAnalyzer()
        
        n_train = 80
        n_val = 20
        n_features = 5
        
        X_train = np.random.randn(n_train, n_features)
        X_val = np.random.randn(n_val, n_features)
        
        train_smiles = [f'TRAIN_{i}' for i in range(n_train)]
        val_smiles = [f'VAL_{i}' for i in range(n_val)]
        
        # Create data with missing values
        y_train = {'Tg': np.random.randn(n_train)}
        y_train['Tg'][40:60] = np.nan  # 20 missing in train
        
        y_val = {'Tg': np.random.randn(n_val)}
        y_val['Tg'][10:15] = np.nan  # 5 missing in val
        
        # Predictions for all samples
        y_pred_train = {'Tg': np.random.randn(n_train)}
        y_pred_val = {'Tg': np.random.randn(n_val)}
        
        df = analyzer.get_combined_residuals_dataframe(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            y_pred_train=y_pred_train,
            y_pred_val=y_pred_val,
            train_smiles=train_smiles,
            val_smiles=val_smiles,
            target='Tg',
            method='lightgbm',
            is_cv=True,
            fold=0,
            seed=42
        )
        
        # Check train_val column
        assert 'train_val' in df.columns, "Missing train_val column"
        
        # Count train and val samples
        train_count = (~df['train_val']).sum()
        val_count = df['train_val'].sum()
        
        expected_train = (~np.isnan(y_train['Tg'])).sum()
        expected_val = (~np.isnan(y_val['Tg'])).sum()
        
        print("\nCombined dataframe train/val split:")
        print(f"  Train samples: {train_count} (expected: {expected_train})")
        print(f"  Val samples: {val_count} (expected: {expected_val})")
        print(f"  Total samples: {len(df)}")
        print(f"  NaN in residuals: {df['residuals'].isna().sum()}")
        
        assert train_count == expected_train, "Train sample count mismatch"
        assert val_count == expected_val, "Val sample count mismatch"
        assert df['residuals'].isna().sum() == 0, "Should not have NaN in residuals"
        
        # Verify SMILES are correctly assigned
        train_smiles_in_df = df[~df['train_val']]['SMILES'].str.startswith('TRAIN').all()
        val_smiles_in_df = df[df['train_val']]['SMILES'].str.startswith('VAL').all()
        
        assert train_smiles_in_df, "Train SMILES not correctly assigned"
        assert val_smiles_in_df, "Val SMILES not correctly assigned"


if __name__ == "__main__":
    # Run tests
    test_instance = TestNaNResidualsDiagnosis()
    
    print("=" * 60)
    print("Testing residuals with missing targets...")
    test_instance.test_residuals_with_missing_targets()
    
    print("\n" + "=" * 60)
    print("Testing prediction alignment...")
    test_instance.test_prediction_alignment_issue()
    
    print("\n" + "=" * 60)
    print("Testing dataframe structure...")
    test_instance.test_dataframe_structure_validation()
    
    print("\n" + "=" * 60)
    print("Testing combined dataframe train/val split...")
    test_instance.test_combined_dataframe_train_val_split()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")