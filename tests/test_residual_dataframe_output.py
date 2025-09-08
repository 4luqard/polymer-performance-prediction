"""Tests for residual analysis dataframe output modifications"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, '/workspace/kaggle/neurips-open-polymer-prediction-2025')

from src.residual_analysis import ResidualAnalyzer


class TestResidualDataframeOutput:
    """Test cases for residual analysis dataframe output requirements"""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing"""
        np.random.seed(42)
        n_samples = 100
        return {
            'X_train': np.random.randn(n_samples, 10),
            'X_val': np.random.randn(20, 10),
            'y_train': {
                'Tg': np.random.randn(n_samples),
                'FFV': np.random.randn(n_samples),
                'Tc': np.random.randn(n_samples),
                'Density': np.random.randn(n_samples),
                'Rg': np.random.randn(n_samples)
            },
            'y_val': {
                'Tg': np.random.randn(20),
                'FFV': np.random.randn(20),
                'Tc': np.random.randn(20),
                'Density': np.random.randn(20),
                'Rg': np.random.randn(20)
            },
            'y_pred': {
                'Tg': np.random.randn(20),
                'FFV': np.random.randn(20),
                'Tc': np.random.randn(20),
                'Density': np.random.randn(20),
                'Rg': np.random.randn(20)
            },
            'train_smiles': ['C'*i for i in range(1, n_samples+1)],
            'val_smiles': ['CC'*i for i in range(1, 21)]
        }
    
    def test_txt_png_disabled_by_default(self, tmp_path):
        """Test that txt and png files are not created by default"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        # Check that save_txt and save_png default to False
        assert hasattr(analyzer, 'save_txt')
        assert hasattr(analyzer, 'save_png')
        assert analyzer.save_txt == False
        assert analyzer.save_png == False
        
    def test_dataframe_includes_residuals_column(self, mock_data, tmp_path):
        """Test that output dataframe includes residuals column"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        # Mock the get_residuals_dataframe method
        df = analyzer.get_residuals_dataframe(
            X=mock_data['X_val'],
            y_true=mock_data['y_val'],
            y_pred=mock_data['y_pred'],
            smiles=mock_data['val_smiles'],
            target='Tg'
        )
        
        # Check that residuals column exists
        assert 'residuals' in df.columns
        
    def test_dataframe_includes_smiles_features_targets(self, mock_data, tmp_path):
        """Test that dataframe includes SMILES, features, and targets"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        df = analyzer.get_residuals_dataframe(
            X=mock_data['X_val'],
            y_true=mock_data['y_val'],
            y_pred=mock_data['y_pred'],
            smiles=mock_data['val_smiles'],
            target='Tg'
        )
        
        # Check required columns
        assert 'SMILES' in df.columns
        assert 'Tg' in df.columns  # Target column
        assert df.shape[1] > 3  # Should have features too
        
    def test_train_val_column_added(self, mock_data, tmp_path):
        """Test that train_val column is added to distinguish samples"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        # Combine train and val data
        df = analyzer.get_combined_residuals_dataframe(
            X_train=mock_data['X_train'],
            X_val=mock_data['X_val'],
            y_train=mock_data['y_train'],
            y_val=mock_data['y_val'],
            y_pred_train={'Tg': np.random.randn(100)},
            y_pred_val=mock_data['y_pred'],
            train_smiles=mock_data['train_smiles'],
            val_smiles=mock_data['val_smiles'],
            target='Tg'
        )
        
        # Check train_val column exists
        assert 'train_val' in df.columns
        # Check values are boolean
        assert df['train_val'].dtype == bool
        # Check we have both train (False) and val (True) samples
        assert False in df['train_val'].values
        assert True in df['train_val'].values
        
    def test_method_cv_fold_seed_columns(self, mock_data, tmp_path):
        """Test that method, cv, fold, and seed columns are added"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        df = analyzer.get_residuals_dataframe(
            X=mock_data['X_val'],
            y_true=mock_data['y_val'],
            y_pred=mock_data['y_pred'],
            smiles=mock_data['val_smiles'],
            target='Tg',
            method='lightgbm',
            is_cv=True,
            fold=0,
            seed=42
        )
        
        # Check all metadata columns exist
        assert 'method' in df.columns
        assert 'cv' in df.columns
        assert 'fold' in df.columns
        assert 'seed' in df.columns
        
        # Check values
        assert df['method'].iloc[0] == 'lightgbm'
        assert df['cv'].iloc[0] == True
        assert df['fold'].iloc[0] == 0
        assert df['seed'].iloc[0] == 42
        
    def test_dataframe_saved_as_parquet(self, mock_data, tmp_path):
        """Test that dataframes are saved as parquet files"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        df = analyzer.get_residuals_dataframe(
            X=mock_data['X_val'],
            y_true=mock_data['y_val'],
            y_pred=mock_data['y_pred'],
            smiles=mock_data['val_smiles'],
            target='Tg',
            method='lightgbm',
            is_cv=True,
            fold=0,
            seed=42
        )
        
        # Save dataframe
        output_file = analyzer.save_residuals_dataframe(df, 'lightgbm', 'Tg', fold=0)
        
        # Check file exists and is parquet
        assert output_file.endswith('.parquet')
        assert os.path.exists(output_file)
        
        # Check we can read it back
        df_loaded = pd.read_parquet(output_file)
        assert df_loaded.shape == df.shape
        
    def test_preprocessing_method_dataframe_structure(self, tmp_path):
        """Test dataframe structure for preprocessing methods like PCA"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        # Mock data for preprocessing method
        n_samples = 50
        original_features = np.random.randn(n_samples, 20)
        transformed_features = np.random.randn(n_samples, 5)
        smiles = ['C'*i for i in range(1, n_samples+1)]
        targets = {
            'Tg': np.random.randn(n_samples),
            'FFV': np.random.randn(n_samples)
        }
        residuals = np.random.randn(n_samples)
        
        df = analyzer.get_preprocessing_residuals_dataframe(
            original_features=original_features,
            transformed_features=transformed_features,
            smiles=smiles,
            targets=targets,
            residuals=residuals,
            method='pca'
        )
        
        # Check structure
        assert 'SMILES' in df.columns
        assert 'residuals' in df.columns
        assert 'method' in df.columns
        
        # Check we have both original and transformed features
        # Original features (20) + transformed (5) + metadata
        expected_min_cols = 20 + 5 + 4  # +4 for SMILES, residuals, method, targets
        assert df.shape[1] >= expected_min_cols
        
    def test_cv_fold_dataframe_generation(self, mock_data, tmp_path):
        """Test that dataframes are generated for each CV fold"""
        analyzer = ResidualAnalyzer(output_dir=str(tmp_path))
        
        # Simulate CV fold processing
        fold_dataframes = []
        for fold in range(5):
            df = analyzer.get_residuals_dataframe(
                X=mock_data['X_val'],
                y_true=mock_data['y_val'],
                y_pred=mock_data['y_pred'],
                smiles=mock_data['val_smiles'],
                target='Tg',
                method='lightgbm',
                is_cv=True,
                fold=fold,
                seed=42
            )
            fold_dataframes.append(df)
            
        # Check we have 5 dataframes
        assert len(fold_dataframes) == 5
        
        # Check each has correct fold number
        for i, df in enumerate(fold_dataframes):
            assert df['fold'].iloc[0] == i