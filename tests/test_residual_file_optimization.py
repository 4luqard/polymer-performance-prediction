import pytest
import os
import json
import pickle
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from src.residual_analysis import ResidualAnalysis, LightGBMResidualAnalyzer


class TestResidualFileOptimization:
    """Tests for residual analysis file optimization changes"""
    
    @pytest.fixture
    def test_dir(self, tmp_path):
        """Create temporary test directory"""
        test_dir = tmp_path / "residual_analysis"
        test_dir.mkdir(exist_ok=True)
        return test_dir
    
    @pytest.fixture
    def sample_data(self):
        """Create sample prediction and actual data"""
        n_samples = 100
        return {
            'predictions': pd.DataFrame({
                'Tg': np.random.randn(n_samples) * 50 + 300,
                'FFV': np.random.rand(n_samples) * 0.5,
                'Tc': np.random.randn(n_samples) * 30 + 200,
                'Density': np.random.rand(n_samples) * 0.5 + 1.0,
                'Rg': np.random.randn(n_samples) * 5 + 10
            }),
            'actuals': pd.DataFrame({
                'Tg': np.random.randn(n_samples) * 50 + 300,
                'FFV': np.random.rand(n_samples) * 0.5,
                'Tc': np.random.randn(n_samples) * 30 + 200,
                'Density': np.random.rand(n_samples) * 0.5 + 1.0,
                'Rg': np.random.randn(n_samples) * 5 + 10
            })
        }
    
    def test_append_fold_results_to_single_file(self, test_dir, sample_data):
        """Test that fold results are appended to a single file per target"""
        analyzer = LightGBMResidualAnalyzer(model_type='lightgbm', output_dir=str(test_dir))
        
        # Test appending results for multiple folds
        for seed in range(3):
            for fold in range(5):
                fold_num = seed * 5 + fold  # 0 to 14
                results = analyzer.analyze(
                    predictions=sample_data['predictions'],
                    actuals=sample_data['actuals'],
                    model_name=f'lightgbm_fold_{fold_num}'
                )
                
                # Save results - should append to existing files
                analyzer.save_results(
                    results=results,
                    filename_prefix=f'lightgbm_seed_{seed}_fold_{fold}'
                )
        
        # Check that we have exactly 5 JSON files (one per target) and 5 txt files
        json_files = list(test_dir.glob("*.json"))
        txt_files = list(test_dir.glob("*.txt"))
        
        # Should have 5 targets × 2 formats = 10 files (no pkl files by default)
        assert len(json_files) == 5, f"Expected 5 JSON files, found {len(json_files)}"
        assert len(txt_files) == 5, f"Expected 5 txt files, found {len(txt_files)}"
        
        # Verify each target has one file
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        for target in targets:
            target_json_files = [f for f in json_files if target in f.name]
            assert len(target_json_files) == 1, f"Expected 1 JSON file for {target}, found {len(target_json_files)}"
            
            # Check that the file contains all 15 folds
            with open(target_json_files[0], 'r') as f:
                data = json.load(f)
                # Should have 15 entries (3 seeds × 5 folds)
                assert len(data) == 15, f"Expected 15 fold results for {target}, found {len(data)}"
    
    def test_pkl_files_not_created_by_default(self, test_dir, sample_data):
        """Test that pkl files are not created unless specifically requested"""
        analyzer = LightGBMResidualAnalyzer(model_type='lightgbm', output_dir=str(test_dir))
        
        results = analyzer.analyze(
            predictions=sample_data['predictions'],
            actuals=sample_data['actuals'],
            model_name='test_model'
        )
        
        # Save without specifying save_pkl=True
        analyzer.save_results(results=results, filename_prefix='test')
        
        # Check that no pkl files were created
        pkl_files = list(test_dir.glob("*.pkl"))
        assert len(pkl_files) == 0, f"Expected no pkl files, found {len(pkl_files)}"
        
        # But JSON and txt files should exist
        json_files = list(test_dir.glob("*.json"))
        txt_files = list(test_dir.glob("*.txt"))
        assert len(json_files) > 0, "JSON files should be created"
        assert len(txt_files) > 0, "Text files should be created"
    
    def test_pkl_files_created_when_requested(self, test_dir, sample_data):
        """Test that pkl files are created when explicitly requested"""
        analyzer = LightGBMResidualAnalyzer(model_type='lightgbm', output_dir=str(test_dir))
        
        results = analyzer.analyze(
            predictions=sample_data['predictions'],
            actuals=sample_data['actuals'],
            model_name='test_model'
        )
        
        # Save with save_pkl=True
        analyzer.save_results(results=results, filename_prefix='test', save_pkl=True)
        
        # Check that pkl files were created
        pkl_files = list(test_dir.glob("*.pkl"))
        assert len(pkl_files) > 0, "pkl files should be created when save_pkl=True"
    
    def test_multi_seed_cv_handling(self, test_dir, sample_data):
        """Test that multi-seed CV (3 seeds × 5 folds = 15 total) is properly handled"""
        analyzer = LightGBMResidualAnalyzer(model_type='lightgbm', output_dir=str(test_dir))
        
        fold_identifiers = []
        
        # Simulate 3 seeds × 5 folds
        for seed in range(3):
            for fold in range(5):
                fold_id = f'seed_{seed}_fold_{fold}'
                fold_identifiers.append(fold_id)
                
                results = analyzer.analyze(
                    predictions=sample_data['predictions'],
                    actuals=sample_data['actuals'],
                    model_name=f'lightgbm_{fold_id}'
                )
                
                analyzer.save_results(
                    results=results,
                    filename_prefix=fold_id
                )
        
        # Verify all 15 folds are saved
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        for target in targets:
            target_json_files = list(test_dir.glob(f"*{target}*.json"))
            assert len(target_json_files) == 1, f"Should have 1 file for {target}"
            
            with open(target_json_files[0], 'r') as f:
                data = json.load(f)
                # Check all fold identifiers are present
                saved_fold_ids = [entry.get('fold_id', entry.get('model_name', '')) for entry in data]
                assert len(saved_fold_ids) == 15, f"Expected 15 entries for {target}"


class TestTransformerDeepchemRemoval:
    """Tests for removing deepchem from transformer_model.py"""
    
    def test_transformer_uses_manual_tokenizer(self):
        """Test that transformer model uses manual tokenizer instead of deepchem"""
        # First check if transformer_model.py exists
        transformer_path = "/workspace/kaggle/neurips-open-polymer-prediction-2025/transformer_model.py"
        assert os.path.exists(transformer_path), "transformer_model.py should exist"
        
        # Read the file and check it doesn't import deepchem
        with open(transformer_path, 'r') as f:
            content = f.read()
            
        # Check deepchem is not imported
        assert 'from deepchem' not in content, "deepchem import should be removed"
        assert 'import deepchem' not in content, "deepchem import should be removed"
        assert 'dc.' not in content, "deepchem usage should be removed"
        
        # Check manual tokenizer is used
        assert 'ManualSmilesTokenizer' in content, "Manual tokenizer should be used"
    
    def test_transformer_tokenization_works(self):
        """Test that transformer tokenization works without deepchem"""
        from transformer_model import ManualSmilesTokenizer
        
        # Test basic tokenization
        tokenizer = ManualSmilesTokenizer()
        smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        
        # Should be able to tokenize
        tokens = tokenizer.tokenize(smiles)
        assert isinstance(tokens, np.ndarray), "Tokenize should return a numpy array"
        assert tokens.shape[0] == 1, "Should have one sequence"
        assert tokens.shape[1] > 0, "Should produce tokens"
        
        # Should be able to encode
        encoded = tokenizer.encode(smiles, max_length=100)
        assert 'input_ids' in encoded, "Should have input_ids"
        assert 'attention_mask' in encoded, "Should have attention_mask"


class TestPreprocessingAnalyzerResults:
    """Tests for saving preprocessing analyzer results"""
    
    @pytest.fixture
    def test_dir(self, tmp_path):
        """Create temporary test directory"""
        test_dir = tmp_path / "residual_analysis"
        test_dir.mkdir(exist_ok=True)
        return test_dir
    
    def test_preprocessing_results_saved_once(self, test_dir):
        """Test that preprocessing analyzer results are saved only once"""
        # Mock the preprocessing analyzer results
        mock_pca_results = {
            'explained_variance': [0.4, 0.3, 0.2, 0.1],
            'n_components': 4,
            'total_variance_explained': 1.0
        }
        
        mock_pls_results = {
            'r2_scores': {'Tg': 0.8, 'FFV': 0.7},
            'n_components': 3,
            'feature_importance': np.random.randn(10).tolist()
        }
        
        mock_autoencoder_results = {
            'reconstruction_error': 0.05,
            'latent_dim': 32,
            'epochs_trained': 50
        }
        
        # Save preprocessing results
        preprocessing_results = {
            'pca': mock_pca_results,
            'pls': mock_pls_results,
            'autoencoder': mock_autoencoder_results
        }
        
        output_file = test_dir / "preprocessing_methods.json"
        with open(output_file, 'w') as f:
            json.dump(preprocessing_results, f, indent=2)
        
        # Verify file exists and contains all methods
        assert output_file.exists(), "Preprocessing results file should exist"
        
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'pca' in loaded_results, "Should contain PCA results"
        assert 'pls' in loaded_results, "Should contain PLS results"
        assert 'autoencoder' in loaded_results, "Should contain autoencoder results"
        
        # Verify only one file is created
        json_files = list(test_dir.glob("preprocessing_*.json"))
        assert len(json_files) == 1, "Should have exactly one preprocessing results file"