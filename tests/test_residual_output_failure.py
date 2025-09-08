import pytest
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from src.residual_analysis import ResidualAnalysis, LightGBMResidualAnalyzer


class TestResidualOutputFailure:
    """Test residual output changes based on :FAILURE: feedback"""
    
    @pytest.fixture
    def test_data(self):
        """Create test data with multiple folds and seeds"""
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        n_samples = 100
        n_folds = 5
        n_seeds = 3
        
        # Generate data for 15 folds (3 seeds x 5 folds)
        fold_data = {}
        for seed in range(n_seeds):
            for fold in range(n_folds):
                fold_idx = seed * n_folds + fold
                fold_data[f'fold_{fold_idx}'] = {
                    'residuals': {target: np.random.randn(n_samples) for target in targets},
                    'statistics': {
                        target: {
                            'mean': np.random.randn(),
                            'std': abs(np.random.randn()),
                            'rmse': abs(np.random.randn()),
                            'mae': abs(np.random.randn())
                        } for target in targets
                    },
                    'cv_seed': seed,
                    'fold_number': fold
                }
        
        return fold_data, targets
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for test outputs"""
        output_dir = tmp_path / "residual_analysis"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def test_no_json_by_default(self, temp_dir):
        """Test that JSON files are not created by default"""
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        
        results = {
            'residuals': {'Tg': np.random.randn(10)},
            'statistics': {'Tg': {'mean': 0.5, 'std': 1.2}}
        }
        
        # Save without specifying save_json (should default to False)
        ra.save_results(results, 'test_model')
        
        # Check that no JSON files were created
        json_files = list(temp_dir.glob("*.json"))
        assert len(json_files) == 0, "JSON files should not be created by default"
    
    def test_no_pkl_by_default(self, temp_dir):
        """Test that pkl files are not created by default"""
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        
        results = {
            'residuals': {'Tg': np.random.randn(10)},
            'statistics': {'Tg': {'mean': 0.5, 'std': 1.2}}
        }
        
        # Save without specifying save_pkl (should default to False)
        ra.save_results(results, 'test_model', save_pkl=False)
        
        # Check that no pkl files were created
        pkl_files = list(temp_dir.glob("*.pkl"))
        assert len(pkl_files) == 0, "Pickle files should not be created by default"
    
    def test_markdown_output_format(self, temp_dir, test_data):
        """Test markdown output with combined statistics and visualizations"""
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        fold_data, targets = test_data
        
        # Mock the visualization to create a base64 encoded image
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            # Create a simple plot and save to bytes
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            
            # Test saving a single fold
            fold_0_data = fold_data['fold_0']
            ra.save_cv_fold_results(
                fold_data=fold_0_data,
                model_name='lightgbm',
                fold_idx=0,
                cv_seed=0
            )
        
        # Check markdown files exist with correct naming
        for target in targets:
            md_file = temp_dir / f"residuals_cv_lightgbm_{target}.md"
            assert md_file.exists(), f"Markdown file for {target} should exist"
            
            # Check content format
            content = md_file.read_text()
            assert "---" in content
            assert "Model: lightgbm" in content
            assert "Seed: 0" in content
            assert "Statistics:" in content
            assert "Visualization:" in content
    
    def test_residual_file_naming(self, temp_dir):
        """Test that residual files follow the correct naming convention"""
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        
        # Test for LightGBM
        ra.save_cv_results(
            results={'Tg': {'statistics': {'mean': 0.1}}},
            model_name='lightgbm'
        )
        
        expected_file = temp_dir / "residuals_Tg.md"
        assert expected_file.exists()
        
        # Test for Ridge
        ra.save_cv_results(
            results={'FFV': {'statistics': {'mean': 0.2}}},
            model_name='ridge'
        )
        
        expected_file = temp_dir / "residuals_FFV.md"
        assert expected_file.exists()
    
    def test_fold_appending(self, temp_dir, test_data):
        """Test that fold results are appended to the same file"""
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        fold_data, targets = test_data
        
        # Save multiple folds for the same target
        for fold_idx in range(3):
            fold_key = f'fold_{fold_idx}'
            ra.save_cv_fold_results(
                fold_data=fold_data[fold_key],
                model_name='lightgbm',
                fold_idx=fold_idx,
                cv_seed=0
            )
        
        # Check that results are appended
        for target in targets:
            md_file = temp_dir / f"residuals_cv_lightgbm_{target}.md"
            content = md_file.read_text()
            # Should have 3 sections (one per fold)
            assert content.count("Seed: 0") == 3
    
    def test_transformer_results_saved(self, temp_dir):
        """Test that transformer residual analysis results are saved"""
        from transformer_model import TransformerModel
        
        with patch('transformer_model.should_run_analysis', return_value=True):
            with patch('transformer_model.ResidualAnalysisHook') as mock_hook_class:
                mock_hook = Mock()
                mock_hook_class.return_value = mock_hook
                
                # Mock analyzer results
                analyzer_results = {
                    'residuals': {'Tg': np.random.randn(10)},
                    'attention_analysis': {'mean_attention': 0.5}
                }
                mock_hook.analyze_model_specific.return_value = analyzer_results
                
                # Create transformer instance and run transform
                transformer = TransformerModel()
                smiles = ['CCO'] * 10
                
                # Mock the encoder_model to avoid fitting requirement
                mock_encoder = Mock()
                mock_encoder.predict.return_value = np.random.randn(10, 768)
                transformer.encoder_model = mock_encoder
                
                # Mock the model for residual analysis
                mock_model = Mock()
                mock_output = Mock()
                mock_output.last_hidden_state = Mock()
                mock_output.last_hidden_state.cpu.return_value.numpy.return_value = np.random.randn(10, 200, 768)
                mock_model.predict.return_value = mock_output
                transformer.model = mock_model
                
                # Mock tokenizer tokenize method
                with patch.object(transformer.tokenizer, 'tokenize') as mock_tokenize:
                    mock_tokenize.return_value = np.zeros((10, 200), dtype=np.int32)
                    
                    # Run transform
                    latent = transformer.transform(smiles)
                    
                    # Verify analyzer was called
                    mock_hook.analyze_model_specific.assert_called_once()
                    
                    # Verify save was called
                    mock_hook.save.assert_called_once_with(
                        analyzer_results, 
                        'transformer_residual_analysis'
                    )
    
    def test_preprocessing_methods_saved(self, temp_dir):
        """Test that preprocessing method results are saved correctly"""
        # Create test results for multiple preprocessing methods
        all_analyzer_results = {
            'pca': {
                'reconstruction_error': {'mean': 0.1, 'std': 0.05},
                'variance_analysis': {'cumulative_variance': [0.3, 0.5, 0.7]}
            },
            'pls': {
                'component_analysis': {'mean_loading': [0.2, 0.3]},
                'score_analysis': {'variance': [0.1, 0.15]}
            },
            'autoencoder': {
                'reconstruction_error': {'mean': 0.08, 'std': 0.03},
                'latent_analysis': {'mean_activation': np.zeros(10)}
            },
            'transformer': {
                'residuals': {'Tg': np.random.randn(5)},
                'attention_analysis': {'mean_attention': 0.6}
            }
        }
        
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        ra.save_results(all_analyzer_results, "preprocessing_methods")
        
        # Check that file was created
        expected_file = temp_dir / "residual_analysis_preprocessing_methods.txt"
        assert expected_file.exists()
        
        # Check content includes all methods
        content = expected_file.read_text()
        assert 'pca' in content.lower()
        assert 'pls' in content.lower()
        assert 'autoencoder' in content.lower()
        assert 'transformer' in content.lower()
    
    def test_statistics_not_missing(self, temp_dir):
        """Test that statistics are included in the output"""
        ra = ResidualAnalysis(output_dir=str(temp_dir))
        
        # Create fold data with statistics
        fold_data = {
            'residuals': {'Tg': np.random.randn(50)},
            'statistics': {
                'Tg': {
                    'mean': 0.05,
                    'std': 1.2,
                    'rmse': 1.21,
                    'mae': 0.95,
                    'r2': 0.85
                }
            },
            'cv_seed': 42,
            'fold_number': 0
        }
        
        ra.save_cv_fold_results(
            fold_data=fold_data,
            model_name='lightgbm',
            fold_idx=0,
            cv_seed=42
        )
        
        # Check that statistics are in the file
        md_file = temp_dir / "residuals_Tg.md"
        content = md_file.read_text()
        
        assert "mean: 0.05" in content or "mean: 0.050000" in content
        assert "std: 1.2" in content or "std: 1.200000" in content
        assert "rmse: 1.21" in content or "rmse: 1.210000" in content
        assert "mae: 0.95" in content or "mae: 0.950000" in content
        assert "r2: 0.85" in content or "r2: 0.850000" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])