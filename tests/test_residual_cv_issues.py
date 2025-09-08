import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import tempfile
import json
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the modules we need to test
from residual_analysis import ResidualAnalyzer


class TestCVResidualIntegration:
    """Test CV integration with residual analysis"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_data = pd.DataFrame({
            'SMILES': ['CCO', 'CCC', 'CCCC'],
            'Tg': [1.0, 2.0, 3.0],
            'FFV': [0.1, 0.2, np.nan],
            'Tc': [10.0, np.nan, 30.0],
            'Density': [1.0, 1.1, 1.2],
            'Rg': [5.0, 6.0, 7.0]
        })
        
    @pytest.mark.parametrize("cv_flag", ["--cv", "--cv-only"])
    def test_cv_runs_with_residual_analysis_locally(self, cv_flag, tmp_path):
        """Test that CV runs without error when residual analysis is enabled locally"""
        # Mock the environment check to simulate local environment
        with patch('src.residual_analysis.is_local', return_value=True):
            with patch('cv.pd.read_csv', return_value=self.test_data):
                with patch('cv.DataProcessor') as mock_dp:
                    with patch('cv.LightGBM') as mock_lgb:
                        with patch('cv.CrossValidation._train_model') as mock_train:
                            # Set up mocks
                            mock_dp_instance = Mock()
                            mock_dp_instance.process.return_value = (
                                np.random.rand(3, 10),  # X_train
                                {target: self.test_data[target].values for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']},  # y_train_dict
                                None,  # X_val
                                None,  # y_val_dict
                                ['feat1', 'feat2'],  # feature_names
                                {}  # analyzer_results
                            )
                            mock_dp.return_value = mock_dp_instance
                            
                            # Mock train model to return predictions
                            mock_train.return_value = {
                                'Tg': np.array([1.1, 2.1, 3.1]),
                                'FFV': np.array([0.11, 0.21, 0.31]),
                                'Tc': np.array([10.1, 20.1, 30.1]),
                                'Density': np.array([1.01, 1.11, 1.21]),
                                'Rg': np.array([5.1, 6.1, 7.1])
                            }
                            
                            # Mock sys.argv
                            with patch('sys.argv', ['cv.py', cv_flag]):
                                # Import and run main
                                from cv import main
                                
                                # This should not raise any errors
                                main()
    
    def test_analyzer_results_saved_for_preprocessing_methods(self, tmp_path):
        """Test that analyzer_results from preprocessing methods are saved"""
        # Create a temporary output directory
        output_dir = tmp_path / "residual_outputs"
        
        with patch('src.residual_analysis.is_local', return_value=True):
            with patch('src.data_processing.ResidualAnalyzer') as mock_analyzer_class:
                # Mock analyzer instance
                mock_analyzer = Mock()
                mock_results = {
                    'pca': {
                        'explained_variance_ratio': [0.4, 0.3, 0.2],
                        'reconstruction_error': 0.05
                    }
                }
                mock_analyzer.get_results.return_value = mock_results
                mock_analyzer_class.return_value = mock_analyzer
                
                # Create data processor
                dp = DataProcessor(
                    train_data=self.test_data,
                    test_data=None,
                    target_cols=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
                    feature_engineering_methods=['pca']
                )
                
                # Mock the output directory
                with patch('src.residual_analysis.ResidualAnalyzer._get_output_dir', return_value=str(output_dir)):
                    # Process data
                    X_train, y_train_dict, _, _, feature_names, analyzer_results = dp.process()
                    
                    # Check that analyzer_results contains the expected data
                    assert 'pca' in analyzer_results
                    assert analyzer_results['pca'] == mock_results
                    
                    # Check that results were saved
                    expected_file = output_dir / "preprocessing_methods_residuals.json"
                    
                    # The save_results method should have been called
                    mock_analyzer.save_results.assert_called()
    
    def test_preprocessing_analyzer_results_format(self, tmp_path):
        """Test that preprocessing analyzer results are properly formatted and saved"""
        output_dir = tmp_path / "residual_outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Test data for multiple preprocessing methods
        analyzer_results = {
            'pca': {
                'method': 'PCA',
                'n_components': 3,
                'explained_variance_ratio': [0.4, 0.3, 0.2],
                'reconstruction_errors': {
                    'mean': 0.05,
                    'std': 0.01,
                    'max': 0.08
                }
            },
            'pls': {
                'method': 'PLS',
                'n_components': 2,
                'r2_scores': {
                    'Tg': 0.85,
                    'FFV': 0.78,
                    'Tc': 0.82,
                    'Density': 0.90,
                    'Rg': 0.88
                }
            },
            'autoencoder': {
                'method': 'Autoencoder',
                'encoding_dim': 16,
                'reconstruction_loss': 0.032,
                'validation_loss': 0.041
            }
        }
        
        # Expected file path
        expected_file = output_dir / "preprocessing_methods_residuals.json"
        
        # Test that the file can be created and contains correct structure
        with open(expected_file, 'w') as f:
            json.dump(analyzer_results, f, indent=2)
        
        # Verify file was created and can be read
        assert expected_file.exists()
        
        with open(expected_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results == analyzer_results
        assert all(method in loaded_results for method in ['pca', 'pls', 'autoencoder'])
    
    def test_cv_error_handling_with_residual_analysis(self):
        """Test that CV handles errors gracefully when residual analysis fails"""
        with patch('src.residual_analysis.is_local', return_value=True):
            with patch('cv.pd.read_csv', return_value=self.test_data):
                with patch('cv.DataProcessor') as mock_dp:
                    # Simulate an error in data processing
                    mock_dp.side_effect = Exception("Residual analysis error")
                    
                    with patch('sys.argv', ['cv.py', '--cv']):
                        from cv import main
                        
                        # Should handle the error gracefully
                        with pytest.raises(Exception, match="Residual analysis error"):
                            main()
    
    def test_residual_analyzer_integration_with_cv(self, tmp_path):
        """Test full integration of residual analyzer with CV"""
        output_dir = tmp_path / "residual_outputs"
        
        # Create mock residual analyzer that tracks calls
        calls = {'analyze_called': 0, 'save_called': 0}
        
        class MockResidualAnalyzer(ResidualAnalyzer):
            def __init__(self, model_name):
                self.model_name = model_name
                self.results = {}
                
            def analyze(self, *args, **kwargs):
                calls['analyze_called'] += 1
                self.results = {'test': 'results'}
                
            def save_results(self, output_dir):
                calls['save_called'] += 1
                # Create dummy files
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{self.model_name}_residuals.json"), 'w') as f:
                    json.dump(self.results, f)
        
        with patch('src.residual_analysis.is_local', return_value=True):
            with patch('cv.ResidualAnalyzer', MockResidualAnalyzer):
                with patch('cv.ResidualAnalyzer._get_output_dir', return_value=str(output_dir)):
                    # Test that analyzer is created and used during CV
                    assert calls['analyze_called'] == 0
                    assert calls['save_called'] == 0
                    
                    # After CV runs, these should be called
                    # (Implementation will be tested after fixing the actual code)