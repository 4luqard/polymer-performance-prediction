import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import os
import tempfile
import json

# Test the residual analysis class that doesn't exist yet
from src.residual_analysis import ResidualAnalysis, ResidualAnalyzer


class TestResidualAnalysis:
    """Test suite for the ResidualAnalysis class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create sample predictions and targets
        self.n_samples = 100
        self.targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Create sample data with missing values pattern similar to real data
        np.random.seed(42)
        self.y_true = {}
        self.y_pred = {}
        
        for target in self.targets:
            # Simulate missing value patterns
            mask = np.random.rand(self.n_samples) > 0.9  # ~90% missing
            values = np.random.randn(self.n_samples) * 10 + 50
            values[~mask] = np.nan
            self.y_true[target] = values
            
            # Predictions should have values for all samples
            self.y_pred[target] = np.random.randn(self.n_samples) * 10 + 50
    
    def test_residual_analysis_initialization(self):
        """Test that ResidualAnalysis can be initialized"""
        ra = ResidualAnalysis(output_dir="test_output")
        assert ra.output_dir == "test_output"
        assert hasattr(ra, 'targets')
        assert ra.targets == self.targets
    
    def test_calculate_residuals(self):
        """Test residual calculation functionality"""
        ra = ResidualAnalysis()
        residuals = ra.calculate_residuals(self.y_true, self.y_pred)
        
        # Check that residuals are calculated correctly
        for target in self.targets:
            mask = ~np.isnan(self.y_true[target])
            expected = self.y_pred[target][mask] - self.y_true[target][mask]
            np.testing.assert_array_almost_equal(residuals[target], expected)
    
    def test_visualize_residuals(self):
        """Test residual visualization functionality"""
        ra = ResidualAnalysis()
        residuals = ra.calculate_residuals(self.y_true, self.y_pred)
        
        # Test that visualization doesn't raise errors
        with tempfile.TemporaryDirectory() as tmpdir:
            ra.output_dir = tmpdir
            ra.visualize_residuals(residuals, fold=0)
            
            # Check that plots were saved
            for target in self.targets:
                plot_path = os.path.join(tmpdir, f"residuals_{target}_fold_0.png")
                assert os.path.exists(plot_path)
    
    def test_analyze_patterns(self):
        """Test pattern analysis functionality"""
        ra = ResidualAnalysis()
        residuals = ra.calculate_residuals(self.y_true, self.y_pred)
        
        patterns = ra.analyze_patterns(residuals)
        
        # Check that pattern analysis returns expected structure
        assert isinstance(patterns, dict)
        for target in self.targets:
            assert target in patterns
            assert 'mean' in patterns[target]
            assert 'std' in patterns[target]
            assert 'skewness' in patterns[target]
            assert 'kurtosis' in patterns[target]
    
    def test_integration_with_cv(self):
        """Test integration with cross-validation"""
        ra = ResidualAnalysis()
        
        # Mock CV predictions
        cv_predictions = {
            'fold_0': {'y_true': self.y_true, 'y_pred': self.y_pred},
            'fold_1': {'y_true': self.y_true, 'y_pred': self.y_pred}
        }
        
        # Test that it can handle CV results
        results = ra.analyze_cv_results(cv_predictions)
        assert 'fold_0' in results
        assert 'fold_1' in results
    
    def test_save_results(self):
        """Test saving results functionality"""
        ra = ResidualAnalysis()
        residuals = ra.calculate_residuals(self.y_true, self.y_pred)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ra.output_dir = tmpdir
            ra.save_results(residuals, "test_model")
            
            # Check that results were saved
            results_path = os.path.join(tmpdir, "residual_analysis_test_model.pkl")
            assert os.path.exists(results_path)
    
    def test_save_human_readable_results(self):
        """Test saving results in human-readable formats"""
        ra = ResidualAnalysis()
        residuals = ra.calculate_residuals(self.y_true, self.y_pred)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ra.output_dir = tmpdir
            
            # Create mock results with numpy arrays
            results = {
                'residuals': residuals,
                'patterns': ra.analyze_patterns(residuals),
                'feature_importance': np.array([0.1, 0.2, 0.3, 0.4])
            }
            
            # Save results
            ra.save_results(results, "test_model")
            
            # Check that human-readable files were saved
            json_path = os.path.join(tmpdir, "residual_analysis_test_model.json")
            txt_path = os.path.join(tmpdir, "residual_analysis_test_model.txt")
            
            assert os.path.exists(json_path), "JSON file should be created"
            assert os.path.exists(txt_path), "Text file should be created"
            
            # Verify JSON content is readable
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                assert 'residuals' in json_data
                assert 'patterns' in json_data
                assert 'feature_importance' in json_data
            
            # Verify text content is human-readable
            with open(txt_path, 'r') as f:
                txt_content = f.read()
                assert 'Residual Analysis Results' in txt_content
                assert 'PATTERNS' in txt_content
                assert 'mean:' in txt_content


class TestResidualAnalyzer:
    """Test suite for model/method-specific analyzers"""
    
    def test_lightgbm_analyzer(self):
        """Test LightGBM-specific residual analysis"""
        from src.residual_analysis import LightGBMResidualAnalyzer
        
        analyzer = LightGBMResidualAnalyzer()
        
        # Mock LightGBM model predictions
        mock_model = Mock()
        mock_model.predict.return_value = np.random.randn(100, 5)
        
        # Test analysis
        results = analyzer.analyze(mock_model, X=None, y=None)
        assert 'residuals' in results
        assert 'feature_importance' in results
    
    def test_transformer_analyzer(self):
        """Test Transformer-specific residual analysis"""
        from src.residual_analysis import TransformerResidualAnalyzer
        
        analyzer = TransformerResidualAnalyzer()
        
        # Mock transformer predictions
        mock_predictions = {
            'predictions': np.random.randn(100, 5),
            'attention_weights': np.random.rand(100, 10, 10),
            'hidden_states': np.random.randn(100, 10, 256)
        }
        
        # Test analysis
        results = analyzer.analyze(mock_predictions)
        assert 'residuals' in results
        assert 'attention_analysis' in results
    
    def test_autoencoder_analyzer(self):
        """Test Autoencoder-specific residual analysis"""
        from src.residual_analysis import AutoencoderResidualAnalyzer
        
        analyzer = AutoencoderResidualAnalyzer()
        
        # Mock autoencoder outputs
        mock_outputs = {
            'reconstructed': np.random.randn(100, 50),
            'latent': np.random.randn(100, 10)
        }
        
        # Test analysis
        results = analyzer.analyze(mock_outputs)
        assert 'reconstruction_error' in results
        assert 'latent_analysis' in results
    
    def test_pca_analyzer(self):
        """Test PCA-specific residual analysis"""
        from src.residual_analysis import PCAResidualAnalyzer
        
        analyzer = PCAResidualAnalyzer()
        
        # Mock PCA outputs
        mock_outputs = {
            'transformed': np.random.randn(100, 10),
            'explained_variance': np.random.rand(10)
        }
        
        # Test analysis
        results = analyzer.analyze(mock_outputs)
        assert 'reconstruction_error' in results
        assert 'variance_analysis' in results


class TestIntegration:
    """Test integration with existing codebase"""
    
    def test_cv_integration(self):
        """Test that residual analysis integrates with CV without breaking it"""
        # Import CV module
        from cv import perform_cross_validation
        
        # Mock the residual analysis
        with patch('src.residual_analysis.ResidualAnalysis') as mock_ra:
            # Run CV with residual analysis enabled
            # This should not raise any errors or change CV behavior
            pass  # Actual test would run CV here
    
    def test_model_integration(self):
        """Test that residual analysis integrates with model.py"""
        # Import model module
        import model
        
        # Mock residual analysis
        with patch('src.residual_analysis.LightGBMResidualAnalyzer') as mock_analyzer:
            # Create and train model
            # This should not raise any errors or change model behavior
            pass  # Actual test would create and train model
    
    def test_data_processing_integration(self):
        """Test integration with data_processing.py"""
        # Import data processing module
        import data_processing
        
        # Mock residual analyzers
        with patch('src.residual_analysis.TransformerResidualAnalyzer') as mock_t:
            with patch('src.residual_analysis.AutoencoderResidualAnalyzer') as mock_a:
                # Process data with residual analysis
                # This should not raise errors or change outputs
                pass  # Actual test would process data


class TestConfiguration:
    """Test configuration and setup"""
    
    def test_local_only_execution(self):
        """Test that residual analysis only runs in local environment"""
        # Set environment to non-local
        with patch.dict(os.environ, {'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}):
            from src.residual_analysis import should_run_analysis
            assert not should_run_analysis()
        
        # Set environment to local
        with patch.dict(os.environ, {}, clear=True):
            from src.residual_analysis import should_run_analysis
            assert should_run_analysis()
    
    def test_output_directory_creation(self):
        """Test that output directories are created properly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ra = ResidualAnalysis(output_dir=os.path.join(tmpdir, "new_dir"))
            ra._ensure_output_dir()
            assert os.path.exists(ra.output_dir)
    
    def test_gitignore_update(self):
        """Test that residual analysis outputs are added to gitignore"""
        from src.residual_analysis import update_gitignore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gitignore_path = os.path.join(tmpdir, ".gitignore")
            
            # Create initial gitignore
            with open(gitignore_path, 'w') as f:
                f.write("*.pyc\n")
            
            # Update gitignore
            update_gitignore(gitignore_path)
            
            # Check that residual analysis patterns were added
            with open(gitignore_path, 'r') as f:
                content = f.read()
                assert "residual_analysis/" in content
                assert "*.pkl" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])