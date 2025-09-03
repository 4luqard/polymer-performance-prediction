"""
Test integration of residual analysis functionality.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.residual_analysis import ResidualAnalyzer, ResidualMetrics
from transformer_model import TransformerModel


class TestResidualAnalysis:
    """Test residual analysis functionality."""
    
    def test_residual_analyzer_initialization(self, tmp_path):
        """Test ResidualAnalyzer initialization."""
        analyzer = ResidualAnalyzer(output_dir=tmp_path)
        assert analyzer.output_dir == tmp_path
        assert len(analyzer.residuals_store) == 0
    
    def test_compute_residuals(self, tmp_path):
        """Test residual computation."""
        analyzer = ResidualAnalyzer(output_dir=tmp_path)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        residuals = analyzer.compute_residuals(y_true, y_pred, "test_model")
        
        assert residuals.shape == y_true.shape
        np.testing.assert_array_almost_equal(residuals, y_true - y_pred)
        assert "test_model" in analyzer.residuals_store
    
    def test_analyze_residuals(self, tmp_path):
        """Test residual analysis metrics."""
        analyzer = ResidualAnalyzer(output_dir=tmp_path)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        residuals = y_true - y_pred
        
        metrics = analyzer.analyze_residuals(residuals, y_true, y_pred, "test_target")
        
        assert isinstance(metrics, ResidualMetrics)
        assert metrics.target == "test_target"
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert -1 <= metrics.r2 <= 1
    
    def test_multi_target_analysis(self, tmp_path):
        """Test multi-target residual analysis."""
        analyzer = ResidualAnalyzer(output_dir=tmp_path)
        
        y_true = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_pred = np.array([[1.1, 2.1], [2.2, 2.9], [2.9, 4.1], [4.1, 4.9], [4.8, 6.1]])
        
        results = analyzer.analyze_multi_target(
            y_true, y_pred, 
            ["target1", "target2"], 
            "test_model"
        )
        
        assert len(results) == 2
        assert "target1" in results
        assert "target2" in results
        assert isinstance(results["target1"], ResidualMetrics)


class TestTransformerResiduals:
    """Test transformer model residual functionality."""
    
    def test_predict_with_residuals(self, sample_smiles, sample_targets):
        """Test transformer predict with residuals."""
        model = TransformerModel(
            vocab_size=100,
            target_dim=5,
            latent_dim=32,
            num_heads=2,
            ff_dim=64,
            max_length=50
        )
        
        # Build model
        model.build_model()
        
        # Mock training (just to initialize weights)
        model.fit(sample_smiles[:3], sample_targets[:3], epochs=1, verbose=0)
        
        # Test predict with residuals
        predictions, residuals = model.predict(
            sample_smiles[:3], 
            return_residuals=True,
            y_true=sample_targets[:3]
        )
        
        assert predictions.shape == sample_targets[:3].shape
        assert residuals.shape == sample_targets[:3].shape
        np.testing.assert_array_almost_equal(
            residuals, 
            sample_targets[:3] - predictions
        )
    
    def test_fit_predict_with_residuals(self, sample_smiles, sample_targets):
        """Test transformer fit_predict with residuals."""
        model = TransformerModel(
            vocab_size=100,
            target_dim=5,
            latent_dim=32,
            num_heads=2,
            ff_dim=64,
            max_length=50
        )
        
        # Test fit_predict with residuals
        predictions, residuals, history = model.fit_predict(
            sample_smiles[:3], 
            sample_targets[:3],
            epochs=1,
            verbose=0,
            return_residuals=True
        )
        
        assert predictions.shape == sample_targets[:3].shape
        assert residuals.shape == sample_targets[:3].shape
        assert history is not None
    
    def test_get_training_residuals(self, sample_smiles, sample_targets):
        """Test getting training residuals."""
        model = TransformerModel(
            vocab_size=100,
            target_dim=5,
            latent_dim=32,
            num_heads=2,
            ff_dim=64,
            max_length=50
        )
        
        # Train model
        model.build_model()
        model.fit(sample_smiles[:3], sample_targets[:3], epochs=1, verbose=0)
        
        # Get training residuals
        residuals = model.get_training_residuals(sample_smiles[:3])
        
        assert residuals.shape == sample_targets[:3].shape
        
        # Verify residuals match expected calculation
        predictions = model.predict(sample_smiles[:3])
        expected_residuals = sample_targets[:3] - predictions
        np.testing.assert_array_almost_equal(residuals, expected_residuals)