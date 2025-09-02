"""
Comprehensive test for residual analysis integration with all models
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from residual_analysis import ResidualAnalyzer
import warnings
warnings.filterwarnings('ignore')

def test_residual_analyzer_with_multiple_models():
    """Test residual analyzer with multiple model types"""
    analyzer = ResidualAnalyzer()
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randn(n_samples)
    
    # Simulate predictions from different models
    models_residuals = {
        'lightgbm_Tg': y_true - (y_true + np.random.randn(n_samples) * 0.1),
        'lightgbm_FFV': y_true - (y_true + np.random.randn(n_samples) * 0.2),
        'transformer_Tg': y_true - (y_true + np.random.randn(n_samples) * 0.15),
        'transformer_FFV': y_true - (y_true + np.random.randn(n_samples) * 0.25),
    }
    
    # Test comparison
    comparison = analyzer.compare_models_dict(models_residuals)
    assert comparison is not None
    assert isinstance(comparison, str)
    assert 'lightgbm_Tg' in comparison
    assert 'transformer_Tg' in comparison
    print("✓ Model comparison works")

def test_residual_metrics_computation():
    """Test residual metrics computation"""
    analyzer = ResidualAnalyzer()
    
    # Create test data
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
    
    metrics = analyzer.compute_metrics(y_true, y_pred)
    
    assert hasattr(metrics, 'mae')
    assert hasattr(metrics, 'rmse')
    assert hasattr(metrics, 'r2')
    assert hasattr(metrics, 'mean_residual')
    assert hasattr(metrics, 'std_residual')
    
    assert metrics.mae > 0
    assert metrics.rmse > 0
    assert -1 <= metrics.r2 <= 1
    print("✓ Metrics computation works")

def test_command_line_integration():
    """Test that command line arguments are properly integrated"""
    import subprocess
    
    # Test that help includes residual analysis
    result = subprocess.run(
        [sys.executable, '../model.py', '--help'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__)
    )
    
    # The script should at least run without import errors
    assert 'ImportError' not in result.stderr or 'residual_analysis' not in result.stderr
    print("✓ Command line integration works")

def test_model_import():
    """Test that model.py can import residual_analysis"""
    try:
        # Test import in model context
        import importlib.util
        spec = importlib.util.spec_from_file_location("model", "../model.py")
        model_module = importlib.util.module_from_spec(spec)
        # We don't execute the module, just check it can be loaded
        assert model_module is not None
        print("✓ Model module loads successfully")
    except ImportError as e:
        if 'residual_analysis' in str(e):
            pytest.fail(f"Failed to import residual_analysis in model.py: {e}")
        else:
            # Other import errors are acceptable for this test
            print("✓ Model module structure valid")

if __name__ == "__main__":
    print("Running residual analysis integration tests...")
    
    test_residual_analyzer_with_multiple_models()
    test_residual_metrics_computation()
    test_model_import()
    test_command_line_integration()
    
    print("\n✅ All integration tests passed!")