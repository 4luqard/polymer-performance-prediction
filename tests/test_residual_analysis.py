import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from data_processing import apply_autoencoder

def test_residual_analysis_quick():
    """Quick test for residual analysis functionality"""
    np.random.seed(42)
    
    # Enable residual analysis
    os.environ['ENABLE_RESIDUAL_ANALYSIS'] = 'true'
    
    # Create synthetic data
    n_samples = 100
    n_features = 20
    n_targets = 5
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples, n_targets)
    
    # Add some NaN values to targets (simulating missing data)
    mask = np.random.random((n_samples, n_targets)) < 0.3
    y_train[mask] = np.nan
    
    X_test = np.random.randn(30, n_features)
    
    # Run autoencoder with residual analysis
    X_train_encoded, X_test_encoded = apply_autoencoder(
        X_train, X_test, y_train, 
        latent_dim=10, epochs=2, batch_size=32
    )
    
    # Check that encoded outputs have correct shape
    assert X_train_encoded.shape == (n_samples, 10)
    assert X_test_encoded.shape == (30, 10)
    
    # Check that residual file was created
    residual_file = 'autoencoder_residuals.parquet'
    assert os.path.exists(residual_file), f"Residual file {residual_file} not found"
    
    # Load and verify residual data
    residual_df = pd.read_parquet(residual_file)
    
    print(f"\nResidual DataFrame shape: {residual_df.shape}")
    print(f"Columns: {residual_df.columns.tolist()[:10]}...")
    print(f"Split counts: {residual_df['split'].value_counts().to_dict()}")
    
    # Verify structure
    assert 'split' in residual_df.columns
    assert residual_df['split'].isin(['train', 'val', 'test']).all()
    
    # Check for feature columns
    for i in range(n_features):
        assert f'feature_{i}' in residual_df.columns
    
    # Check for target, prediction, and residual columns
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    for name in target_names:
        assert f'{name}_actual' in residual_df.columns
        assert f'{name}_pred' in residual_df.columns
        assert f'{name}_residual' in residual_df.columns
    
    # Clean up
    os.environ['ENABLE_RESIDUAL_ANALYSIS'] = 'false'
    if os.path.exists(residual_file):
        os.remove(residual_file)
    
    print("\nResidual analysis test passed!")

if __name__ == "__main__":
    test_residual_analysis_quick()
