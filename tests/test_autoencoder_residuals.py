import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from data_processing import apply_autoencoder

def test_current_autoencoder_behavior():
    """Test the current apply_autoencoder function behavior"""
    np.random.seed(42)
    
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
    
    # Test with test data
    X_train_encoded, X_test_encoded = apply_autoencoder(
        X_train, X_test, y_train, 
        latent_dim=10, epochs=2, batch_size=32
    )
    
    assert X_train_encoded.shape == (n_samples, 10)
    assert X_test_encoded.shape == (30, 10)
    
    # Test without test data
    X_train_only = apply_autoencoder(
        X_train, None, y_train,
        latent_dim=10, epochs=2, batch_size=32
    )
    
    assert X_train_only.shape == (n_samples, 10)

if __name__ == "__main__":
    test_current_autoencoder_behavior()
    print("Current behavior test passed!")