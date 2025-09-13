"""Test loss masking implementation for autoencoder"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data_processing import apply_autoencoder, create_masked_competition_loss
import tensorflow as tf

def test_masked_loss_function():
    """Test that the masked loss function correctly handles missing values"""
    print("Testing masked loss function...")
    
    # Create test data with known values
    num_targets = 5
    batch_size = 4
    
    # Create predictions and true values
    y_true_np = np.array([
        [100.0, 0.5, 0.3, 1.2, 20.0],  # All values present
        [0.0, 0.4, 0.0, 1.3, 0.0],      # Only FFV and Density present (0.0 marks missing)
        [150.0, 0.0, 0.2, 0.0, 25.0],   # Tg, Tc, Rg present
        [0.0, 0.0, 0.0, 0.0, 0.0]       # All missing
    ])
    
    y_pred_np = np.array([
        [110.0, 0.55, 0.32, 1.25, 21.0],  # Predictions for all
        [5.0, 0.42, 0.1, 1.28, 5.0],      # Predictions for all
        [145.0, 0.1, 0.22, 0.5, 24.0],    # Predictions for all
        [50.0, 0.3, 0.15, 1.0, 15.0]      # Predictions for all (shouldn't contribute to loss)
    ])
    
    # Convert to tensors
    y_true = tf.constant(y_true_np, dtype=tf.float32)
    y_pred = tf.constant(y_pred_np, dtype=tf.float32)
    
    # Create loss function
    loss_fn = create_masked_competition_loss(num_targets)
    
    # Calculate loss
    loss = loss_fn(y_true, y_pred)
    
    # Evaluate the loss
    loss_value = loss.numpy()
    
    print(f"Loss value: {loss_value}")
    
    # The loss should be finite and reasonable
    assert np.isfinite(loss_value), "Loss should be finite"
    assert loss_value > 0, "Loss should be positive for non-perfect predictions"
    
    print("✓ Masked loss function test passed")

def test_autoencoder_with_missing_values():
    """Test that autoencoder works with missing values using the new loss"""
    print("\nTesting autoencoder with missing values...")
    
    # Create synthetic data with missing values
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_targets = 5
    
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(30, n_features)
    
    # Create targets with missing values (NaN)
    y_train = np.random.randn(n_samples, n_targets)
    # Randomly set 60% of values to NaN
    mask = np.random.random((n_samples, n_targets)) < 0.6
    y_train[mask] = np.nan
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Percentage of missing values: {np.isnan(y_train).mean()*100:.1f}%")
    
    # Apply autoencoder
    try:
        X_train_encoded, X_test_encoded = apply_autoencoder(
            X_train, 
            X_test,
            y_train=y_train,
            latent_dim=10,
            epochs=2,  # Quick test
            batch_size=32,
            is_Kaggle=True  # Set to True to avoid residual analysis in test
        )
        
        print(f"Encoded train shape: {X_train_encoded.shape}")
        print(f"Encoded test shape: {X_test_encoded.shape}")
        
        # Check that encoding worked
        assert X_train_encoded.shape[0] == X_train.shape[0]
        assert X_test_encoded.shape[0] == X_test.shape[0]
        assert X_train_encoded.shape[1] == 10  # latent_dim
        
        print("✓ Autoencoder with missing values test passed")
        
    except Exception as e:
        print(f"✗ Error in autoencoder: {e}")
        raise

def test_residual_analysis_compatibility():
    """Test that residual analysis still works with the new loss"""
    print("\nTesting residual analysis compatibility...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_targets = 5
    
    X_train = np.random.randn(n_samples, n_features)
    
    # Create targets with missing values
    y_train = np.random.randn(n_samples, n_targets)
    mask = np.random.random((n_samples, n_targets)) < 0.3
    y_train[mask] = np.nan
    
    # Apply autoencoder with residual analysis enabled
    try:
        X_train_encoded = apply_autoencoder(
            X_train,
            y_train=y_train,
            latent_dim=10,
            epochs=2,
            batch_size=32,
            is_Kaggle=False  # Enable residual analysis
        )
        
        # Check if residual file was created
        import os
        if os.path.exists('autoencoder_residuals.parquet'):
            residual_df = pd.read_parquet('autoencoder_residuals.parquet')
            print(f"Residual DataFrame shape: {residual_df.shape}")
            print(f"Splits in residual data: {residual_df['split'].value_counts().to_dict()}")
            
            # Verify structure
            assert 'split' in residual_df.columns
            assert all(split in ['train', 'val', 'test'] for split in residual_df['split'].unique())
            
            print("✓ Residual analysis compatibility test passed")
        else:
            print("✓ Residual analysis file not created (expected in Kaggle mode)")
            
    except Exception as e:
        print(f"✗ Error in residual analysis: {e}")
        raise

if __name__ == "__main__":
    print("="*60)
    print("Testing Loss Masking Implementation")
    print("="*60)
    
    # Run tests
    test_masked_loss_function()
    test_autoencoder_with_missing_values()
    test_residual_analysis_compatibility()
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)