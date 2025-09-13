import inspect
import numpy as np
import pytest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_processing import apply_autoencoder


def test_apply_autoencoder_signature():
    sig = inspect.signature(apply_autoencoder)
    assert 'supervised' not in sig.parameters


def test_apply_autoencoder_requires_y_train():
    X = np.random.rand(4, 3).astype(np.float32)
    with pytest.raises(ValueError):
        apply_autoencoder(X)


def test_apply_autoencoder_output_shape():
    X = np.random.rand(10, 4).astype(np.float32)
    y = np.random.rand(10, 2).astype(np.float32)
    y[0, 1] = np.nan
    encoded = apply_autoencoder(X, y_train=y, latent_dim=3, epochs=1, batch_size=5)
    assert encoded.shape == (10, 3)
