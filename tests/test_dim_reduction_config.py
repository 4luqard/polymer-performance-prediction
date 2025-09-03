import pytest
from config import CONFIG


def test_dim_reduction_defaults():
    """Dimensionality reduction flags should have expected defaults."""
    dr_cfg = CONFIG.dim_reduction
    assert dr_cfg.use_autoencoder is False
    assert dr_cfg.use_pls is False
    assert dr_cfg.pca_variance_threshold == pytest.approx(0.99999)
