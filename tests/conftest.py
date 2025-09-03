"""
Shared pytest fixtures and configuration for test suite.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.constants import TARGET_COLUMNS


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        'CC(C)C',  # isobutane
        'CCO',  # ethanol
        'CCC',  # propane
        'c1ccccc1',  # benzene
        'CC(=O)O',  # acetic acid
        'CC(C)(C)CC(C)(C)CC(C)(C)C',  # complex polymer
    ]


@pytest.fixture(scope="session") 
def sample_targets():
    """Sample target values with missing data patterns."""
    return np.array([
        [1.0, np.nan, np.nan, np.nan, np.nan],  # Only Tg
        [2.0, 3.0, np.nan, np.nan, np.nan],  # Tg and FFV
        [np.nan, np.nan, 4.0, 5.0, np.nan],  # Tc and Density
        [1.5, 2.5, 3.5, 4.5, 5.5],  # All targets
        [np.nan, np.nan, np.nan, 6.0, 7.0],  # Density and Rg
        [8.0, np.nan, 9.0, np.nan, 10.0],  # Mixed pattern
    ])


@pytest.fixture
def sample_dataframe(sample_smiles, sample_targets):
    """Create a sample DataFrame with SMILES and targets."""
    df = pd.DataFrame({
        'SMILES': sample_smiles,
        'Tg': sample_targets[:, 0],
        'FFV': sample_targets[:, 1],
        'Tc': sample_targets[:, 2],
        'Density': sample_targets[:, 3],
        'Rg': sample_targets[:, 4]
    })
    return df


@pytest.fixture
def mock_model_params():
    """Standard model parameters for testing."""
    return {
        'random_state': 42,
        'epochs': 2,
        'batch_size': 2,
        'verbose': 0
    }


@pytest.fixture
def target_columns():
    """Target column names."""
    return TARGET_COLUMNS


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Path to test data directory."""
    return project_root / "data" / "test"


@pytest.fixture(scope="session")
def train_data_dir(project_root):
    """Path to training data directory."""
    return project_root / "data" / "train"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def suppress_tf_warnings():
    """Suppress TensorFlow warnings during tests."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*softmax.*')
    warnings.filterwarnings('ignore', message='.*__array__.*')