"""
Centralized configuration for the NeurIPS polymer prediction project.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class DatasetPaths:
    train: Path
    test: Path
    submission: Path
    supplementary: List[Path]


class EnvironmentConfig:
    """Environment-specific configuration."""
    
    def __init__(self):
        self.is_kaggle = os.path.exists('/kaggle/input')
        self.project_root = Path(__file__).parent
        self._setup_paths()
        self._setup_model_params()
    
    def _setup_paths(self):
        """Setup data paths based on environment."""
        if self.is_kaggle:
            self.data_dir = Path('/kaggle/input/neurips-open-polymer-prediction-2025')
            self.output_dir = Path('/kaggle/working')
            self.cv_dir = None  # CV not available on Kaggle
        else:
            self.data_dir = self.project_root / 'data'
            self.output_dir = self.project_root / 'output'
            self.cv_dir = self.data_dir / 'cv'

        # Common paths
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.sample_submission_file = 'sample_submission.csv'

        path_map = {
            True: DatasetPaths(
                train=self.data_dir / self.train_file,
                test=self.data_dir / self.test_file,
                submission=self.output_dir / 'submission.csv',
                supplementary=[
                    self.data_dir / 'train_supplement/dataset1.csv',
                    self.data_dir / 'train_supplement/dataset2.csv',
                    self.data_dir / 'train_supplement/dataset3.csv',
                    self.data_dir / 'train_supplement/dataset4.csv',
                    Path('/kaggle/input/extra-dataset-with-smilestgpidpolimers-class/TgSS_enriched_cleaned.csv'),
                    Path('/kaggle/input/polymer-tg-density-excerpt/tg_density.csv')
                ]
            ),
            False: DatasetPaths(
                train=self.data_dir / 'raw/train.csv',
                test=self.data_dir / 'raw/test.csv',
                submission=self.output_dir / 'submission.csv',
                supplementary=[
                    self.data_dir / 'raw/train_supplement/dataset1.csv',
                    self.data_dir / 'raw/train_supplement/dataset2.csv',
                    self.data_dir / 'raw/train_supplement/dataset3.csv',
                    self.data_dir / 'raw/train_supplement/dataset4.csv',
                    self.data_dir / 'raw/extra_datasets/TgSS_enriched_cleaned.csv',
                    self.data_dir / 'raw/extra_datasets/tg_density.csv'
                ]
            )
        }

        self.dataset_paths = path_map[self.is_kaggle]
    
    def _setup_model_params(self):
        """Setup model parameters based on environment."""
        # LightGBM parameters
        self.lgb_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 500 if self.is_kaggle else 200,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Transformer parameters
        self.transformer_params = {
            'vocab_size': None,  # Set dynamically
            'target_dim': 5,
            'latent_dim': 32,
            'num_heads': 4,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'ff_dim': 64,
            'dropout_rate': 0.1,
            'max_length': 150,
            'random_state': 42
        }
        
        # Training parameters
        self.training_params = {
            'epochs': 20 if self.is_kaggle else 10,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 5,
            'reduce_lr_patience': 3
        }
        
        # Dimensionality reduction
        self.dim_reduction = DimensionalityReductionConfig()
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path for a data file."""
        return self.data_dir / filename

    def get_dataset_paths(self) -> DatasetPaths:
        """Return dataset paths for current environment."""
        return self.dataset_paths
    
    def get_cv_path(self, fold: int) -> Dict[str, Path]:
        """Get CV fold paths."""
        if not self.cv_dir:
            raise ValueError("CV data not available in Kaggle environment")
        
        return {
            'train': self.cv_dir / f'fold_{fold}_train.csv',
            'val': self.cv_dir / f'fold_{fold}_val.csv'
        }
    
    @property
    def use_cv(self) -> bool:
        """Check if cross-validation is available."""
        return self.cv_dir is not None and self.cv_dir.exists()
    
    @property
    def callback_monitor(self) -> str:
        """Get the metric to monitor for callbacks."""
        return 'val_loss' if self.training_params['validation_split'] > 0 else 'loss'


class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction methods."""
    
    # Available methods
    METHODS = {
        'none': None,
        'pca': 'PCA',
        'autoencoder': 'Autoencoder',
        'umap': 'UMAP',
        'tsne': 't-SNE'
    }
    
    def __init__(self, method: str = 'pca'):
        self.method = method
        self.params = self._get_method_params(method)
    
    def _get_method_params(self, method: str) -> Dict[str, Any]:
        """Get parameters for specific method."""
        params_map = {
            'pca': {
                'variance_threshold': 0.99999,
                'n_components': None  # Determined by variance threshold
            },
            'autoencoder': {
                'latent_dim': 32,
                'epochs': 20,
                'batch_size': 32
            },
            'umap': {
                'n_components': 50,
                'n_neighbors': 15,
                'min_dist': 0.1
            },
            'tsne': {
                'n_components': 50,
                'perplexity': 30,
                'learning_rate': 200
            },
            'none': {}
        }
        return params_map.get(method, {})
    
    def get_reducer(self):
        """Get the appropriate dimensionality reducer."""
        if self.method == 'pca':
            from sklearn.decomposition import PCA
            return PCA(n_components=self.params['variance_threshold'])
        elif self.method == 'autoencoder':
            # Return custom autoencoder class
            return None  # Implemented separately
        elif self.method == 'umap':
            try:
                import umap
                return umap.UMAP(**self.params)
            except ImportError:
                print("UMAP not available, falling back to PCA")
                return self._get_fallback_reducer()
        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            return TSNE(**self.params)
        else:
            return None
    
    def _get_fallback_reducer(self):
        """Get fallback reducer if primary method unavailable."""
        from sklearn.decomposition import PCA
        return PCA(n_components=0.99999)


class ModelRegistry:
    """Registry for model types and configurations."""

    def __init__(self, config: "EnvironmentConfig"):
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.models = {
            'lgbm': {
                'class': 'lightgbm.LGBMRegressor',
                'params': {t: config.lgb_params.copy() for t in targets},
                'supports_multioutput': False
            },
            'ridge': {
                'class': 'sklearn.linear_model.Ridge',
                'params': {
                    'Tg': {'alpha': 10.0, 'random_state': 42},
                    'FFV': {'alpha': 1.0, 'random_state': 42},
                    'Tc': {'alpha': 10.0, 'random_state': 42},
                    'Density': {'alpha': 5.0, 'random_state': 42},
                    'Rg': {'alpha': 10.0, 'random_state': 42}
                },
                'supports_multioutput': False
            }
        }

    def get_model(self, model_type: str, target: str, random_state: int | None = None):
        """Get model instance by type and target."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_info = self.models[model_type]
        if target not in model_info['params']:
            raise ValueError(f"Unknown target: {target}")

        params = model_info['params'][target].copy()
        if random_state is not None:
            params['random_state'] = random_state

        module_name, class_name = model_info['class'].rsplit('.', 1)

        if module_name == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMRegressor(**params)
        elif module_name.startswith('sklearn'):
            from sklearn.linear_model import Ridge

            if class_name == 'Ridge':
                return Ridge(**params)

        raise ValueError(f"Could not instantiate model: {model_info['class']}")

    def requires_multioutput_wrapper(self, model_type: str) -> bool:
        """Check if model needs MultiOutputRegressor wrapper."""
        return not self.models[model_type]['supports_multioutput']


# Global configuration instance
CONFIG = EnvironmentConfig()
MODEL_REGISTRY = ModelRegistry(CONFIG)

# Legacy support for old code
LIGHTGBM_PARAMS = CONFIG.lgb_params
