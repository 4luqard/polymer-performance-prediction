"""
Helper functions for data handling and processing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from config import CONFIG

__all__ = [
    "load_raw_data",
    "load_preprocessed_data",
    "save_preprocessed_data",
    "split_features_targets",
    "handle_missing_values",
    "load_cv_fold",
    "prepare_submission",
    "check_data_availability",
    "get_data_stats",
    "load_competition_data",
]


def load_raw_data(data_type: str = 'train') -> pd.DataFrame:
    """
    Load raw data from CSV files.
    
    Args:
        data_type: Type of data to load ('train', 'test', 'sample_submission')
    
    Returns:
        DataFrame with loaded data
    """
    file_map = {
        'train': CONFIG.train_file,
        'test': CONFIG.test_file,
        'sample_submission': CONFIG.sample_submission_file
    }
    
    if data_type not in file_map:
        raise ValueError(f"Unknown data type: {data_type}")
    
    file_path = CONFIG.get_data_path(file_map[data_type])
    return pd.read_csv(file_path)


def load_preprocessed_data(filename: str) -> pd.DataFrame:
    """
    Load preprocessed data from parquet or CSV files.
    
    Args:
        filename: Name of the preprocessed file
    
    Returns:
        DataFrame with preprocessed data
    """
    file_path = CONFIG.output_dir / filename
    
    if file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_preprocessed_data(df: pd.DataFrame, filename: str, format: str = 'parquet'):
    """
    Save preprocessed data to file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        format: Output format ('parquet' or 'csv')
    """
    output_path = CONFIG.output_dir / filename
    CONFIG.output_dir.mkdir(exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def split_features_targets(df: pd.DataFrame, 
                         feature_cols: Optional[list] = None,
                         target_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into features and targets.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_cols: List of target column names
    
    Returns:
        Tuple of (features_df, targets_df)
    """
    if target_cols is None:
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Find available target columns
    available_targets = [col for col in target_cols if col in df.columns]
    
    if feature_cols is None:
        # All columns except targets and ID columns
        feature_cols = [col for col in df.columns 
                       if col not in available_targets + ['id', 'SMILES']]
    
    X = df[feature_cols] if feature_cols else df.drop(columns=available_targets + ['id', 'SMILES'], errors='ignore')
    y = df[available_targets] if available_targets else pd.DataFrame()
    
    return X, y


def handle_missing_values(X: pd.DataFrame, 
                        y: Optional[pd.DataFrame] = None,
                        strategy: str = 'mean') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Handle missing values in features and targets.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame (optional)
        strategy: Imputation strategy for features ('mean', 'median', 'zero')
    
    Returns:
        Tuple of (X_imputed, y) with missing values handled
    """
    from sklearn.impute import SimpleImputer
    
    # Handle features
    if strategy == 'zero':
        X_imputed = X.fillna(0)
    else:
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    # Targets are not imputed (kept as NaN for multi-output handling)
    return X_imputed, y


def load_cv_fold(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cross-validation fold data.
    
    Args:
        fold: Fold number (0-4)
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    if not CONFIG.use_cv:
        raise ValueError("Cross-validation data not available")
    
    paths = CONFIG.get_cv_path(fold)
    
    # Load train and validation data
    train_df = pd.read_csv(paths['train'])
    val_df = pd.read_csv(paths['val'])
    
    # Split features and targets
    X_train, y_train = split_features_targets(train_df)
    X_val, y_val = split_features_targets(val_df)
    
    return X_train, y_train, X_val, y_val


def prepare_submission(predictions: np.ndarray, 
                      test_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Prepare submission DataFrame from predictions.
    
    Args:
        predictions: Array of predictions (n_samples, 5)
        test_ids: Array of test IDs
    
    Returns:
        DataFrame ready for submission
    """
    if test_ids is None:
        test_df = load_raw_data('test')
        test_ids = test_df['id'].values
    
    submission = pd.DataFrame({
        'id': test_ids,
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    return submission


def check_data_availability(check_cv: bool = False) -> Dict[str, bool]:
    """
    Check which data files are available.
    
    Args:
        check_cv: Whether to check for CV data
    
    Returns:
        Dictionary with availability status
    """
    status = {
        'train': CONFIG.get_data_path(CONFIG.train_file).exists(),
        'test': CONFIG.get_data_path(CONFIG.test_file).exists(),
        'sample_submission': CONFIG.get_data_path(CONFIG.sample_submission_file).exists(),
    }
    
    if check_cv and CONFIG.cv_dir:
        status['cv'] = CONFIG.cv_dir.exists()
        if status['cv']:
            # Check individual folds
            for fold in range(5):
                try:
                    paths = CONFIG.get_cv_path(fold)
                    status[f'cv_fold_{fold}'] = all(p.exists() for p in paths.values())
                except:
                    status[f'cv_fold_{fold}'] = False
    
    return status


def get_data_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with data statistics
    """
    stats = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict()
    }
    
    # Check for target columns
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_targets = [col for col in target_cols if col in df.columns]
    
    if available_targets:
        stats['targets'] = {
            'available': available_targets,
            'missing_patterns': {}
        }
        
        # Analyze missing patterns
        for col in available_targets:
            stats['targets']['missing_patterns'][col] = {
                'n_missing': df[col].isna().sum(),
                'pct_missing': df[col].isna().mean() * 100
            }
    
    return stats


def load_competition_data(train_path, test_path, supp_paths=None, use_supplementary=True):
    """Load main and supplementary competition datasets."""
    print("Loading training data...")

    train_df = pd.read_csv(train_path)
    train_df['new_sim'] = True
    print(f"Main training data shape: {train_df.shape}")

    if use_supplementary and supp_paths:
        print("\nLoading supplementary datasets...")
        all_train_dfs = [train_df]

        for supp_path in supp_paths:
            try:
                supp_df = pd.read_csv(supp_path)
                if 'dataset1.csv' in supp_path and 'TC_mean' in supp_df.columns:
                    supp_df = supp_df.rename(columns={'TC_mean': 'Tc'})
                    print("Renamed TC_mean to Tc in dataset1")

                if 'id' not in supp_df.columns:
                    supp_df['id'] = [
                        f"supp_{supp_path.split('/')[-1].split('.')[0]}_{i}"
                        for i in range(len(supp_df))
                    ]

                supp_df['new_sim'] = False
                print(f"Loaded {supp_path}: {supp_df.shape}")
                all_train_dfs.append(supp_df)
            except Exception as e:
                print(f"Could not load {supp_path}: {e}")

        train_df = pd.concat(all_train_dfs, ignore_index=True)
        print(f"\nCombined training data shape: {train_df.shape}")
    else:
        print("\n** Using ONLY main training data (no supplementary datasets) **")

    print("\nLoading test data...")
    test_df = pd.read_csv(test_path)
    test_df['new_sim'] = True
    print(f"Test data shape: {test_df.shape}")

    print("\nRemoving duplicates from training data...")
    original_count = len(train_df)
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    train_df['target_count'] = train_df[target_columns].notna().sum(axis=1)
    train_df = train_df.sort_values(['target_count', 'new_sim'], ascending=[False, False])
    train_df = train_df.drop_duplicates(subset=['SMILES'], keep='first')
    train_df = train_df.drop('target_count', axis=1)

    duplicate_count = original_count - len(train_df)
    print(f"Removed {duplicate_count} duplicate rows")
    print(f"Final training data shape: {train_df.shape}")

    return train_df, test_df