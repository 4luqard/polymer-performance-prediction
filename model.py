#!/usr/bin/env python3
"""
Target-Specific LightGBM Model for NeurIPS Open Polymer Prediction 2025
Uses non-linear gradient boosting to better capture polymer property relationships
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import lightgbm as lgb
import re
import sys
import os
import math

# Import data processing functions
from data_processing import (
    calculate_main_branch_atoms,
    calculate_backbone_bonds,
    calculate_average_bond_length,
    extract_molecular_features,
    prepare_features,
    apply_autoencoder,
    select_features_for_target,
    preprocess_data,
    load_competition_data,
    TARGET_FEATURES
)

import warnings
warnings.filterwarnings('ignore')

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# PCA variance threshold - set to None to disable PCA
PCA_VARIANCE_THRESHOLD = None

# Autoencoder settings - set to True to use autoencoder instead of PCA
USE_AUTOENCODER = False
AUTOENCODER_LATENT_DIM = 26  # Number of latent dimensions

# Import competition metric and CV functions only if not on Kaggle
if not IS_KAGGLE:
    from src.competition_metric import neurips_polymer_metric
    from src.diagnostics import CVDiagnostics
    from cv import perform_cross_validation, perform_multi_seed_cv
    from config import LIGHTGBM_PARAMS


# Already checked above

# Set paths based on environment
if IS_KAGGLE:
    # Kaggle competition paths
    TRAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    TEST_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
    SUBMISSION_PATH = 'submission.csv'
    
    # Supplementary dataset paths
    SUPP_PATHS = [
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv'
    ]
else:
    # Local paths
    TRAIN_PATH = 'data/raw/train.csv'
    TEST_PATH = 'data/raw/test.csv'
    SUBMISSION_PATH = 'output/submission.csv'
    
    # Supplementary dataset paths
    SUPP_PATHS = [
        'data/raw/train_supplement/dataset1.csv',
        'data/raw/train_supplement/dataset2.csv',
        'data/raw/train_supplement/dataset3.csv',
        'data/raw/train_supplement/dataset4.csv'
    ]

# Target columns
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def main(cv_only=False, use_supplementary=True, model_type='lightgbm'):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
        use_supplementary: If True, include supplementary datasets in training
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
    """
    print(f"=== Separate {model_type.upper()} Models for Polymer Prediction ===")
    
    # Load competition data using imported function
    train_df, test_df = load_competition_data(
        TRAIN_PATH, TEST_PATH, 
        supp_paths=SUPP_PATHS,
        use_supplementary=use_supplementary
    )
    
    # Extract features
    print("\nExtracting features from training data...")
    X_train = prepare_features(train_df)
    
    print("\nExtracting features from test data...")
    X_test = prepare_features(test_df)
    
    # Prepare target variables
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    y_train = train_df[target_columns]
    
    # Print feature statistics
    print(f"\nFeature dimensions: {X_train.shape[1]} features")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # We'll handle missing values and scaling per target
    print("\nPreparing for target-specific training...")
    
    # We don't need to handle missing target values since we train separate models
    # Each model will only use samples with valid values for its specific target
    
    # Print target statistics
    print("\nTarget value statistics:")
    for col in target_columns:
        print(f"{col}: median={y_train[col].median():.4f}, mean={y_train[col].mean():.4f}, std={y_train[col].std():.4f}, "
              f"missing={y_train[col].isna().sum()} ({y_train[col].isna().sum()/len(y_train)*100:.1f}%)")
    
    # Apply preprocessing using imported function
    X_train_preprocessed, X_test_preprocessed = preprocess_data(
        X_train, X_test, 
        use_autoencoder=USE_AUTOENCODER,
        autoencoder_latent_dim=AUTOENCODER_LATENT_DIM,
        pca_variance_threshold=PCA_VARIANCE_THRESHOLD
    )
    
    # Run cross-validation if requested (but not on Kaggle)
    if cv_only:
        if IS_KAGGLE:
            print("\n⚠️  Cross-validation is not available in Kaggle notebooks")
            print("Proceeding with submission generation instead...")
        else:
            print("\n=== Testing with Multiple Random Seeds ===")
            if not use_supplementary:
                print("** Running CV WITHOUT supplementary datasets **")
            else:
                print("** Running CV WITH supplementary datasets **")
            
            # Run multi-seed CV with target-specific features (current implementation)
            multi_seed_result = perform_multi_seed_cv(X_train_preprocessed, y_train, cv_folds=5, 
                                                     target_columns=target_columns,
                                                     enable_diagnostics=False,
                                                     model_type=model_type)
            
            return {
                'multi_seed': multi_seed_result
            }
    
    # Train separate models for each target
    print(f"\n=== Training Separate {model_type.upper()} Models for Each Target ===")
    predictions = np.zeros((len(X_test), len(target_columns)))
    
    # Model parameters
    if model_type == 'lightgbm':
        # Use parameters from config for single source of truth
        if IS_KAGGLE:
            # Need to define params inline for Kaggle since we can't import config
            lgb_params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'max_depth': -1,
                'num_leaves': 31,
                'n_estimators': 200,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            lgb_params = LIGHTGBM_PARAMS.copy()
    else:
        # Ridge parameters
        target_alphas = {
            'Tg': 10.0,      # Higher regularization for sparse target
            'FFV': 1.0,      # Lower regularization for dense target
            'Tc': 10.0,      # Higher regularization for sparse target
            'Density': 5.0,  # Medium regularization
            'Rg': 10.0       # Higher regularization for sparse target
        }
    
    for i, target in enumerate(target_columns):
        print(f"\nTraining model for {target}...")
        
        # Get non-missing samples for this target
        mask = ~y_train[target].isna()
        n_samples = mask.sum()
        print(f"  Available samples: {n_samples} ({n_samples/len(y_train)*100:.1f}%)")
        
        if n_samples > 0:
            # Use preprocessed data directly
            X_target = X_train_preprocessed[mask]
            y_target = y_train[target][mask]
            
            print(f"  Using all {X_target.shape[1]} preprocessed features")
            print(f"  Training samples: {len(X_target)}")
            
            # No need for further preprocessing - data is already scaled and reduced
            X_target_final = X_target
            X_test_final = X_test_preprocessed
            
            # Train model for this target
            if model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**lgb_params)
                # Split data for validation to track overfitting
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_target_final, y_target, 
                    test_size=0.2, random_state=42
                )
                # Train with validation data to see training progress
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_tr, y_tr), (X_val, y_val)],
                    eval_names=['train', 'valid'],
                    eval_metric='mae',
                    callbacks=[lgb.log_evaluation(0)]  # Disable verbose output
                )
                # Get final MAE scores from eval results
                train_mae = model.evals_result_['train']['l1'][-1]
                val_mae = model.evals_result_['valid']['l1'][-1]
                print(f"  Final MAE - Train: {train_mae:.4f}, Valid: {val_mae:.4f}")
            else:
                # Use Ridge with target-specific alpha
                alpha = target_alphas.get(target, 1.0)
                print(f"  Using alpha={alpha}")
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_target_final, y_target)
            
            # Make predictions
            predictions[:, i] = model.predict(X_test_final)
            print(f"  Predictions: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}")
        else:
            # Use median of available values if no samples
            predictions[:, i] = y_train[target].median()
            print(f"  No samples available, using median: {predictions[:, i][0]:.4f}")
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Save submission
    print(f"\nSaving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")
    
    # Display submission preview
    print("\nSubmission preview:")
    print(submission_df)
    
    print("\n=== Model training complete! ===")
    print("Used target-specific features to reduce complexity")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    cv_only = '--cv-only' in sys.argv or '--cv' in sys.argv
    no_supplement = '--no-supplement' in sys.argv or '--no-supp' in sys.argv
    
    # Check for model type
    model_type = 'lightgbm'  # default
    if '--model' in sys.argv:
        model_idx = sys.argv.index('--model')
        if model_idx + 1 < len(sys.argv):
            model_type = sys.argv[model_idx + 1]
    
    main(cv_only=cv_only, use_supplementary=not no_supplement, model_type=model_type)
