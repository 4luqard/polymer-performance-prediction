#!/usr/bin/env python3
"""
Target-Specific LightGBM Model for NeurIPS Open Polymer Prediction 2025
Uses non-linear gradient boosting to better capture polymer property relationships
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import re
import sys
import os
import math
from rich import print

# Import data processing functions
from data_processing import *

import warnings
warnings.filterwarnings('ignore')

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# Autoencoder settings
USE_AUTOENCODER = True
AUTOENCODER_LATENT_DIM = 32  # Number of latent dimensions
EPOCHS = 25

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
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
        '/kaggle/input/extra-dataset-with-smilestgpidpolimers-class/TgSS_enriched_cleaned.csv',
        '/kaggle/input/polymer-tg-density-excerpt/tg_density.csv'
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
        'data/raw/train_supplement/dataset4.csv',
        'data/raw/extra_datasets/TgSS_enriched_cleaned.csv',
        'data/raw/extra_datasets/tg_density.csv'
    ]

def main(cv_only=False, seeds=[42, 123, 456], use_supplementary=True):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
        use_supplementary: If True, include supplementary datasets in training
    """
    print(f"=== Separate {'LIGHTGBM'} Models for Polymer Prediction ===")

    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    # Load competition data using imported function
    train_df, test_df = load_competition_data(
        TRAIN_PATH, TEST_PATH, 
        supp_paths=SUPP_PATHS,
        use_supplementary=use_supplementary
    )

    # Basic outlier removal
    train_df = remove_outliers(train_df, target_columns)

    # Extract features
    print("\nExtracting features from SMILES...")
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    # Prepare target variables
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
    
    # Note: Preprocessing is now done inside CV or during model training
    # to ensure consistency with masked samples

    # Model parameters
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

    # Run cross-validation if requested (but not on Kaggle)
    if cv_only:
        from cv import perform_multi_seed_cv
        print("\n=== Testing with Multiple Random Seeds ===")
        if not use_supplementary:
            print("** Running CV WITHOUT supplementary datasets **")
        else:
            print("** Running CV WITH supplementary datasets **")
            
        # Run multi-seed CV with target-specific features (current implementation)
        multi_seed_result = perform_multi_seed_cv(X_train, y_train, 
                                                  lgb_params=lgb_params,
                                                  cv_folds=3,
                                                  target_columns=target_columns,
                                                  smiles=train_df['SMILES'],
                                                  use_autoencoder=USE_AUTOENCODER,
                                                  autoencoder_latent_dim=AUTOENCODER_LATENT_DIM,
                                                  epochs=EPOCHS,
                                                  seeds=seeds)
            
        return {
            'multi_seed': multi_seed_result
        }
    
    # Train separate models for each target
    print(f"\n=== Training Separate {'LIGHTGBM'} Models for Each Target ===")
    predictions = np.zeros((len(X_test), len(target_columns)))

    for i, target in enumerate(target_columns):
        print(f"\nTraining model for {target}...")
        
        # Get non-missing samples for this target
        mask = ~y_train[target].isna()
        n_samples = mask.sum()
        print(f"  Available samples: {n_samples} ({n_samples/len(y_train)*100:.1f}%)")
        
        # Get masked samples
        mask_indices = np.where(mask)[0]
        X_target = X_train.iloc[mask_indices]
        y_target = y_train[target].iloc[mask_indices]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_target, y_target,
            test_size=0.15, random_state=42
        )

        # Apply preprocessing on masked samples
        y_tr_df = pd.DataFrame({target: y_tr})
        y_val_df = pd.DataFrame({target: y_val})
        X_tr_preprocessed, X_val_preprocessed, X_test_preprocessed = preprocess_data(
            [X_tr, X_val], X_test,
            use_autoencoder=USE_AUTOENCODER,
            autoencoder_latent_dim=AUTOENCODER_LATENT_DIM,
            y_train=[y_tr_df, y_val_df],
            epochs=EPOCHS
        )
            
        print(f"  Using all {X_tr_preprocessed.shape[1]} preprocessed features")
        print(f"  Training samples: {len(X_tr_preprocessed)}")
        print(f"  Validation samples: {len(X_val_preprocessed)}")

        # Train model for this target
        model = lgb.LGBMRegressor(**lgb_params)

        # Train with validation data to see training progress
        model.fit(
            X_tr_preprocessed, y_tr,
            eval_set=[(X_tr_preprocessed, y_tr), (X_val_preprocessed, y_val)],
            eval_names=['train', 'valid'],
            eval_metric='mae',
            callbacks=[lgb.log_evaluation(0)]  # Disable verbose output
        )
        # Get final MAE scores from eval results
        train_mae = model.evals_result_['train']['l1'][-1]
        val_mae = model.evals_result_['valid']['l1'][-1]
        print(f"  Final MAE - Train: {train_mae:.4f}, Valid: {val_mae:.4f}")
            
        # Make predictions
        predictions[:, i] = model.predict(X_test_preprocessed)
        print(f"  Predictions: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}")

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
    seeds = [456] if '--s' in sys.argv else [42, 123, 456]

    main(cv_only=cv_only, seeds=seeds, use_supplementary=not no_supplement)
