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

# Import configuration
from config import EnvironmentConfig
from residual_analysis import ResidualAnalyzer
config = EnvironmentConfig()

# Dimensionality reduction settings (only one method should be enabled at a time)
# PCA variance threshold - set to None to disable PCA
PCA_VARIANCE_THRESHOLD = 0.99999

# Autoencoder settings - set to True to use autoencoder instead of PCA
USE_AUTOENCODER = False
AUTOENCODER_LATENT_DIM = 32  # Number of latent dimensions
EPOCHS = 20

# PLS settings - set to True to use PLS instead of PCA/autoencoder
USE_PLS = False  # Whether to use PLS for dimensionality reduction
PLS_N_COMPONENTS = 86  # Number of PLS components

# Transformer settings - set to True to add transformer features
USE_TRANSFORMER = True  # Whether to add transformer latent features
TRANSFORMER_LATENT_DIM = 32  # Number of transformer latent dimensions

# Import competition metric and CV functions only if not on Kaggle
if not config.is_kaggle:
    from src.competition_metric import neurips_polymer_metric
    from src.diagnostics import CVDiagnostics
    from cv import perform_cross_validation, perform_multi_seed_cv
    from config import LIGHTGBM_PARAMS


# Already checked above

# Set paths based on environment
if config.is_kaggle:
    # Kaggle competition paths
    TRAIN_PATH = config.data_dir / 'train.csv'
    TEST_PATH = config.data_dir / 'test.csv'
    SUBMISSION_PATH = config.output_dir / 'submission.csv'
    
    # Supplementary dataset paths
    SUPP_PATHS = [
        config.data_dir / 'train_supplement/dataset1.csv',
        config.data_dir / 'train_supplement/dataset2.csv',
        config.data_dir / 'train_supplement/dataset3.csv',
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

# Target columns
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def main(cv_only=False, use_supplementary=True, model_type='lightgbm', run_residual_analysis=False):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
        use_supplementary: If True, include supplementary datasets in training
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
        run_residual_analysis: If True, perform residual analysis for all enabled models
    """
    print(f"=== Separate {model_type.upper()} Models for Polymer Prediction ===")
    
    # Validate dimensionality reduction settings
    dim_reduction_methods = sum([USE_AUTOENCODER, USE_PLS, PCA_VARIANCE_THRESHOLD is not None])
    if dim_reduction_methods > 1:
        raise ValueError("Only one dimensionality reduction method should be enabled at a time")
    
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
        pca_variance_threshold=PCA_VARIANCE_THRESHOLD,
        use_pls=USE_PLS,
        pls_n_components=PLS_N_COMPONENTS,
        y_train=y_train,
        epochs=EPOCHS,
        use_transformer=USE_TRANSFORMER,
        transformer_latent_dim=TRANSFORMER_LATENT_DIM,
        smiles_train=train_df['SMILES'],
        smiles_test=test_df['SMILES']
    )
    
    # Run cross-validation if requested (but not on Kaggle)
    if cv_only:
        if config.is_kaggle:
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
    
    # Initialize models dictionary for residual analysis
    models = {} if run_residual_analysis else None
    smiles_train = train_df['SMILES'] if run_residual_analysis and USE_TRANSFORMER else None
    
    # Model parameters
    if model_type == 'lightgbm':
        # Use parameters from config for single source of truth
        if config.is_kaggle:
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
            # Use mask.values to avoid index alignment issues
            X_target = X_train_preprocessed[mask.values]
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
                    test_size=0.15, random_state=42
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
            
            # Store model for residual analysis
            if run_residual_analysis:
                models[target] = (model, X_target_final, y_target)
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
    
    # Perform residual analysis if requested
    if run_residual_analysis:
        print("\n" + "="*60)
        print("=== RESIDUAL ANALYSIS ===")
        print("="*60)
        
        analyzer = ResidualAnalyzer()
        all_residuals = {}
        
        # Analyze LightGBM/Ridge models
        print(f"\n--- {model_type.upper()} Model Residuals ---")
        for target, (model, X_data, y_true) in models.items():
            if len(y_true) > 0:
                y_pred = model.predict(X_data)
                residuals = y_true.values - y_pred
                all_residuals[f"{model_type}_{target}"] = residuals
                
                print(f"\n{target}:")
                metrics = analyzer.compute_metrics(y_true.values, y_pred, target)
                print(f"  MAE: {metrics.mae:.4f}")
                print(f"  RMSE: {metrics.rmse:.4f}")
                print(f"  R²: {metrics.r2:.4f}")
                print(f"  Mean Residual: {metrics.mean_residual:.4f}")
                print(f"  Std Residual: {metrics.std_residual:.4f}")
        
        # Analyze Transformer residuals if enabled
        if USE_TRANSFORMER:
            print("\n--- Transformer Model Residuals ---")
            try:
                from transformer_model import TransformerModel
                
                # Get transformer from preprocessing (if it was saved)
                # For now, we'll train a fresh one for residual analysis
                if smiles_train is not None:
                    print("Training transformer for residual analysis...")
                    transformer = TransformerModel(
                        vocab_size=None,
                        target_dim=5,
                        latent_dim=TRANSFORMER_LATENT_DIM,
                        num_heads=1,
                        num_encoder_layers=1
                    )
                    
                    # Prepare target matrix
                    y_matrix = y_train[target_columns].values
                    
                    # Fit and get residuals
                    y_pred, residuals = transformer.fit_predict(
                        smiles_train.values, y_matrix,
                        epochs=10,
                        batch_size=32,
                        return_residuals=True
                    )
                    
                    # Analyze each target
                    for i, target in enumerate(target_columns):
                        mask = ~np.isnan(y_matrix[:, i])
                        if mask.sum() > 0:
                            y_true_target = y_matrix[mask, i]
                            y_pred_target = y_pred[mask, i]
                            residuals_target = residuals[mask, i]
                            all_residuals[f"transformer_{target}"] = residuals_target
                            
                            print(f"\n{target}:")
                            metrics = analyzer.compute_metrics(y_true_target, y_pred_target, target)
                            print(f"  MAE: {metrics.mae:.4f}")
                            print(f"  RMSE: {metrics.rmse:.4f}")
                            print(f"  R²: {metrics.r2:.4f}")
                            print(f"  Mean Residual: {metrics.mean_residual:.4f}")
                            print(f"  Std Residual: {metrics.std_residual:.4f}")
            except Exception as e:
                print(f"Error analyzing transformer residuals: {e}")
        
        # Analyze PCA residuals if enabled
        if PCA_VARIANCE_THRESHOLD is not None:
            print("\n--- PCA Reconstruction Residuals ---")
            # PCA was applied during preprocessing, we need to get reconstruction error
            # This would require storing the PCA object during preprocessing
            print("PCA reconstruction analysis: Feature reduction was applied, residuals reflect reduced feature space")
        
        # Generate comparison report
        if len(all_residuals) > 1:
            print("\n" + "="*60)
            print("=== MODEL COMPARISON ===")
            print("="*60)
            comparison = analyzer.compare_models_dict(all_residuals)
            print(comparison)
        
        print("\n" + "="*60)
        print("=== END RESIDUAL ANALYSIS ===")
        print("="*60)
    
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
    residual_analysis = '--residual-analysis' in sys.argv or '--residuals' in sys.argv
    
    # Check for model type
    model_type = 'lightgbm'  # default
    if '--model' in sys.argv:
        model_idx = sys.argv.index('--model')
        if model_idx + 1 < len(sys.argv):
            model_type = sys.argv[model_idx + 1]

    # Check for no dimensionality reduction
    if '--no-dim-reduction' in sys.argv:
        USE_PLS = False
        PCA_VARIANCE_THRESHOLD = None
        USE_AUTOENCODER = False
        print("No dimensionality reduction will be used")
    
    main(cv_only=cv_only, use_supplementary=not no_supplement, model_type=model_type, run_residual_analysis=residual_analysis)
