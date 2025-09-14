#!/usr/bin/env python3
"""
Cross-Validation Functions for NeurIPS Open Polymer Prediction 2025
Updated to use LightGBM for better handling of non-linear relationships
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
from datetime import datetime
from rich import print
from sklearn.metrics import mean_absolute_error

# These imports are conditional based on the main model.py imports
from src.competition_metric import neurips_polymer_metric

# Import SHAP for feature importance
import shap

# Feature importance functions
from feature_importance import *

# Import preprocessing function
from data_processing import preprocess_data

def perform_cross_validation(X, y, lgb_params, cv_folds=5, target_columns=None, random_seed=42, smiles=None,
                           use_autoencoder=False, autoencoder_latent_dim=30, epochs=100):
    """
    Perform cross-validation for separate models approach
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
        random_seed: Random seed for reproducibility
        calculate_feature_importance: Whether to calculate SHAP feature importance
    
    Returns:
        Dictionary with CV scores
    """
    print(f"\n=== Cross-Validation ({cv_folds} folds, seed={random_seed}) ===")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    # Store scores for each fold and target
    fold_scores = []
    target_fold_scores = {target: [] for target in target_columns}
    
    # Initialize feature importance tracking
    target_feature_importance = {target: [] for target in target_columns}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{cv_folds}...")
        
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]
        
        # Predictions for this fold
        fold_predictions = np.zeros((len(val_idx), len(target_columns)))
        
        # Train separate models for each target
        for i, target in enumerate(target_columns):
            # =======================================================================
            # Target masking
            # =======================================================================
            # Get non-missing samples for this target
            mask = ~y_fold_train[target].isna()

            # Get samples with valid target values
            mask_indices = np.where(mask)[0]

            X_target = X_fold_train.iloc[mask_indices]
            y_target = y_fold_train[target].iloc[mask_indices]

            val_mask = ~y_fold_val[target].isna()
            val_mask_indices = np.where(val_mask)[0]
            X_val_masked = X_fold_val.iloc[val_mask_indices]

            X_tr, X_val_inner, y_tr, y_val_inner = train_test_split(
                X_target, y_target, test_size=0.15, random_state=random_seed
            )

            # Apply preprocessing on masked samples
            y_tr_df = pd.DataFrame({target: y_tr})
            y_val_df = pd.DataFrame({target: y_val_inner})
            X_tr_final, X_val_final, X_test_final = preprocess_data(
                [X_tr, X_val_inner], X_val_masked,
                use_autoencoder=use_autoencoder,
                autoencoder_latent_dim=autoencoder_latent_dim,
                y_train=[y_tr_df, y_val_df],
                epochs=epochs
            )
            
            val_complete_indices = val_mask_indices
            # =======================================================================
            # Target masking
            # =======================================================================

            # =======================================================================
            # Training + Feature Importance + Predicting
            # =======================================================================
            lgb_params['random_state'] = random_seed  # Override seed for CV
            model = lgb.LGBMRegressor(**lgb_params)
                    
            model.fit(X_tr_final, y_tr,
                      eval_set=[(X_val_final, y_val_inner)],
                      eval_metric='mae',
                      callbacks=[lgb.log_evaluation(0)])

            # Create a DataFrame with proper column names for feature importance
            # X_target_final is already a DataFrame with column names
            feature_df = X_tr_final

            importance = calculate_shap_importance(model, feature_df)
            if importance:
                target_feature_importance[target].append(importance)
                    
            # Initialize predictions with median for all samples
            fold_predictions[:, i] = y_fold_train[target].median()
                    
            # Make predictions only for validation samples with complete features
            predictions = model.predict(X_test_final)
            # Map predictions back to original validation indices
            for idx, pred in zip(val_complete_indices, predictions):
                fold_predictions[idx, i] = pred
            # =======================================================================
            # Training + Feature Importance + Predicting
            # =======================================================================

    # =======================================================================
    # CV Statistics
    # =======================================================================
        # Calculate fold score using competition metric
        # Create predictions DataFrame with proper column names and index
        fold_pred_df = pd.DataFrame(fold_predictions, columns=target_columns, index=y_fold_val.index)
        
        # Calculate competition metric
        fold_score, individual_scores = neurips_polymer_metric(y_fold_val, fold_pred_df, target_columns)
        
        if not np.isnan(fold_score):
            fold_scores.append(fold_score)
            print(f"  Fold {fold + 1} competition score: {fold_score:.4f}")
            
            # Track individual target scores
            if individual_scores:
                print(f"    Individual target scores:")
                for target, score in individual_scores.items():
                    if not np.isnan(score):
                        target_fold_scores[target].append(score)
                        print(f"      {target}: {score:.4f}")

    # -----------------------------------------------------------------------

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\nCross-Validation Score (Competition Metric): {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # Calculate per-target statistics
    print("\n=== Per-Target CV Results ===")
    for target, scores in target_fold_scores.items():
        if scores:
            target_mean = np.mean(scores)
            target_std = np.std(scores)
            print(f"{target}: {target_mean:.4f} (+/- {target_std:.4f})")

    result = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores,
        'target_scores': target_fold_scores
    }
    # =======================================================================
    # CV Statistics
    # =======================================================================

    # =======================================================================
    # Process feature importance if calculated
    # =======================================================================
    aggregated_importance = {}
    if random_seed == 456:
        print("\n=== Processing Feature Importance ===")
        for target, fold_importances in target_feature_importance.items():
            if fold_importances:
                # Aggregate across folds
                aggregated = aggregate_feature_importance(fold_importances)
                aggregated_importance[target] = aggregated

                # Show top features for this target
                if aggregated:
                    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:10]
                    print(f"\nTop 10 features for {target}:")
                    for feat, imp in sorted_features:
                        print(f"  {feat}: {imp:.4f}")

        # Save feature importance to JSON
        # save_feature_importance(aggregated_importance, 'feature_importance.json')

        # Update FEATURES.md
        update_features_md(aggregated_importance)

    # Add feature importance to results if calculated
    if aggregated_importance:
        result['feature_importance'] = aggregated_importance
    # =======================================================================
    # Process feature importance if calculated
    # =======================================================================

    return result


def perform_multi_seed_cv(X, y, lgb_params, cv_folds=5, target_columns=None, smiles=None,
                          use_autoencoder=False, autoencoder_latent_dim=30, epochs=100, seeds=[42, 123, 456]):
    """
    Perform cross-validation with multiple random seeds for more robust results
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names

    Returns:
        Dictionary with aggregated CV scores across all seeds
    """
    print(f"\n=== Multi-Seed Cross-Validation ({len(seeds)} seeds, {cv_folds} folds each) ===")
    print(f"Seeds: {seeds}")

    # ======================================================================
    # Individual CV
    # ======================================================================
    all_scores = []
    seed_results = {}
    
    # Initialize per-target tracking
    target_scores = {target: [] for target in (target_columns or y.columns.tolist())}
    
    for seed in seeds:
        print(f"\n--- Running CV with seed {seed} ---")
        result = perform_cross_validation(X, y, lgb_params, cv_folds=cv_folds,
                                        target_columns=target_columns, 
                                        random_seed=seed,
                                        smiles=smiles,
                                        use_autoencoder=use_autoencoder,
                                        autoencoder_latent_dim=autoencoder_latent_dim,
                                        epochs=epochs)
        
        if result is not None:
            seed_results[seed] = result
            all_scores.extend(result['fold_scores'])
            
            # Track per-target scores if available
            if 'target_scores' in result:
                for target, scores in result['target_scores'].items():
                    target_scores[target].extend(scores)
    # =======================================================================
    # Individual CV
    # =======================================================================

    # =======================================================================
    # Calculate overall statistics
    # =======================================================================
    overall_mean = np.mean(all_scores)
    overall_std = np.std(all_scores)
    
    print(f"\n=== Multi-Seed CV Summary ===")
    for seed, result in seed_results.items():
        print(f"Seed {seed}: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
    
    print(f"\nOverall: {overall_mean:.4f} (+/- {overall_std:.4f})")
    print(f"Total folds evaluated: {len(all_scores)}")
    
    # Per-target analysis
    print(f"\n=== Per-Target Analysis ===")
    for target, scores in target_scores.items():
        if scores:
            target_mean = np.mean(scores)
            target_std = np.std(scores)
            print(f"{target}: {target_mean:.4f} (+/- {target_std:.4f})")
    # ======================================================================
    # Calculate overall statistics
    # ========================================================================

    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'seed_results': seed_results,
        'all_scores': all_scores,
        'target_scores': target_scores
    }
