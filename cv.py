#!/usr/bin/env python3
"""
Cross-Validation Functions for NeurIPS Open Polymer Prediction 2025
Updated to use LightGBM for better handling of non-linear relationships
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# These imports are conditional based on the main model.py imports
from src.competition_metric import neurips_polymer_metric
from utils.diagnostics import CVDiagnostics


def perform_cross_validation(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, random_seed=42, model_type='lightgbm'):
    """
    Perform cross-validation for separate models approach
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
        enable_diagnostics: Enable diagnostic tracking
        random_seed: Random seed for reproducibility
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
    
    Returns:
        Dictionary with CV scores
    """
    # Import select_features_for_target from model
    from model import select_features_for_target
    
    if target_columns is None:
        target_columns = y.columns.tolist()
    
    print(f"\n=== Cross-Validation ({cv_folds} folds, seed={random_seed}) ===")
    
    # Initialize diagnostics if enabled
    cv_diagnostics = None
    if enable_diagnostics:
        cv_diagnostics = CVDiagnostics()
        # Track initial feature statistics
        cv_diagnostics.track_feature_statistics(X)
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    
    # Store scores for each fold and target
    fold_scores = []
    target_fold_scores = {target: [] for target in target_columns}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{cv_folds}...")
        
        X_fold_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_fold_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]
        
        # Track data split if diagnostics enabled
        if cv_diagnostics:
            cv_diagnostics.track_data_split(fold, train_idx, val_idx, y_fold_train, y_fold_val)
        
        # Predictions for this fold
        fold_predictions = np.zeros((len(val_idx), len(target_columns)))
        
        # Train separate models for each target
        for i, target in enumerate(target_columns):
            # Get non-missing samples for this target
            mask = ~y_fold_train[target].isna()
            
            if mask.sum() > 0:
                # Get samples with valid target values
                mask_indices = np.where(mask)[0]
                # Select features for this target
                X_fold_train_selected = select_features_for_target(X_fold_train, target)
                X_fold_val_selected = select_features_for_target(X_fold_val, target)
                
                X_target = X_fold_train_selected.iloc[mask_indices] if hasattr(X_fold_train_selected, 'iloc') else X_fold_train_selected[mask_indices]
                y_target = y_fold_train[target].iloc[mask_indices]
                
                # Further filter to only keep rows with no missing features
                if isinstance(X_target, pd.DataFrame):
                    feature_complete_mask = ~X_target.isnull().any(axis=1)
                else:
                    # For numpy arrays
                    feature_complete_mask = ~np.isnan(X_target).any(axis=1)
                
                X_target_complete = X_target[feature_complete_mask]
                y_target_complete = y_target[feature_complete_mask]
                
                if len(X_target_complete) > 0:
                    # Scale features (no imputation needed)
                    scaler = StandardScaler()
                    X_target_scaled = scaler.fit_transform(X_target_complete)
                    
                    # For validation set, apply same filtering as training
                    # Only evaluate on samples with non-missing target values
                    val_mask = ~y_fold_val[target].isna()
                    val_mask_indices = np.where(val_mask)[0]
                    X_val_complete = None
                    val_complete_indices = np.array([])
                    
                    if len(val_mask_indices) > 0:
                        X_val_target = X_fold_val_selected.iloc[val_mask_indices] if hasattr(X_fold_val_selected, 'iloc') else X_fold_val_selected[val_mask_indices]
                        
                        # Further filter to only keep rows with no missing features
                        if isinstance(X_val_target, pd.DataFrame):
                            val_feature_complete_mask = ~X_val_target.isnull().any(axis=1)
                        else:
                            val_feature_complete_mask = ~np.isnan(X_val_target).any(axis=1)
                        
                        X_val_complete = X_val_target[val_feature_complete_mask]
                        val_complete_indices = val_mask_indices[val_feature_complete_mask]
                        
                        if len(X_val_complete) > 0:
                            # Scale validation features
                            X_val_scaled = scaler.transform(X_val_complete)
                    
                    # Train model based on type
                    if model_type == 'lightgbm':
                        # LightGBM parameters as requested
                        lgb_params = {
                            'objective': 'regression',
                            'metric': 'mae',  # Changed from rmse to mae for competition alignment
                            'boosting_type': 'gbdt',
                            'max_depth': -1,  # No limit
                            'num_leaves': 31,
                            'n_estimators': 200,
                            'learning_rate': 0.1,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'verbose': -1,
                            'random_state': random_seed
                        }
                        model = lgb.LGBMRegressor(**lgb_params)
                    else:
                        # Ridge model with target-specific alpha
                        target_alphas = {
                            'Tg': 10.0,
                            'FFV': 1.0,
                            'Tc': 10.0,
                            'Density': 5.0,
                            'Rg': 10.0
                        }
                        alpha = target_alphas.get(target, 1.0)
                        model = Ridge(alpha=alpha, random_state=random_seed)
                    
                    model.fit(X_target_scaled, y_target_complete)
                    
                    # Initialize predictions with median for all samples
                    fold_predictions[:, i] = y_fold_train[target].median()
                    
                    # Make predictions only for validation samples with complete features
                    if len(val_mask_indices) > 0 and X_val_complete is not None and len(X_val_complete) > 0:
                        predictions = model.predict(X_val_scaled)
                        # Map predictions back to original validation indices
                        for idx, pred in zip(val_complete_indices, predictions):
                            fold_predictions[idx, i] = pred
                    
                    # Track target training if diagnostics enabled
                    if cv_diagnostics:
                        features_used = list(X_fold_train_selected.columns) if hasattr(X_fold_train_selected, 'columns') else [f'feature_{j}' for j in range(X_fold_train_selected.shape[1])]
                        # Pass alpha value for diagnostics (use 1.0 for LightGBM as placeholder)
                        alpha_for_diagnostics = alpha if model_type == 'ridge' else 1.0
                        cv_diagnostics.track_target_training(
                            fold, target, 
                            len(X_target_complete), 
                            len(X_val_complete) if len(val_mask_indices) > 0 and X_val_complete is not None else 0,
                            features_used, 
                            alpha_for_diagnostics
                        )
                    
                    # Track predictions if diagnostics enabled
                    if cv_diagnostics and len(val_mask_indices) > 0 and X_val_complete is not None and len(X_val_complete) > 0:
                        cv_diagnostics.track_predictions(
                            fold, target,
                            val_idx[val_complete_indices],
                            fold_predictions[val_complete_indices, i],
                            y_fold_val[target].iloc[val_complete_indices].values
                        )
                else:
                    # No complete samples, use median
                    fold_predictions[:, i] = y_fold_train[target].median()
            else:
                # No samples available, use median
                fold_predictions[:, i] = y_fold_train[target].median()
        
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
            
            # Track fold score if diagnostics enabled
            if cv_diagnostics:
                cv_diagnostics.track_fold_score(fold, fold_score, individual_scores)
    
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
    
    # Finalize diagnostics if enabled
    if cv_diagnostics:
        cv_diagnostics.finalize_session(cv_mean, cv_std, fold_scores)
        report = cv_diagnostics.generate_summary_report()
        print("\n" + report)
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores,
        'target_scores': target_fold_scores
    }


def perform_multi_seed_cv(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, seeds=None, per_target_analysis=True, model_type='lightgbm'):
    """
    Perform cross-validation with multiple random seeds for more robust results
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
        enable_diagnostics: Enable diagnostic tracking
        seeds: List of random seeds to use (default: [42, 123, 456])
        per_target_analysis: Whether to perform per-target analysis
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
    
    Returns:
        Dictionary with aggregated CV scores across all seeds
    """
    if seeds is None:
        # Use 3 reproducible seeds as requested
        seeds = [42, 123, 456]
    
    print(f"\n=== Multi-Seed Cross-Validation ({len(seeds)} seeds, {cv_folds} folds each) ===")
    print(f"Seeds: {seeds}")
    
    all_scores = []
    seed_results = {}
    
    # Initialize per-target tracking
    target_scores = {target: [] for target in (target_columns or y.columns.tolist())}
    
    for seed in seeds:
        print(f"\n--- Running CV with seed {seed} ---")
        result = perform_cross_validation(X, y, cv_folds=cv_folds, 
                                        target_columns=target_columns, 
                                        enable_diagnostics=enable_diagnostics,
                                        random_seed=seed,
                                        model_type=model_type)
        
        if result is not None:
            seed_results[seed] = result
            all_scores.extend(result['fold_scores'])
            
            # Track per-target scores if available
            if 'target_scores' in result and per_target_analysis:
                for target, scores in result['target_scores'].items():
                    target_scores[target].extend(scores)
    
    # Calculate overall statistics
    overall_mean = np.mean(all_scores)
    overall_std = np.std(all_scores)
    
    print(f"\n=== Multi-Seed CV Summary ===")
    for seed, result in seed_results.items():
        print(f"Seed {seed}: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
    
    print(f"\nOverall: {overall_mean:.4f} (+/- {overall_std:.4f})")
    print(f"Total folds evaluated: {len(all_scores)}")
    
    # Per-target analysis
    if per_target_analysis and any(target_scores.values()):
        print(f"\n=== Per-Target Analysis ===")
        for target, scores in target_scores.items():
            if scores:
                target_mean = np.mean(scores)
                target_std = np.std(scores)
                print(f"{target}: {target_mean:.4f} (+/- {target_std:.4f})")
    
    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'seed_results': seed_results,
        'all_scores': all_scores,
        'target_scores': target_scores if per_target_analysis else None
    }