#!/usr/bin/env python3
"""
Cross-Validation Functions for NeurIPS Open Polymer Prediction 2025
Extracted from model.py for better readability and organization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# These imports are conditional based on the main model.py imports
from src.competition_metric import neurips_polymer_metric
from utils.diagnostics import CVDiagnostics


# Import the select_features_for_target function from model.py
# This will be done dynamically in the functions that need it


def perform_cross_validation(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, random_seed=42):
    """
    Perform cross-validation for separate models approach
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
        enable_diagnostics: Enable diagnostic tracking
        random_seed: Random seed for reproducibility
    
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
                        
                        val_complete_indices = val_mask_indices[val_feature_complete_mask]
                        
                        if len(val_complete_indices) > 0:
                            X_val_complete = X_val_target[val_feature_complete_mask]
                        
                    if X_val_complete is not None and len(X_val_complete) > 0:
                        X_val_scaled = scaler.transform(X_val_complete)
                        
                        # Train model
                        model = Ridge(alpha=1.0, random_state=42)
                        model.fit(X_target_scaled, y_target_complete)
                        
                        # Predict for validation samples with complete features
                        y_pred_target = model.predict(X_val_scaled)
                        
                        # Track predictions and errors
                        if cv_diagnostics:
                            cv_diagnostics.track_predictions(
                                fold, target, 
                                y_fold_val[target].iloc[val_complete_indices], 
                                y_pred_target
                            )
                        
                        # Store predictions
                        relative_indices = [np.where(val_idx == idx)[0][0] for idx in val_complete_indices if idx in val_idx]
                        fold_predictions[relative_indices, i] = y_pred_target
                        
                        # Calculate and store target-specific scores
                        target_score = mean_squared_error(
                            y_fold_val[target].iloc[val_complete_indices], 
                            y_pred_target, 
                            squared=False
                        )
                        target_fold_scores[target].append(target_score)
                        
                        print(f"  {target}: {len(X_target_complete)} train, {len(X_val_complete)} val samples, RMSE = {target_score:.4f}")
                    else:
                        print(f"  {target}: No valid validation samples with complete features")
                else:
                    print(f"  {target}: No valid training samples with complete features")
            else:
                print(f"  {target}: No valid training samples")
        
        # Calculate fold score using competition metric
        # Create DataFrame with predictions
        fold_pred_df = pd.DataFrame(fold_predictions, columns=target_columns)
        
        # Add fold predictions and true values if diagnostics enabled
        if cv_diagnostics:
            cv_diagnostics.track_fold_results(fold, y_fold_val, fold_pred_df)
        
        # Get true values for validation fold
        y_val_true = y_fold_val[target_columns]
        
        # Calculate competition metric for this fold
        fold_score = neurips_polymer_metric(y_val_true, fold_pred_df)
        fold_scores.append(fold_score)
        
        print(f"\nFold {fold + 1} Score: {fold_score:.4f}")
    
    # Calculate overall CV score
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    # Finalize diagnostics
    if cv_diagnostics:
        cv_diagnostics.generate_report(cv_mean, cv_std)
    
    # Print target-specific scores
    print(f"\n=== Target-Specific CV Results (seed={random_seed}) ===")
    for target in target_columns:
        if target_fold_scores[target]:
            target_mean = np.mean(target_fold_scores[target])
            target_std = np.std(target_fold_scores[target])
            print(f"{target}: {target_mean:.4f} (+/- {target_std:.4f})")
    
    print(f"\n=== Overall CV Results (seed={random_seed}) ===")
    print(f"CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"Fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores,
        'target_scores': target_fold_scores
    }


def perform_multi_seed_cv(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, seeds=None, per_target_analysis=True):
    """
    Perform cross-validation with multiple random seeds for more robust estimates
    
    Args:
        X: Features
        y: Targets
        cv_folds: Number of CV folds
        target_columns: Target column names
        enable_diagnostics: Enable diagnostic tracking
        seeds: List of random seeds to use (default: [42, 123, 456])
        per_target_analysis: Whether to perform per-target analysis
    
    Returns:
        Dictionary with aggregated results
    """
    if seeds is None:
        seeds = [42, 123, 456]  # Default seeds for reproducibility
    
    print(f"\n{'='*60}")
    print(f"MULTI-SEED CROSS-VALIDATION")
    print(f"Seeds: {seeds}")
    print(f"Folds per seed: {cv_folds}")
    print(f"{'='*60}")
    
    all_results = []
    all_fold_scores = []
    target_results = {target: [] for target in (target_columns or y.columns)}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'='*50}")
        
        # Run CV with this seed
        cv_result = perform_cross_validation(
            X, y,
            cv_folds=cv_folds,
            target_columns=target_columns,
            enable_diagnostics=enable_diagnostics and (seed_idx == 0),  # Only first seed
            random_seed=seed
        )
        
        all_results.append(cv_result)
        all_fold_scores.extend(cv_result['fold_scores'])
        
        # Collect target-specific results
        for target, scores in cv_result['target_scores'].items():
            target_results[target].extend(scores)
    
    # Aggregate results across all seeds
    overall_mean = np.mean(all_fold_scores)
    overall_std = np.std(all_fold_scores)
    
    # Per-seed statistics
    seed_means = [r['cv_mean'] for r in all_results]
    seed_stds = [r['cv_std'] for r in all_results]
    
    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"Overall mean: {overall_mean:.4f}")
    print(f"Overall std:  {overall_std:.4f}")
    print(f"Range: [{np.min(all_fold_scores):.4f}, {np.max(all_fold_scores):.4f}]")
    print(f"\nPer-seed means: {[f'{m:.4f}' for m in seed_means]}")
    print(f"Mean of means: {np.mean(seed_means):.4f}")
    print(f"Std of means:  {np.std(seed_means):.4f}")
    
    if per_target_analysis:
        print(f"\n{'='*60}")
        print(f"PER-TARGET ANALYSIS (across all seeds)")
        print(f"{'='*60}")
        for target, scores in target_results.items():
            if scores:
                print(f"{target}:")
                print(f"  Mean RMSE: {np.mean(scores):.4f}")
                print(f"  Std RMSE:  {np.std(scores):.4f}")
                print(f"  Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # Check stability
    if np.std(seed_means) > 0.01:
        print(f"\n⚠️  WARNING: High variance across seeds detected!")
        print(f"   Consider using more seeds or investigating data issues.")
    
    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'seed_results': all_results,
        'all_fold_scores': all_fold_scores,
        'seed_means': seed_means,
        'target_results': target_results
    }