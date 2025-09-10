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
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# These imports are conditional based on the main model.py imports
from src.competition_metric import neurips_polymer_metric
from src.diagnostics import CVDiagnostics
from config import LIGHTGBM_PARAMS
from src.residual_analysis import ResidualAnalysisHook, should_run_analysis

# PCA variance threshold - should match the one in model.py
PCA_VARIANCE_THRESHOLD = None


def create_newsim_stratified_splits(X, y, cv_folds=5, random_seed=42):
    """
    Create custom CV splits where validation sets only contain samples with new_sim == 1
    
    Args:
        X: Features DataFrame (must contain 'new_sim' column)
        y: Targets DataFrame
        cv_folds: Number of cross-validation folds
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (train_idx, val_idx) tuples
    """
    np.random.seed(random_seed)
    
    # Check if new_sim column exists
    if 'new_sim' not in X.columns:
        raise ValueError("X must contain 'new_sim' column for stratified splitting")
    
    # Get indices for new_sim samples
    newsim_mask = X['new_sim'] == 1
    newsim_indices = np.where(newsim_mask)[0]
    non_newsim_indices = np.where(~newsim_mask)[0]
    
    # Shuffle new_sim indices for random fold assignment
    newsim_indices_shuffled = newsim_indices.copy()
    np.random.shuffle(newsim_indices_shuffled)
    
    # Calculate fold sizes for new_sim samples
    n_newsim = len(newsim_indices_shuffled)
    fold_sizes = np.full(cv_folds, n_newsim // cv_folds)
    fold_sizes[:n_newsim % cv_folds] += 1
    
    # Create folds
    splits = []
    current_idx = 0
    
    for fold in range(cv_folds):
        # Validation indices (only from new_sim == 1)
        val_start = current_idx
        val_end = current_idx + fold_sizes[fold]
        val_idx = newsim_indices_shuffled[val_start:val_end]
        
        # Training indices (all non-new_sim + remaining new_sim)
        train_newsim = np.concatenate([
            newsim_indices_shuffled[:val_start],
            newsim_indices_shuffled[val_end:]
        ])
        train_idx = np.concatenate([non_newsim_indices, train_newsim])
        
        # Shuffle training indices
        np.random.shuffle(train_idx)
        
        splits.append((train_idx, val_idx))
        current_idx = val_end
    
    return splits


def perform_cross_validation(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, random_seed=42, model_type='lightgbm', preprocessed=True, smiles=None):
    """
    Perform cross-validation for separate models approach
    
    Args:
        X: Features (numpy array or DataFrame) - should be preprocessed if preprocessed=True
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
        enable_diagnostics: Enable diagnostic tracking
        random_seed: Random seed for reproducibility
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
        preprocessed: Whether data is already preprocessed (default: True)
    
    Returns:
        Dictionary with CV scores
    """
    # Import select_features_for_target from model only if not preprocessed
    if not preprocessed:
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
    
    # Use custom splitting if new_sim column is present
    if hasattr(X, 'columns') and 'new_sim' in X.columns:
        print("Using new_sim stratified splitting (validation only from new_sim == 1)")
        splits = create_newsim_stratified_splits(X, y, cv_folds=cv_folds, random_seed=random_seed)
    else:
        # Fallback to regular KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        splits = list(kf.split(X))
    
    # Store scores for each fold and target
    fold_scores = []
    target_fold_scores = {target: [] for target in target_columns}
    
    for fold, (train_idx, val_idx) in enumerate(splits):
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
                
                if preprocessed:
                    # Use preprocessed data directly
                    X_target = X_fold_train.iloc[mask_indices] if hasattr(X_fold_train, 'iloc') else X_fold_train[mask_indices]
                    y_target = y_fold_train[target].iloc[mask_indices]
                    
                    # Data is already complete and preprocessed
                    X_target_final = X_target
                    y_target_complete = y_target
                else:
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
                        
                        # Apply PCA if enabled
                        pca = None
                        if PCA_VARIANCE_THRESHOLD is not None:
                            pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=random_seed)
                            X_target_pca = pca.fit_transform(X_target_scaled)
                            X_target_final = X_target_pca
                        else:
                            X_target_final = X_target_scaled
                    
                if preprocessed:
                    # For preprocessed data, just filter by target availability
                    val_mask = ~y_fold_val[target].isna()
                    val_mask_indices = np.where(val_mask)[0]
                    
                    if len(val_mask_indices) > 0:
                        X_val_final = X_fold_val.iloc[val_mask_indices] if hasattr(X_fold_val, 'iloc') else X_fold_val[val_mask_indices]
                        val_complete_indices = val_mask_indices
                    else:
                        X_val_final = None
                        val_complete_indices = np.array([])
                    X_val_complete = None  # Initialize for non-preprocessed case
                else:
                    # For non-preprocessed data, apply the same preprocessing as training
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
                            
                            # Apply PCA to validation set if enabled
                            if pca is not None:
                                X_val_final = pca.transform(X_val_scaled)
                            else:
                                X_val_final = X_val_scaled
                
                if len(X_target_final) > 0:
                    
                    # Train model based on type
                    if model_type == 'lightgbm':
                        # Use parameters from config for single source of truth
                        lgb_params = LIGHTGBM_PARAMS.copy()
                        lgb_params['random_state'] = random_seed  # Override seed for CV
                        model = lgb.LGBMRegressor(**lgb_params)
                        
                        # For LightGBM, create a validation split from training data
                        from sklearn.model_selection import train_test_split
                        if len(X_target_final) > 20:  # Only split if we have enough data
                            if target in ['FFV', 'Tc']:
                                #y_target_complete = np.log1p(y_target_complete)
                                y_target_complete = 1 / (1 + np.exp(-y_target_complete))
                            X_tr, X_val_inner, y_tr, y_val_inner = train_test_split(
                                X_target_final, y_target_complete, test_size=0.15, random_state=random_seed
                            )
                            model.fit(X_tr, y_tr, eval_set=[(X_val_inner, y_val_inner)], eval_metric='mae', callbacks=[lgb.log_evaluation(0)])
                        else:
                            model.fit(X_target_final, y_target_complete)
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
                        model.fit(X_target_final, y_target_complete)
                    
                    # Initialize predictions with median for all samples
                    fold_predictions[:, i] = y_fold_train[target].median()
                    
                    # Make predictions only for validation samples with complete features
                    if preprocessed and len(val_mask_indices) > 0 and X_val_final is not None:
                        predictions = model.predict(X_val_final)
                        if target in ['FFV', 'Tc']:
                            #predictions = np.expm1(predictions)
                            predictions = -np.log((1 - predictions) / predictions)
                        # Map predictions back to original validation indices
                        for idx, pred in zip(val_complete_indices, predictions):
                            fold_predictions[idx, i] = pred
                    elif not preprocessed and len(val_mask_indices) > 0 and X_val_complete is not None and len(X_val_complete) > 0:
                        predictions = model.predict(X_val_final)
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
        
        # Run residual analysis if enabled
        if should_run_analysis():
            residual_hook = ResidualAnalysisHook()
            
            # Original residual analysis (for compatibility)
            residual_hook.analyze_predictions(
                y_true=y_fold_val,
                y_pred=fold_pred_df,
                model_name=f'cv_{model_type}',
                fold=fold
            )
            
            # Generate dataframes if SMILES are available
            if smiles is not None:
                # Get train and val SMILES
                train_smiles = smiles.iloc[train_idx] if hasattr(smiles, 'iloc') else [smiles[i] for i in train_idx]
                val_smiles = smiles.iloc[val_idx] if hasattr(smiles, 'iloc') else [smiles[i] for i in val_idx]
                
                # Get train predictions (using the trained models)
                y_pred_train = pd.DataFrame(index=train_idx, columns=target_columns)
                for i, target in enumerate(target_columns):
                    # Use median as prediction for simplicity in train set
                    # Handle cases where all values are NaN
                    median_val = y_fold_train[target].median()
                    if pd.isna(median_val):
                        # If all values are NaN, use 0 as default
                        y_pred_train[target] = 0
                    else:
                        y_pred_train[target] = median_val
                
                # Generate and save dataframes
                residual_hook.generate_residuals_dataframe(
                    X_train=X_fold_train.values if hasattr(X_fold_train, 'values') else X_fold_train,
                    X_val=X_fold_val.values if hasattr(X_fold_val, 'values') else X_fold_val,
                    y_train=y_fold_train,
                    y_val=y_fold_val,
                    y_pred_train=y_pred_train,
                    y_pred_val=fold_pred_df,
                    train_smiles=train_smiles.tolist() if hasattr(train_smiles, 'tolist') else train_smiles,
                    val_smiles=val_smiles.tolist() if hasattr(val_smiles, 'tolist') else val_smiles,
                    method=model_type,
                    is_cv=True,
                    fold=fold,
                    seed=random_seed
                )
        
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


def perform_multi_seed_cv(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, seeds=None, per_target_analysis=True, model_type='lightgbm', preprocessed=True, smiles=None):
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
                                        model_type=model_type,
                                        preprocessed=preprocessed,
                                        smiles=smiles)
        
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
