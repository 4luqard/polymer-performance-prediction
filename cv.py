#!/usr/bin/env python3
"""
Cross-Validation Functions for NeurIPS Open Polymer Prediction 2025
Updated to use LightGBM for better handling of non-linear relationships
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
from datetime import datetime

# These imports are conditional based on the main model.py imports
from src.competition_metric import neurips_polymer_metric
from src.diagnostics import CVDiagnostics
from src.residual_analysis import ResidualAnalysisHook, should_run_analysis

# Import SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature importance analysis will be skipped.")


def calculate_shap_importance(model, X, sample_size=100):
    """
    Calculate SHAP-based feature importance for a trained model
    
    Args:
        model: Trained model (LightGBM or similar)
        X: Feature data (DataFrame or array)
        sample_size: Number of samples to use for SHAP calculation (for efficiency)
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if not SHAP_AVAILABLE:
        return {}
    
    try:
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Sample data if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # Multi-class case
            shap_values = shap_values[0]
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dictionary
        importance = {}
        for i, col in enumerate(X.columns):
            importance[col] = float(mean_abs_shap[i])
        
        return importance
    
    except Exception as e:
        print(f"Warning: Failed to calculate SHAP importance: {e}")
        return {}


def aggregate_feature_importance(fold_importances):
    """
    Aggregate feature importance across multiple folds
    
    Args:
        fold_importances: List of importance dictionaries from each fold
    
    Returns:
        Dictionary with mean importance for each feature
    """
    if not fold_importances:
        return {}
    
    # Get all features
    all_features = set()
    for importance in fold_importances:
        all_features.update(importance.keys())
    
    # Calculate mean importance
    aggregated = {}
    for feature in all_features:
        values = [imp.get(feature, 0) for imp in fold_importances]
        aggregated[feature] = np.mean(values)
    
    return aggregated


def save_feature_importance(feature_importance, output_path):
    """
    Save feature importance to JSON file
    
    Args:
        feature_importance: Dictionary of feature importance
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    importance_json = convert_types(feature_importance)
    
    with open(output_path, 'w') as f:
        json.dump(importance_json, f, indent=2)


def update_features_md(feature_importance, features_path=None):
    """
    Update FEATURES.md with feature importance information
    
    Args:
        feature_importance: Dictionary with target as key and feature importance dict as value
        features_path: Path to FEATURES.md (defaults to project FEATURES.md)
    """
    if features_path is None:
        features_path = Path(__file__).parent / 'FEATURES.md'
    else:
        features_path = Path(features_path)
    
    start_marker = "<!-- FEATURE_IMPORTANCE_START -->"
    end_marker = "<!-- FEATURE_IMPORTANCE_END -->"

    if features_path.exists():
        with open(features_path, 'r') as f:
            content = f.read()
        if start_marker in content and end_marker in content:
            start = content.index(start_marker)
            end = content.index(end_marker) + len(end_marker)
            content = content[:start].rstrip() + "\n\n" + content[end:].lstrip()
        else:
            content = "# Features\n\n"
    else:
        content = "# Features\n\n"

    importance_lines = [start_marker,
                        "## Feature Importance (SHAP-based)",
                        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                        ""]

    for target, importance in feature_importance.items():
        if not importance:
            continue
        importance_lines.append(f"### {target}")

        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        importance_lines.append("| Feature | SHAP Importance |")
        importance_lines.append("|---------|----------------|")
        for feature, score in sorted_features[:20]:
            importance_lines.append(f"| {feature} | {score:.4f} |")
        importance_lines.append("")

    importance_lines.append(end_marker)
    importance_section = "\n".join(importance_lines)

    content = content.rstrip() + "\n\n" + importance_section + "\n"

    with open(features_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {features_path} with feature importance")


def perform_cross_validation(X, y, lgb_params, cv_folds=5, target_columns=None, enable_diagnostics=True, random_seed=42, smiles=None, calculate_feature_importance=True):
    """
    Perform cross-validation for separate models approach
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
        enable_diagnostics: Enable diagnostic tracking
        random_seed: Random seed for reproducibility
        calculate_feature_importance: Whether to calculate SHAP feature importance
    
    Returns:
        Dictionary with CV scores
    """
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
    
    # Initialize feature importance tracking
    target_feature_importance = {target: [] for target in target_columns} if calculate_feature_importance else None
    
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
                
                X_target = X_fold_train.iloc[mask_indices] if hasattr(X_fold_train, 'iloc') else X_fold_train[mask_indices]
                y_target = y_fold_train[target].iloc[mask_indices]
                    
                X_target_final = X_target
                y_target_complete = y_target

                val_mask = ~y_fold_val[target].isna()
                val_mask_indices = np.where(val_mask)[0]

                if len(val_mask_indices) > 0:
                    X_val_final = X_fold_val.iloc[val_mask_indices] if hasattr(X_fold_val, 'iloc') else X_fold_val[val_mask_indices]
                    val_complete_indices = val_mask_indices
                else:
                    X_val_final = None
                    val_complete_indices = np.array([])

                if len(X_target_final) > 0:
                    # Train LightGBM model
                    lgb_params['random_state'] = random_seed  # Override seed for CV
                    model = lgb.LGBMRegressor(**lgb_params)
                    
                    # For LightGBM, create a validation split from training data
                    from sklearn.model_selection import train_test_split
                    if len(X_target_final) > 20:  # Only split if we have enough data
                        X_tr, X_val_inner, y_tr, y_val_inner = train_test_split(
                            X_target_final, y_target_complete, test_size=0.15, random_state=random_seed
                        )
                        model.fit(X_tr, y_tr, eval_set=[(X_val_inner, y_val_inner)], eval_metric='mae', callbacks=[lgb.log_evaluation(0)])
                    else:
                        model.fit(X_target_final, y_target_complete)
                    
                    # Calculate feature importance if requested
                    if calculate_feature_importance:
                        # Create a DataFrame with proper column names for feature importance
                        # X_target_final is already a DataFrame with column names
                        feature_df = X_target_final

                        importance = calculate_shap_importance(model, feature_df)
                        if importance:
                            target_feature_importance[target].append(importance)
                    
                    # Initialize predictions with median for all samples
                    fold_predictions[:, i] = y_fold_train[target].median()
                    
                    # Make predictions only for validation samples with complete features
                    predictions = model.predict(X_val_final)
                    # Map predictions back to original validation indices
                    for idx, pred in zip(val_complete_indices, predictions):
                        fold_predictions[idx, i] = pred

                    # Track target training if diagnostics enabled
                    if cv_diagnostics:
                        # Use X_target_final which is always defined
                        features_used = list(X_target_final.columns) if hasattr(X_target_final, 'columns') else [f'feature_{j}' for j in range(X_target_final.shape[1])]
                        # Pass alpha value for diagnostics (use 1.0 for LightGBM as placeholder)
                        alpha_for_diagnostics = 1.0  # Default value for LightGBM
                        n_train_samples = len(X_target_final) if X_target_final is not None else len(train_mask_complete)
                        n_val_samples = len(X_val_final) if X_val_final is not None else 0
                        cv_diagnostics.track_target_training(
                            fold, target, 
                            n_train_samples, 
                            n_val_samples,
                            features_used, 
                            alpha_for_diagnostics
                        )
                    
                    # Track predictions if diagnostics enabled
                    if cv_diagnostics and len(val_mask_indices) > 0 and X_val_final is not None and len(X_val_final) > 0:
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
                model_name='cv_lightgbm',
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
                    method='lightgbm',
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
    
    # Process feature importance if calculated
    aggregated_importance = {}
    if calculate_feature_importance and target_feature_importance and random_seed == 456:
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
    
    # Finalize diagnostics if enabled
    if cv_diagnostics:
        cv_diagnostics.finalize_session(cv_mean, cv_std, fold_scores)
        report = cv_diagnostics.generate_summary_report()
        print("\n" + report)
    
    result = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores,
        'target_scores': target_fold_scores
    }
    
    # Add feature importance to results if calculated
    if aggregated_importance:
        result['feature_importance'] = aggregated_importance
    
    return result


def perform_multi_seed_cv(X, y, lgb_params, cv_folds=5, target_columns=None, enable_diagnostics=True, seeds=None, per_target_analysis=True, smiles=None):
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
        result = perform_cross_validation(X, y, lgb_params, cv_folds=cv_folds,
                                        target_columns=target_columns, 
                                        enable_diagnostics=enable_diagnostics,
                                        random_seed=seed,
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
