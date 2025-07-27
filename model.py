#!/usr/bin/env python3
"""
Target-Specific Ridge Regression Model for NeurIPS Open Polymer Prediction 2025
Uses feature selection per target to reduce complexity
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import re
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# Import competition metric only if not on Kaggle
if not IS_KAGGLE:
    from src.competition_metric import neurips_polymer_metric
    from utils.diagnostics import CVDiagnostics


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

def extract_molecular_features(smiles):
    """Extract features from SMILES string without external libraries"""
    features = {}
    
    # Basic string features
    features['length'] = len(smiles)
    
    # Count different atoms (case-sensitive for aromatic vs non-aromatic)
    features['num_C'] = len(re.findall(r'C', smiles))
    features['num_c'] = len(re.findall(r'c', smiles))  # aromatic carbon
    features['num_O'] = len(re.findall(r'O', smiles))
    features['num_o'] = len(re.findall(r'o', smiles))  # aromatic oxygen
    features['num_N'] = len(re.findall(r'N', smiles))
    features['num_n'] = len(re.findall(r'n', smiles))  # aromatic nitrogen
    features['num_S'] = len(re.findall(r'S', smiles))
    features['num_s'] = len(re.findall(r's', smiles))  # aromatic sulfur
    features['num_F'] = smiles.count('F')
    features['num_Cl'] = smiles.count('Cl')
    features['num_Br'] = smiles.count('Br')
    features['num_I'] = smiles.count('I')
    features['num_P'] = smiles.count('P')
    
    # Count bonds
    features['num_single_bonds'] = smiles.count('-')
    features['num_double_bonds'] = smiles.count('=')
    features['num_triple_bonds'] = smiles.count('#')
    features['num_aromatic_bonds'] = smiles.count(':')
    
    # Count structural features
    features['num_rings'] = sum(smiles.count(str(i)) for i in range(1, 10))
    features['num_branches'] = smiles.count('(')
    features['num_chiral_centers'] = smiles.count('@')
    
    # Polymer-specific features
    # features['has_polymer_end'] = int('*' in smiles)  # Removed - zero importance
    # features['num_polymer_ends'] = smiles.count('*')  # Removed - may cause overfitting
    
    # Functional group patterns
    features['has_carbonyl'] = int('C(=O)' in smiles or 'C=O' in smiles)
    # features['has_hydroxyl'] = int('OH' in smiles or 'O[H]' in smiles)  # Removed - may cause overfitting
    features['has_ether'] = int('COC' in smiles or 'cOc' in smiles)
    features['has_amine'] = int('N' in smiles)
    features['has_sulfone'] = int('S(=O)(=O)' in smiles)
    features['has_ester'] = int('C(=O)O' in smiles or 'COO' in smiles)
    features['has_amide'] = int('C(=O)N' in smiles or 'CON' in smiles)
    
    # Aromatic features
    features['num_aromatic_atoms'] = features['num_c'] + features['num_n'] + features['num_o'] + features['num_s']
    features['aromatic_ratio'] = features['num_aromatic_atoms'] / max(features['length'], 1)
    
    # Calculate derived features
    features['heavy_atom_count'] = (features['num_C'] + features['num_c'] + 
                                   features['num_O'] + features['num_o'] + 
                                   features['num_N'] + features['num_n'] + 
                                   features['num_S'] + features['num_s'] + 
                                   features['num_F'] + features['num_Cl'] + 
                                   features['num_Br'] + features['num_I'] + 
                                   features['num_P'])
    
    features['heteroatom_count'] = (features['num_O'] + features['num_o'] + 
                                    features['num_N'] + features['num_n'] + 
                                    features['num_S'] + features['num_s'] + 
                                    features['num_F'] + features['num_Cl'] + 
                                    features['num_Br'] + features['num_I'] + 
                                    features['num_P'])
    
    features['heteroatom_ratio'] = features['heteroatom_count'] / max(features['heavy_atom_count'], 1)
    
    # Flexibility indicators
    features['rotatable_bond_estimate'] = max(0, features['num_single_bonds'] - features['num_rings'])
    features['flexibility_score'] = features['rotatable_bond_estimate'] / max(features['heavy_atom_count'], 1)
    
    # Size and complexity
    features['molecular_complexity'] = (features['num_rings'] + features['num_branches'] + 
                                       features['num_chiral_centers'])
    
    # Additional polymer-specific patterns
    features['has_phenyl'] = int('c1ccccc1' in smiles or 'c1ccc' in smiles)
    features['has_cyclohexyl'] = int('C1CCCCC1' in smiles)
    features['has_methyl'] = int(bool(re.search(r'C(?![a-zA-Z])', smiles)))
    features['chain_length_estimate'] = max(len(x) for x in re.split(r'[\(\)]', smiles) if x)
    
    # Additional structural patterns
    features['has_fused_rings'] = int(bool(re.search(r'[0-9].*c.*[0-9]', smiles)))
    features['has_spiro'] = int('@' in smiles and smiles.count('@') > 1)
    features['has_bridge'] = int(bool(re.search(r'[0-9].*[0-9]', smiles)))
    
    return features

# Define target-specific features to use
TARGET_FEATURES = {
    'Tg': ['num_C', 'num_n'],  # Top 2 atom features
    'FFV': ['num_S', 'num_n'], 
    'Tc': ['num_C', 'num_S'],
    'Density': ['num_Cl', 'num_Br'],
    'Rg': ['num_F', 'num_Cl']
}

def prepare_features(df):
    """Convert SMILES to molecular features"""
    print("Extracting molecular features...")
    features_list = []
    
    for idx, smiles in enumerate(df['SMILES']):
        if idx % 1000 == 0:
            print(f"Processing molecule {idx}/{len(df)}...")
        features = extract_molecular_features(smiles)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    return features_df

def select_features_for_target(X, target):
    """Select features for a specific target"""
    # Get all non-atom features
    atom_features = ['num_C', 'num_c', 'num_O', 'num_o', 'num_N', 'num_n', 
                    'num_S', 'num_s', 'num_F', 'num_Cl', 'num_Br', 'num_I', 'num_P']
    non_atom_features = [col for col in X.columns if col not in atom_features]
    
    # Select features: all non-atom features + target-specific atom features
    selected_features = non_atom_features + TARGET_FEATURES[target]
    return X[selected_features]


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
    if IS_KAGGLE:
        print("Cross-validation is not available on Kaggle")
        return None
        
    from sklearn.model_selection import KFold
    
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
                    
                    # Use target-specific alpha
                    target_alphas = {
                        'Tg': 10.0,
                        'FFV': 1.0,
                        'Tc': 10.0,
                        'Density': 5.0,
                        'Rg': 10.0
                    }
                    alpha = target_alphas.get(target, 1.0)
                    
                    # Train Ridge model
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
                        cv_diagnostics.track_target_training(
                            fold, target, 
                            len(X_target_complete), 
                            len(X_val_complete) if len(val_mask_indices) > 0 and X_val_complete is not None else 0,
                            features_used, 
                            alpha
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


def perform_multi_seed_cv(X, y, cv_folds=5, target_columns=None, enable_diagnostics=True, seeds=None, per_target_analysis=True):
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
        result = perform_cross_validation(X, y, cv_folds=cv_folds, 
                                        target_columns=target_columns, 
                                        enable_diagnostics=enable_diagnostics,
                                        random_seed=seed)
        
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


def main(cv_only=False, use_supplementary=True):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
        use_supplementary: If True, include supplementary datasets in training
    """
    print("=== Separate Ridge Models for Polymer Prediction ===")
    print("Loading training data...")
    
    # Load main training data
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Main training data shape: {train_df.shape}")
    
    if use_supplementary:
        # Load and combine supplementary datasets
        print("\nLoading supplementary datasets...")
        all_train_dfs = [train_df]
        
        for supp_path in SUPP_PATHS:
            try:
                supp_df = pd.read_csv(supp_path)
                print(f"Loaded {supp_path}: {supp_df.shape}")
                all_train_dfs.append(supp_df)
            except Exception as e:
                print(f"Could not load {supp_path}: {e}")
        
        # Combine all training data
        train_df = pd.concat(all_train_dfs, ignore_index=True)
        print(f"\nCombined training data shape: {train_df.shape}")
    else:
        print("\n** Using ONLY main training data (no supplementary datasets) **")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test data shape: {test_df.shape}")
    
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
        print(f"{col}: median={y_train[col].median():.4f}, "
              f"missing={y_train[col].isna().sum()} ({y_train[col].isna().sum()/len(y_train)*100:.1f}%)")
    
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
            multi_seed_result = perform_multi_seed_cv(X_train, y_train, cv_folds=5, 
                                                     target_columns=target_columns,
                                                     enable_diagnostics=True)
            
            # Also run single seed CV for comparison if needed
            print("\n=== Single Seed Comparison ===")
            single_seed_result = perform_cross_validation(X_train, y_train, cv_folds=5,
                                                         target_columns=target_columns,
                                                         enable_diagnostics=False,
                                                         random_seed=42)
            
            print(f"\n=== Final Comparison ===")
            print(f"Single seed (42) CV: {single_seed_result['cv_mean']:.4f} (+/- {single_seed_result['cv_std']:.4f})")
            print(f"Multi-seed CV: {multi_seed_result['overall_mean']:.4f} (+/- {multi_seed_result['overall_std']:.4f})")
            
            return {
                'single_seed': single_seed_result,
                'multi_seed': multi_seed_result
            }
    
    # Train separate Ridge regression models for each target
    print("\n=== Training Separate Models for Each Target ===")
    predictions = np.zeros((len(X_test), len(target_columns)))
    
    # Try different alpha values for each target
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
            # Select features for this target
            X_train_selected = select_features_for_target(X_train, target)
            X_test_selected = select_features_for_target(X_test, target)
            
            print(f"  Using {len(X_train_selected.columns)} features (reduced from {len(X_train.columns)})")
            
            # Get samples with valid target values
            X_target = X_train_selected[mask]
            y_target = y_train[target][mask]
            
            # Further filter to only keep rows with no missing features
            feature_complete_mask = ~X_target.isnull().any(axis=1)
            X_target_complete = X_target[feature_complete_mask]
            y_target_complete = y_target[feature_complete_mask]
            
            print(f"  Complete samples (no missing features): {len(X_target_complete)} ({len(X_target_complete)/len(y_train)*100:.1f}%)")
            
            if len(X_target_complete) > 0:
                # Scale features (no imputation needed)
                scaler = StandardScaler()
                X_target_scaled = scaler.fit_transform(X_target_complete)
                
                # For test set, we need to handle missing values somehow
                # Use zero imputation only for test set predictions
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                imputer.fit(X_target_complete)  # Fit on complete training data
                X_test_imputed = imputer.transform(X_test_selected)
                X_test_scaled = scaler.transform(X_test_imputed)
                
                # Use target-specific alpha
                alpha = target_alphas.get(target, 1.0)
                print(f"  Using alpha={alpha}")
                
                # Train Ridge model for this target
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_target_scaled, y_target_complete)
                
                # Make predictions
                predictions[:, i] = model.predict(X_test_scaled)
                print(f"  Predictions: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}")
            else:
                # No complete samples available, use median
                predictions[:, i] = y_train[target].median()
                print(f"  No complete samples available, using median: {predictions[:, i][0]:.4f}")
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
    multiple_cv = '--multiple-cv' in sys.argv
    no_supplement = '--no-supplement' in sys.argv or '--no-supp' in sys.argv
    
    if multiple_cv and not IS_KAGGLE:
        # Run multiple CV runs for robust results
        from utils.cv_runner import run_cv_multiple_times
        
        print("=== Running Multiple CV Runs ===")
        
        # Load and prepare data as in main()
        train_df = pd.read_csv(TRAIN_PATH)
        all_train_dfs = [train_df]
        
        if not no_supplement:
            for supp_path in SUPP_PATHS:
                try:
                    supp_df = pd.read_csv(supp_path)
                    all_train_dfs.append(supp_df)
                except:
                    pass
        
        combined_train = pd.concat(all_train_dfs, ignore_index=True)
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        feature_columns = [col for col in combined_train.columns if col not in ['id'] + target_columns]
        
        X_train = combined_train[feature_columns]
        y_train = combined_train[target_columns]
        X_features = extract_features(X_train['SMILES'])
        
        # Run multiple CV
        n_runs = int(sys.argv[sys.argv.index('--multiple-cv') + 1]) if len(sys.argv) > sys.argv.index('--multiple-cv') + 1 else 5
        run_cv_multiple_times(perform_cross_validation, X_features, y_train, n_runs=n_runs)
    else:
        main(cv_only=cv_only, use_supplementary=not no_supplement)