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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import re
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# Import competition metric and CV functions only if not on Kaggle
if not IS_KAGGLE:
    from src.competition_metric import neurips_polymer_metric
    from utils.diagnostics import CVDiagnostics
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
    
    # Molecular weight estimation (for Tg prediction based on Newton's second law)
    # Atomic weights with 0.1 significance as requested
    atomic_weights = {
        'C': 12.0, 'c': 12.0,  # Carbon
        'O': 16.0, 'o': 16.0,  # Oxygen
        'N': 14.0, 'n': 14.0,  # Nitrogen
        'S': 32.1, 's': 32.1,  # Sulfur
        'F': 19.0,             # Fluorine
        'Cl': 35.5,            # Chlorine
        'Br': 79.9,            # Bromine
        'I': 126.9,            # Iodine
        'P': 31.0,             # Phosphorus
        'H': 1.0               # Hydrogen (implicit)
    }
    
    # Calculate molecular weight estimate
    mol_weight = 0.0
    mol_weight += features['num_C'] * atomic_weights['C']
    mol_weight += features['num_c'] * atomic_weights['c']
    mol_weight += features['num_O'] * atomic_weights['O']
    mol_weight += features['num_o'] * atomic_weights['o']
    mol_weight += features['num_N'] * atomic_weights['N']
    mol_weight += features['num_n'] * atomic_weights['n']
    mol_weight += features['num_S'] * atomic_weights['S']
    mol_weight += features['num_s'] * atomic_weights['s']
    mol_weight += features['num_F'] * atomic_weights['F']
    mol_weight += features['num_Cl'] * atomic_weights['Cl']
    mol_weight += features['num_Br'] * atomic_weights['Br']
    mol_weight += features['num_I'] * atomic_weights['I']
    mol_weight += features['num_P'] * atomic_weights['P']
    
    # Round to 0.1 significance
    features['molecular_weight'] = round(mol_weight, 1)
    
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
    
    # Since molecular_weight is commented out, no need to exclude features
    
    # Select features: all non-atom features + target-specific atom features
    selected_features = non_atom_features + TARGET_FEATURES[target]
    return X[selected_features]


def main(cv_only=False, use_supplementary=True, model_type='lightgbm'):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
        use_supplementary: If True, include supplementary datasets in training
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
    """
    print(f"=== Separate {model_type.upper()} Models for Polymer Prediction ===")
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
                                                     enable_diagnostics=True,
                                                     model_type=model_type)
            
            # Also run single seed CV for comparison if needed
            print("\n=== Single Seed Comparison ===")
            single_seed_result = perform_cross_validation(X_train, y_train, cv_folds=5,
                                                         target_columns=target_columns,
                                                         enable_diagnostics=False,
                                                         random_seed=42,
                                                         model_type=model_type)
            
            print(f"\n=== Final Comparison ===")
            print(f"Single seed (42) CV: {single_seed_result['cv_mean']:.4f} (+/- {single_seed_result['cv_std']:.4f})")
            print(f"Multi-seed CV: {multi_seed_result['overall_mean']:.4f} (+/- {multi_seed_result['overall_std']:.4f})")
            
            return {
                'single_seed': single_seed_result,
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
                'objective': 'regression',
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
                'random_state': 42
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
                
                # Train model for this target
                if model_type == 'lightgbm':
                    model = lgb.LGBMRegressor(**lgb_params)
                else:
                    # Use Ridge with target-specific alpha
                    alpha = target_alphas.get(target, 1.0)
                    print(f"  Using alpha={alpha}")
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
    no_supplement = '--no-supplement' in sys.argv or '--no-supp' in sys.argv
    
    main(cv_only=cv_only, use_supplementary=not no_supplement)