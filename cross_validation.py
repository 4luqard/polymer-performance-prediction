#!/usr/bin/env python3
"""
Cross-validation script for NeurIPS Open Polymer Prediction 2025
Uses the official competition metric for evaluation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import re
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import competition metric
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.competition_metric import neurips_polymer_metric, display_metric_results
except ImportError:
    # Fallback if running as script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    try:
        from src.competition_metric import neurips_polymer_metric, display_metric_results
    except ImportError:
        # If still can't import, define inline
        MINMAX_DICT = {
            'Tg': [-148.0297376, 472.25],
            'FFV': [0.2269924, 0.77709707],
            'Tc': [0.0465, 0.524],
            'Density': [0.748691234, 1.840998909],
            'Rg': [9.7283551, 34.672905605],
        }
        
        def scaling_error(labels, preds, property):
            """Calculate scaled error for a property"""
            error = np.abs(labels - preds)
            min_val, max_val = MINMAX_DICT[property]
            label_range = max_val - min_val
            return np.mean(error / label_range)
        
        def get_property_weights(labels):
            """Calculate property weights based on inverse square root of valid samples"""
            property_weight = []
            for property in MINMAX_DICT.keys():
                if isinstance(labels, pd.DataFrame):
                    valid_num = np.sum(labels[property] != -9999)
                else:
                    valid_num = np.sum(labels[property] != -9999) if property in labels else 0
                property_weight.append(valid_num)
            property_weight = np.array(property_weight)
            property_weight = np.sqrt(1 / np.maximum(property_weight, 1))
            return (property_weight / np.sum(property_weight)) * len(property_weight)
        
        def neurips_polymer_metric(y_true, y_pred, target_names=None):
            """NeurIPS Open Polymer Prediction 2025 competition metric."""
            if target_names is None:
                target_names = list(MINMAX_DICT.keys())
            
            if isinstance(y_true, np.ndarray):
                y_true = pd.DataFrame(y_true, columns=target_names)
            if isinstance(y_pred, np.ndarray):
                y_pred = pd.DataFrame(y_pred, columns=target_names)
            
            property_maes = []
            property_weights = get_property_weights(y_true)
            individual_scores = {}
            
            for i, property in enumerate(target_names):
                is_labeled = y_true[property].notna()
                
                if is_labeled.sum() > 0:
                    mae = scaling_error(
                        y_true.loc[is_labeled, property].values,
                        y_pred.loc[is_labeled, property].values,
                        property
                    )
                    property_maes.append(mae)
                    individual_scores[property] = mae
                else:
                    individual_scores[property] = np.nan
            
            if len(property_maes) == 0:
                return np.nan, individual_scores
            
            final_score = float(np.average(property_maes, weights=property_weights[:len(property_maes)]))
            return final_score, individual_scores
        
        def display_metric_results(score, individual, name="Competition Metric (wMAE)"):
            """Display metric results in a formatted way."""
            print(f"\n{name} Results:")
            print("=" * 50)
            print(f"Overall Score: {score:.4f}")
            print("\nIndividual Target Scores (scaled MAE):")
            for target, score in individual.items():
                if not np.isnan(score):
                    print(f"  {target}: {score:.4f} ({score*100:.2f}% of range)")
                else:
                    print(f"  {target}: No data")

# Local paths
TRAIN_PATH = 'data/raw/train.csv'
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
    features['has_polymer_end'] = int('*' in smiles)
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

def train_separate_models(X_train, y_train, target_names, use_separate=True):
    """
    Train separate Ridge models for each target or a single MultiOutputRegressor
    
    Args:
        X_train: Training features
        y_train: Training targets (DataFrame)
        target_names: List of target column names
        use_separate: If True, train separate models; if False, use MultiOutputRegressor
    
    Returns:
        Dictionary of models or MultiOutputRegressor
    """
    if not use_separate:
        # Original approach
        model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
        model.fit(X_train, y_train.fillna(y_train.median()))
        return model
    
    # Separate models approach
    models = {}
    target_alphas = {
        'Tg': 10.0,      # Higher regularization for sparse target
        'FFV': 1.0,      # Lower regularization for dense target
        'Tc': 10.0,      # Higher regularization for sparse target
        'Density': 5.0,  # Medium regularization
        'Rg': 10.0       # Higher regularization for sparse target
    }
    
    for target in target_names:
        # Get non-missing samples for this target
        mask = ~y_train[target].isna()
        n_samples = mask.sum()
        
        if n_samples > 0:
            # Train on samples with valid target values
            X_target = X_train[mask]
            y_target = y_train[target][mask]
            
            # Use target-specific alpha
            alpha = target_alphas.get(target, 1.0)
            
            # Train Ridge model for this target
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_target, y_target)
            models[target] = model
        else:
            # No samples available
            models[target] = None
    
    return models

def predict_with_models(models, X_test, y_train, target_names):
    """
    Make predictions using separate models or MultiOutputRegressor
    
    Args:
        models: Dictionary of models or MultiOutputRegressor
        X_test: Test features
        y_train: Training targets (for median fallback)
        target_names: List of target column names
    
    Returns:
        Predictions as numpy array
    """
    if isinstance(models, MultiOutputRegressor):
        # Original approach
        return models.predict(X_test)
    
    # Separate models approach
    predictions = np.zeros((len(X_test), len(target_names)))
    
    for i, target in enumerate(target_names):
        if target in models and models[target] is not None:
            predictions[:, i] = models[target].predict(X_test)
        else:
            # Use median if no model available
            predictions[:, i] = y_train[target].median()
    
    return predictions

def perform_cross_validation(X, y, model_params=None, cv_folds=5, holdout_size=0.2, test_size=0.1, use_separate_models=False):
    """
    Perform cross-validation with train/val/test/holdout splits
    
    Args:
        X: Features
        y: Targets
        model_params: Model parameters
        cv_folds: Number of CV folds
        holdout_size: Fraction to hold out completely (never seen during CV)
        test_size: Fraction for test set (within CV)
        use_separate_models: If True, train separate models for each target
    """
    if model_params is None:
        model_params = {'alpha': 1.0, 'random_state': 42}
    
    print(f"\n=== Advanced Cross-Validation Setup ===")
    print(f"Total samples: {len(X)}")
    
    # Step 1: Create holdout set (completely unseen data)
    from sklearn.model_selection import train_test_split
    X_cv, X_holdout, y_cv, y_holdout = train_test_split(
        X, y, test_size=holdout_size, random_state=42, stratify=None
    )
    print(f"Holdout set: {len(X_holdout)} samples ({holdout_size*100:.0f}%)")
    print(f"CV set: {len(X_cv)} samples ({(1-holdout_size)*100:.0f}%)")
    
    # Step 2: Within CV set, create test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_cv, y_cv, test_size=test_size/(1-holdout_size), random_state=42
    )
    print(f"Test set: {len(X_test)} samples ({test_size*100:.0f}% of total)")
    print(f"Train+Val set: {len(X_trainval)} samples")
    
    # Get target column names
    if isinstance(y, pd.DataFrame):
        target_names = y.columns.tolist()
    else:
        target_names = [f'Target_{i}' for i in range(y.shape[1])]
    
    print(f"\nPerforming {cv_folds}-fold cross-validation on train+val set...")
    print(f"Model parameters: {model_params}")
    
    # Step 3: Perform CV on train+val set
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    cv_individual_scores = {col: [] for col in target_names}
    best_fold_model = None
    best_fold_score = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\nFold {fold + 1}/{cv_folds}...")
        
        # Split data
        X_fold_train = X_trainval[train_idx]
        X_fold_val = X_trainval[val_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        y_fold_val = y_trainval.iloc[val_idx]
        
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Train model(s)
        if use_separate_models:
            fold_model = train_separate_models(X_fold_train, y_fold_train, target_names, use_separate=True)
        else:
            fold_model = MultiOutputRegressor(Ridge(**model_params))
            fold_model.fit(X_fold_train, y_fold_train.fillna(y_fold_train.median()))
        
        # Predict on validation
        y_pred = predict_with_models(fold_model, X_fold_val, y_fold_train, target_names)
        y_pred_df = pd.DataFrame(y_pred, columns=target_names, index=y_fold_val.index)
        
        # Calculate competition metric
        score, individual = neurips_polymer_metric(y_fold_val, y_pred_df, target_names)
        
        if not np.isnan(score):
            cv_scores.append(score)
            print(f"  Fold {fold + 1} validation score: {score:.4f}")
            
            # Track best model
            if score < best_fold_score:
                best_fold_score = score
                best_fold_model = fold_model
            
            for target in target_names:
                if target in individual and not np.isnan(individual[target]):
                    cv_individual_scores[target].append(individual[target])
    
    # Step 4: Evaluate best model on test set
    print(f"\n=== Test Set Evaluation ===")
    y_test_pred = predict_with_models(best_fold_model, X_test, y_trainval, target_names)
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=target_names, index=y_test.index)
    test_score, test_individual = neurips_polymer_metric(y_test, y_test_pred_df, target_names)
    print(f"Test set score: {test_score:.4f}")
    
    # Step 5: Train final model on all train+val+test and evaluate on holdout
    print(f"\n=== Holdout Set Evaluation ===")
    if use_separate_models:
        final_model = train_separate_models(X_cv, y_cv, target_names, use_separate=True)
    else:
        final_model = MultiOutputRegressor(Ridge(**model_params))
        final_model.fit(X_cv, y_cv.fillna(y_cv.median()))
    
    y_holdout_pred = predict_with_models(final_model, X_holdout, y_cv, target_names)
    y_holdout_pred_df = pd.DataFrame(y_holdout_pred, columns=target_names, index=y_holdout.index)
    holdout_score, holdout_individual = neurips_polymer_metric(y_holdout, y_holdout_pred_df, target_names)
    print(f"Holdout set score: {holdout_score:.4f}")
    
    # Calculate summary statistics
    results = {
        'cv_mean_score': np.mean(cv_scores) if cv_scores else np.nan,
        'cv_std_score': np.std(cv_scores) if cv_scores else np.nan,
        'cv_all_scores': cv_scores,
        'test_score': test_score,
        'holdout_score': holdout_score,
        'individual_targets': {},
        'split_sizes': {
            'total': len(X),
            'holdout': len(X_holdout),
            'test': len(X_test),
            'train_val': len(X_trainval),
            'cv_train_avg': len(X_trainval) * (cv_folds-1) / cv_folds,
            'cv_val_avg': len(X_trainval) / cv_folds
        }
    }
    
    # Individual target results
    for target in target_names:
        results['individual_targets'][target] = {
            'cv_mean': np.mean(cv_individual_scores[target]) if cv_individual_scores[target] else np.nan,
            'cv_std': np.std(cv_individual_scores[target]) if cv_individual_scores[target] else np.nan,
            'test': test_individual.get(target, np.nan),
            'holdout': holdout_individual.get(target, np.nan)
        }
    
    return results

def main():
    """Main function to run cross-validation"""
    print("=== Cross-Validation for NeurIPS Polymer Prediction ===")
    print("Loading training data...")
    
    # Load main training data
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Main training data shape: {train_df.shape}")
    
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
    
    # Extract features
    print("\nExtracting features from training data...")
    X_train = prepare_features(train_df)
    
    # Prepare target variables
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    y_train = train_df[target_columns]
    
    # Print feature statistics
    print(f"\nFeature dimensions: {X_train.shape[1]} features")
    print(f"Training samples: {X_train.shape[0]}")
    
    # Handle missing values in features
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Handle missing values in targets
    print("Handling missing target values...")
    y_train_filled = y_train.fillna(y_train.median())
    
    # Print target statistics
    print("\nTarget value statistics:")
    for col in target_columns:
        print(f"{col}: median={y_train[col].median():.4f}, "
              f"missing={y_train[col].isna().sum()} ({y_train[col].isna().sum()/len(y_train)*100:.1f}%)")
    
    # Perform cross-validation with different alpha values
    alpha_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    best_alpha = None
    best_score = float('inf')
    
    print("\n=== Testing different Ridge alpha values ===")
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        cv_results = perform_cross_validation(
            X_train_scaled, y_train_filled, 
            model_params={'alpha': alpha, 'random_state': 42},
            cv_folds=5
        )
        
        mean_score = cv_results['cv_mean_score']
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = alpha
        
        print(f"\nAlpha {alpha} - CV: {mean_score:.4f} (+/- {cv_results['cv_std_score']:.4f}), "
              f"Test: {cv_results['test_score']:.4f}, Holdout: {cv_results['holdout_score']:.4f}")
    
    # Run final cross-validation with best alpha
    print(f"\n=== Final Cross-Validation with best alpha = {best_alpha} ===")
    cv_results = perform_cross_validation(
        X_train_scaled, y_train_filled,
        model_params={'alpha': best_alpha, 'random_state': 42},
        cv_folds=5
    )
    
    # Test separate models approach
    print(f"\n=== Testing Separate Models Approach ===")
    cv_results_separate = perform_cross_validation(
        X_train_scaled, y_train,  # Use original y_train with NaN values
        model_params={'alpha': best_alpha, 'random_state': 42},
        cv_folds=5,
        use_separate_models=True
    )
    
    print(f"\n=== Comparison: MultiOutput vs Separate Models ===")
    print(f"MultiOutput - CV: {cv_results['cv_mean_score']:.4f}, Test: {cv_results['test_score']:.4f}, Holdout: {cv_results['holdout_score']:.4f}")
    print(f"Separate    - CV: {cv_results_separate['cv_mean_score']:.4f}, Test: {cv_results_separate['test_score']:.4f}, Holdout: {cv_results_separate['holdout_score']:.4f}")
    
    # Display results
    print(f"\n=== Final Results ===")
    print(f"Cross-Validation Score: {cv_results['cv_mean_score']:.4f} (+/- {cv_results['cv_std_score']:.4f})")
    print(f"Test Set Score: {cv_results['test_score']:.4f}")
    print(f"Holdout Set Score: {cv_results['holdout_score']:.4f}")
    
    print("\nIndividual Target Scores:")
    print(f"{'Target':<10} {'CV Mean':>10} {'CV Std':>10} {'Test':>10} {'Holdout':>10}")
    print("-" * 54)
    for target, scores in cv_results['individual_targets'].items():
        print(f"{target:<10} {scores['cv_mean']:>10.4f} {scores['cv_std']:>10.4f} "
              f"{scores['test']:>10.4f} {scores['holdout']:>10.4f}")
    
    print(f"\nData Split Sizes:")
    splits = cv_results['split_sizes']
    print(f"  Total samples: {splits['total']}")
    print(f"  Holdout: {splits['holdout']} (20%)")
    print(f"  Test: {splits['test']} (10%)")
    print(f"  Train+Val: {splits['train_val']} (70%)")
    print(f"  CV avg train: {splits['cv_train_avg']:.0f}")
    print(f"  CV avg val: {splits['cv_val_avg']:.0f}")
    
    print(f"\nCV Fold scores: {[f'{s:.4f}' for s in cv_results['cv_all_scores']]}")
    
    # Use holdout score for LB estimation
    print(f"\n💡 For LB estimation, use the holdout score: {cv_results['holdout_score']:.4f}")
    
    print("\n=== Cross-validation complete ===")
    return cv_results

if __name__ == "__main__":
    results = main()