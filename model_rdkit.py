#!/usr/bin/env python3
"""
Ridge Regression Model with RDKit Feature Extraction for NeurIPS Open Polymer Prediction 2025
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available, falling back to manual feature extraction")
    import re

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# Import competition metric only if not on Kaggle
if not IS_KAGGLE:
    from src.competition_metric import neurips_polymer_metric

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

def extract_molecular_features_rdkit(smiles):
    """Extract features from SMILES string using RDKit"""
    features = {}
    
    # Handle polymer notation - replace * with dummy atom
    polymer_smiles = smiles.replace('*', '[At]')
    
    try:
        mol = Chem.MolFromSmiles(polymer_smiles)
        if mol is None:
            raise ValueError("Failed to parse SMILES")
            
        # Basic molecular descriptors
        features['MolWt'] = Descriptors.MolWt(mol)
        features['MolLogP'] = Descriptors.MolLogP(mol)
        features['MolMR'] = Descriptors.MolMR(mol)
        features['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
        
        # Atom counts
        features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        features['NumHDonors'] = Descriptors.NumHDonors(mol)
        features['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        features['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
        features['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
        features['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
        features['RingCount'] = Descriptors.RingCount(mol)
        
        # Complexity measures
        features['BertzCT'] = Descriptors.BertzCT(mol)
        features['Chi0'] = Descriptors.Chi0(mol)
        features['Chi1'] = Descriptors.Chi1(mol)
        features['Chi0n'] = Descriptors.Chi0n(mol)
        features['Chi1n'] = Descriptors.Chi1n(mol)
        features['Chi0v'] = Descriptors.Chi0v(mol)
        features['Chi1v'] = Descriptors.Chi1v(mol)
        
        # Surface area and volume
        features['TPSA'] = Descriptors.TPSA(mol)
        features['LabuteASA'] = Descriptors.LabuteASA(mol)
        
        # Crippen descriptors - using MolLogP and MolMR instead
        # features['CrippenLogP'] = Descriptors.MolLogP(mol)  # Already added above
        # features['CrippenMR'] = Descriptors.MolMR(mol)  # Already added above
        
        # Electronic features - these often return NaN for complex molecules
        # features['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
        # features['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
        # features['MaxAbsPartialCharge'] = Descriptors.MaxAbsPartialCharge(mol)
        # features['MinAbsPartialCharge'] = Descriptors.MinAbsPartialCharge(mol)
        
        # Fragment-based features
        # features['FractionCsp3'] = Descriptors.FractionCsp3(mol)  # Not available in all RDKit versions
        features['NumSpiroAtoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        features['NumBridgeheadAtoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        
        # Polymer-specific features
        features['has_polymer_end'] = int('[At]' in polymer_smiles)
        features['num_polymer_ends'] = polymer_smiles.count('[At]')
        
        # Functional group counts
        features['num_carbonyl'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]')))
        features['num_ester'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6](=[#8])-[#8]')))
        features['num_ether'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]-[#8]-[#6]')))
        features['num_amine'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]')))
        features['num_amide'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6](=[#8])-[#7]')))
        features['num_sulfone'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#16](=[#8])(=[#8])')))
        
        # Aromatic features
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features['num_aromatic_atoms'] = aromatic_atoms
        features['aromatic_ratio'] = aromatic_atoms / max(mol.GetNumAtoms(), 1)
        
        # Chain flexibility
        features['flexibility_score'] = features['NumRotatableBonds'] / max(features['HeavyAtomCount'], 1)
        
        # Size features
        features['NumAtoms'] = mol.GetNumAtoms()
        features['NumBonds'] = mol.GetNumBonds()
        
        # Additional polymer patterns
        features['has_phenyl'] = int(len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1ccccc1'))) > 0)
        features['has_cyclohexyl'] = int(len(mol.GetSubstructMatches(Chem.MolFromSmarts('C1CCCCC1'))) > 0)
        
    except Exception as e:
        # If RDKit fails, return NaN features
        # Silently handle exceptions - most likely due to complex polymer structures
        pass
        # Return dict with NaN values for all expected features
        feature_names = ['MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors', 
                        'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumAromaticRings',
                        'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles',
                        'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'RingCount',
                        'BertzCT', 'Chi0', 'Chi1', 'Chi0n', 'Chi1n', 'Chi0v', 'Chi1v',
                        'TPSA', 'LabuteASA',
                        'NumSpiroAtoms', 'NumBridgeheadAtoms', 'has_polymer_end',
                        'num_polymer_ends', 'num_carbonyl', 'num_ester', 'num_ether', 'num_amine',
                        'num_amide', 'num_sulfone', 'num_aromatic_atoms', 'aromatic_ratio',
                        'flexibility_score', 'NumAtoms', 'NumBonds', 'has_phenyl', 'has_cyclohexyl']
        features = {name: np.nan for name in feature_names}
    
    return features

def extract_molecular_features_manual(smiles):
    """Extract features from SMILES string without external libraries (fallback)"""
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
    features['num_polymer_ends'] = smiles.count('*')
    
    # Functional group patterns
    features['has_carbonyl'] = int('C(=O)' in smiles or 'C=O' in smiles)
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
    failed_count = 0
    
    for idx, smiles in enumerate(df['SMILES']):
        if idx % 1000 == 0:
            print(f"Processing molecule {idx}/{len(df)}...")
        
        if RDKIT_AVAILABLE:
            features = extract_molecular_features_rdkit(smiles)
            # Check if parsing failed (MolWt is NaN - indicates parsing failure)
            if pd.isna(features.get('MolWt', np.nan)):
                failed_count += 1
        else:
            features = extract_molecular_features_manual(smiles)
            
        features_list.append(features)
    
    if RDKIT_AVAILABLE and failed_count > 0:
        print(f"Warning: RDKit failed to parse {failed_count}/{len(df)} molecules ({failed_count/len(df)*100:.1f}%)")
    
    features_df = pd.DataFrame(features_list)
    return features_df

def perform_cross_validation(X, y, cv_folds=5, target_columns=None):
    """
    Perform cross-validation for separate models approach
    
    Args:
        X: Features (numpy array or DataFrame)
        y: Targets (DataFrame with multiple columns)
        cv_folds: Number of cross-validation folds
        target_columns: List of target column names
    
    Returns:
        Dictionary with CV scores
    """
    if IS_KAGGLE:
        print("Cross-validation is not available on Kaggle")
        return None
        
    if target_columns is None:
        target_columns = y.columns.tolist()
    
    print(f"\n=== Cross-Validation ({cv_folds} folds) ===")
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Store scores for each fold and target
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{cv_folds}...")
        
        X_fold_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_fold_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]
        
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
                    
                    # For validation set, we need to handle missing values
                    imputer = SimpleImputer(strategy='mean')
                    imputer.fit(X_target_complete)
                    X_val_imputed = imputer.transform(X_fold_val)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
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
                    model = Ridge(alpha=alpha, random_state=42)
                    model.fit(X_target_scaled, y_target_complete)
                    
                    # Predict on validation
                    fold_predictions[:, i] = model.predict(X_val_scaled)
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
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\nCross-Validation Score (Competition Metric): {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores
    }

def main(cv_only=False):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
    """
    print("=== Ridge Models with RDKit Feature Extraction ===")
    print(f"RDKit available: {RDKIT_AVAILABLE}")
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
            cv_results = perform_cross_validation(X_train, y_train, cv_folds=5, target_columns=target_columns)
            print(f"\n=== Cross-Validation Complete ===")
            return cv_results
    
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
            # Get samples with valid target values
            X_target = X_train[mask]
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
                # Use mean imputation only for test set predictions
                imputer = SimpleImputer(strategy='mean')
                imputer.fit(X_target_complete)  # Fit on complete training data
                X_test_imputed = imputer.transform(X_test)
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

if __name__ == "__main__":
    # Check for command line arguments
    cv_only = '--cv-only' in sys.argv or '--cv' in sys.argv
    
    main(cv_only=cv_only)