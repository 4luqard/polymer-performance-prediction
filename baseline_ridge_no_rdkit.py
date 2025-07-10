#!/usr/bin/env python3
"""
Baseline Ridge Regression Model for NeurIPS Open Polymer Prediction 2025
This script uses only libraries available in Kaggle notebooks without internet access.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import re
import warnings
warnings.filterwarnings('ignore')

# Kaggle competition paths
TRAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
TEST_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
SUBMISSION_PATH = '/kaggle/working/submission.csv'

# Supplementary dataset paths
SUPP_PATHS = [
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv',
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv',
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv',
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv'
]

def extract_molecular_features(smiles):
    """Extract features from SMILES string without RDKit"""
    features = {}
    
    # Basic string features
    features['length'] = len(smiles)
    
    # Count different atoms
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
    features['num_rings'] = smiles.count('1') + smiles.count('2') + smiles.count('3') + \
                           smiles.count('4') + smiles.count('5') + smiles.count('6') + \
                           smiles.count('7') + smiles.count('8') + smiles.count('9')
    features['num_branches'] = smiles.count('(')
    features['num_chiral_centers'] = smiles.count('@')
    
    # Polymer-specific features
    features['has_polymer_end'] = int('*' in smiles)
    features['num_polymer_ends'] = smiles.count('*')
    
    # Functional group patterns
    features['has_carbonyl'] = int('C(=O)' in smiles or 'C=O' in smiles)
    features['has_hydroxyl'] = int('O' in smiles and not 'C(=O)' in smiles)
    features['has_ether'] = int('COC' in smiles or 'cOc' in smiles)
    features['has_amine'] = int('N' in smiles)
    features['has_sulfone'] = int('S(=O)(=O)' in smiles)
    features['has_ester'] = int('C(=O)O' in smiles or 'COO' in smiles)
    
    # Aromatic features
    features['num_aromatic_atoms'] = features['num_c'] + features['num_n'] + features['num_o'] + features['num_s']
    features['aromatic_ratio'] = features['num_aromatic_atoms'] / max(features['length'], 1)
    
    # Calculate some derived features
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
    features['rotatable_bond_estimate'] = features['num_single_bonds'] - features['num_rings']
    features['flexibility_score'] = features['rotatable_bond_estimate'] / max(features['heavy_atom_count'], 1)
    
    # Size and complexity
    features['molecular_complexity'] = (features['num_rings'] + features['num_branches'] + 
                                       features['num_chiral_centers'])
    
    # Additional polymer-specific patterns
    features['has_phenyl'] = int('c1ccccc1' in smiles or 'c1ccc' in smiles)
    features['has_cyclohexyl'] = int('C1CCCCC1' in smiles)
    features['has_methyl'] = int('C' in smiles and not 'CC' in smiles)
    features['chain_length_estimate'] = max(len(x) for x in re.split(r'[\(\)]', smiles))
    
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

def main():
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
        except:
            print(f"Could not load {supp_path}")
    
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
    
    # Handle missing values in features
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Handle missing values in targets (fill with median)
    print("Handling missing target values...")
    y_train_filled = y_train.fillna(y_train.median())
    
    # Train Ridge regression model
    print("\nTraining Ridge regression model...")
    model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    model.fit(X_train_scaled, y_train_filled)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    
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
    print(submission_df.head())
    
    # Display feature importance (coefficients)
    print("\nTop 5 most important features for each target:")
    feature_names = X_train.columns
    for idx, target in enumerate(target_columns):
        coefs = model.estimators_[idx].coef_
        top_features = np.argsort(np.abs(coefs))[-5:][::-1]
        print(f"\n{target}:")
        for feat_idx in top_features:
            print(f"  {feature_names[feat_idx]}: {coefs[feat_idx]:.4f}")

if __name__ == "__main__":
    main()