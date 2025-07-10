#!/usr/bin/env python3
"""
Baseline Ridge Regression Model for NeurIPS Open Polymer Prediction 2025
This script is designed to run directly in a Kaggle notebook without any modifications.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
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
    """Extract molecular descriptors from SMILES string"""
    features = {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return default values if SMILES parsing fails
            return {
                'MolWt': 0, 'LogP': 0, 'NumHDonors': 0, 'NumHAcceptors': 0,
                'NumRotatableBonds': 0, 'NumHeteroatoms': 0, 'NumAromaticRings': 0,
                'TPSA': 0, 'NumAtoms': 0, 'NumHeavyAtoms': 0,
                'NumAliphaticRings': 0, 'NumSaturatedRings': 0, 'RingCount': 0,
                'MolMR': 0, 'FractionCsp3': 0
            }
        
        # Basic molecular properties
        features['MolWt'] = Descriptors.MolWt(mol)
        features['LogP'] = Descriptors.MolLogP(mol)
        features['NumHDonors'] = Descriptors.NumHDonors(mol)
        features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        features['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        features['TPSA'] = Descriptors.TPSA(mol)
        features['NumAtoms'] = mol.GetNumAtoms()
        features['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
        features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        features['RingCount'] = Descriptors.RingCount(mol)
        features['MolMR'] = Crippen.MolMR(mol)
        features['FractionCsp3'] = Descriptors.FractionCsp3(mol)
        
    except:
        # Return default values if any error occurs
        features = {
            'MolWt': 0, 'LogP': 0, 'NumHDonors': 0, 'NumHAcceptors': 0,
            'NumRotatableBonds': 0, 'NumHeteroatoms': 0, 'NumAromaticRings': 0,
            'TPSA': 0, 'NumAtoms': 0, 'NumHeavyAtoms': 0,
            'NumAliphaticRings': 0, 'NumSaturatedRings': 0, 'RingCount': 0,
            'MolMR': 0, 'FractionCsp3': 0
        }
    
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