#!/usr/bin/env python3
"""
Local test version of the baseline Ridge regression model
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

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
    """Extract simple features from SMILES string without RDKit"""
    features = {}
    
    # Simple features based on SMILES string
    features['length'] = len(smiles)
    features['num_c'] = smiles.lower().count('c')
    features['num_o'] = smiles.lower().count('o')
    features['num_n'] = smiles.lower().count('n')
    features['num_s'] = smiles.lower().count('s')
    features['num_f'] = smiles.lower().count('f')
    features['num_cl'] = smiles.count('Cl')
    features['num_br'] = smiles.count('Br')
    features['num_rings'] = smiles.count('1') + smiles.count('2') + smiles.count('3')
    features['num_double_bonds'] = smiles.count('=')
    features['num_triple_bonds'] = smiles.count('#')
    features['num_aromatic'] = smiles.count('c') + smiles.count('n') + smiles.count('o')
    features['num_branches'] = smiles.count('(')
    features['has_polymer_end'] = int('*' in smiles)
    features['num_polymer_ends'] = smiles.count('*')
    
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