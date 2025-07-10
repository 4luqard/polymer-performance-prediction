#!/usr/bin/env python3
"""
Test file for the polymer prediction model.
This file contains all tests for local development and validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import re
import warnings
warnings.filterwarnings('ignore')

# Import functions from model.py
from model import extract_molecular_features, prepare_features, perform_cross_validation

# Local paths for testing
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

def test_feature_extraction():
    """Test the feature extraction function with sample SMILES"""
    print("Testing feature extraction...")
    
    # Test cases
    test_smiles = [
        "*CC(*)c1ccccc1C(=O)OCCCCCC",  # Polymer with aromatic ring
        "C1CCCCC1",  # Simple cyclohexane
        "*Nc1ccc(N*)cc1",  # Polymer with amine groups
        "CC(=O)OC",  # Simple ester
        "c1ccccc1",  # Benzene
    ]
    
    for smiles in test_smiles:
        features = extract_molecular_features(smiles)
        print(f"\nSMILES: {smiles}")
        print(f"Number of features: {len(features)}")
        print(f"Key features: C={features.get('num_C', 0)}, "
              f"c={features.get('num_c', 0)}, "
              f"rings={features.get('num_rings', 0)}, "
              f"polymer_ends={features.get('num_polymer_ends', 0)}")
    
    print("\n✓ Feature extraction test passed")

def test_model_training():
    """Test the full model training pipeline locally"""
    print("\n" + "="*50)
    print("Testing full model training pipeline...")
    
    try:
        # Load training data
        print("\nLoading training data...")
        train_df = pd.read_csv(TRAIN_PATH)
        print(f"Main training data shape: {train_df.shape}")
        
        # Load supplementary datasets
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
        print("\nExtracting features...")
        X_train = prepare_features(train_df.head(100))  # Use subset for faster testing
        X_test = prepare_features(test_df)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        
        # Prepare targets
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        y_train = train_df.head(100)[target_columns]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Handle missing targets
        y_train_filled = y_train.fillna(y_train.median())
        
        # Train model
        print("\nTraining model...")
        model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
        model.fit(X_train_scaled, y_train_filled)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        print("\nPredictions shape:", predictions.shape)
        print("Sample predictions:")
        for i, target in enumerate(target_columns):
            print(f"  {target}: {predictions[0, i]:.4f}")
        
        print("\n✓ Model training test passed")
        
    except Exception as e:
        print(f"\n✗ Model training test failed: {e}")
        raise

def test_cross_validation():
    """Test the cross-validation functionality"""
    print("\n" + "="*50)
    print("Testing cross-validation...")
    
    try:
        # Create small test dataset
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        n_targets = 5
        
        # Generate random data
        X = np.random.randn(n_samples, n_features)
        y = pd.DataFrame(
            np.random.randn(n_samples, n_targets),
            columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        )
        
        # Add some missing values to test handling
        y.iloc[::10, 0] = np.nan  # Add NaN to Tg
        y.iloc[5::10, 2] = np.nan  # Add NaN to Tc
        
        # Perform cross-validation
        print("Running cross-validation on synthetic data...")
        cv_scores = perform_cross_validation(X, y, Ridge(alpha=1.0), cv_folds=3)
        
        # Check results
        print("\nCross-validation results:")
        for target, scores in cv_scores.items():
            if not np.isnan(scores['mean_rmse']):
                print(f"  {target}: RMSE = {scores['mean_rmse']:.4f} (+/- {scores['std_rmse']:.4f})")
                assert len(scores['all_scores']) > 0, f"No scores for {target}"
            else:
                print(f"  {target}: No valid scores")
        
        print("\n✓ Cross-validation test passed")
        
    except Exception as e:
        print(f"\n✗ Cross-validation test failed: {e}")
        raise

def test_submission_format():
    """Test that submission file has correct format"""
    print("\n" + "="*50)
    print("Testing submission format...")
    
    try:
        # Check if submission exists
        if os.path.exists(SUBMISSION_PATH):
            submission_df = pd.read_csv(SUBMISSION_PATH)
            
            # Check columns
            expected_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
            assert list(submission_df.columns) == expected_columns, \
                f"Columns don't match. Expected {expected_columns}, got {list(submission_df.columns)}"
            
            # Check shape
            assert submission_df.shape[0] == 3, \
                f"Expected 3 rows, got {submission_df.shape[0]}"
            
            # Check no missing values
            assert not submission_df.isnull().any().any(), \
                "Submission contains missing values"
            
            print(f"✓ Submission format is correct")
            print(f"  Shape: {submission_df.shape}")
            print(f"  Columns: {list(submission_df.columns)}")
            
        else:
            print("⚠ No submission file found. Run the model first.")
            
    except Exception as e:
        print(f"\n✗ Submission format test failed: {e}")
        raise

def run_all_tests():
    """Run all tests"""
    print("Running all tests for polymer prediction model")
    print("="*50)
    
    # Test 1: Feature extraction
    test_feature_extraction()
    
    # Test 2: Cross-validation functionality
    test_cross_validation()
    
    # Test 3: Model training pipeline
    test_model_training()
    
    # Test 4: Submission format
    test_submission_format()
    
    print("\n" + "="*50)
    print("All tests completed!")

if __name__ == "__main__":
    run_all_tests()