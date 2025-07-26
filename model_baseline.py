#!/usr/bin/env python3
"""
Simple baseline model for NeurIPS Open Polymer Prediction 2025
Uses median predictions to establish a baseline
"""

import pandas as pd
import numpy as np
import os

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# Set paths based on environment
if IS_KAGGLE:
    TRAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    TEST_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
    SUBMISSION_PATH = 'submission.csv'
    
    SUPP_PATHS = [
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv'
    ]
else:
    TRAIN_PATH = 'data/raw/train.csv'
    TEST_PATH = 'data/raw/test.csv'
    SUBMISSION_PATH = 'output/submission_baseline.csv'
    
    SUPP_PATHS = [
        'data/raw/train_supplement/dataset1.csv',
        'data/raw/train_supplement/dataset2.csv',
        'data/raw/train_supplement/dataset3.csv',
        'data/raw/train_supplement/dataset4.csv'
    ]

def main():
    print("=== Simple Baseline Model ===")
    print("Using median values for all predictions")
    
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
    
    # Calculate medians for each target
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("\nCalculating median values:")
    medians = {}
    for col in target_columns:
        median_val = train_df[col].median()
        medians[col] = median_val
        non_null = train_df[col].notna().sum()
        print(f"{col}: median={median_val:.4f} (from {non_null} samples)")
    
    # Create submission
    print("\nCreating submission...")
    submission_df = pd.DataFrame({
        'id': test_df['id']
    })
    
    for col in target_columns:
        submission_df[col] = medians[col]
    
    # Save submission
    print(f"\nSaving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print("\nSubmission preview:")
    print(submission_df.head(10))
    
    print("\n=== Baseline model complete! ===")
    print("This should establish a proper baseline score on the leaderboard")

if __name__ == "__main__":
    main()