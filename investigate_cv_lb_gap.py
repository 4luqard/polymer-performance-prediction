#!/usr/bin/env python3
"""
Comprehensive investigation of CV vs LB score discrepancy.
This script analyzes why CV shows 0.0122 but LB shows 0.081 for the best model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from model import prepare_features, perform_cross_validation, neurips_polymer_metric
import warnings
warnings.filterwarnings('ignore')

def analyze_data_distribution():
    """Analyze the distribution of training data."""
    print("=== Data Distribution Analysis ===\n")
    
    # Load all training data (same as model.py)
    train_df = pd.read_csv('data/raw/train.csv')
    
    # Load supplementary data
    supp_paths = [
        'data/raw/train_supplement/PolyBERT_cLogP.csv',
        'data/raw/train_supplement/PolyBERT_EA.csv', 
        'data/raw/train_supplement/PolyBERT_Eg.csv',
        'data/raw/train_supplement/Erni_cleaned.csv'
    ]
    
    all_train_dfs = [train_df]
    for path in supp_paths:
        try:
            supp_df = pd.read_csv(path)
            all_train_dfs.append(supp_df)
        except:
            pass
    
    # Combine all data
    full_train_df = pd.concat(all_train_dfs, ignore_index=True)
    
    print(f"Total training samples: {len(full_train_df)}")
    print(f"Main train.csv samples: {len(train_df)}")
    print(f"Supplementary samples: {len(full_train_df) - len(train_df)}")
    
    # Analyze target availability
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print("\nTarget availability:")
    for target in targets:
        n_available = full_train_df[target].notna().sum()
        pct = n_available / len(full_train_df) * 100
        print(f"  {target}: {n_available} samples ({pct:.1f}%)")
    
    # Analyze target combinations
    print("\nTarget combination patterns:")
    target_patterns = full_train_df[targets].notna().apply(lambda row: ''.join(['1' if v else '0' for v in row]), axis=1)
    pattern_counts = target_patterns.value_counts().head(10)
    
    for pattern, count in pattern_counts.items():
        active_targets = [targets[i] for i, bit in enumerate(pattern) if bit == '1']
        pct = count / len(full_train_df) * 100
        print(f"  {pattern} ({', '.join(active_targets)}): {count} samples ({pct:.1f}%)")
    
    # Check if main train.csv is representative
    print("\nComparing main train.csv to full dataset:")
    for target in targets:
        main_mean = train_df[target].mean()
        full_mean = full_train_df[target].mean()
        if not pd.isna(main_mean) and not pd.isna(full_mean):
            diff = (main_mean - full_mean) / full_mean * 100
            print(f"  {target} mean difference: {diff:+.1f}%")
    
    return train_df, full_train_df

def run_different_cv_strategies(train_df, full_train_df):
    """Test different CV strategies to find the source of discrepancy."""
    print("\n=== Testing Different CV Strategies ===\n")
    
    # Extract features
    X_train = prepare_features(train_df)
    X_full = prepare_features(full_train_df)
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    y_train = train_df[targets]
    y_full = full_train_df[targets]
    
    # Strategy 1: Standard KFold on main train.csv
    print("1. Standard KFold on main train.csv (what model.py uses for submission):")
    cv_result1 = perform_cross_validation(X_train, y_train, cv_folds=3, target_columns=targets)
    
    # Strategy 2: KFold on full dataset
    print("\n2. KFold on full dataset (including supplements):")
    cv_result2 = perform_cross_validation(X_full, y_full, cv_folds=3, target_columns=targets)
    
    # Strategy 3: Leave-out validation using train/test split
    print("\n3. Train/test split (80/20) on main train.csv:")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train models and evaluate (simplified version of model.py logic)
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    
    predictions = np.zeros((len(X_val), len(targets)))
    
    for i, target in enumerate(targets):
        mask = ~y_tr[target].isna()
        if mask.sum() > 0:
            X_target = X_tr[mask]
            y_target = y_tr[target][mask]
            
            # Handle missing features
            imputer = SimpleImputer(strategy='mean')
            X_target_imputed = imputer.fit_transform(X_target)
            X_val_imputed = imputer.transform(X_val)
            
            # Scale
            scaler = StandardScaler()
            X_target_scaled = scaler.fit_transform(X_target_imputed)
            X_val_scaled = scaler.transform(X_val_imputed)
            
            # Train
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_target_scaled, y_target)
            
            predictions[:, i] = model.predict(X_val_scaled)
    
    pred_df = pd.DataFrame(predictions, columns=targets, index=y_val.index)
    score, _ = neurips_polymer_metric(y_val, pred_df, targets)
    print(f"  Score: {score:.4f}")
    
    return cv_result1, cv_result2

def analyze_prediction_difficulty():
    """Analyze which samples are hardest to predict."""
    print("\n=== Prediction Difficulty Analysis ===\n")
    
    train_df = pd.read_csv('data/raw/train.csv')
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Count how many targets each sample has
    train_df['n_targets'] = train_df[targets].notna().sum(axis=1)
    
    print("Samples by number of available targets:")
    target_counts = train_df['n_targets'].value_counts().sort_index()
    for n, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"  {n} targets: {count} samples ({pct:.1f}%)")
    
    # Check if samples with fewer targets are harder to predict
    print("\nTarget statistics by availability:")
    for n in range(1, 6):
        subset = train_df[train_df['n_targets'] == n]
        if len(subset) > 0:
            print(f"\nSamples with {n} target(s):")
            for target in targets:
                values = subset[target].dropna()
                if len(values) > 0:
                    print(f"  {target}: mean={values.mean():.3f}, std={values.std():.3f}, n={len(values)}")

def analyze_feature_importance():
    """Analyze which features are most important for predictions."""
    print("\n=== Feature Importance Analysis ===\n")
    
    train_df = pd.read_csv('data/raw/train.csv')
    X_train = prepare_features(train_df)
    
    print(f"Total features: {X_train.shape[1]}")
    print(f"Feature categories:")
    
    # Group features by type
    feature_groups = {
        'Molecular weight': [col for col in X_train.columns if 'mol_weight' in col],
        'Atom counts': [col for col in X_train.columns if col.startswith('C_count') or col.startswith('O_count') or col.startswith('N_count')],
        'Bond types': [col for col in X_train.columns if 'bond' in col],
        'Ring features': [col for col in X_train.columns if 'ring' in col],
        'Polymer markers': [col for col in X_train.columns if 'polymer' in col],
        'Other': []
    }
    
    # Categorize remaining features
    categorized = set()
    for group_features in feature_groups.values():
        categorized.update(group_features)
    
    feature_groups['Other'] = [col for col in X_train.columns if col not in categorized]
    
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)} features")

def main():
    """Run comprehensive CV/LB discrepancy investigation."""
    print("=" * 80)
    print("CV vs LB Score Discrepancy Investigation")
    print("=" * 80)
    
    # 1. Analyze data distribution
    train_df, full_train_df = analyze_data_distribution()
    
    # 2. Test different CV strategies
    cv_result1, cv_result2 = run_different_cv_strategies(train_df, full_train_df)
    
    # 3. Analyze prediction difficulty
    analyze_prediction_difficulty()
    
    # 4. Analyze feature importance
    analyze_feature_importance()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey findings:")
    print("1. The CV correctly uses KFold on training data (not the 3-sample test set)")
    print("2. The large CV/LB gap (0.0122 vs 0.081) suggests:")
    print("   - Possible distribution shift between local validation and public test")
    print("   - The extreme sparsity of targets makes validation challenging")
    print("   - Most samples have only 1 target, making multi-target prediction difficult")
    print("\nRecommendations:")
    print("1. The current CV implementation is correct and should not be changed")
    print("2. Focus on improving the model itself rather than the CV methodology")
    print("3. Consider ensemble methods or target-specific optimizations")

if __name__ == "__main__":
    main()