#!/usr/bin/env python3
"""
Diagnose the CV/LB gap issue
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
sys.path.append('.')
from src.competition_metric import neurips_polymer_metric

# Load data
print("Loading data...")
train_df = pd.read_csv('data/raw/train.csv')

# Analyze sparsity
print("\n=== Data Sparsity Analysis ===")
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for col in target_cols:
    non_null = train_df[col].notna().sum()
    total = len(train_df)
    print(f"{col}: {non_null}/{total} ({non_null/total*100:.1f}%)")

# Check overlap - how many samples have all targets?
complete_samples = train_df[target_cols].notna().all(axis=1).sum()
print(f"\nComplete samples (all 5 targets): {complete_samples}/{len(train_df)} ({complete_samples/len(train_df)*100:.1f}%)")

# Check pairwise overlap
print("\n=== Target Overlap Matrix ===")
print("How many samples have both targets non-null:")
overlap_matrix = pd.DataFrame(index=target_cols, columns=target_cols)
for t1 in target_cols:
    for t2 in target_cols:
        overlap = (train_df[t1].notna() & train_df[t2].notna()).sum()
        overlap_matrix.loc[t1, t2] = overlap
print(overlap_matrix)

# Test different CV strategies
print("\n=== Testing Different CV Strategies ===")

# Extract features (simplified for speed)
from model import extract_molecular_features
X_features = []
for smiles in train_df['SMILES']:
    X_features.append(extract_molecular_features(smiles))
X = pd.DataFrame(X_features)

# Prepare data
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(imputer.fit_transform(X))
y = train_df[target_cols]

# Strategy 1: Standard KFold (what we use)
print("\n1. Standard KFold (current approach):")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train
    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_train, y_train.fillna(y_train.median()))
    
    # Predict
    y_pred = pd.DataFrame(model.predict(X_val), columns=target_cols, index=val_idx)
    
    # Score
    score, _ = neurips_polymer_metric(y_val, y_pred)
    scores.append(score)
    
    # Check how many validation samples have each target
    val_counts = y_val.notna().sum()
    print(f"  Fold: val_size={len(val_idx)}, non-null per target: {val_counts.to_dict()}")

print(f"Average CV score: {np.mean(scores):.4f} (±{np.std(scores):.4f})")

# Strategy 2: Only score on complete samples
print("\n2. CV scoring only on complete samples:")
complete_mask = train_df[target_cols].notna().all(axis=1)
if complete_mask.sum() > 10:
    # Only run if we have enough complete samples
    X_complete = X_scaled[complete_mask]
    y_complete = y[complete_mask]
    
    kf2 = KFold(n_splits=min(5, complete_mask.sum() // 2), shuffle=True, random_state=42)
    scores2 = []
    
    for train_idx, val_idx in kf2.split(X_complete):
        X_train, X_val = X_complete[train_idx], X_complete[val_idx]
        y_train, y_val = y_complete.iloc[train_idx], y_complete.iloc[val_idx]
        
        model = MultiOutputRegressor(Ridge(alpha=1.0))
        model.fit(X_train, y_train)
        
        y_pred = pd.DataFrame(model.predict(X_val), columns=target_cols)
        score, _ = neurips_polymer_metric(y_val, y_pred)
        scores2.append(score)
    
    print(f"Average CV score (complete only): {np.mean(scores2):.4f} (±{np.std(scores2):.4f})")
else:
    print("Not enough complete samples for this strategy")

# Check prediction variance
print("\n=== Prediction Analysis ===")
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_scaled, y.fillna(y.median()))
y_pred_all = pd.DataFrame(model.predict(X_scaled), columns=target_cols)

print("\nPrediction statistics (on training data):")
for col in target_cols:
    pred_std = y_pred_all[col].std()
    pred_range = y_pred_all[col].max() - y_pred_all[col].min()
    actual_std = y[col].std()
    print(f"{col}: pred_std={pred_std:.4f}, pred_range={pred_range:.4f}, actual_std={actual_std:.4f}")

print("\n=== Hypothesis ===")
print("The 13x CV/LB gap might be because:")
print("1. CV uses many almost-empty validation folds (sparse targets)")
print("2. Test set might have different sparsity pattern")
print("3. Our predictions have very low variance (close to median)")
print("4. Competition metric weights by valid samples differently")