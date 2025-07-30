#!/usr/bin/env python3
"""
Visualize LightGBM trees for polymer prediction models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import os
import json
from pathlib import Path

# Import from model
from model import (
    TRAIN_PATH, TEST_PATH, SUPP_PATHS,
    prepare_features, select_features_for_target
)

def train_and_visualize_lgbm():
    """Train LightGBM models and visualize their trees"""
    
    print("=== LightGBM Tree Visualization ===")
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
    
    # LightGBM parameters
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
    
    # Create output directory for trees
    output_dir = Path("output/lgbm_trees")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train separate models for each target
    print("\n=== Training LightGBM Models and Extracting Trees ===")
    
    for i, target in enumerate(target_columns):
        print(f"\n{target}:")
        print("-" * 50)
        
        # Get non-missing samples for this target
        mask = ~y_train[target].isna()
        n_samples = mask.sum()
        print(f"Available samples: {n_samples} ({n_samples/len(y_train)*100:.1f}%)")
        
        if n_samples > 0:
            # Select features for this target
            X_train_selected = select_features_for_target(X_train, target)
            X_test_selected = select_features_for_target(X_test, target)
            
            print(f"Using {len(X_train_selected.columns)} features")
            print(f"Features: {', '.join(X_train_selected.columns[:10])}" + 
                  (" ..." if len(X_train_selected.columns) > 10 else ""))
            
            # Get samples with valid target values
            X_target = X_train_selected[mask]
            y_target = y_train[target][mask]
            
            # Further filter to only keep rows with no missing features
            feature_complete_mask = ~X_target.isnull().any(axis=1)
            X_target_complete = X_target[feature_complete_mask]
            y_target_complete = y_target[feature_complete_mask]
            
            print(f"Complete samples (no missing features): {len(X_target_complete)}")
            
            if len(X_target_complete) > 0:
                # Scale features
                scaler = StandardScaler()
                X_target_scaled = scaler.fit_transform(X_target_complete)
                
                # Train LightGBM model
                print(f"Training LightGBM model for {target}...")
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X_target_scaled, y_target_complete)
                
                # Get feature importances
                feature_names = list(X_train_selected.columns)
                importances = model.feature_importances_
                
                # Sort features by importance
                feature_importance_pairs = sorted(
                    zip(feature_names, importances), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                print(f"\nTop 10 most important features:")
                for feat, imp in feature_importance_pairs[:10]:
                    print(f"  {feat}: {imp:.2f}")
                
                # Save tree structure as text
                tree_file = output_dir / f"{target}_tree_structure.txt"
                with open(tree_file, 'w') as f:
                    f.write(f"LightGBM Tree Structure for {target}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Get tree structure for first few trees
                    n_trees_to_show = min(5, model.n_estimators)
                    f.write(f"Showing first {n_trees_to_show} trees out of {model.n_estimators}\n\n")
                    
                    for tree_idx in range(n_trees_to_show):
                        f.write(f"\n--- Tree {tree_idx} ---\n")
                        tree_str = model.booster_.dump_model()['tree_info'][tree_idx]
                        f.write(json.dumps(tree_str, indent=2))
                        f.write("\n")
                
                print(f"Tree structure saved to: {tree_file}")
                
                # Save feature importance
                importance_file = output_dir / f"{target}_feature_importance.txt"
                with open(importance_file, 'w') as f:
                    f.write(f"Feature Importance for {target}\n")
                    f.write("=" * 40 + "\n")
                    for feat, imp in feature_importance_pairs:
                        f.write(f"{feat}: {imp:.4f}\n")
                
                # Create a simple tree visualization for the first tree
                print(f"\nFirst tree summary:")
                tree_info = model.booster_.dump_model()['tree_info'][0]
                
                def print_tree_node(node, depth=0):
                    """Recursively print tree structure"""
                    indent = "  " * depth
                    
                    if 'leaf_value' in node:
                        # Leaf node
                        print(f"{indent}Leaf: value={node['leaf_value']:.4f}")
                    else:
                        # Split node
                        split_feature = feature_names[node['split_feature']]
                        threshold = node['threshold']
                        print(f"{indent}Split: {split_feature} <= {threshold:.4f}")
                        
                        # Print left child
                        if 'left_child' in node:
                            print_tree_node(node['left_child'], depth + 1)
                        
                        # Print right child
                        if 'right_child' in node:
                            print_tree_node(node['right_child'], depth + 1)
                
                print_tree_node(tree_info['tree_structure'])
                
            else:
                print(f"No complete samples available for {target}")
        else:
            print(f"No samples available for {target}")
    
    print(f"\n\nAll tree structures and feature importances saved to: {output_dir}")
    print("\nNote: Tree structures are saved as JSON format in text files.")
    print("You can use external tools like Graphviz to create visual tree diagrams.")

if __name__ == "__main__":
    train_and_visualize_lgbm()