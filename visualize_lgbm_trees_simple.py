#!/usr/bin/env python3
"""
Generate human-readable final tree visualization for LightGBM models
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

def format_tree_node(node, feature_names, depth=0, max_depth=5):
    """Format tree node in human-readable format"""
    indent = "  " * depth
    output = []
    
    if depth > max_depth:
        output.append(f"{indent}... (tree continues)")
        return output
    
    if 'leaf_value' in node:
        # Leaf node
        output.append(f"{indent}→ Prediction: {node['leaf_value']:.4f}")
    else:
        # Split node
        split_feature = feature_names[node['split_feature']]
        threshold = node['threshold']
        
        output.append(f"{indent}IF {split_feature} <= {threshold:.4f}:")
        
        # Left child (condition is true)
        if 'left_child' in node:
            output.extend(format_tree_node(node['left_child'], feature_names, depth + 1, max_depth))
        
        output.append(f"{indent}ELSE ({split_feature} > {threshold:.4f}):")
        
        # Right child (condition is false)  
        if 'right_child' in node:
            output.extend(format_tree_node(node['right_child'], feature_names, depth + 1, max_depth))
    
    return output

def train_and_visualize_final_trees():
    """Train LightGBM models and visualize only the final tree in human-readable format"""
    
    print("=== LightGBM Final Tree Visualization (Human-Readable) ===")
    print("Loading training data...")
    
    # Load main training data
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Load and combine supplementary datasets
    all_train_dfs = [train_df]
    
    for supp_path in SUPP_PATHS:
        try:
            supp_df = pd.read_csv(supp_path)
            all_train_dfs.append(supp_df)
        except Exception as e:
            pass
    
    # Combine all training data
    train_df = pd.concat(all_train_dfs, ignore_index=True)
    
    # Load test data
    test_df = pd.read_csv(TEST_PATH)
    
    # Extract features
    X_train = prepare_features(train_df)
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
    print("\n=== Training Models and Extracting Final Trees ===")
    
    for i, target in enumerate(target_columns):
        print(f"\n{target}:")
        print("-" * 50)
        
        # Get non-missing samples for this target
        mask = ~y_train[target].isna()
        n_samples = mask.sum()
        
        if n_samples > 0:
            # Select features for this target
            X_train_selected = select_features_for_target(X_train, target)
            X_test_selected = select_features_for_target(X_test, target)
            
            # Get samples with valid target values
            X_target = X_train_selected[mask]
            y_target = y_train[target][mask]
            
            # Further filter to only keep rows with no missing features
            feature_complete_mask = ~X_target.isnull().any(axis=1)
            X_target_complete = X_target[feature_complete_mask]
            y_target_complete = y_target[feature_complete_mask]
            
            if len(X_target_complete) > 0:
                # Scale features
                scaler = StandardScaler()
                X_target_scaled = scaler.fit_transform(X_target_complete)
                
                # Train LightGBM model
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X_target_scaled, y_target_complete)
                
                # Get feature names
                feature_names = list(X_train_selected.columns)
                
                # Get feature importances
                importances = model.feature_importances_
                feature_importance_pairs = sorted(
                    zip(feature_names, importances), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Save final tree in human-readable format
                tree_file = output_dir / f"{target}_final_tree.txt"
                with open(tree_file, 'w') as f:
                    f.write(f"LightGBM Final Tree for {target}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Model summary
                    f.write(f"Training samples: {len(X_target_complete)}\n")
                    f.write(f"Number of features: {len(feature_names)}\n")
                    f.write(f"Total trees in ensemble: {model.n_estimators}\n\n")
                    
                    # Top features
                    f.write("Top 5 Most Important Features:\n")
                    for feat, imp in feature_importance_pairs[:5]:
                        f.write(f"  - {feat}: {imp:.2f}\n")
                    f.write("\n")
                    
                    # Final tree structure (last tree in the ensemble)
                    f.write("Final Tree Structure (Tree 199):\n")
                    f.write("=" * 40 + "\n\n")
                    
                    # Get the last tree
                    tree_info = model.booster_.dump_model()['tree_info'][-1]
                    tree_lines = format_tree_node(tree_info['tree_structure'], feature_names, 0, 5)
                    
                    for line in tree_lines:
                        f.write(line + "\n")
                    
                    f.write("\n" + "=" * 40 + "\n")
                    f.write("Note: This is the final tree (200th) in the ensemble.\n")
                    f.write("The actual prediction is the sum of all 200 trees.\n")
                    f.write("Tree is truncated at depth 5 for readability.\n")
                
                print(f"✓ Final tree saved to: {tree_file}")
                
                # Print a preview
                print(f"\nPreview of final tree for {target}:")
                tree_info = model.booster_.dump_model()['tree_info'][-1]
                preview_lines = format_tree_node(tree_info['tree_structure'], feature_names, 0, 3)
                for line in preview_lines[:10]:
                    print(line)
                if len(preview_lines) > 10:
                    print("  ... (truncated)")
                
            else:
                print(f"✗ No complete samples available for {target}")
        else:
            print(f"✗ No samples available for {target}")
    
    print(f"\n\nAll final trees saved to: {output_dir}")
    print("Files are now in human-readable format with IF-ELSE structure.")

if __name__ == "__main__":
    train_and_visualize_final_trees()