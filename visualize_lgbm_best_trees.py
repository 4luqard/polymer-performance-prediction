#!/usr/bin/env python3
"""
Visualize the best LightGBM tree (based on training performance) for each target
Shows the tree with lowest training error instead of the last tree
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import json
import os
import sys
from model import prepare_features, select_features_for_target
from config import LIGHTGBM_PARAMS

# Define feature selection for targets
TARGET_FEATURES = {
    'Tg': ['num_C', 'num_n'],
    'FFV': ['num_F', 'num_S'],
    'Tc': ['num_C', 'num_Cl'],
    'Density': ['num_Cl', 'num_F'],
    'Rg': ['num_F', 'num_Cl']
}

def format_tree_node(node, depth=0, max_depth=5, feature_names=None):
    """
    Format a tree node in human-readable IF-ELSE format
    """
    if depth >= max_depth:
        return "    " * depth + "... (tree continues)\n"
    
    indent = "  " * depth
    
    # Check if it's a leaf node
    if 'leaf_index' in node:
        # Leaf node
        return f"{indent}→ Prediction: {node['leaf_value']:.4f}\n"
    else:
        # Decision node
        feature_idx = node['split_feature']
        if feature_names and 0 <= feature_idx < len(feature_names):
            feature = feature_names[feature_idx]
        else:
            feature = f"feature_{feature_idx}"
        
        threshold = node['threshold']
        left_child = node['left_child']
        right_child = node['right_child']
        
        result = f"{indent}IF {feature} <= {threshold:.4f}:\n"
        result += format_tree_node(left_child, depth + 1, max_depth, feature_names)
        result += f"{indent}ELSE ({feature} > {threshold:.4f}):\n"
        result += format_tree_node(right_child, depth + 1, max_depth, feature_names)
        
        return result

def find_best_tree(model):
    """
    Find the best tree based on training performance.
    Returns the tree index with the lowest training error.
    """
    # Get training results
    results = model.evals_result_
    
    if not results:
        # No eval results available, return last tree
        return model.n_estimators_ - 1
    
    # Get the validation scores (MAE)
    valid_scores = results.get('valid_0', {}).get('l1', [])
    
    if not valid_scores:
        # No validation scores, return last tree
        return model.n_estimators_ - 1
    
    # Find the tree with minimum MAE
    best_tree_idx = np.argmin(valid_scores)
    return best_tree_idx

def main():
    # Load data
    print("Loading data...")
    train = pd.read_csv('data/raw/train.csv')
    
    # Extract features
    print("Extracting features...")
    X_train = prepare_features(train)
    y_train = train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    
    # Create output directory
    output_dir = 'output/lgbm_best_trees'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models and save best trees for each target
    for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        print(f"\nProcessing {target}...")
        
        # Get non-missing samples
        mask = ~y_train[target].isna()
        if mask.sum() == 0:
            print(f"  No samples available for {target}")
            continue
            
        # Select features for this target
        X_selected = select_features_for_target(X_train, target)
        
        # Get samples with valid target values
        X_target = X_selected[mask]
        y_target = y_train[target][mask]
        
        # Filter to only keep rows with no missing features
        feature_complete_mask = ~X_target.isnull().any(axis=1)
        X_complete = X_target[feature_complete_mask]
        y_complete = y_target[feature_complete_mask]
        
        print(f"  Training with {len(X_complete)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_complete)
        
        # Train model with eval set for finding best iteration
        X_train_split = X_scaled[:int(len(X_scaled)*0.85)]
        y_train_split = y_complete.iloc[:int(len(y_complete)*0.85)]
        X_valid_split = X_scaled[int(len(X_scaled)*0.85):]
        y_valid_split = y_complete.iloc[int(len(y_complete)*0.85):]
        
        # Train LightGBM
        model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_valid_split, y_valid_split)],
            eval_metric='mae'
        )
        
        # Find best tree
        best_tree_idx = find_best_tree(model)
        print(f"  Best tree: {best_tree_idx} (out of {model.n_estimators_} trees)")
        
        # Get tree structure
        booster = model.booster_
        best_tree_structure = booster.dump_model()['tree_info'][best_tree_idx]
        
        # Get feature importances
        feature_importances = model.feature_importances_
        feature_names = X_selected.columns.tolist()
        
        # Sort features by importance
        importance_pairs = list(zip(feature_names, feature_importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Create human-readable output
        output_file = os.path.join(output_dir, f"{target}_best_tree.txt")
        with open(output_file, 'w') as f:
            f.write(f"LightGBM Best Tree for {target}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training samples: {len(X_complete)}\n")
            f.write(f"Number of features: {len(feature_names)}\n")
            f.write(f"Total trees in ensemble: {model.n_estimators_}\n")
            f.write(f"Best tree index: {best_tree_idx}\n")
            if best_tree_idx < model.n_estimators_ - 1:
                f.write(f"Early stopping occurred at iteration {best_tree_idx + 1}\n")
            f.write("\nTop 5 Most Important Features:\n")
            for feat, imp in importance_pairs[:5]:
                f.write(f"  - {feat}: {imp:.2f}\n")
            
            f.write("\nBest Tree Structure:\n")
            f.write("=" * 40 + "\n\n")
            
            # Format tree in readable format
            tree_str = format_tree_node(best_tree_structure['tree_structure'], 
                                       feature_names=feature_names)
            f.write(tree_str)
            
            f.write("\n" + "=" * 40 + "\n")
            f.write("Note: This is the best performing tree based on validation MAE.\n")
            f.write("The actual prediction is the sum of all trees up to this index.\n")
            f.write("Tree is truncated at depth 5 for readability.\n")
        
        print(f"  Saved best tree visualization to {output_file}")

if __name__ == "__main__":
    main()