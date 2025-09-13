#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
from datetime import datetime
import shap


def calculate_shap_importance(model, X, sample_size=100):
    """
    Calculate SHAP-based feature importance for a trained model

    Args:
        model: Trained model (LightGBM or similar)
        X: Feature data (DataFrame or array)
        sample_size: Number of samples to use for SHAP calculation (for efficiency)

    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Sample data if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # Multi-class case
            shap_values = shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dictionary
        importance = {}
        for i, col in enumerate(X.columns):
            importance[col] = float(mean_abs_shap[i])

        return importance

    except Exception as e:
        print(f"Warning: Failed to calculate SHAP importance: {e}")
        return {}


def aggregate_feature_importance(fold_importances):
    """
    Aggregate feature importance across multiple folds

    Args:
        fold_importances: List of importance dictionaries from each fold

    Returns:
        Dictionary with mean importance for each feature
    """
    if not fold_importances:
        return {}

    # Get all features
    all_features = set()
    for importance in fold_importances:
        all_features.update(importance.keys())

    # Calculate mean importance
    aggregated = {}
    for feature in all_features:
        values = [imp.get(feature, 0) for imp in fold_importances]
        aggregated[feature] = np.mean(values)

    return aggregated


def save_feature_importance(feature_importance, output_path):
    """
    Save feature importance to JSON file

    Args:
        feature_importance: Dictionary of feature importance
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    importance_json = convert_types(feature_importance)

    with open(output_path, 'w') as f:
        json.dump(importance_json, f, indent=2)


def update_features_md(feature_importance, features_path=None):
    """
    Update FEATURES.md with feature importance information

    Args:
        feature_importance: Dictionary with target as key and feature importance dict as value
        features_path: Path to FEATURES.md (defaults to project FEATURES.md)
    """
    if features_path is None:
        features_path = Path(__file__).parent / 'FEATURES.md'
    else:
        features_path = Path(features_path)

    start_marker = "<!-- FEATURE_IMPORTANCE_START -->"
    end_marker = "<!-- FEATURE_IMPORTANCE_END -->"

    if features_path.exists():
        with open(features_path, 'r') as f:
            content = f.read()
        if start_marker in content and end_marker in content:
            start = content.index(start_marker)
            end = content.index(end_marker) + len(end_marker)
            content = content[:start].rstrip() + "\n\n" + content[end:].lstrip()
        else:
            content = "# Features\n\n"
    else:
        content = "# Features\n\n"

    importance_lines = [start_marker,
                        "## Feature Importance (SHAP-based)",
                        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                        ""]

    for target, importance in feature_importance.items():
        if not importance:
            continue
        importance_lines.append(f"### {target}")

        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        importance_lines.append("| Feature | SHAP Importance |")
        importance_lines.append("|---------|----------------|")
        for feature, score in sorted_features[:20]:
            importance_lines.append(f"| {feature} | {score:.4f} |")
        importance_lines.append("")

    importance_lines.append(end_marker)
    importance_section = "\n".join(importance_lines)

    content = content.rstrip() + "\n\n" + importance_section + "\n"

    with open(features_path, 'w') as f:
        f.write(content)

    print(f"Updated {features_path} with feature importance")
