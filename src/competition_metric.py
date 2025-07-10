"""
Competition evaluation metric for NeurIPS Open Polymer Prediction 2025

The competition uses a weighted Mean Absolute Error (wMAE) metric with:
1. Scaling by the range of each property (min-max normalization)
2. Weighting by the inverse square root of the number of valid samples per property
3. Handling of missing values marked as -9999

Source: https://www.kaggle.com/code/metric/open-polymer-2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def competition_metric(y_true, y_pred, target_names=None):
    """
    Calculate the competition metric for polymer property predictions.
    
    Currently implements weighted average RMSE across all targets.
    This should be updated with the exact competition metric.
    
    Args:
        y_true: DataFrame or array with true values
        y_pred: DataFrame or array with predicted values
        target_names: List of target column names (optional)
    
    Returns:
        float: Competition score (lower is better)
    """
    if target_names is None:
        target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Convert to DataFrames if needed
    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true, columns=target_names)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=target_names)
    
    # Calculate RMSE for each target
    scores = {}
    weights = {}
    
    for target in target_names:
        # Only calculate for non-null values
        mask = y_true[target].notna()
        if mask.sum() > 0:
            true_values = y_true.loc[mask, target]
            pred_values = y_pred.loc[mask, target]
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            scores[target] = rmse
            
            # Weight by number of samples (can be adjusted)
            weights[target] = mask.sum()
        else:
            scores[target] = 0
            weights[target] = 0
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for target in weights:
            weights[target] /= total_weight
    
    # Calculate weighted average
    weighted_score = sum(scores[target] * weights[target] for target in target_names)
    
    return weighted_score, scores


def normalized_competition_metric(y_true, y_pred, target_names=None):
    """
    Calculate a normalized version of the competition metric.
    
    Normalizes each target's RMSE by its standard deviation before averaging.
    This accounts for different scales of the targets.
    
    Args:
        y_true: DataFrame or array with true values
        y_pred: DataFrame or array with predicted values
        target_names: List of target column names (optional)
    
    Returns:
        float: Normalized competition score (lower is better)
    """
    if target_names is None:
        target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Convert to DataFrames if needed
    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true, columns=target_names)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=target_names)
    
    # Calculate normalized RMSE for each target
    scores = {}
    weights = {}
    
    for target in target_names:
        # Only calculate for non-null values
        mask = y_true[target].notna()
        if mask.sum() > 1:  # Need at least 2 samples for std
            true_values = y_true.loc[mask, target]
            pred_values = y_pred.loc[mask, target]
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            
            # Normalize by standard deviation
            std = true_values.std()
            if std > 0:
                normalized_rmse = rmse / std
            else:
                normalized_rmse = rmse
            
            scores[target] = normalized_rmse
            weights[target] = mask.sum()
        else:
            scores[target] = 0
            weights[target] = 0
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for target in weights:
            weights[target] /= total_weight
    
    # Calculate weighted average
    weighted_score = sum(scores[target] * weights[target] for target in target_names)
    
    return weighted_score, scores


def display_metric_results(weighted_score, individual_scores, metric_name="Competition Metric (wMAE)"):
    """
    Display metric results in a formatted way.
    
    Args:
        weighted_score: Overall weighted score
        individual_scores: Dictionary of scores per target
        metric_name: Name of the metric for display
    """
    print(f"\n{metric_name} Results:")
    print("=" * 50)
    print(f"Overall Score: {weighted_score:.4f}")
    print("\nIndividual Target Scores (scaled MAE):")
    for target, score in individual_scores.items():
        if not np.isnan(score):
            # Show percentage for easier interpretation
            print(f"  {target}: {score:.4f} ({score*100:.2f}% of range)")
        else:
            print(f"  {target}: No data")


# Competition constants from the official metric
MINMAX_DICT = {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}
NULL_FOR_SUBMISSION = -9999


def scaling_error(labels, preds, property):
    """Calculate scaled error for a property"""
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)


def get_property_weights(labels):
    """Calculate property weights based on inverse square root of valid samples"""
    property_weight = []
    for property in MINMAX_DICT.keys():
        if isinstance(labels, pd.DataFrame):
            valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        else:
            # Handle dict input
            valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION) if property in labels else 0
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / np.maximum(property_weight, 1))  # Avoid division by zero
    return (property_weight / np.sum(property_weight)) * len(property_weight)


def neurips_polymer_metric(y_true, y_pred, target_names=None):
    """
    NeurIPS Open Polymer Prediction 2025 competition metric.
    
    Implements weighted Mean Absolute Error (wMAE) with:
    - Scaling by property range
    - Weighting by inverse square root of valid samples
    
    Args:
        y_true: DataFrame with true values
        y_pred: DataFrame with predicted values
        target_names: List of target column names (default uses MINMAX_DICT keys)
    
    Returns:
        tuple: (overall_score, individual_scores)
    """
    if target_names is None:
        target_names = list(MINMAX_DICT.keys())
    
    # Convert to DataFrames if needed
    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true, columns=target_names)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=target_names)
    
    property_maes = []
    property_weights = get_property_weights(y_true)
    individual_scores = {}
    
    for i, property in enumerate(target_names):
        # Handle missing values - only evaluate where we have true labels
        is_labeled = y_true[property].notna()
        
        if is_labeled.sum() > 0:
            mae = scaling_error(
                y_true.loc[is_labeled, property].values,
                y_pred.loc[is_labeled, property].values,
                property
            )
            property_maes.append(mae)
            individual_scores[property] = mae
        else:
            individual_scores[property] = np.nan
    
    if len(property_maes) == 0:
        return np.nan, individual_scores
    
    # Calculate weighted average
    final_score = float(np.average(property_maes, weights=property_weights[:len(property_maes)]))
    
    return final_score, individual_scores


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy data
    n_samples = 100
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    y_true = pd.DataFrame(
        np.random.randn(n_samples, 5) * [20, 0.1, 0.1, 0.2, 2] + [80, 0.3, 0.2, 1.0, 15],
        columns=target_names
    )
    
    # Add some noise for predictions
    y_pred = y_true + np.random.randn(n_samples, 5) * [2, 0.01, 0.01, 0.02, 0.2]
    
    # Calculate metrics
    score, individual = competition_metric(y_true, y_pred)
    display_metric_results(score, individual, "Weighted RMSE")
    
    norm_score, norm_individual = normalized_competition_metric(y_true, y_pred)
    display_metric_results(norm_score, norm_individual, "Normalized Weighted RMSE")