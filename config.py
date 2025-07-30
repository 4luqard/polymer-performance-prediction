"""
Configuration for LightGBM parameters - Single source of truth
"""

LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',  # Changed from rmse to mae for competition alignment
    'boosting_type': 'gbdt',
    'max_depth': -1,  # No limit
    'num_leaves': 31,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    # Early stopping parameters
    'n_iter_no_change': 50,  # Stop if no improvement for 50 iterations
    'validation_fraction': 0.15,  # Use 15% of data for validation
    'tol': 0.001  # Minimum improvement threshold
}