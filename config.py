"""
Configuration for LightGBM parameters - Single source of truth
"""

LIGHTGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'extra_trees': True,
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
