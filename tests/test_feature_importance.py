import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests for feature importance functionality

def test_shap_feature_importance_calculation():
    """Test that SHAP feature importance can be calculated correctly"""
    from cv import calculate_shap_importance
    import lightgbm as lgb
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.random.randn(n_samples)
    
    # Train a simple model
    model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y)
    
    # Calculate SHAP importance
    importance = calculate_shap_importance(model, X)
    
    # Verify output format
    assert isinstance(importance, dict)
    assert len(importance) == n_features
    assert all(col in importance for col in X.columns)
    assert all(isinstance(val, (int, float)) for val in importance.values())
    assert all(val >= 0 for val in importance.values())


def test_feature_importance_aggregation():
    """Test aggregation of feature importance across folds"""
    from cv import aggregate_feature_importance
    
    # Create mock importance data from multiple folds
    fold_importances = [
        {'feature_1': 0.3, 'feature_2': 0.5, 'feature_3': 0.2},
        {'feature_1': 0.4, 'feature_2': 0.4, 'feature_3': 0.2},
        {'feature_1': 0.35, 'feature_2': 0.45, 'feature_3': 0.2},
    ]
    
    # Aggregate importance
    aggregated = aggregate_feature_importance(fold_importances)
    
    # Verify aggregation
    assert isinstance(aggregated, dict)
    assert len(aggregated) == 3
    assert abs(aggregated['feature_1'] - 0.35) < 0.01  # Mean of [0.3, 0.4, 0.35]
    assert abs(aggregated['feature_2'] - 0.45) < 0.01  # Mean of [0.5, 0.4, 0.45]
    assert abs(aggregated['feature_3'] - 0.2) < 0.01   # Mean of [0.2, 0.2, 0.2]


def test_features_md_update():
    """Test updating FEATURES.md with feature importance"""
    from cv import update_features_md
    
    # Create temporary FEATURES.md
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Features\n\n")
        f.write("## Current Features\n\n")
        f.write("Some existing content\n\n")
        features_path = f.name
    
    try:
        # Mock feature importance data
        feature_importance = {
            'Tg': {
                'feature_1': 0.35,
                'feature_2': 0.45,
                'feature_3': 0.2
            },
            'FFV': {
                'feature_a': 0.6,
                'feature_b': 0.3,
                'feature_c': 0.1
            }
        }
        
        # Update FEATURES.md
        update_features_md(feature_importance, features_path)
        
        # Read updated content
        with open(features_path, 'r') as f:
            content = f.read()
        
        # Verify content was updated
        assert '# Feature Importance' in content or '## Feature Importance' in content
        assert 'SHAP' in content or 'Shapley' in content
        assert 'Tg' in content
        assert 'FFV' in content
        assert 'feature_1' in content
        assert 'feature_a' in content
        
    finally:
        # Clean up
        if os.path.exists(features_path):
            os.remove(features_path)


def test_feature_importance_integration_cv():
    """Test feature importance integration in cross-validation"""
    from cv import perform_cross_validation
    import lightgbm as lgb
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['feature_1', 'feature_2', 'feature_3']
    )
    
    # Create target with some missing values using a real target name
    y = pd.DataFrame({
        'Tg': np.where(np.random.rand(n_samples) > 0.3, 
                      np.random.randn(n_samples), np.nan)
    })
    
    # Run CV with feature importance tracking
    results = perform_cross_validation(
        X, y,
        cv_folds=2,
        target_columns=['Tg'],
        model_type='lightgbm',
        random_seed=42,
        calculate_feature_importance=True,
        preprocessed=True  # Set to True to avoid feature selection
    )
    
    # Verify feature importance is included
    assert 'feature_importance' in results
    assert isinstance(results['feature_importance'], dict)
    assert all(feat in results['feature_importance'] for feat in X.columns)


def test_shap_importance_with_missing_data():
    """Test SHAP importance calculation with missing data"""
    from cv import calculate_shap_importance
    import lightgbm as lgb
    
    # Create data with missing values
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some missing values
    X.loc[0:10, 'feature_0'] = np.nan
    X.loc[20:30, 'feature_2'] = np.nan
    
    y = np.random.randn(n_samples)
    
    # Train model
    model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y)
    
    # Calculate importance
    importance = calculate_shap_importance(model, X)
    
    # Verify all features have importance values
    assert len(importance) == n_features
    assert all(col in importance for col in X.columns)
    assert all(val >= 0 for val in importance.values())


def test_feature_importance_save_to_file():
    """Test saving feature importance to JSON file"""
    from cv import save_feature_importance
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock feature importance data
        feature_importance = {
            'Tg': {
                'feature_1': 0.35,
                'feature_2': 0.45,
                'feature_3': 0.2
            },
            'FFV': {
                'feature_a': 0.6,
                'feature_b': 0.3,
                'feature_c': 0.1
            }
        }
        
        # Save to file
        output_path = os.path.join(tmpdir, 'feature_importance.json')
        save_feature_importance(feature_importance, output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Load and verify content
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == feature_importance


if __name__ == "__main__":
    # Run tests
    test_shap_feature_importance_calculation()
    print("✓ SHAP feature importance calculation test passed")
    
    test_feature_importance_aggregation()
    print("✓ Feature importance aggregation test passed")
    
    test_features_md_update()
    print("✓ FEATURES.md update test passed")
    
    test_feature_importance_integration_cv()
    print("✓ Feature importance CV integration test passed")
    
    test_shap_importance_with_missing_data()
    print("✓ SHAP importance with missing data test passed")
    
    test_feature_importance_save_to_file()
    print("✓ Feature importance save to file test passed")
    
    print("\nAll feature importance tests passed!")