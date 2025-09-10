import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv import create_newsim_stratified_splits, perform_cross_validation


def test_newsim_stratified_splits_function():
    """Test the custom splitting function directly"""
    n_samples = 300
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'new_sim': np.concatenate([np.zeros(200), np.ones(100)])  # 100 new_sim samples
    })
    y = pd.DataFrame({'target': np.random.randn(n_samples)})
    
    # Create splits
    splits = create_newsim_stratified_splits(X, y, cv_folds=3, random_seed=42)
    
    assert len(splits) == 3, "Should create 3 folds"
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Check validation only has new_sim == 1
        X_val = X.iloc[val_idx]
        assert X_val['new_sim'].sum() == len(X_val), \
            f"Fold {fold_idx}: Not all validation samples have new_sim == 1"
        
        # Check training has both
        X_train = X.iloc[train_idx]
        assert X_train['new_sim'].sum() < len(X_train), \
            f"Fold {fold_idx}: Training should contain non-new_sim samples"
        
        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0, \
            f"Fold {fold_idx}: Train and validation indices overlap"


def test_cv_with_newsim_column():
    """Test that perform_cross_validation uses new_sim when present"""
    n_samples = 200
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'new_sim': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    })
    y = pd.DataFrame({
        'Tg': np.random.randn(n_samples),
        'FFV': np.random.randn(n_samples),
        'Tc': np.random.randn(n_samples),
        'Density': np.random.randn(n_samples),
        'Rg': np.random.randn(n_samples)
    })
    
    # This should use the new_sim stratified splitting
    # We can't easily test the internal behavior, but it should run without error
    result = perform_cross_validation(
        X, y, cv_folds=3, 
        target_columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
        enable_diagnostics=False,
        random_seed=42,
        model_type='ridge'  # Use ridge for faster testing
    )
    
    assert 'cv_mean' in result
    assert 'target_scores' in result
    assert 'Tg' in result['target_scores']
    assert 'FFV' in result['target_scores']


def test_cv_without_newsim_column():
    """Test that perform_cross_validation works without new_sim column"""
    n_samples = 100
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
        # No new_sim column
    })
    y = pd.DataFrame({
        'Tg': np.random.randn(n_samples),
        'FFV': np.random.randn(n_samples),
        'Tc': np.random.randn(n_samples),
        'Density': np.random.randn(n_samples),
        'Rg': np.random.randn(n_samples)
    })
    
    # Should fallback to regular KFold
    result = perform_cross_validation(
        X, y, cv_folds=3,
        target_columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
        enable_diagnostics=False,
        random_seed=42,
        model_type='ridge'
    )
    
    assert 'cv_mean' in result
    assert 'target_scores' in result


def test_newsim_edge_cases():
    """Test edge cases for new_sim splitting"""
    # Case 1: Very few new_sim samples
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'new_sim': np.concatenate([np.zeros(95), np.ones(5)])
    })
    y = pd.DataFrame({'target': np.random.randn(100)})
    
    splits = create_newsim_stratified_splits(X, y, cv_folds=3, random_seed=42)
    
    # Should still work, but some folds may have very small validation sets
    total_val_samples = sum(len(val_idx) for _, val_idx in splits)
    assert total_val_samples == 5, "All new_sim samples should be used for validation"
    
    # Case 2: All samples are new_sim
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'new_sim': np.ones(100)
    })
    y = pd.DataFrame({'target': np.random.randn(100)})
    
    splits = create_newsim_stratified_splits(X, y, cv_folds=3, random_seed=42)
    
    for _, val_idx in splits:
        assert len(val_idx) > 0, "Each fold should have validation samples"


def test_model_validation_concept():
    """Test the concept of validation split for model.py"""
    # Simulate what happens in model.py
    n_samples = 200
    train_df = pd.DataFrame({
        'new_sim': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    })
    
    # Simulate mask for target-specific samples
    mask = np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
    original_indices = np.where(mask)[0]
    
    # Check new_sim values for these samples
    newsim_values = train_df.iloc[original_indices]['new_sim'].values
    newsim_mask = newsim_values == 1
    
    # Separate new_sim and non-new_sim samples
    newsim_indices = np.where(newsim_mask)[0]
    non_newsim_indices = np.where(~newsim_mask)[0]
    
    # Calculate validation size
    target_val_size = int(len(original_indices) * 0.15)
    
    if len(newsim_indices) >= target_val_size:
        np.random.seed(42)
        np.random.shuffle(newsim_indices)
        val_indices = newsim_indices[:target_val_size]
        train_indices = np.concatenate([newsim_indices[target_val_size:], non_newsim_indices])
        
        # Check that validation only has new_sim == 1
        val_original_indices = original_indices[val_indices]
        assert train_df.iloc[val_original_indices]['new_sim'].sum() == len(val_indices), \
            "Validation should only contain new_sim == 1 samples"


if __name__ == "__main__":
    print("Testing new_sim stratified splits function...")
    try:
        test_newsim_stratified_splits_function()
        print("✓ Stratified splits function test passed")
    except AssertionError as e:
        print(f"✗ Stratified splits function test failed: {e}")
    
    print("\nTesting CV with new_sim column...")
    try:
        test_cv_with_newsim_column()
        print("✓ CV with new_sim test passed")
    except AssertionError as e:
        print(f"✗ CV with new_sim test failed: {e}")
    
    print("\nTesting CV without new_sim column...")
    try:
        test_cv_without_newsim_column()
        print("✓ CV without new_sim test passed")
    except AssertionError as e:
        print(f"✗ CV without new_sim test failed: {e}")
    
    print("\nTesting edge cases...")
    try:
        test_newsim_edge_cases()
        print("✓ Edge cases test passed")
    except AssertionError as e:
        print(f"✗ Edge cases test failed: {e}")
    
    print("\nTesting model validation concept...")
    try:
        test_model_validation_concept()
        print("✓ Model validation concept test passed")
    except AssertionError as e:
        print(f"✗ Model validation concept test failed: {e}")
    
    print("\n=== All tests completed ===")