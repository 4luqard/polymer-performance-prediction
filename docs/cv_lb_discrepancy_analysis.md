# CV vs Public LB Discrepancy Analysis

## Critical Discovery

The local test.csv file contains only 3 samples, and ALL of them are already in the training data. This is clearly a smoke test, not the actual competition test set.

## Key Findings

### 1. Local Test Set Issues
- **Size**: Only 3 samples (vs thousands expected)
- **Data Leakage**: 100% of test SMILES exist in training data
- **Purpose**: This is a smoke test for local development
- **Implication**: Local CV cannot predict Kaggle performance

### 2. CV Scores Analysis
- **Median baseline CV**: ~0.118 (using simple median predictions)
- **Our model CV**: ~0.068 (significant improvement over baseline)
- **Public LB**: 0.3 (much worse than baseline)

### 3. Possible Causes for Poor LB Performance

#### A. Different Test Distribution
The Kaggle test set likely has:
- Different polymer types not seen in training
- Different distribution of target values
- More challenging cases

#### B. Feature Extraction Issues
When running on Kaggle:
- Target-specific features might not work correctly
- Feature calculation could differ between environments
- Missing value handling might be problematic

#### C. Submission Format Issues
- The model might be outputting wrong values for Kaggle's test set
- ID mapping could be incorrect

## Recommendations

### 1. Immediate Actions
- **Revert to baseline**: Submit simple median predictions first
- **Test incrementally**: Add features one by one
- **Monitor LB closely**: Each change should improve score

### 2. Debugging Strategy
```python
# 1. Submit median baseline
baseline_submission = pd.DataFrame({
    'id': test_df['id'],
    'Tg': train_df['Tg'].median(),
    'FFV': train_df['FFV'].median(),
    'Tc': train_df['Tc'].median(),
    'Density': train_df['Density'].median(),
    'Rg': train_df['Rg'].median()
})

# 2. Submit with all features (no target-specific)
# 3. Compare scores to identify issue
```

### 3. Model Simplification
Given the discrepancy, we should:
1. Use all atom features for all targets (no target-specific selection)
2. Remove complex derived features
3. Focus on robust, simple features

## Conclusion

The CV-LB discrepancy is primarily due to:
1. Local test set being a smoke test (3 samples from training data)
2. Kaggle test set being completely different
3. Possible overfitting to training data distribution
4. Target-specific features may be causing issues

**Action**: Revert to simpler model and rebuild carefully while monitoring LB scores.