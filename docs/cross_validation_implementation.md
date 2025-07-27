# Cross-Validation Implementation Documentation

## Overview

The cross-validation (CV) implementation in our polymer prediction model uses K-fold cross-validation with a single Ridge regression model trained on all targets simultaneously. This approach balances computational efficiency with robust performance estimation.

## Key Design Decisions

### 1. Single Model for All Targets
Instead of training separate models for each target variable (Tg, FFV, Tc, Density, Rg), we use `MultiOutputRegressor` with a single Ridge model. This approach:
- Reduces training time by 5x (one model instead of five)
- Captures potential correlations between targets
- Simplifies the codebase and maintenance

### 2. Weighted Average Alpha
Since scikit-learn's Ridge doesn't support per-target alpha values in multi-output scenarios, we calculate a weighted average alpha based on target sparsity:

```python
# Base alpha values for each target
base_alphas = {
    'Tg': 10.0,      # 96.7% missing data → higher regularization
    'FFV': 1.0,      # 53.5% missing data → lower regularization  
    'Tc': 10.0,      # 95.7% missing data → higher regularization
    'Density': 5.0,  # 96.4% missing data → medium regularization
    'Rg': 10.0       # 96.4% missing data → higher regularization
}

# Weight by number of available samples
weighted_alpha = Σ(alpha[target] × n_samples[target]) / total_samples
```

This ensures targets with more data (like FFV) have proportionally more influence on the regularization strength.

## Implementation Details

### Function: `perform_cross_validation(X, y, model, cv_folds=5)`

**Parameters:**
- `X`: Feature matrix (numpy array) - already scaled and imputed
- `y`: Target DataFrame with columns ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
- `model`: Not used (kept for API compatibility)
- `cv_folds`: Number of cross-validation folds (default: 5)

**Process Flow:**

1. **Calculate Weighted Alpha**
   - Count non-missing samples for each target
   - Apply weighted average formula
   - Result: Single alpha value representing all targets

2. **K-Fold Split**
   - Uses `KFold` with shuffle=True and random_state=42
   - Ensures reproducible splits across runs
   - Each fold maintains the same proportion of samples

3. **Training Loop**
   For each fold:
   - Split data into train/validation sets
   - Fill missing values in training data with median
   - Train `MultiOutputRegressor(Ridge(alpha=weighted_alpha))`
   - Predict on validation set

4. **Evaluation**
   - Calculate RMSE for each target using only non-missing validation samples
   - This prevents bias from imputed values affecting the score
   - Store scores for each fold

5. **Results Aggregation**
   - Compute mean and standard deviation of RMSE across folds
   - Return dictionary with scores for each target

### Handling Missing Data

The implementation carefully handles missing values:
- **Training**: Missing values are filled with target medians
- **Validation**: Only non-missing samples are used for scoring
- This approach ensures we evaluate the model's true predictive performance

## Example Usage

```python
# Prepare data
X_train_scaled = scaler.fit_transform(X_train_imputed)
y_train = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Run cross-validation
cv_results = perform_cross_validation(
    X_train_scaled, 
    y_train,
    model=None,  # Not used internally
    cv_folds=5
)

# Access results
for target, scores in cv_results.items():
    print(f"{target}: RMSE = {scores['mean_rmse']:.4f} (+/- {scores['std_rmse']:.4f})")
```

## Performance Characteristics

- **Time Complexity**: O(k × n × m × p) where k=folds, n=samples, m=features, p=targets
- **Space Complexity**: O(n × m) for feature matrix storage
- **Typical Runtime**: ~30 seconds for full dataset with 5 folds

## Advantages of This Approach

1. **Efficiency**: Single model training per fold instead of five
2. **Consistency**: Same regularization approach as final model
3. **Robustness**: Handles missing data appropriately
4. **Simplicity**: Clean, maintainable code structure

## Future Improvements

Potential enhancements could include:
- Stratified sampling based on target availability
- Custom scoring metrics that weight targets differently
- Hyperparameter tuning for the weighted alpha calculation
- Parallel fold processing for faster execution