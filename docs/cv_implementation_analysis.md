# Cross-Validation Implementation Analysis

This document provides a detailed analysis of the current cross-validation implementation in the model, explaining each component and identifying potential issues that might cause CV/LB score discrepancy.

## Overview

The CV implementation uses a K-Fold cross-validation approach with separate Ridge regression models for each target variable. The key characteristics are:

- **K-Fold Split**: 5-fold cross-validation with shuffling (random_state=42)
- **Target-Specific Models**: Separate models trained for each of the 5 targets (Tg, FFV, Tc, Density, Rg)
- **Feature Selection**: Target-specific feature selection applied before training
- **Missing Value Handling**: Only samples with non-missing target values are used for training
- **Imputation**: Zero imputation applied to validation set
- **Scoring**: Uses the competition metric (neurips_polymer_metric)

## Detailed Component Analysis

### 1. Data Splitting (Lines 207-218)

```python
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
```

**What it does**: 
- Splits the data into 5 folds
- Shuffles the data before splitting to ensure random distribution
- Uses fixed random_state for reproducibility

**Potential Issues**:
- Random shuffling might not respect the temporal or structural patterns in polymer data
- No stratification based on target value distributions
- All targets share the same fold splits, which might not be optimal given different sparsity levels

### 2. Target-Specific Training (Lines 224-280)

For each target, the process is:

#### a) Sample Selection (Lines 226-236)
```python
mask = ~y_fold_train[target].isna()
X_fold_train_selected = select_features_for_target(X_fold_train, target)
```

**What it does**:
- Filters training samples to only those with non-missing values for the current target
- Applies target-specific feature selection (only top 2 atom count features + all non-atom features)

**Potential Issues**:
- Different targets have vastly different sample sizes (3.3% for Tg vs 46.5% for FFV)
- Feature selection is applied BEFORE filtering for non-missing samples, which might not be optimal

#### b) Feature Completeness Check (Lines 238-246)
```python
feature_complete_mask = ~X_target.isnull().any(axis=1)
X_target_complete = X_target[feature_complete_mask]
```

**What it does**:
- Further filters to only keep samples with no missing features
- This is a secondary filter after the target value filter

**Potential Issues**:
- This could significantly reduce the already small sample sizes for sparse targets
- No information logged about how many samples are dropped at this stage

#### c) Scaling and Imputation (Lines 249-257)
```python
scaler = StandardScaler()
X_target_scaled = scaler.fit_transform(X_target_complete)
imputer = SimpleImputer(strategy='constant', fill_value=0)
imputer.fit(X_target_complete)
X_val_imputed = imputer.transform(X_fold_val_selected)
```

**What it does**:
- Fits StandardScaler on complete training samples only
- Fits imputer on complete training samples
- Applies imputation to entire validation fold (which may have missing features)

**Potential Issues**:
- Scaler is fit on potentially very small subset of data (complete samples only)
- Imputation strategy (zero-fill) applied to validation but NOT to training
- This creates a train/validation distribution mismatch

#### d) Model Training (Lines 260-274)
```python
target_alphas = {
    'Tg': 10.0,
    'FFV': 1.0,
    'Tc': 10.0,
    'Density': 5.0,
    'Rg': 10.0
}
model = Ridge(alpha=alpha, random_state=42)
```

**What it does**:
- Uses target-specific alpha values for Ridge regression
- Higher alpha for sparse targets (Tg, Tc, Rg) and lower for dense target (FFV)

**Potential Issues**:
- Alpha values are hard-coded, not tuned through nested CV
- Same alpha used regardless of actual sample size in each fold

### 3. Prediction Handling (Lines 274-280)

**What it does**:
- If no complete samples available, uses median of fold training data
- Predictions made on imputed and scaled validation data

**Potential Issues**:
- Median fallback might not be representative if fold has skewed distribution
- No tracking of how often median fallback is used

### 4. Scoring (Lines 282-291)

```python
fold_pred_df = pd.DataFrame(fold_predictions, columns=target_columns, index=y_fold_val.index)
fold_score, individual_scores = neurips_polymer_metric(y_fold_val, fold_pred_df, target_columns)
```

**What it does**:
- Creates prediction DataFrame with all 5 targets
- Calculates competition metric on validation fold

**Potential Issues**:
- Metric calculated on ALL validation samples, including those with missing targets
- No analysis of per-target performance

## Key Issues Identified

### 1. **Training Data Leakage/Bias**
- The model is trained only on samples with non-missing target values
- But validated on ALL samples in the validation fold
- This creates a distribution mismatch between training and validation

### 2. **Imputation Inconsistency**
- Training uses only complete samples (no imputation)
- Validation uses imputed features
- This could lead to optimistic CV scores if imputation works well on validation

### 3. **Sample Size Variability**
- Some folds might have very few samples for sparse targets
- No minimum sample size checks
- Could lead to unstable model training

### 4. **Feature Scaling Issues**
- Scaler fit on subset of training data (complete samples only)
- May not capture true feature distributions
- Different scales applied per target due to different training subsets

### 5. **No Data Leakage Prevention**
- Supplementary datasets are mixed with main training data
- No temporal validation considered
- Feature selection not nested within CV

## Recommendations for Improvement

1. **Consistent Sample Selection**:
   - Apply same filtering to both train and validation sets
   - Or use imputation consistently for both

2. **Nested Cross-Validation**:
   - Tune alpha values within each fold
   - Perform feature selection within each fold

3. **Stratified Splitting**:
   - Consider stratification based on target availability patterns
   - Or use GroupKFold if there are natural groups in data

4. **Better Imputation Strategy**:
   - Use target-specific imputation strategies
   - Consider more sophisticated imputation methods

5. **Diagnostic Tracking**:
   - Log sample sizes at each stage
   - Track how often fallback strategies are used
   - Monitor per-target CV scores

## Data Analysis Findings

### Missing Value Patterns
- **No complete samples**: 0% of samples have all 5 targets
- **Dominant pattern**: 47.6% of samples have NO target values (likely structure-only data)
- **Second pattern**: 44.5% have only FFV values
- **High correlation**: Density and Rg are almost always missing together (r=0.994)

### Data Source Issues
- **dataset1**: 100% missing for ALL targets (structure-only)
- **dataset2**: 100% missing for ALL targets (structure-only)  
- **dataset3**: Only has Tg values (100% missing for other targets)
- **dataset4**: Only has FFV values (100% missing for other targets)

This means supplementary datasets are single-target datasets, not multi-target!

### CV Fold Distribution
- Each fold maintains roughly proportional target availability
- But validation folds include many samples with NO target values
- Example: Fold 1 has 3393 validation samples but only 124 have Tg values

## Root Cause Analysis

The main issue is **evaluating on samples without ground truth**:

1. **47.6% of data has NO target values** - these are included in validation folds
2. **Supplementary datasets are single-target** - not representative of test distribution
3. **CV metric includes predictions for missing targets** - artificially inflates score

## Conclusion

The CV score is misleadingly optimistic because:
1. **Phantom Validation**: CV evaluates on samples with no ground truth values
2. **Data Distribution Mismatch**: Supplementary single-target datasets don't match test distribution
3. **Metric Calculation Error**: Competition metric may be calculated incorrectly on missing values

The real issue isn't imputation or feature scaling - it's that we're validating on samples where we can't actually measure performance!

## Metric Implementation Check

The `neurips_polymer_metric` correctly handles missing values:
- Line 223: `is_labeled = y_true[property].notna()`
- Only evaluates on samples with non-missing ground truth

So the metric itself is NOT the issue. The problem lies elsewhere.

## The Real Problem: Data Contamination

After deeper analysis, the issue is **data source contamination**:

1. **Supplementary datasets are structure-only or single-target**
   - dataset1 & dataset2: No targets at all (8,082 samples)
   - dataset3: Only Tg values (46 samples)
   - dataset4: Only FFV values (862 samples)

2. **These contaminate the training process**:
   - Models trained on mixed data learn different patterns
   - Single-target data doesn't represent multi-target test distribution
   - Structure-only data adds noise to feature statistics

3. **CV appears good because**:
   - It validates on the same contaminated distribution
   - But test set likely has different characteristics

## Recommended Fix

1. **Use only main training data for CV**: Remove supplementary datasets from CV
2. **Train separate models**: One for main data, separate for supplementary
3. **Weighted ensemble**: Combine predictions based on data quality