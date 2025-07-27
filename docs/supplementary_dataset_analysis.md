# Supplementary Dataset Impact Analysis

## Summary

We compared cross-validation results with and without supplementary datasets to investigate if they contribute to overly optimistic CV scores that don't align with public leaderboard performance.

## Results

### CV Scores Comparison

| Configuration | Multi-Seed CV Score | Std Dev | Public LB Score |
|--------------|-------------------|---------|-----------------|
| **With Supplementary** | 0.0680 | ±0.0025 | 0.3 |
| **Without Supplementary** | 0.0682 | ±0.0028 | TBD |

### Key Findings

1. **Minimal CV Score Difference**: The CV score WITHOUT supplementary datasets (0.0682) is actually slightly HIGHER than with them (0.0680), though the difference is negligible (0.0002).

2. **No Evidence of Optimistic Bias**: The supplementary datasets do NOT appear to be making the CV score artificially optimistic. If anything, they slightly reduce the CV score.

3. **CV/LB Gap Persists**: The large gap between CV score (~0.068) and public LB score (0.3) is NOT explained by the inclusion of supplementary datasets.

### Dataset Composition

#### With Supplementary Datasets:
- Total samples: 16,963
- Main dataset: 7,973 samples
- Supplementary: 8,990 samples (53% of total)
  - dataset1: 874 samples (structure-only)
  - dataset2: 7,208 samples (structure-only)
  - dataset3: 46 samples (Tg only)
  - dataset4: 862 samples (FFV only)

#### Without Supplementary Datasets:
- Total samples: 7,973 (main dataset only)
- Better target coverage per sample

### Per-Target Performance

| Target | With Supp (CV) | Without Supp (CV) | Difference |
|--------|----------------|-------------------|------------|
| Tg | 0.0963 | 0.0954 | -0.0009 |
| FFV | 0.0260 | 0.0263 | +0.0003 |
| Tc | 0.0768 | 0.0767 | -0.0001 |
| Density | 0.0316 | 0.0325 | +0.0009 |
| Rg | 0.1094 | 0.1099 | +0.0005 |

All targets show minimal differences, with no consistent pattern of improvement or degradation.

## Analysis

### Why Supplementary Datasets Don't Help

1. **Structure-Only Data**: 90% of supplementary data (datasets 1 & 2) contains NO target values
   - These 8,082 samples only contribute SMILES structures
   - Cannot directly improve prediction accuracy
   - May add noise to feature statistics

2. **Single-Target Data**: Remaining supplementary datasets are single-target
   - dataset3: Only Tg values (46 samples)
   - dataset4: Only FFV values (862 samples)
   - Don't represent the multi-target nature of the test set

3. **Missing Value Patterns**: Main dataset has more complete multi-target patterns
   - Without supplementary: 93.6% missing Tg, 11.8% missing FFV
   - With supplementary: 96.7% missing Tg, 53.5% missing FFV
   - Supplementary data dilutes the target availability

## Conclusions

1. **Supplementary datasets are NOT the cause of CV/LB discrepancy**
   - CV scores are nearly identical with and without them
   - The 0.068 vs 0.3 gap has another root cause

2. **Structure-only data provides minimal value**
   - 8,082 samples with no targets don't improve predictions
   - May even add noise to feature distributions

3. **Real issue likely lies elsewhere**:
   - Test set distribution differences
   - Feature engineering limitations
   - Model complexity (Ridge may be too simple)
   - Validation methodology issues

## Recommendations

1. **Keep supplementary datasets for final training**: They don't hurt CV and may help with feature robustness
2. **Investigate other causes**: Focus on feature engineering and model complexity
3. **Consider ensemble methods**: Combine models trained on different data subsets
4. **Analyze test set**: The large CV/LB gap suggests fundamental distribution differences