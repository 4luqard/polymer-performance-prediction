
# PCA vs Autoencoder Comparison Report

## Summary
- PCA (32 components) vs Autoencoder (32-dim encoding) comparison
- Dataset: 7973 polymer samples with 36 features

## Reconstruction Performance (MAE)
- **PCA**: 0.0008
- **Autoencoder**: 0.2620
- **Winner**: PCA (by 31104.9%)

## Spearman Correlation Analysis
Correlations between corresponding components:
- Mean: 0.0587
- Max: 0.5216
- Min: -0.3127
- Std: 0.2212

## Variance Captured
- PCA: 33.9991 (explained variance ratio: 1.0000)
- Autoencoder: 30.3291
- Autoencoder captures 89.2% of PCA's variance

## Conclusions
1. **Reconstruction**: PCA performs better with 99.7% lower MAE
2. **Correlation**: The mean Spearman correlation of 0.0587 indicates weak alignment between methods
3. **Variance**: PCA captures more variance
