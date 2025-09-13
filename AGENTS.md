# AGENTS.md - Project Documentation

## Project Overview
**Competition**: NeurIPS Open Polymer Prediction 2025
**Type**: Kaggle Code Competition (offline execution required)
**Task**: Multi-target regression to predict polymer properties from SMILES molecular representations

## Problem Description
- **Input**: SMILES molecular representations (text strings describing polymer structure)
- **Output**: 5 polymer properties predictions
  - Tg: Glass transition temperature (K)
  - FFV: Fractional free volume (dimensionless)
  - Tc: Crystallization temperature (K)
  - Density: Material density (g/cm³)
  - Rg: Radius of gyration (Å)
- **Challenge**: Significant missing values in training data (~91% samples have only 1 target value)
- **Evaluation Metric**: Weighted Mean Absolute Error (wMAE) with property-specific weights

## Project Structure
```
neurips-open-polymer-prediction-2025/
├── data/                       # Competition datasets
│   └── raw/                    # Original data files
├── src/                        # Source code modules
│   ├── competition_metric.py   # Custom metric implementation
│   └── diagnostics/            # CV diagnostics tools
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── output/                     # Model outputs and submissions
├── notebooks/                  # Jupyter notebooks
├── residual_analysis/          # Analysis results
├── model.py                    # Main model implementation
├── data_processing.py          # Feature engineering pipeline
├── cv.py                       # Cross-validation logic
├── config.py                   # Configuration parameters
```

## Technology Stack
### Core Libraries
- **pandas**: Data manipulation (DataFrames)
- **numpy**: Numerical operations
- **scikit-learn**: ML utilities and preprocessing
  - StandardScaler for normalization
  - Ridge regression for baseline models
  - PCA for dimensionality reduction experiments
  - SimpleImputer for missing values
- **lightgbm**: Gradient boosting models (main algorithm)
- **keras/tensorflow**: Deep learning experiments (autoencoder)
- **tqdm**: Progress tracking

### Key Dependencies
- Python 3.11+ (based on venv directory)
- Standard ML stack (sklearn, pandas, numpy)
- Gradient boosting framework (lightgbm)
- Deep learning framework (keras)

## Code Style and Conventions
### File Organization
- Main execution files at root level (model.py, cv.py)
- Utility functions in data_processing.py
- Source modules in src/ directory
- Configuration centralized in config.py

### Naming Conventions
- Snake_case for functions and variables
- ALL_CAPS for constants (e.g., LIGHTGBM_PARAMS)
- Descriptive function names (e.g., extract_smiles_features, calculate_wMAE)

### Code Patterns
- Docstrings for all major functions
- Type hints in critical functions
- Defensive programming with try-except blocks
- Extensive logging and warnings
- Modular design with clear separation of concerns

## Model Architecture
### Feature Engineering Pipeline
- 52+ molecular features extracted from SMILES
- Custom feature extractors for polymer-specific properties
- Target-specific feature selection
- Automatic deduplication of training samples

### Model Strategy
- Separate LightGBM models per target property
- Gradient boosting with MAE objective
- No dimensionality reduction (degrades performance)
- Cross-validation for hyperparameter tuning
- Multi-seed validation for robustness

### Configuration (config.py)
```python
LIGHTGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
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
```

## Execution Environment
### Local Development
- Command: `python3 model.py`
- CV mode: `python3 model.py --cv` or `--cv-only`
- Uses local data paths and full preprocessing

### Kaggle Competition
- Offline execution (no internet access)
- Kaggle notebook environment constraints
- Automatic environment detection in code
- Submission format: CSV with all 5 predictions per sample

## Key Implementation Details
### Data Handling
- Automatic environment detection (Kaggle vs local)
- Handles sparse data with ~91% missing values
- Deduplication removes ~6,600 duplicate SMILES
- Full dataset preprocessing before CV

### Performance Optimizations
- Sparse matrix representations for memory efficiency
- Batch processing for large datasets
- Optimized feature extraction pipeline
- Caching of preprocessed features

### Validation Strategy
- K-fold cross-validation
- Competition metric (wMAE) for evaluation
- Residual analysis for model diagnostics
- Multi-seed runs for stability testing

## Current Performance
- Cross-validation score: 0.0541 (competition metric)
- Improved from baseline through feature engineering
- Stable performance across different random seeds

## Important Notes
- Code must run offline in Kaggle environment
- All dependencies must be pre-installed
- Submission file must be named "submission.csv"
- Test set requires predictions for all 5 targets
- Missing value patterns show strong correlations between targets