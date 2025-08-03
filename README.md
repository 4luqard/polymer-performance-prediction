# NeurIPS Open Polymer Prediction 2025

This repository contains a LightGBM-based solution for the NeurIPS 2025 Open Polymer Prediction competition, featuring target-specific feature selection and comprehensive molecular feature engineering.

## Competition Overview

- **Task**: Multi-target regression to predict 5 polymer properties from SMILES molecular representations
- **Target Properties**:
  - **Tg**: Glass transition temperature (K)
  - **FFV**: Fractional free volume (dimensionless)
  - **Tc**: Crystallization temperature (K)
  - **Density**: Material density (g/cm³)
  - **Rg**: Radius of gyration (Å)

## Key Challenge

The training data has significant missing values (~91% of samples have only 1 target value), making this a challenging multi-target prediction problem with partial labels.

## Project Structure

```
├── model.py                   # Main model implementation with LightGBM
├── cv.py                      # Cross-validation functions
├── config.py                  # LightGBM configuration parameters
├── src/                       # Source code modules
│   └── competition_metric.py  # Competition evaluation metric
├── scripts/                   # Utility scripts
│   ├── submit_to_kaggle.py    # Kaggle submission script
│   └── visualize_lgbm_trees.py # LightGBM tree visualization
├── utils/                     # Utility functions
│   ├── diagnostics/           # CV diagnostics utilities
│   └── test/                  # Model testing utilities
├── data/                      # Competition data
│   └── raw/                   # Original data files
│       ├── train.csv          # Training data with partial targets
│       ├── test.csv           # Test data (SMILES only)
│       └── train_supplement/  # Additional training datasets
└── output/                    # Model outputs
    ├── submission.csv         # Competition submission
    └── lgbm_trees/           # Tree visualizations
```

## Model Approach

- **Algorithm**: LightGBM gradient boosting with target-specific models
- **Features**: Comprehensive molecular features extracted from SMILES:
  - Atom counts by type (C, N, O, S, F, Cl, Br, I, P) and aromaticity
  - Bond counts (single, double, triple, aromatic)
  - Structural features (rings, branches, chiral centers)
  - Functional group indicators (carbonyl, ether, amine, sulfone, ester, amide)
  - Molecular weight and complexity metrics
  - Van der Waals volume and density estimation
  - Main branch atom analysis
  - Polymer-specific patterns (phenyl, cyclohexyl groups)
  - FFV (Fractional Free Volume) estimation
  - Dataset source indicator (new_sim feature)
- **Cross-validation**: Multi-seed CV (seeds: 42, 123, 456) with competition metric
- **Current Performance**: CV score of 0.0540 (+/- 0.0021)

## Quick Start

1. Install dependencies:
```bash
pip install --user pandas numpy scikit-learn lightgbm rdkit
```

2. Download competition data (requires Kaggle API):
```bash
kaggle competitions download -c neurips-open-polymer-prediction-2025 -p data/
cd data/raw && unzip neurips-open-polymer-prediction-2025.zip
```

3. Run the model:
```bash
# For training and submission
python model.py

# For cross-validation only
python model.py --cv

# Without supplementary data
python model.py --no-supplement
```

4. Visualize decision trees:
```bash
python scripts/visualize_lgbm_trees.py
```

## Key Implementation Details

- **Target-specific feature selection**: Optimized atom features for each target:
  - Tg: num_C, num_n (aromatic nitrogen)
  - FFV: num_S, num_n
  - Tc: num_C, num_S
  - Density: num_Cl, num_Br
  - Rg: num_F, num_Cl
- **Separate models**: Individual LightGBM model trained for each target
- **Missing value handling**: Models trained only on samples with available target values
- **Supplementary datasets**: Integrates dataset1 from train_supplement (renaming TC_mean to Tc)
- **Feature engineering**: 49 total features including molecular descriptors and structural patterns

## Configuration

LightGBM parameters (defined in `config.py`):
- `objective`: regression
- `metric`: mae
- `boosting_type`: gbdt
- `max_depth`: -1 (no limit)
- `num_leaves`: 31
- `n_estimators`: 200
- `learning_rate`: 0.1
- `feature_fraction`: 0.9
- `bagging_fraction`: 0.8

## Missing Value Patterns

- Each polymer typically has only 1 out of 5 target values present
- Missing patterns show strong correlations (e.g., Density & Rg almost always missing/present together)
- Missing values are due to measurement limitations in molecular dynamics simulations