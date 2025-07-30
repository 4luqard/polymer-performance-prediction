# NeurIPS Open Polymer Prediction 2025

This repository contains code for the NeurIPS 2025 Open Polymer Prediction competition using LightGBM models with target-specific feature selection.

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
│   └── visualization/         # Visualization scripts
│       └── visualize_trees.py # LightGBM tree visualization
├── utils/                     # Utility functions
│   ├── diagnostics/           # CV diagnostics (disabled)
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

- **Algorithm**: LightGBM with target-specific models
- **Features**: Molecular features extracted from SMILES including:
  - Atom counts and ratios
  - Structural features (rings, branches, bonds)
  - Functional group indicators
  - Molecular weight and complexity
- **Cross-validation**: Multi-seed CV with competition metric (MAE-based)
- **Score**: CV score of 0.0599 (+/- 0.0025)

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
cd scripts/visualization
python visualize_trees.py
```

## Key Features

- **Target-specific feature selection**: Different atom features selected for each target based on importance
- **Separate models**: Individual LightGBM model trained for each of the 5 targets
- **Missing value handling**: Models trained only on samples with available target values
- **Supplementary data**: Optional inclusion of additional training datasets

## Missing Value Patterns

- Each polymer typically has only 1 out of 5 target values present
- Missing patterns show strong correlations (e.g., Density & Rg almost always missing/present together)
- Missing values are due to measurement limitations, not intentional design