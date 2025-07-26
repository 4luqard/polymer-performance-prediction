# NeurIPS Open Polymer Prediction 2025

This repository contains code and analysis for the NeurIPS 2025 Open Polymer Prediction competition.

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
├── data/                      # Competition data
│   └── raw/                   # Original data files
│       ├── train.csv          # Training data with partial targets
│       ├── test.csv           # Test data (SMILES only)
│       └── train_supplement/  # Additional training datasets
├── src/                       # Source code modules
├── utils/                     # Utility functions
│   ├── diagnostics/          # Model diagnostics tools
│   ├── test/                 # Testing utilities
│   └── tracking/             # Submission tracking tools
├── output/                    # Model outputs and submissions
├── model.py                   # Main model implementation
└── findings.md               # Research findings and investigations
```

## Quick Start

1. Install dependencies:
```bash
pip install --user pandas numpy scikit-learn rdkit torch
```

2. Download competition data (requires Kaggle API):
```bash
kaggle competitions download -c neurips-open-polymer-prediction-2025 -p data/
```

3. Run the model:
```bash
python model.py
```

## Missing Value Patterns

- Each polymer typically has only 1 out of 5 target values present
- Missing patterns show strong correlations (e.g., Density & Rg almost always missing/present together)
- Missing values are due to measurement limitations, not intentional design

## Documentation Structure

The project documentation is organized by category for easy navigation:

- `docs/polymer_properties.md` - Detailed information about the 5 target properties (Tg, FFV, Tc, Density, Rg)
- `docs/smiles_notation.md` - Guide to understanding and analyzing polymer SMILES notation
- `docs/crystallinity_analysis.md` - Research findings on polymer crystallinity and its relationship to properties
- `docs/property_relationships.md` - Analysis of interdependencies between polymer properties

For specific topics:
- **Understanding target properties?** → See `docs/polymer_properties.md`
- **Working with SMILES?** → See `docs/smiles_notation.md`
- **Crystallinity questions?** → See `docs/crystallinity_analysis.md`
- **Property correlations?** → See `docs/property_relationships.md`