# NeurIPS Open Polymer Prediction 2025

A machine learning solution for predicting polymer properties from SMILES molecular representations.

## Competition Overview

This project tackles the NeurIPS 2025 Open Polymer Prediction challenge, which involves predicting 5 key polymer properties:
- **Tg**: Glass transition temperature (K)
- **FFV**: Fractional free volume (dimensionless)
- **Tc**: Crystallization temperature (K)  
- **Density**: Material density (g/cm³)
- **Rg**: Radius of gyration (Å)

## Key Features

### Advanced Feature Engineering
- **52 molecular features** extracted from SMILES representations
- **Target-specific feature selection** for optimal performance
- **SMILES pattern extraction** from repeating polymer units
- Custom features including:
  - Molecular weight estimation
  - Backbone bonds counting
  - Average bond length calculation
  - Radius of gyration estimation
  - Van der Waals volume
  - Density estimation

### Data Processing
- **Automatic deduplication** of training data (removes ~6,600 duplicate SMILES)
- **Sparse feature representation** for memory efficiency
- **Full dataset preprocessing** before CV (uses ~17,000 samples)
- Handles missing values effectively (~91% of samples have only 1 target)

### Model Architecture
- **LightGBM gradient boosting** for non-linear relationships
- Separate models for each target property
- Integrates supplementary datasets for improved training
- Optimized preprocessing pipeline for efficiency

### Performance
- Cross-validation score: **0.0541** (competition metric)
- No dimensionality reduction (tested PCA/Autoencoder but degrades performance)
- Multi-seed validation for robust evaluation
- Improved index alignment for accurate predictions

## Project Structure

```
├── data/                    # Competition data
│   └── raw/                 
│       ├── train.csv        # Training data with partial targets
│       ├── test.csv         # Test data for predictions
│       └── train_supplement/# Additional training datasets
├── src/                     # Source code modules
│   ├── competition_metric.py# Official competition metric
│   └── diagnostics/         # CV diagnostics tools
├── scripts/                 # Utility scripts
│   ├── submit_to_kaggle.py  # Kaggle submission script
│   └── visualize_lgbm_trees.py # Tree visualization
├── output/                  # Model outputs
│   └── submission.csv       # Competition submission file
├── model.py                 # Main model implementation
├── data_processing.py       # Data loading and feature engineering
├── cv.py                    # Cross-validation functions
├── config.py                # Model configuration
├── FEATURES.md              # Detailed feature documentation
└── README.md                # This file
```

## Usage

### Training and Prediction
```bash
# Run full training and generate submission
python model.py

# Run cross-validation only
python model.py --cv

# Use Ridge model instead of LightGBM
python model.py --model ridge

# Exclude supplementary datasets
python model.py --no-supplement
```

### Feature Importance
See [FEATURES.md](FEATURES.md) for detailed feature rankings and descriptions.

## Requirements
- Python 3.8+
- pandas
- numpy 
- scikit-learn
- lightgbm
- kaggle (for submissions)

## Model Details

### LightGBM Parameters
- Objective: regression_l1 (MAE)
- Max depth: -1 (no limit)
- Num leaves: 31
- Estimators: 200
- Learning rate: 0.1
- Feature fraction: 0.9
- Bagging fraction: 0.8

### Data Processing Pipeline
1. Load all data sources (train, test, supplementary)
2. Remove duplicate SMILES (prioritizing samples with more targets)
3. Extract 52 molecular features from SMILES
4. Apply preprocessing (scaling, optional dimensionality reduction)
5. Train separate models for each target property

### Feature Selection Strategy
Different atom count features are selected for each target based on importance:
- **Tg**: num_C, num_n
- **FFV**: num_S, num_n
- **Tc**: num_C, num_S
- **Density**: num_Cl, num_Br
- **Rg**: num_F, num_Cl

## Results Summary

| Model Configuration | CV Score |
|-------------------|----------|
| LightGBM (current) | 0.0541   |
| LightGBM (no preprocessing) | 0.0542   |
| Ridge (no PCA)    | 0.0674   |
| LightGBM + PCA    | 0.0598   |
| Ridge + PCA       | 0.0687   |

## Recent Improvements
- Fixed index alignment bug between preprocessed features and target masks
- Implemented efficient deduplication strategy
- Moved preprocessing before CV for better efficiency
- Added SMILES pattern extraction from repeating units
- Optimized sparse feature representation

## License
MIT License - see [LICENSE](LICENSE) file for details.