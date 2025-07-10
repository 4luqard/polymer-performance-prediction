# NeurIPS - Open Polymer Prediction 2025

This repository contains a baseline Ridge regression model for the [NeurIPS - Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) Kaggle competition.

## Competition Overview

The goal is to predict five polymer properties from SMILES molecular representations:
- **Tg**: Glass transition temperature
- **FFV**: Fractional free volume
- **Tc**: Crystallization temperature
- **Density**: Material density
- **Rg**: Radius of gyration

The competition uses a weighted Mean Absolute Error (wMAE) metric that:
- Normalizes each property's error by its min-max range
- Weights properties by the inverse square root of valid samples
- Handles missing values marked as -9999

## Project Structure

```
neurips-open-polymer-prediction-2025/
├── model.py           # Main model for Kaggle submission
├── test/
│   └── test_model.py  # Local tests and validation
├── data/
│   ├── raw/          # Competition data (train.csv, test.csv, etc.)
│   └── processed/    # Processed datasets
├── notebooks/        # Jupyter notebooks for exploration
├── models/           # Saved model artifacts
├── output/           # Predictions and submissions
└── src/              # Additional source code
```

## Model Description

The baseline model (`model.py`) uses:
- **Feature Engineering**: Extracts 45+ molecular features from SMILES strings without external dependencies
- **Algorithm**: Ridge regression with multi-output wrapper
- **Data**: Combines main training data with 4 supplementary datasets (~17k samples total)
- **Validation**: 5-fold cross-validation for quick performance feedback

### Key Features Extracted:
- Atom counts (C, O, N, S, F, Cl, Br, etc.)
- Bond types (single, double, triple, aromatic)
- Structural features (rings, branches, chiral centers)
- Functional groups (carbonyl, hydroxyl, ether, amine, etc.)
- Polymer-specific features (polymer ends, chain length estimates)
- Derived features (flexibility score, heteroatom ratio, etc.)

### Baseline Performance (5-fold CV):
- **Competition Metric (wMAE)**: TBD (run model.py to calculate)
- Individual property scores shown during cross-validation

## Usage

### For Kaggle Submission

The `model.py` script is designed to run directly in a Kaggle notebook without any modifications or internet access:

```python
# Simply run in a Kaggle notebook
!python model.py
```

The script will:
1. Load all training data (including supplementary datasets)
2. Extract molecular features from SMILES
3. Perform 5-fold cross-validation for baseline performance
4. Train a Ridge regression model on full dataset
5. Generate predictions for test set
6. Save submission to `/kaggle/working/submission.csv`

### Local Development

For local testing and development:

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run cross-validation only (quick feedback)
python model.py --cv-only

# Run full model training and prediction
python model.py

# Run tests
python test/test_model.py
```

## Data

The competition provides:
- `train.csv`: 7,973 training samples with SMILES and target values
- `test.csv`: 3 test samples for prediction
- `train_supplement/`: 4 additional datasets with partial target values
  - `dataset1.csv`: 874 samples
  - `dataset2.csv`: 7,208 samples
  - `dataset3.csv`: 46 samples
  - `dataset4.csv`: 862 samples

## Testing

Run the test suite to validate:
- Feature extraction functionality
- Cross-validation functionality
- Model training pipeline
- Submission format compliance

```bash
python test/test_model.py
```

## Requirements

The model uses only standard libraries available in Kaggle notebooks:
- pandas
- numpy  
- scikit-learn
- re (built-in)

No external dependencies or internet connection required during submission.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.