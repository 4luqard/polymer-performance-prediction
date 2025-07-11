# NeurIPS - Open Polymer Prediction 2025

A clean, organized solution for the [NeurIPS - Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) Kaggle competition.

## 🚀 Quick Start

1. **Generate submission:**
   ```bash
   python model.py
   ```

2. **Evaluate locally:**
   ```bash
   python cross_validation.py
   ```

3. **Track CV/LB correlation** (after submitting to Kaggle):
   ```bash
   python sync_cv_lb.py 0.158 "your description"
   ```

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
├── model.py              # 🎯 Main model - generates submission.csv
├── cross_validation.py   # 📊 Evaluation with proper train/val/test/holdout splits
├── sync_cv_lb.py        # 🔄 Track CV vs LB scores after submission
├── data/                # 📁 Competition data
├── output/              # 📤 Submission files
├── src/                 # 📦 Core modules (competition metric)
├── utils/               # 🛠️ Additional tools (see utils/README.md)
└── docs/                # 📚 Documentation
```

## Model Description

The current best model (`model.py`) uses:
- **Feature Engineering**: Extracts 43 molecular features from SMILES strings without external dependencies
- **Algorithm**: Separate Ridge regression models for each target property
- **Data**: Combines main training data with 4 supplementary datasets (~17k samples total)
- **Key Innovation**: Trains on only non-missing values for each target (avoiding imputation noise)

### Key Features Extracted:
- Atom counts (C, O, N, S, F, Cl, Br, etc.)
- Bond types (single, double, triple, aromatic)
- Structural features (rings, branches, chiral centers)
- Functional groups (carbonyl, hydroxyl, ether, amine, etc.)
- Polymer-specific features (polymer ends, chain length estimates)
- Derived features (flexibility score, heteroatom ratio, etc.)

### Current Performance:
- **MultiOutput Model**: CV: 0.0122, Holdout: 0.0118, LB: 0.158
- **Separate Models**: CV: 0.0709, Holdout: 0.0616, LB: 0.081 ✨
- **Best LB**: 0.081 (using separate Ridge models per target)

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

## Key Findings

1. **Extreme Data Sparsity**: 
   - 0 samples have all 5 targets
   - Only FFV has 46.5% coverage, others 3-4%
   - Almost no target overlap (Tg and FFV share only 1 sample)

2. **Separate Models Work Better**:
   - Training individual models per target improved LB from 0.158 to 0.081
   - Holdout score (0.0616) is a better LB predictor than CV score
   - Gap reduced from 13x to 1.3x (holdout vs LB)

3. **Target-Specific Training**:
   - Each model trains only on available data for that target
   - Uses different regularization (alpha) per target based on data availability
   - Avoids noise from imputed values

## Requirements

The model uses only standard libraries available in Kaggle notebooks:
- pandas
- numpy  
- scikit-learn
- re (built-in)

No external dependencies or internet connection required during submission.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.