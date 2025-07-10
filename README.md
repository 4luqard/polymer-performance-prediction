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

### Current Performance:
- **Local CV**: 0.0122 (±0.0002)
- **Local Holdout**: 0.0118
- **Public LB**: 0.158 (~13x higher - see Key Findings)

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
   - Only FFV has 88% coverage, others <10%
   - Almost no target overlap (Tg and FFV share only 1 sample)

2. **CV/LB Gap (13x)**:
   - Local scores ~0.012, LB scores ~0.158
   - Likely due to test set distribution differences
   - Use `sync_cv_lb.py` to track correlation

3. **Target LB Scores** (based on current ratio):
   - LB 0.15 → Need CV 0.0114
   - LB 0.14 → Need CV 0.0106
   - LB 0.10 → Need CV 0.0076

## Requirements

The model uses only standard libraries available in Kaggle notebooks:
- pandas
- numpy  
- scikit-learn
- re (built-in)

No external dependencies or internet connection required during submission.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.