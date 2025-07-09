# polymer-performance-prediction

This repository provides a minimal baseline for the **NeurIPS - Open Polymer Prediction 2025** Kaggle competition.

The `baseline.py` script trains a ridge regression model using only numeric features and
creates a submission file.

## Usage

```bash
python baseline.py --train path/to/train.csv --test path/to/test.csv --output submission.csv
```

By default it expects `id` and `target` columns. The predictions will be written to
`submission.csv` in the required Kaggle format.

