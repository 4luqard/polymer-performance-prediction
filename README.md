# polymer-performance-prediction

This repository provides a minimal baseline for the **NeurIPS - Open Polymer Prediction 2025** Kaggle competition.

The `baseline.py` script trains a ridge regression model using only numeric features and
creates a submission file.

## Usage

```bash
python baseline.py
```

When executed inside Kaggle, the script automatically reads the competition
data from `/kaggle/input/neurips-open-polymer-prediction-2025/` and writes the
submission file to `/kaggle/working/submission.csv`. If you want to supply
custom paths, use the `--train`, `--test` and `--output` options.

