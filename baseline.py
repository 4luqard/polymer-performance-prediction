import argparse
from pathlib import Path

# Default paths for running inside Kaggle
DEFAULT_TRAIN_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
DEFAULT_TEST_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
DEFAULT_OUTPUT_PATH = "/kaggle/working/submission.csv"

import pandas as pd
from sklearn.linear_model import Ridge


def train_ridge_and_predict(train_path: str,
                            test_path: str,
                            output_path: str = "submission.csv",
                            target_column: str = "target",
                            id_column: str = "id",
                            alpha: float = 1.0) -> pd.DataFrame:
    """Train a ridge regression model and generate predictions.

    Only numeric feature columns present in both train and test are used.
    The resulting predictions are saved to ``output_path``.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in {target_column, id_column}]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    X_train = train_df[feature_cols]
    y_train = train_df[target_column]
    X_test = test_df[feature_cols]

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({id_column: test_df[id_column], target_column: predictions})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline for Open Polymer Prediction 2025")
    parser.add_argument("--train", default=DEFAULT_TRAIN_PATH, help="Path to training CSV")
    parser.add_argument("--test", default=DEFAULT_TEST_PATH, help="Path to test CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Where to write predictions")
    parser.add_argument("--target", default="target", help="Name of target column")
    parser.add_argument("--id", dest="id_column", default="id", help="Name of ID column")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    args = parser.parse_args()
    train_ridge_and_predict(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        target_column=args.target,
        id_column=args.id_column,
        alpha=args.alpha,
    )

