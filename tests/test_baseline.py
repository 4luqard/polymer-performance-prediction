import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from baseline import train_ridge_and_predict


def test_ridge_baseline_creates_prediction_file(tmp_path):
    """Given simple numeric data, predictions file is created with expected values."""
    train_df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "feature1": [1, 2, 3, 4],
        "feature2": [0.1, 0.2, 0.3, 0.4],
    })
    train_df["target"] = 2 * train_df["feature1"] + 0.5 * train_df["feature2"]

    test_df = pd.DataFrame({
        "id": [5, 6],
        "feature1": [5, 6],
        "feature2": [0.5, 0.6],
    })

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    output_path = tmp_path / "pred.csv"

    preds = train_ridge_and_predict(
        train_path=str(train_path),
        test_path=str(test_path),
        output_path=str(output_path),
        alpha=0.0,  # makes linear regression for deterministic results
    )

    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(test_df)
    np.testing.assert_allclose(
        loaded["target"],
        2 * test_df["feature1"] + 0.5 * test_df["feature2"],
        rtol=1e-5,
    )
    assert list(loaded.columns) == ["id", "target"]


def test_ridge_baseline_ignores_non_numeric_columns(tmp_path):
    """Given categorical columns, they are ignored during training."""
    train_df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "category": ["a", "b", "c", "d"],
    })
    train_df["target"] = 3 * train_df["feature1"]

    test_df = pd.DataFrame({
        "id": [5, 6],
        "feature1": [5.0, 6.0],
        "category": ["e", "f"],
    })

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    output_path = tmp_path / "pred.csv"
    preds = train_ridge_and_predict(
        train_path=str(train_path),
        test_path=str(test_path),
        output_path=str(output_path),
        alpha=0.0,
    )
    expected = 3 * test_df["feature1"]
    np.testing.assert_allclose(preds["target"], expected, rtol=1e-5)

