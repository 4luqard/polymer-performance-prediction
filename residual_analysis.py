#!/usr/bin/env python3

"""Residual analysis utilities for the autoencoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def residual_analysis(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    batch_size: int,
    epochs: int,
    random_state: int,
    verbose: int,
    output_path: str = "autoencoder_residuals.parquet",
) -> None:
    """Run residual analysis for an autoencoder model.

    This function splits the provided data into train/validation/test sets,
    fits the given ``model`` on the training portion, computes predictions for
    all splits, calculates residuals and writes them to ``output_path``.

    Parameters
    ----------
    model:
        Compiled Keras model to train.
    X:
        Feature matrix as a pandas DataFrame.
    y:
        Target matrix as a pandas DataFrame or numpy array.
    batch_size:
        Batch size used for training.
    epochs:
        Number of epochs to train for.
    random_state:
        Random seed for the train/validation/test split.
    verbose:
        Verbosity level passed to Keras ``fit`` and ``predict``.
    output_path:
        Path where the residual DataFrame will be stored.
    """

    # Create train/val/test split for residual analysis
    X_train_split, X_temp, y_train_split, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    X_val_split, X_test_split, y_val_split, y_test_split = train_test_split(
        X_temp, y_temp, test_size=0.30, random_state=random_state
    )

    # Train without validation_split for residual analysis
    model.fit(
        X_train_split,
        y_train_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(X_val_split, np.nan_to_num(y_val_split, nan=0.0)),
    )

    # Compute predictions for all splits
    train_pred = model.predict(X_train_split, verbose=verbose)
    val_pred = model.predict(X_val_split, verbose=verbose)
    test_pred = model.predict(X_test_split, verbose=verbose)

    # Calculate residuals (only for non-NaN values)
    train_residuals = np.where(np.isnan(y_train_split), np.nan, train_pred - y_train_split)
    val_residuals = np.where(np.isnan(y_val_split), np.nan, val_pred - y_val_split)
    test_residuals = np.where(np.isnan(y_test_split), np.nan, test_pred - y_test_split)

    # Create DataFrame with all information
    residual_data: dict[str, list | np.ndarray] = {
        "split": (
            ["train"] * len(X_train_split)
            + ["val"] * len(X_val_split)
            + ["test"] * len(X_test_split)
        )
    }

    # Add features
    X_all = np.vstack([X_train_split, X_val_split, X_test_split])
    for i, col in enumerate(X.columns):
        residual_data[col] = X_all[:, i]

    # Add targets and predictions
    y_all = np.vstack([y_train_split, y_val_split, y_test_split])
    pred_all = np.vstack([train_pred, val_pred, test_pred])
    residuals_all = np.vstack([train_residuals, val_residuals, test_residuals])

    num_targets = y.shape[1] if len(y.shape) > 1 else 1
    target_names = (
        ["Tg", "FFV", "Tc", "Density", "Rg"]
        if num_targets == 5
        else [f"target_{i}" for i in range(num_targets)]
    )

    for i, name in enumerate(target_names[:num_targets]):
        residual_data[f"{name}_actual"] = y_all[:, i]
        residual_data[f"{name}_pred"] = pred_all[:, i]
        residual_data[f"{name}_residual"] = residuals_all[:, i]

    # Save to parquet
    residual_df = pd.DataFrame(residual_data)
    residual_df.to_parquet(output_path, index=False)

    return None
