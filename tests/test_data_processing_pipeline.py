import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_processing as dp


@pytest.fixture
def pipeline():
    return dp.DataProcessingPipeline()


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("*SCCCCC*", 6),
        ("*OCC1C(C1)C*", 5),
        ("*CC(CC)CC*", 4),
    ],
)
def test_calculate_main_branch_atoms(pipeline, smiles, expected):
    assert pipeline.calculate_main_branch_atoms(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("*CCCCCC*", 5),
        ("*CC(C)CC*", 3),
        ("*CC(=O)CC*", 3),
    ],
)
def test_calculate_backbone_bonds(pipeline, smiles, expected):
    assert pipeline.calculate_backbone_bonds(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("*CCO*", 1.485),
        ("*C=C*", 1.34),
        ("*C#N*", 1.2),
    ],
)
def test_calculate_average_bond_length(pipeline, smiles, expected):
    assert pipeline.calculate_average_bond_length(smiles) == expected


def test_extract_molecular_features_basic(pipeline):
    features = pipeline.extract_molecular_features("*CCO*", False)
    assert features["length"] == 5
    assert features["num_C"] == 2
    assert features["num_O"] == 1


def test_prepare_features_single(pipeline):
    df = pd.DataFrame({"SMILES": ["*CCO*"], "new_sim": [True]})
    features_df = pipeline.prepare_features(df)
    assert features_df.loc[0, "new_sim"] == 1
    assert features_df.loc[0, "length"] == 5


def test_apply_pca(pipeline):
    X_train = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    X_test = pd.DataFrame([[7, 8, 9]], columns=["a", "b", "c"])

    imputer = dp.SimpleImputer(strategy="constant", fill_value=0)
    scaler = dp.StandardScaler()

    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    X_train_p, X_test_p = pipeline.apply_pca(X_train_scaled, X_test_scaled, 0.95)

    expected_train = np.array([[-0.70710678], [0.70710678]])
    expected_test = np.array([[2.12132034]])

    assert np.allclose(X_train_p, expected_train)
    assert np.allclose(X_test_p, expected_test)


def test_apply_pls(pipeline):
    X_train = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 10]], columns=["a", "b", "c"])
    X_test = pd.DataFrame([[2, 3, 4]], columns=["a", "b", "c"])
    y_train = pd.DataFrame({"Tg": [1, 2, 3]})

    imputer = dp.SimpleImputer(strategy="constant", fill_value=0)
    scaler = dp.StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    X_train_p, X_test_p = pipeline.apply_pls(X_train_scaled, X_test_scaled, y_train, 2)

    expected_train = np.array([
        [-2.08544816, 0.04972642],
        [-0.06696393, -0.09495174],
        [2.15241209, 0.04522532],
    ])
    expected_test = np.array([[-1.41262008, 0.00150037]])

    assert np.allclose(X_train_p, expected_train)
    assert np.allclose(X_test_p, expected_test)


def test_apply_autoencoder(pipeline):
    X_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X_test = np.array([[7.0, 8.0, 9.0]])
    y_train = np.array([[0.0], [1.0]])

    train_encoded, test_encoded = pipeline.apply_autoencoder(
        X_train, X_test, y_train=y_train, latent_dim=2, epochs=1, batch_size=1
    )

    assert train_encoded.shape == (2, 2)
    assert test_encoded.shape == (1, 2)


def test_add_transformer_features(pipeline):
    pytest.importorskip("deepchem")

    train_df = pd.DataFrame({"SMILES": ["*CCO*", "*CCC*"], "new_sim": [True, True]})
    test_df = pd.DataFrame({"SMILES": ["*CCO*"], "new_sim": [True]})
    X_train = pipeline.prepare_features(train_df)
    X_test = pipeline.prepare_features(test_df)
    smiles_train = train_df["SMILES"]
    smiles_test = test_df["SMILES"]
    y_train = pd.DataFrame({"Tg": [1.0, 2.0]})

    X_train_p, X_test_p = pipeline.preprocess_data(
        X_train,
        X_test,
        pca_variance_threshold=0.95,
        use_transformer=True,
        transformer_latent_dim=2,
        y_train=y_train,
        smiles_train=smiles_train,
        smiles_test=smiles_test,
        epochs=1,
    )

    assert X_train_p.shape[1] == 3  # 1 PCA component + 2 transformer dims
    assert X_test_p.shape[1] == 3
