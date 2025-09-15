"""
Data processing functions for NeurIPS Open Polymer Prediction 2025
Extracted from model.py for better code organization
"""

import os
# Configure TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from rich import print
from extract_features import *

from rich import console
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve

console = console.Console(width=120)
from datetime import datetime

from rich.jupyter import print as show

def instrument(fn):
    def tracked_fn(*args, **kwargs):
        before = datetime.now()
        results = fn(*args, **kwargs)
        console.log(f"[bold]{fn.__name__}[/bold] took {datetime.now() - before}")
        return results

    return tracked_fn


import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LayerNormalization
from keras.layers import BatchNormalization as BN

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# Competition metric constants for autoencoder loss
MINMAX_DICT = {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}

def create_masked_competition_loss(num_targets):
    """Create a custom loss function based on the competition metric with masking for missing values."""
    def masked_competition_loss(y_true, y_pred):
        import tensorflow as tf
        
        # Create mask for non-NaN values (1 where valid, 0 where NaN/-10000.0)
        mask = tf.cast(tf.not_equal(y_true, -10000.0), tf.float32)
        
        # Calculate absolute errors
        abs_errors = tf.abs(y_pred - y_true)
        
        # Scale errors by property ranges
        target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg'][:num_targets]
        scaled_errors = []
        
        for i, prop in enumerate(target_names):
            if i < num_targets:
                min_val, max_val = MINMAX_DICT[prop]
                label_range = max_val - min_val
                # Scale the error for this property and apply mask
                scaled_error = abs_errors[:, i] / label_range
                masked_error = scaled_error * mask[:, i]
                scaled_errors.append(masked_error)
        
        # Stack errors and compute mean per sample
        if len(scaled_errors) > 0:
            stacked_errors = tf.stack(scaled_errors, axis=1)
            # Count valid values per sample
            valid_counts = tf.reduce_sum(mask, axis=1, keepdims=True)
            # Avoid division by zero
            valid_counts = tf.maximum(valid_counts, 1.0)
            # Mean of valid errors per sample
            sample_losses = tf.reduce_sum(stacked_errors, axis=1) / tf.squeeze(valid_counts, axis=1)
            return tf.reduce_mean(sample_losses)
        else:
            return tf.constant(0.0)
    
    return masked_competition_loss

def extract_molecular_features(smiles, rpt):
    """Extract features from SMILES string without external libraries"""
    features = {}
    elements = ['B', 'C', 'N', 'O', 'F', 'Na', 'Si', 'P', 'S',
                'Cl', 'Ca', 'Ge', 'Se', 'Br', 'Cd', 'Sn', 'Te']
    aromatic_atoms = ['b', 'c', 'n', 'o', 'p', 's']

    # Basic string features
    features['length'] = len(smiles) # No units

    # Number of atoms on the longest chain of atoms
    features['longest_chain_atom_count'] = longest_chain_atom_count(smiles) # No units

    # Ring features
    features['num_fused_rings'] = num_fused_rings(smiles) # No units
    features['num_rings'] = num_rings(smiles) # No units

    # Atom count features, excluding Hydrogen
    features |= element_count(smiles, elements) # No units
    features |= element_count(smiles, aromatic_atoms) # No units

    # Number of Hydrogens
    features['H'] = hydrogen_amount(smiles) # No units

    # Atom count derived features
    features['heavy_atom_amount'] = heavy_atom_amount(features) # No units
    features['heteroatom_amount'] = heteroatom_amount(features) # No units

    # Molecular weights
    features['molecular_weight'] = molecular_weight(features) # grams per mole

    # Van der Waals radius, molecular
    features['vdw_volume'] = vdw_volume(features) # centimeters cubed per mole

    # Stereochemistry
    features['num_tetrahedral_carbon'] = num_tetrahedral_carbon(smiles)

    # Target estimates
    features['density_estimate'] = density_estimate(features) # grams per centimeter cubed

    # Count bonds
    features['num_single_bonds'] = smiles.count('-')
    features['num_double_bonds'] = smiles.count('=')
    features['num_triple_bonds'] = smiles.count('#')
    features['num_aromatic_bonds'] = smiles.count(':')
    
    # Count structural features
    features['num_branches'] = smiles.count('(')

    # Polymer-specific features
    features['has_polymer_end'] = int('*' in smiles)
    features['num_polymer_ends'] = smiles.count('*')
    features['has_polymer_end_branch'] = int('(*)' in smiles)
    
    # Functional group patterns
    features['has_carbonyl'] = int('C(=O)' in smiles or 'C=O' in smiles)
    # features['has_cfthree'] = int('C(F)(F)F' in smiles)
    # features['has_hydroxyl'] = int('OH' in smiles or 'O[H]' in smiles)  # Removed - may cause overfitting
    features['has_ether'] = int('COC' in smiles or 'cOc' in smiles)
    features['has_amine'] = int('N' in smiles)
    features['has_sulfone'] = int('S(=O)(=O)' in smiles)
    features['has_ester'] = int('C(=O)O' in smiles or 'COO' in smiles)
    features['has_amide'] = int('C(=O)N' in smiles or 'CON' in smiles)
    
    # features['carbon_percent'] = (features['num_C'] + features['num_c']) / (features['heavy_atom_count'] + features['ion_count'])
    # features['aromatic_ratio'] = features['num_aromatic_atoms'] / (features['heavy_atom_count'] + features['ion_count'])
    
    # Flexibility indicators
    # features['rotatable_bond_estimate'] = max(0, features['num_single_bonds'] - features['num_rings'])
    # features['flexibility_score'] = features['rotatable_bond_estimate'] / max(features['heavy_atom_count'], 1)

    # Size and complexity
    features['molecular_complexity'] = (features['num_rings'] + features['num_branches'] + 
                                       features['num_tetrahedral_carbon'])
    
    # Additional polymer-specific patterns
    # Phenyl: aromatic 6-membered ring patterns
    features['has_phenyl'] = int(bool(re.search(r'c1ccccc1|c1ccc.*cc1', smiles)))
    features['has_cyclohexyl'] = int('C1CCCCC1' in smiles)
    features['has_methyl'] = int(bool(re.search(r'C(?![a-zA-Z])', smiles)))
    segments = [x for x in re.split(r'[\(\)]', smiles) if x]
    features['chain_length_estimate'] = max(len(x) for x in segments) if segments else 0
    
    # Bridge: multiple different ring numbers
    ring_numbers = set(re.findall(r'[0-9]', smiles))
    features['has_bridge'] = int(len(ring_numbers) >= 2)
    
    # Main branch atom count
    features['main_branch_atoms'] = calculate_main_branch_atoms(smiles)
    
    # Main branch atom ratio (main branch atoms / total heavy atoms)
    features['main_branch_atom_ratio'] = round(features['main_branch_atoms'] / max(features['heavy_atom_amount'], 1), 3)

    # Backbone bonds count
    features['backbone_bonds'] = calculate_backbone_bonds(smiles)
    
    # Average bond length
    features['avg_bond_length'] = calculate_average_bond_length(smiles)
    
    # Rg estimation using formula: Rg = sqrt((N × b²) / 6)
    # where N is backbone bond count and b is average bond length
    N = features['backbone_bonds']
    b = features['avg_bond_length']
    if N > 0 and b > 0:
        rg_squared = (N * b * b) / 6.0
        features['rg_estimate'] = round(math.sqrt(rg_squared), 3)
    else:
        features['rg_estimate'] = 0.0
    
    if rpt:
        feature_names = list(features.keys())
        for feature in feature_names:
            features[f'{feature}_rpt'] = features[feature]
            del features[feature]

    return features


@instrument
def prepare_features(df):
    """Convert SMILES to molecular features"""
    print("Extracting molecular features...")
    features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        if '*' in row['SMILES']:
            row['SMILES_rpt'] = row['SMILES'].split('*')[1]
            for col in ['SMILES', 'SMILES_rpt']:
                rpt = False if col == 'SMILES' else True
                features = extract_molecular_features(row[col], rpt)
                # Add new_sim feature
                features['new_sim'] = int(row['new_sim'])  # Convert boolean to int (0 or 1)
                if rpt:
                    features_rpt = features
                else:
                   features_full = features
            features_list.append(features_full | features_rpt)
        else:
            features = extract_molecular_features(row['SMILES'], False)
            features['new_sim'] = int(row['new_sim'])
            features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    return features_df


@instrument
def apply_autoencoder(X_train, X_test=None, y_train=None, random_state=42):
    """
    Apply supervised encoder for dimensionality reduction.

    Args:
        X_train: Training+Validation data (n_samples, n_features)
        X_test: Test data (optional)
        y_train: Target values for supervised learning
        random_state: Random seed for reproducibility

    Returns:
        X_tr_encoded, X_val_encoded, X_test_encoded

    """
    import numpy as np
    import random
    import os
    from sklearn.model_selection import train_test_split

    if y_train[0] is None or y_train[1] is None:
        raise ValueError("y_train must be provided for supervised autoencoder")

    np.random.seed(random_state)
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)

    import tensorflow as tf
    tf.random.set_seed(random_state)
    if hasattr(tf, 'set_random_seed'):
        tf.set_random_seed(random_state)

    verbose = 1
    latent_dim = 32
    epochs = 25
    batch_size = 64
    drop_rate = 0.1

    input_dim = X_train[0].shape[1]
    encoder = Sequential([
        LayerNormalization(),

        Dense(latent_dim*32, activation='leaky_relu', input_shape=(input_dim,)),
        Dropout(drop_rate),
        LayerNormalization(),

        Dense(latent_dim*16, activation='leaky_relu'),
        Dropout(drop_rate),
        LayerNormalization(),

        Dense(latent_dim*8, activation='leaky_relu'),
        Dropout(drop_rate),
        LayerNormalization(),

        Dense(latent_dim*4, activation='leaky_relu'),
        Dropout(drop_rate),
        LayerNormalization(),

        Dense(latent_dim*2, activation='leaky_relu'),
        Dropout(drop_rate),
        LayerNormalization(),

        Dense(latent_dim, activation='linear'),
    ])

    decoder = Sequential([
        LayerNormalization(),

        Dense(latent_dim*2, activation='leaky_relu', input_shape=(latent_dim,)),
        LayerNormalization(),

        Dense(latent_dim*4, activation='leaky_relu'),
        Dropout(drop_rate),
        LayerNormalization(),

        Dense(latent_dim*8, activation='leaky_relu'),
        LayerNormalization(),

        Dense(latent_dim*16, activation='leaky_relu'),
        LayerNormalization(),

        Dense(latent_dim*32, activation='leaky_relu'),
        LayerNormalization(),

        Dense(input_dim, activation='linear'),
    ], name='decoded')

    input_layer = Input(shape=(input_dim,))
    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    predictions = Sequential([
        # BN(),
        LayerNormalization(),
        Dense(int(latent_dim/4), activation='leaky_relu', input_shape=(latent_dim,)),
        LayerNormalization(),
        # BN(),
        Dense(input_dim, activation='linear')
    ], name="predictions")
    predictions = predictions(encoded)

    model = Model(inputs=input_layer, outputs=[decoded, predictions])
    
    # Use custom loss with masking based on competition metric
    model.compile(optimizer=Adam(learning_rate=3e-4), loss={"decoded": 'mse', "predictions": 'mae'})

    # Original training with validation_split
    model.fit(
        X_train[0],
        [X_train[0], y_train[0]],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(X_train[1], [X_train[1], y_train[1]])
    )

    X_train_preds = model.predict(X_train[0], verbose=verbose)

    encoder_model = Model(inputs=input_layer, outputs=encoded)

    X_tr_encoded = encoder_model.predict(X_train[0], verbose=verbose)
    X_val_encoded = encoder_model.predict(X_train[1], verbose=verbose)
    X_test_encoded = encoder_model.predict(X_test, verbose=verbose)

    return X_tr_encoded, X_val_encoded, X_test_encoded


def preprocess_data(X_train, X_test, use_autoencoder=False, y_train=None):
    """
    Preprocess training and test data including:
    - Dropping columns with all NaN values
    - Imputing missing values
    - Scaling features
    - Applying dimensionality reduction (autoencoder)
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        use_autoencoder: Whether to use autoencoder for dimensionality reduction
        autoencoder_latent_dim: Number of latent dimensions for autoencoder
        y_train: Target values DataFrame (required for PLS)
    
    Returns:
        Tuple of (X_train_preprocessed, X_test_preprocessed)
    """
    # print("\n=== Preprocessing Data ===")

    # Impute missing values with zeros (fit on train, transform both)
    # print("Imputing missing values with zeros...")
    # imputer = SimpleImputer(strategy='constant', fill_value=0).set_output(transform='pandas')
    # X_train_imputed = X_train #imputer.fit_transform(X_train)
    # X_test_imputed = X_test #imputer.transform(X_test)

    # Drop constant features
    cons_cols = X_train[0].nunique() > 1
    X_train[0] = X_train[0].loc[:, cons_cols]
    X_train[1] = X_train[1].loc[:, cons_cols]
    X_test = X_test.loc[:, cons_cols]

    # Scale features (fit on train, transform both)
    # print("Scaling features...")
    global_scaler = StandardScaler().set_output(transform='pandas')
    X_train[0] = global_scaler.fit_transform(X_train[0])
    X_train[1] = global_scaler.transform(X_train[1])
    X_test = global_scaler.transform(X_test)

    # Apply dimensionality reduction if enabled
    if use_autoencoder:
        # print(f"Applying supervised autoencoder: {X_train_scaled.shape[1]} features -> {autoencoder_latent_dim} dimensions")
        X_train_reduced, X_val_reduced, X_test_reduced = apply_autoencoder(X_train, X_test, y_train=y_train)
        X_tr_preprocessed = pd.DataFrame(X_train_reduced)
        X_val_preprocessed = pd.DataFrame(X_val_reduced)
        X_test_preprocessed = pd.DataFrame(X_test_reduced)

    # print(f"Final dimensions: Train {X_train_preprocessed.shape}, Test {X_test_preprocessed.shape}")
    
    return X_tr_preprocessed, X_val_preprocessed, X_test_preprocessed

@instrument
def load_competition_data(train_path, test_path, supp_paths=None, use_supplementary=True):
    """
    Load competition data including main training, test, and optional supplementary datasets
    
    Args:
        train_path: Path to main training data
        test_path: Path to test data
        supp_paths: List of paths to supplementary datasets
        use_supplementary: Whether to include supplementary datasets
    
    Returns:
        Tuple of (train_df, test_df)
    """
    print("Loading training data...")
    
    # Load main training data
    train_df = pd.read_csv(train_path)
    # Add new_sim flag - True for main dataset
    train_df['new_sim'] = True
    print(f"Main training data shape: {train_df.shape}")
    
    if use_supplementary and supp_paths:
        # Load and combine supplementary datasets
        print("\nLoading supplementary datasets...")
        all_train_dfs = [train_df]
        
        for supp_path in supp_paths:
            try:
                supp_df = pd.read_csv(supp_path)
                
                # Handle dataset1 with Tc_mean column
                if 'dataset1.csv' in supp_path and 'TC_mean' in supp_df.columns:
                    # Rename TC_mean to Tc
                    supp_df = supp_df.rename(columns={'TC_mean': 'Tc'})
                    print(f"Renamed TC_mean to Tc in dataset1")
                
                # Add id column if missing
                if 'id' not in supp_df.columns:
                    # Create synthetic ids for supplementary data
                    supp_df['id'] = [f'supp_{supp_path.split("/")[-1].split(".")[0]}_{i}' 
                                     for i in range(len(supp_df))]
                
                # Add new_sim flag - False for supplementary datasets
                supp_df['new_sim'] = False
                print(f"Loaded {supp_path}: {supp_df.shape}")
                all_train_dfs.append(supp_df)
            except Exception as e:
                print(f"Could not load {supp_path}: {e}")
        
        # Combine all training data
        train_df = pd.concat(all_train_dfs, ignore_index=True)
        print(f"\nCombined training data shape: {train_df.shape}")
    else:
        print("\n** Using ONLY main training data (no supplementary datasets) **")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(test_path)
    # Add new_sim flag - True for test dataset (as it's from the competition)
    test_df['new_sim'] = True
    print(f"Test data shape: {test_df.shape}")
    
    # Remove duplicates from training data
    print("\nRemoving duplicates from training data...")
    original_count = len(train_df)
    
    # Count non-null target values for each row
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    train_df['target_count'] = train_df[target_columns].notna().sum(axis=1)
    
    # Sort by target_count (descending) and new_sim (True first) to prioritize rows with more data
    train_df = train_df.sort_values(['target_count', 'new_sim'], ascending=[False, False])
    
    # Keep first occurrence of each SMILES (which has the most target values)
    train_df = train_df.drop_duplicates(subset=['SMILES'], keep='first')
    
    # Remove the temporary target_count column
    train_df = train_df.drop('target_count', axis=1)
    
    duplicate_count = original_count - len(train_df)
    print(f"Removed {duplicate_count} duplicate rows")
    print(f"Final training data shape: {train_df.shape}")
    
    return train_df, test_df


@instrument
def remove_outliers(train_df, target_columns):
    for target in target_columns:
        mean = train_df[target].mean()
        std = train_df[target].std()

        # Z-score
        train_df[target] = (train_df[target] - mean) / std

        # Remove samples with Z-score below -3 and above 3
        train_df.loc[train_df[target].abs() >= 3, target] = np.nan

        # Inverse transform
        train_df[target] = train_df[target] * std + mean

    return train_df
