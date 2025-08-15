#!/usr/bin/env python3
"""
Target-Specific LightGBM Model for NeurIPS Open Polymer Prediction 2025
Uses non-linear gradient boosting to better capture polymer property relationships
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import lightgbm as lgb
import re
import sys
import os
import math

# TensorFlow/Keras imports for autoencoder
try:
    # Try standalone Keras first
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input
    KERAS_AVAILABLE = True
    print("Using standalone Keras")
except ImportError:
    # Fall back to TensorFlow Keras
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Input
        KERAS_AVAILABLE = True
        print("Using TensorFlow Keras")
    except ImportError:
        KERAS_AVAILABLE = False
        print("Warning: Neither Keras nor TensorFlow available, autoencoder will not work")
import warnings
warnings.filterwarnings('ignore')

# Check if running on Kaggle or locally
IS_KAGGLE = os.path.exists('/kaggle/input')

# PCA variance threshold - set to None to disable PCA
PCA_VARIANCE_THRESHOLD = None

# Autoencoder settings - set to True to use autoencoder instead of PCA
USE_AUTOENCODER = False
AUTOENCODER_LATENT_DIM = 30  # Number of latent dimensions

# Import competition metric and CV functions only if not on Kaggle
if not IS_KAGGLE:
    from src.competition_metric import neurips_polymer_metric
    from src.diagnostics import CVDiagnostics
    from cv import perform_cross_validation, perform_multi_seed_cv
    from config import LIGHTGBM_PARAMS


# Already checked above

# Set paths based on environment
if IS_KAGGLE:
    # Kaggle competition paths
    TRAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    TEST_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
    SUBMISSION_PATH = 'submission.csv'
    
    # Supplementary dataset paths
    SUPP_PATHS = [
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv',
        '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv'
    ]
else:
    # Local paths
    TRAIN_PATH = 'data/raw/train.csv'
    TEST_PATH = 'data/raw/test.csv'
    SUBMISSION_PATH = 'output/submission.csv'
    
    # Supplementary dataset paths
    SUPP_PATHS = [
        'data/raw/train_supplement/dataset1.csv',
        'data/raw/train_supplement/dataset2.csv',
        'data/raw/train_supplement/dataset3.csv',
        'data/raw/train_supplement/dataset4.csv'
    ]

def calculate_main_branch_atoms(smiles):
    """
    Calculate the number of atoms in the main branch of a polymer.
    The main branch excludes atoms inside parentheses (branches).
    
    Examples:
    *SCCCCC* -> 6 (all carbons and sulfur in main chain)
    *OCC1C(C1)C* -> 5 (O-C-C-C-C, one C is in branch)
    *CC(CC)CC* -> 4 (C-C-C-C, CC is a branch)
    """
    # Remove polymer end markers for processing
    clean_smiles = smiles.replace('*', '')
    
    # Count atoms not in branches (not inside parentheses)
    main_chain_atoms = 0
    in_branch = False
    branch_depth = 0
    
    i = 0
    while i < len(clean_smiles):
        char = clean_smiles[i]
        
        if char == '(':
            branch_depth += 1
            in_branch = True
            i += 1
            continue
        elif char == ')':
            branch_depth -= 1
            if branch_depth == 0:
                in_branch = False
            i += 1
            continue
        
        # Check for two-letter atoms
        if i + 1 < len(clean_smiles) and clean_smiles[i:i+2] in ['Cl', 'Br']:
            if not in_branch:
                main_chain_atoms += 1
            i += 2
        elif char in 'CNOSFIPcnos':
            if not in_branch:
                main_chain_atoms += 1
            i += 1
        else:
            # Skip bonds, numbers, and other symbols
            i += 1
    
    return main_chain_atoms

def calculate_backbone_bonds(smiles):
    """
    Calculate the number of bonds in the main backbone of a polymer.
    The main backbone excludes bonds inside parentheses (branches).
    
    Examples:
    *CCCCCC* -> 5 bonds (C-C-C-C-C-C)
    *CC(C)CC* -> 3 bonds (C-C-C-C, excluding C branch)
    *CC(=O)CC* -> 3 bonds (C-C-C-C, excluding =O branch)
    """
    # Remove polymer end markers for processing
    clean_smiles = smiles.replace('*', '')
    
    backbone_bonds = 0
    in_branch = False
    branch_depth = 0
    prev_atom = False
    
    i = 0
    while i < len(clean_smiles):
        char = clean_smiles[i]
        
        if char == '(':
            branch_depth += 1
            in_branch = True
            prev_atom = False
            i += 1
            continue
        elif char == ')':
            branch_depth -= 1
            if branch_depth == 0:
                in_branch = False
                # After closing branch, next bond connects to backbone
                prev_atom = True
            i += 1
            continue
        
        # Skip bonds and numbers when in branch
        if in_branch:
            i += 1
            continue
            
        # Check for bonds
        if char in '-=#:':
            if prev_atom and i + 1 < len(clean_smiles):
                # Check if next character is an atom
                next_char = clean_smiles[i + 1]
                if i + 2 < len(clean_smiles) and clean_smiles[i + 1:i + 3] in ['Cl', 'Br']:
                    backbone_bonds += 1
                elif next_char in 'CNOSFIPcnos':
                    backbone_bonds += 1
            i += 1
            continue
        
        # Check for atoms
        if i + 1 < len(clean_smiles) and clean_smiles[i:i+2] in ['Cl', 'Br']:
            if not in_branch:
                if prev_atom:
                    # Implicit single bond
                    backbone_bonds += 1
                prev_atom = True
            i += 2
        elif char in 'CNOSFIPcnos':
            if not in_branch:
                if prev_atom and i > 0:
                    prev_char = clean_smiles[i-1]
                    # Count implicit bond if no explicit bond symbol
                    if prev_char not in '-=#:':
                        backbone_bonds += 1
                prev_atom = True
            i += 1
        else:
            # Skip other characters (numbers, etc.)
            i += 1
    
    return backbone_bonds

def calculate_average_bond_length(smiles):
    """
    Calculate the average bond length in Angstroms for a polymer.
    Uses standard bond lengths for different bond types and atom pairs.
    """
    # Standard bond lengths in Angstroms
    bond_lengths = {
        ('C', 'C', '-'): 1.54,
        ('C', 'C', '='): 1.34,
        ('C', 'C', '#'): 1.20,
        ('C', 'N', '-'): 1.47,
        ('C', 'N', '='): 1.29,
        ('C', 'O', '-'): 1.43,
        ('C', 'O', '='): 1.23,
        ('C', 'S', '-'): 1.82,
        ('C', 'F', '-'): 1.35,
        ('C', 'Cl', '-'): 1.77,
        ('C', 'Br', '-'): 1.94,
        ('C', 'I', '-'): 2.14,
        ('C', 'P', '-'): 1.84,
        ('N', 'N', '-'): 1.45,
        ('N', 'O', '-'): 1.40,
        ('O', 'O', '-'): 1.48,
        ('S', 'S', '-'): 2.05,
        ('c', 'c', ':'): 1.40,  # Aromatic C-C
        ('c', 'n', ':'): 1.34,  # Aromatic C-N
        ('c', 'o', ':'): 1.36,  # Aromatic C-O
        ('c', 's', ':'): 1.71,  # Aromatic C-S
    }
    
    # Remove polymer end markers
    clean_smiles = smiles.replace('*', '')
    
    # Parse bonds and atoms
    bond_list = []
    i = 0
    prev_atom = None
    prev_pos = -1
    
    while i < len(clean_smiles):
        char = clean_smiles[i]
        
        # Skip parentheses
        if char in '()':
            i += 1
            continue
            
        # Skip ring numbers
        if char in '123456789':
            i += 1
            continue
        
        # Check for two-letter atoms
        atom = None
        if i + 1 < len(clean_smiles) and clean_smiles[i:i+2] in ['Cl', 'Br']:
            atom = clean_smiles[i:i+2]
            i += 2
        elif char in 'CNOSFIPcnos':
            atom = char
            i += 1
        elif char in '-=#:':
            # Explicit bond
            i += 1
            continue
        else:
            i += 1
            continue
            
        if atom and prev_atom:
            # Determine bond type between prev_atom and current atom
            bond_type = '-'  # Default single bond
            
            # Look for explicit bond between prev_pos and current pos
            for j in range(prev_pos + 1, i - len(atom)):
                if clean_smiles[j] in '-=#:':
                    bond_type = clean_smiles[j]
                    break
            
            # For aromatic atoms, use aromatic bond
            if prev_atom.islower() and atom.islower() and prev_atom in 'cnos' and atom in 'cnos':
                bond_type = ':'
                
            # Normalize atom names for lookup
            atom1 = prev_atom.upper() if prev_atom.islower() and prev_atom not in 'cnos' else prev_atom
            atom2 = atom.upper() if atom.islower() and atom not in 'cnos' else atom
            
            # Try both orders
            key1 = (atom1, atom2, bond_type)
            key2 = (atom2, atom1, bond_type)
            
            if key1 in bond_lengths:
                bond_list.append(bond_lengths[key1])
            elif key2 in bond_lengths:
                bond_list.append(bond_lengths[key2])
            else:
                # Default bond lengths
                default_lengths = {'-': 1.50, '=': 1.30, '#': 1.20, ':': 1.40}
                bond_list.append(default_lengths.get(bond_type, 1.50))
        
        if atom:
            prev_atom = atom
            prev_pos = i - len(atom)
    
    if not bond_list:
        return 0.0
    
    return round(sum(bond_list) / len(bond_list), 3)

def extract_molecular_features(smiles):
    """Extract features from SMILES string without external libraries"""
    features = {}
    
    # Basic string features
    features['length'] = len(smiles)
    
    # Count different atoms (case-sensitive for aromatic vs non-aromatic)
    # First count two-letter atoms to avoid double counting
    features['num_Cl'] = smiles.count('Cl')
    features['num_Br'] = smiles.count('Br')
    
    # Remove two-letter atoms before counting single letters
    smiles_no_cl_br = smiles.replace('Cl', '').replace('Br', '')
    
    features['num_C'] = len(re.findall(r'C', smiles_no_cl_br))
    features['num_c'] = len(re.findall(r'c', smiles_no_cl_br))  # aromatic carbon
    features['num_O'] = len(re.findall(r'O', smiles_no_cl_br))
    features['num_o'] = len(re.findall(r'o', smiles_no_cl_br))  # aromatic oxygen
    features['num_N'] = len(re.findall(r'N', smiles_no_cl_br))
    features['num_n'] = len(re.findall(r'n', smiles_no_cl_br))  # aromatic nitrogen
    features['num_S'] = len(re.findall(r'S', smiles_no_cl_br))
    features['num_s'] = len(re.findall(r's', smiles_no_cl_br))  # aromatic sulfur
    features['num_F'] = smiles_no_cl_br.count('F')
    features['num_I'] = smiles_no_cl_br.count('I')
    features['num_P'] = smiles_no_cl_br.count('P')
    
    # Count bonds
    features['num_single_bonds'] = smiles.count('-')
    features['num_double_bonds'] = smiles.count('=')
    features['num_triple_bonds'] = smiles.count('#')
    features['num_aromatic_bonds'] = smiles.count(':')
    
    # Count structural features
    # Count unique ring identifiers (each ring is closed once)
    ring_closures = set()
    for i in range(1, 10):
        if str(i) in smiles:
            ring_closures.add(str(i))
    features['num_rings'] = len(ring_closures)
    features['num_branches'] = smiles.count('(')
    features['num_chiral_centers'] = smiles.count('@')
    
    # Polymer-specific features
    # features['has_polymer_end'] = int('*' in smiles)  # Removed - zero importance
    # features['num_polymer_ends'] = smiles.count('*')  # Removed - may cause overfitting
    
    # Functional group patterns
    features['has_carbonyl'] = int('C(=O)' in smiles or 'C=O' in smiles)
    # features['has_hydroxyl'] = int('OH' in smiles or 'O[H]' in smiles)  # Removed - may cause overfitting
    features['has_ether'] = int('COC' in smiles or 'cOc' in smiles)
    features['has_amine'] = int('N' in smiles)
    features['has_sulfone'] = int('S(=O)(=O)' in smiles)
    features['has_ester'] = int('C(=O)O' in smiles or 'COO' in smiles)
    features['has_amide'] = int('C(=O)N' in smiles or 'CON' in smiles)
    
    # Aromatic features
    features['num_aromatic_atoms'] = features['num_c'] + features['num_n'] + features['num_o'] + features['num_s']
    features['aromatic_ratio'] = features['num_aromatic_atoms'] / max(features['length'], 1)
    
    # Calculate derived features
    features['heavy_atom_count'] = (features['num_C'] + features['num_c'] + 
                                   features['num_O'] + features['num_o'] + 
                                   features['num_N'] + features['num_n'] + 
                                   features['num_S'] + features['num_s'] + 
                                   features['num_F'] + features['num_Cl'] + 
                                   features['num_Br'] + features['num_I'] + 
                                   features['num_P'])
    
    features['heteroatom_count'] = (features['num_O'] + features['num_o'] + 
                                    features['num_N'] + features['num_n'] + 
                                    features['num_S'] + features['num_s'] + 
                                    features['num_F'] + features['num_Cl'] + 
                                    features['num_Br'] + features['num_I'] + 
                                    features['num_P'])
    
    features['heteroatom_ratio'] = features['heteroatom_count'] / max(features['heavy_atom_count'], 1)
    
    # Flexibility indicators
    features['rotatable_bond_estimate'] = max(0, features['num_single_bonds'] - features['num_rings'])
    features['flexibility_score'] = features['rotatable_bond_estimate'] / max(features['heavy_atom_count'], 1)
    
    # Size and complexity
    features['molecular_complexity'] = (features['num_rings'] + features['num_branches'] + 
                                       features['num_chiral_centers'])
    
    # Molecular weight estimation (for Tg prediction based on Newton's second law)
    # Atomic weights with 0.1 significance as requested
    atomic_weights = {
        'C': 12.0, 'c': 12.0,  # Carbon
        'O': 16.0, 'o': 16.0,  # Oxygen
        'N': 14.0, 'n': 14.0,  # Nitrogen
        'S': 32.1, 's': 32.1,  # Sulfur
        'F': 19.0,             # Fluorine
        'Cl': 35.5,            # Chlorine
        'Br': 79.9,            # Bromine
        'I': 126.9,            # Iodine
        'P': 31.0,             # Phosphorus
        'H': 1.0               # Hydrogen (implicit)
    }
    
    # Calculate molecular weight estimate
    mol_weight = 0.0
    mol_weight += features['num_C'] * atomic_weights['C']
    mol_weight += features['num_c'] * atomic_weights['c']
    mol_weight += features['num_O'] * atomic_weights['O']
    mol_weight += features['num_o'] * atomic_weights['o']
    mol_weight += features['num_N'] * atomic_weights['N']
    mol_weight += features['num_n'] * atomic_weights['n']
    mol_weight += features['num_S'] * atomic_weights['S']
    mol_weight += features['num_s'] * atomic_weights['s']
    mol_weight += features['num_F'] * atomic_weights['F']
    mol_weight += features['num_Cl'] * atomic_weights['Cl']
    mol_weight += features['num_Br'] * atomic_weights['Br']
    mol_weight += features['num_I'] * atomic_weights['I']
    mol_weight += features['num_P'] * atomic_weights['P']
    
    # Round to 0.1 significance
    features['molecular_weight'] = round(mol_weight, 1)
    
    # Van der Waals volume calculation (heavy atoms only, no H estimation)
    # Van der Waals volumes in Angstrom^3 (Bondi, 1964)
    vdw_volumes = {
        'C': 20.58, 'c': 20.58,  # Carbon (both aliphatic and aromatic)
        'N': 15.60, 'n': 15.60,  # Nitrogen
        'O': 14.71, 'o': 14.71,  # Oxygen  
        'S': 24.43, 's': 24.43,  # Sulfur
        'F': 13.31,              # Fluorine
        'Cl': 22.45,             # Chlorine
        'Br': 26.52,             # Bromine
        'I': 32.52,              # Iodine
        'P': 24.43               # Phosphorus
    }
    
    # Calculate total Van der Waals volume
    vdw_volume = 0.0
    vdw_volume += features['num_C'] * vdw_volumes['C']
    vdw_volume += features['num_c'] * vdw_volumes['c']
    vdw_volume += features['num_O'] * vdw_volumes['O']
    vdw_volume += features['num_o'] * vdw_volumes['o']
    vdw_volume += features['num_N'] * vdw_volumes['N']
    vdw_volume += features['num_n'] * vdw_volumes['n']
    vdw_volume += features['num_S'] * vdw_volumes['S']
    vdw_volume += features['num_s'] * vdw_volumes['s']
    vdw_volume += features['num_F'] * vdw_volumes['F']
    vdw_volume += features['num_Cl'] * vdw_volumes['Cl']
    vdw_volume += features['num_Br'] * vdw_volumes['Br']
    vdw_volume += features['num_I'] * vdw_volumes['I']
    vdw_volume += features['num_P'] * vdw_volumes['P']
    
    # Round to 0.1 significance  
    features['vdw_volume'] = round(vdw_volume, 1)
    
    # Density estimate: molecular weight / volume
    # Convert from g/mol/Å³ to g/cm³
    # 1 Å³ = 10⁻²⁴ cm³, 1 mol = 6.022 × 10²³ molecules
    # Conversion factor: 10²⁴ / 6.022 × 10²³ = 1.66054
    if vdw_volume > 0:
        density_g_mol_A3 = mol_weight / vdw_volume
        features['density_estimate'] = round(density_g_mol_A3 * 1.66054, 3)
        
        # FFV estimation using original density units
        # FFV = (V - 1.3 × Vw) / V where V = 1/density in Å³/(g/mol)
        specific_volume = 1.0 / density_g_mol_A3  # Å³/(g/mol)
        features['ffv_estimate'] = round((specific_volume - 1.3 * vdw_volume) / specific_volume, 3)
    else:
        features['density_estimate'] = 0.0
        features['ffv_estimate'] = 0.0
    
    # Additional polymer-specific patterns
    # Phenyl: aromatic 6-membered ring patterns
    features['has_phenyl'] = int(bool(re.search(r'c1ccccc1|c1ccc.*cc1', smiles)))
    features['has_cyclohexyl'] = int('C1CCCCC1' in smiles)
    features['has_methyl'] = int(bool(re.search(r'C(?![a-zA-Z])', smiles)))
    segments = [x for x in re.split(r'[\(\)]', smiles) if x]
    features['chain_length_estimate'] = max(len(x) for x in segments) if segments else 0
    
    # Additional structural patterns
    features['has_fused_rings'] = int(bool(re.search(r'[0-9].*c.*[0-9]', smiles)))
    features['has_spiro'] = int('@' in smiles and smiles.count('@') > 1)
    # Bridge: multiple different ring numbers
    ring_numbers = set(re.findall(r'[0-9]', smiles))
    features['has_bridge'] = int(len(ring_numbers) >= 2)
    
    # Main branch atom count
    features['main_branch_atoms'] = calculate_main_branch_atoms(smiles)
    
    # Main branch atom ratio (main branch atoms / total heavy atoms)
    features['main_branch_atom_ratio'] = round(features['main_branch_atoms'] / max(features['heavy_atom_count'], 1), 3)
    
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
    
    return features

# Define target-specific features to use
TARGET_FEATURES = {
    'Tg': ['num_C', 'num_n'],  # Top 2 atom features
    'FFV': ['num_S', 'num_n'], 
    'Tc': ['num_C', 'num_S'],
    'Density': ['num_Cl', 'num_Br'],
    'Rg': ['num_F', 'num_Cl']
}

def prepare_features(df):
    """Convert SMILES to molecular features"""
    print("Extracting molecular features...")
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing molecule {idx}/{len(df)}...")
        features = extract_molecular_features(row['SMILES'])
        # Add new_sim feature
        features['new_sim'] = int(row['new_sim'])  # Convert boolean to int (0 or 1)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    return features_df

def apply_autoencoder(X_train, X_test=None, latent_dim=30, epochs=100, batch_size=32):
    """
    Apply autoencoder for dimensionality reduction.
    
    Args:
        X_train: Training data (n_samples, n_features)
        X_test: Test data (optional)
        latent_dim: Number of dimensions in the latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        If X_test is None: X_train_encoded
        If X_test is provided: (X_train_encoded, X_test_encoded)
    
    """
    if not KERAS_AVAILABLE:
        print("Keras/TensorFlow not available, returning original data")
        if X_test is None:
            return X_train
        else:
            return X_train, X_test
    
    # Define encoder architecture
    input_dim = X_train.shape[1]
    # encoder = Sequential([
    #     Dense(32, activation='leaky_relu', input_shape=(input_dim,)),
    #     Dense(16, activation='leaky_relu'),
    #     Dense(10, activation='leaky_relu')
    # ])
    # input_dim = X_train.shape[1]
    # encoded = keras.Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu', input_shape=(input_dim,))

    # Define decoder architecture
    # decoder = Sequential([
    #     Dense(16, activation='linear', input_shape=(10,)),
    #     Dense(32, activation='linear'),
    #     Dense(input_dim, activation='linear')
    # ])
    # decoded = keras.Input(shape=(input_dim/2,))
    decoder = Dense(input_dim, activation='relu', input_shape=(32,))

    # Combine into autoencoder model
    # Create input layer
    input_layer = Input(shape=(input_dim,))
    # Pass through encoder
    encoded = encoder(input_layer)
    # Pass through decoder
    decoded = decoder(encoded)
    # Create autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Compile and train the model
    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Create encoder model for extracting latent representations
    encoder_model = Model(inputs=input_layer, outputs=encoded)
    
    # Extract encoder and apply to data
    X_train_encoded = encoder_model.predict(X_train)
    
    if X_test is None:
        return X_train_encoded
    else:
        X_test_encoded = encoder_model.predict(X_test)
        return X_train_encoded, X_test_encoded


def select_features_for_target(X, target):
    """Select features for a specific target"""
    # Get all non-atom features
    atom_features = ['num_C', 'num_c', 'num_O', 'num_o', 'num_N', 'num_n', 
                    'num_S', 'num_s', 'num_F', 'num_Cl', 'num_Br', 'num_I', 'num_P']
    non_atom_features = [col for col in X.columns if col not in atom_features]
    
    # Since molecular_weight is commented out, no need to exclude features
    
    # Select features: all non-atom features + target-specific atom features
    selected_features = non_atom_features + TARGET_FEATURES[target]
    return X[selected_features]


def main(cv_only=False, use_supplementary=True, model_type='lightgbm'):
    """
    Main function to train model and make predictions
    
    Args:
        cv_only: If True, only run cross-validation and skip submission generation
        use_supplementary: If True, include supplementary datasets in training
        model_type: 'ridge' or 'lightgbm' (default: 'lightgbm')
    """
    print(f"=== Separate {model_type.upper()} Models for Polymer Prediction ===")
    print("Loading training data...")
    
    # Load main training data
    train_df = pd.read_csv(TRAIN_PATH)
    # Add new_sim flag - True for main dataset
    train_df['new_sim'] = True
    print(f"Main training data shape: {train_df.shape}")
    
    if use_supplementary:
        # Load and combine supplementary datasets
        print("\nLoading supplementary datasets...")
        all_train_dfs = [train_df]
        
        for supp_path in SUPP_PATHS:
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
    test_df = pd.read_csv(TEST_PATH)
    # Add new_sim flag - True for test dataset (as it's from the competition)
    test_df['new_sim'] = True
    print(f"Test data shape: {test_df.shape}")
    
    # Extract features
    print("\nExtracting features from training data...")
    X_train = prepare_features(train_df)
    
    print("\nExtracting features from test data...")
    X_test = prepare_features(test_df)
    
    # Prepare target variables
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    y_train = train_df[target_columns]
    
    # Print feature statistics
    print(f"\nFeature dimensions: {X_train.shape[1]} features")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # We'll handle missing values and scaling per target
    print("\nPreparing for target-specific training...")
    
    # We don't need to handle missing target values since we train separate models
    # Each model will only use samples with valid values for its specific target
    
    # Print target statistics
    print("\nTarget value statistics:")
    for col in target_columns:
        print(f"{col}: median={y_train[col].median():.4f}, "
              f"missing={y_train[col].isna().sum()} ({y_train[col].isna().sum()/len(y_train)*100:.1f}%)")
    
    # Run cross-validation if requested (but not on Kaggle)
    if cv_only:
        if IS_KAGGLE:
            print("\n⚠️  Cross-validation is not available in Kaggle notebooks")
            print("Proceeding with submission generation instead...")
        else:
            print("\n=== Testing with Multiple Random Seeds ===")
            if not use_supplementary:
                print("** Running CV WITHOUT supplementary datasets **")
            else:
                print("** Running CV WITH supplementary datasets **")
            
            # Run multi-seed CV with target-specific features (current implementation)
            multi_seed_result = perform_multi_seed_cv(X_train, y_train, cv_folds=5, 
                                                     target_columns=target_columns,
                                                     enable_diagnostics=False,
                                                     model_type=model_type)
            
            # Also run single seed CV for comparison if needed
            print("\n=== Single Seed Comparison ===")
            single_seed_result = perform_cross_validation(X_train, y_train, cv_folds=5,
                                                         target_columns=target_columns,
                                                         enable_diagnostics=False,
                                                         random_seed=42,
                                                         model_type=model_type)
            
            print(f"\n=== Final Comparison ===")
            print(f"Single seed (42) CV: {single_seed_result['cv_mean']:.4f} (+/- {single_seed_result['cv_std']:.4f})")
            print(f"Multi-seed CV: {multi_seed_result['overall_mean']:.4f} (+/- {multi_seed_result['overall_std']:.4f})")
            
            return {
                'single_seed': single_seed_result,
                'multi_seed': multi_seed_result
            }
    
    # Train separate models for each target
    print(f"\n=== Training Separate {model_type.upper()} Models for Each Target ===")
    predictions = np.zeros((len(X_test), len(target_columns)))
    
    # Model parameters
    if model_type == 'lightgbm':
        # Use parameters from config for single source of truth
        if IS_KAGGLE:
            # Need to define params inline for Kaggle since we can't import config
            lgb_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'max_depth': -1,
                'num_leaves': 31,
                'n_estimators': 2000,
                'learning_rate': 0.001,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            lgb_params = LIGHTGBM_PARAMS.copy()
    else:
        # Ridge parameters
        target_alphas = {
            'Tg': 10.0,      # Higher regularization for sparse target
            'FFV': 1.0,      # Lower regularization for dense target
            'Tc': 10.0,      # Higher regularization for sparse target
            'Density': 5.0,  # Medium regularization
            'Rg': 10.0       # Higher regularization for sparse target
        }
    
    for i, target in enumerate(target_columns):
        print(f"\nTraining model for {target}...")
        
        # Get non-missing samples for this target
        mask = ~y_train[target].isna()
        n_samples = mask.sum()
        print(f"  Available samples: {n_samples} ({n_samples/len(y_train)*100:.1f}%)")
        
        if n_samples > 0:
            # Select features for this target
            X_train_selected = select_features_for_target(X_train, target)
            X_test_selected = select_features_for_target(X_test, target)
            
            print(f"  Using {len(X_train_selected.columns)} features (reduced from {len(X_train.columns)})")
            
            # Get samples with valid target values
            X_target = X_train_selected[mask]
            y_target = y_train[target][mask]
            
            # Further filter to only keep rows with no missing features
            feature_complete_mask = ~X_target.isnull().any(axis=1)
            X_target_complete = X_target[feature_complete_mask]
            y_target_complete = y_target[feature_complete_mask]
            
            print(f"  Complete samples (no missing features): {len(X_target_complete)} ({len(X_target_complete)/len(y_train)*100:.1f}%)")
            
            if len(X_target_complete) > 0:
                # Scale features (no imputation needed)
                scaler = StandardScaler()
                X_target_scaled = scaler.fit_transform(X_target_complete)
                
                # Apply dimensionality reduction if enabled
                pca = None
                if USE_AUTOENCODER:
                    print(f"  Applying autoencoder: {X_target_scaled.shape[1]} features -> {AUTOENCODER_LATENT_DIM} dimensions")
                    X_target_final = apply_autoencoder(X_target_scaled, latent_dim=AUTOENCODER_LATENT_DIM)
                elif PCA_VARIANCE_THRESHOLD is not None:
                    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=42)
                    X_target_pca = pca.fit_transform(X_target_scaled)
                    print(f"  PCA: {X_target_scaled.shape[1]} features -> {X_target_pca.shape[1]} components")
                    print(f"  Variance preserved: {pca.explained_variance_ratio_.sum():.4f}")
                    X_target_final = X_target_pca
                else:
                    X_target_final = X_target_scaled
                
                # For test set, we need to handle missing values somehow
                # Use zero imputation only for test set predictions
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                imputer.fit(X_target_complete)  # Fit on complete training data
                X_test_imputed = imputer.transform(X_test_selected)
                X_test_scaled = scaler.transform(X_test_imputed)
                
                # Apply dimensionality reduction to test set if enabled
                if USE_AUTOENCODER:
                    _, X_test_final = apply_autoencoder(X_target_scaled, X_test_scaled, latent_dim=AUTOENCODER_LATENT_DIM)
                elif pca is not None:
                    X_test_final = pca.transform(X_test_scaled)
                else:
                    X_test_final = X_test_scaled
                
                # Train model for this target
                if model_type == 'lightgbm':
                    model = lgb.LGBMRegressor(**lgb_params)
                    # Split data for validation to track overfitting
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_target_final, y_target_complete, 
                        test_size=0.2, random_state=42
                    )
                    # Train with validation data to see training progress
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_tr, y_tr), (X_val, y_val)],
                        eval_names=['train', 'valid'],
                        eval_metric='mae',
                        callbacks=[lgb.log_evaluation(0)]  # Disable verbose output
                    )
                    # Get final MAE scores from eval results
                    train_mae = model.evals_result_['train']['l1'][-1]
                    val_mae = model.evals_result_['valid']['l1'][-1]
                    print(f"  Final MAE - Train: {train_mae:.4f}, Valid: {val_mae:.4f}")
                else:
                    # Use Ridge with target-specific alpha
                    alpha = target_alphas.get(target, 1.0)
                    print(f"  Using alpha={alpha}")
                    model = Ridge(alpha=alpha, random_state=42)
                    model.fit(X_target_final, y_target_complete)
                
                # Make predictions
                predictions[:, i] = model.predict(X_test_final)
                print(f"  Predictions: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}")
            else:
                # No complete samples available, use median
                predictions[:, i] = y_train[target].median()
                print(f"  No complete samples available, using median: {predictions[:, i][0]:.4f}")
        else:
            # Use median of available values if no samples
            predictions[:, i] = y_train[target].median()
            print(f"  No samples available, using median: {predictions[:, i][0]:.4f}")
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Save submission
    print(f"\nSaving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")
    
    # Display submission preview
    print("\nSubmission preview:")
    print(submission_df)
    
    print("\n=== Model training complete! ===")
    print("Used target-specific features to reduce complexity")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    cv_only = '--cv-only' in sys.argv or '--cv' in sys.argv
    no_supplement = '--no-supplement' in sys.argv or '--no-supp' in sys.argv
    
    # Check for model type
    model_type = 'lightgbm'  # default
    if '--model' in sys.argv:
        model_idx = sys.argv.index('--model')
        if model_idx + 1 < len(sys.argv):
            model_type = sys.argv[model_idx + 1]
    
    main(cv_only=cv_only, use_supplementary=not no_supplement, model_type=model_type)
