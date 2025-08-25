"""
Data processing functions for NeurIPS Open Polymer Prediction 2025
Extracted from model.py for better code organization
"""

import pandas as pd
import numpy as np
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

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



def extract_molecular_features(smiles, rpt):
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
    features['has_polymer_end'] = int('*' in smiles)
    features['num_polymer_ends'] = smiles.count('*')
    
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
    features['molecular_weight'] = round(mol_weight, 3)
    
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
    features['vdw_volume'] = round(vdw_volume, 3)
    
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
    
    if rpt:
        feature_names = list(features.keys())
        for feature in feature_names:
            features[f'{feature}_rpt'] = features[feature]
            del features[feature]

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



def apply_autoencoder(X_train, X_test=None, latent_dim=26, epochs=100, batch_size=128):
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
    # Define encoder architecture
    input_dim = X_train.shape[1]
    encoder = Sequential([
        Dense(int(latent_dim+(input_dim-latent_dim)/1.5), activation='relu', input_shape=(input_dim,)),
        Dense(int(latent_dim+(input_dim-latent_dim)/3), activation='tanh'),
        Dense(latent_dim, activation='linear')
    ])
    # input_dim = X_train.shape[1]
    # encoded = keras.Input(shape=(input_dim,))
    # encoder = Dense(latent_dim, activation='linear', input_shape=(input_dim,))

    # Define decoder architecture
    decoder = Sequential([
        Dense(int(latent_dim+(input_dim-latent_dim)/3), activation='relu', input_shape=(latent_dim,)),
        Dense(int(latent_dim+(input_dim-latent_dim)/1.5), activation='tanh'),
        Dense(input_dim, activation='linear')
    ])
    # decoded = keras.Input(shape=(input_dim/2,))
    # decoder = Dense(input_dim, activation='relu', input_shape=(32,))

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
    loss = keras.losses.MeanAbsoluteError()
    autoencoder.compile(optimizer='adam', loss=loss)
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


def preprocess_data(X_train, X_test, use_autoencoder=False, autoencoder_latent_dim=30, 
                    pca_variance_threshold=None, use_pls=False, pls_n_components=50,
                    y_train=None):
    """
    Preprocess training and test data including:
    - Dropping columns with all NaN values
    - Imputing missing values
    - Scaling features
    - Applying dimensionality reduction (PCA, PLS, or autoencoder)
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        use_autoencoder: Whether to use autoencoder for dimensionality reduction
        autoencoder_latent_dim: Number of latent dimensions for autoencoder
        pca_variance_threshold: Variance threshold for PCA (None to disable)
        use_pls: Whether to use PLS for dimensionality reduction
        pls_n_components: Number of PLS components
        y_train: Target values DataFrame (required for PLS)
    
    Returns:
        Tuple of (X_train_preprocessed, X_test_preprocessed)
    """
    print("\n=== Preprocessing Data ===")
    
    # Drop columns with all NaN values in training data
    nan_cols = X_train.columns[X_train.isna().all()]
    if len(nan_cols) > 0:
        print(f"Dropping {len(nan_cols)} columns with all NaN values")
        X_train = X_train.drop(columns=nan_cols)
        X_test = X_test.drop(columns=nan_cols)
    
    # Impute missing values with zeros (fit on train, transform both)
    print("Imputing missing values with zeros...")
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features (fit on train, transform both)
    print("Scaling features...")
    global_scaler = StandardScaler()
    X_train_scaled = global_scaler.fit_transform(X_train_imputed)
    X_test_scaled = global_scaler.transform(X_test_imputed)
    
    # Apply dimensionality reduction if enabled
    global_pca = None
    if use_autoencoder:
        print(f"Applying autoencoder: {X_train_scaled.shape[1]} features -> {autoencoder_latent_dim} dimensions")
        X_train_reduced, X_test_reduced = apply_autoencoder(X_train_scaled, X_test_scaled, latent_dim=autoencoder_latent_dim)
        X_train_preprocessed = pd.DataFrame(X_train_reduced)
        X_test_preprocessed = pd.DataFrame(X_test_reduced)
    elif use_pls:
        if y_train is None:
            raise ValueError("y_train is required for PLS dimensionality reduction")
        
        # Validate number of components
        max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
        if pls_n_components > max_components:
            print(f"WARNING: Requested {pls_n_components} PLS components exceeds maximum ({max_components})")
            pls_n_components = max_components
        
        print(f"Applying PLS: {X_train_scaled.shape[1]} features -> {pls_n_components} components")
        
        # Handle missing values in y_train for PLS
        # Create a mask for samples with at least one non-missing target
        y_mask = ~y_train.isna().all(axis=1)
        
        # Check if we have any samples with targets
        if not y_mask.any():
            print("WARNING: No samples with non-missing targets found for PLS fitting.")
            print("Falling back to PCA for dimensionality reduction.")
            # Fall back to PCA
            pca = PCA(n_components=pls_n_components, random_state=42, whiten=True)
            X_train_reduced = pca.fit_transform(X_train_scaled)
            X_test_reduced = pca.transform(X_test_scaled)
            print(f"PCA: {X_train_scaled.shape[1]} features -> {X_train_reduced.shape[1]} components")
        else:
            # Filter data for PLS fitting
            X_train_pls = X_train_scaled[y_mask]
            y_train_pls = y_train[y_mask].fillna(y_train[y_mask].mean())
            
            # Fit PLS model
            pls = PLSRegression(n_components=pls_n_components, scale=False)
            pls.fit(X_train_pls, y_train_pls)
            
            # Transform both train and test data
            X_train_reduced = pls.transform(X_train_scaled)
            X_test_reduced = pls.transform(X_test_scaled)
            
            print(f"PLS fitted on {X_train_pls.shape[0]} samples with non-missing targets")
        
        X_train_preprocessed = pd.DataFrame(X_train_reduced, index=X_train.index)
        X_test_preprocessed = pd.DataFrame(X_test_reduced, index=X_test.index)
    elif pca_variance_threshold is not None:
        print(f"Applying PCA with variance threshold {pca_variance_threshold}...")
        global_pca = PCA(n_components=pca_variance_threshold, random_state=42, whiten=True)
        X_train_reduced = global_pca.fit_transform(X_train_scaled)
        X_test_reduced = global_pca.transform(X_test_scaled)
        print(f"PCA: {X_train_scaled.shape[1]} features -> {X_train_reduced.shape[1]} components")
        print(f"Variance preserved: {global_pca.explained_variance_ratio_.sum():.4f}")
        X_train_preprocessed = pd.DataFrame(X_train_reduced)
        X_test_preprocessed = pd.DataFrame(X_test_reduced)
    else:
        X_train_preprocessed = pd.DataFrame(X_train_scaled, index=X_train.index)
        X_test_preprocessed = pd.DataFrame(X_test_scaled, index=X_test.index)
    
    print(f"Final dimensions: Train {X_train_preprocessed.shape}, Test {X_test_preprocessed.shape}")
    
    return X_train_preprocessed, X_test_preprocessed


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




