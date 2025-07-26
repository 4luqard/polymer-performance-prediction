import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from collections import Counter
import re

def extract_polymer_features(smiles):
    """Extract structural features from polymer SMILES that may indicate crystallinity."""
    features = {}
    
    # Remove polymer markers
    clean_smiles = smiles.replace('*', '')
    
    # Parse molecule
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        return None
    
    # Basic molecular properties
    features['num_atoms'] = mol.GetNumAtoms()
    features['num_bonds'] = mol.GetNumBonds()
    features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
    features['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
    features['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
    
    # Chain flexibility indicators
    features['rotatable_bond_fraction'] = features['num_rotatable_bonds'] / max(features['num_bonds'], 1)
    
    # Symmetry and regularity indicators
    features['has_symmetry'] = check_symmetry(smiles)
    features['repeat_unit_regularity'] = calculate_repeat_regularity(smiles)
    
    # Branching analysis
    features['branching_points'] = count_branching_points(mol)
    features['branching_density'] = features['branching_points'] / max(features['num_atoms'], 1)
    
    # Intermolecular interaction potential
    features['num_h_donors'] = Descriptors.NumHDonors(mol)
    features['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
    features['h_bonding_potential'] = features['num_h_donors'] + features['num_h_acceptors']
    
    # Steric hindrance indicators
    features['bulky_groups'] = count_bulky_groups(mol)
    features['molar_volume'] = Descriptors.MolMR(mol)  # Molar refractivity as proxy for volume
    
    # Chain stiffness indicators
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    features['aromatic_fraction'] = aromatic_atoms / max(features['num_atoms'], 1)
    features['sp2_fraction'] = count_sp2_carbons(mol) / max(count_carbons(mol), 1)
    
    # Polarity and dipole
    features['tpsa'] = Descriptors.TPSA(mol)  # Topological polar surface area
    features['logp'] = Descriptors.MolLogP(mol)
    
    return features

def check_symmetry(smiles):
    """Check if the polymer SMILES suggests symmetrical structure."""
    # Remove polymer markers
    clean = smiles.replace('*', '')
    
    # Simple check: is the SMILES palindromic or has repeating patterns?
    if clean == clean[::-1]:
        return 1.0
    
    # Check for repeating subunits
    for length in range(2, len(clean)//2 + 1):
        if len(clean) % length == 0:
            chunks = [clean[i:i+length] for i in range(0, len(clean), length)]
            if len(set(chunks)) == 1:
                return 0.8
    
    return 0.0

def calculate_repeat_regularity(smiles):
    """Calculate how regular the repeat unit pattern is."""
    # Look for patterns between polymer markers
    if '*' not in smiles:
        return 0.0
    
    # Extract the main chain between markers
    pattern = re.findall(r'\*([^*]+)\*', smiles)
    if not pattern:
        return 0.0
    
    main_chain = pattern[0]
    
    # Check for repeating motifs
    motif_scores = []
    for length in range(2, min(len(main_chain)//2 + 1, 10)):
        motifs = [main_chain[i:i+length] for i in range(0, len(main_chain)-length+1)]
        motif_counts = Counter(motifs)
        if motif_counts:
            regularity = max(motif_counts.values()) / len(motifs)
            motif_scores.append(regularity)
    
    return max(motif_scores) if motif_scores else 0.0

def count_branching_points(mol):
    """Count the number of branching points (atoms with >2 connections)."""
    branching = 0
    for atom in mol.GetAtoms():
        # Count non-hydrogen neighbors
        degree = len([n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1])
        if degree > 2:
            branching += 1
    return branching

def count_bulky_groups(mol):
    """Count bulky substituents that might hinder crystallization."""
    bulky_count = 0
    
    # SMARTS patterns for bulky groups
    bulky_patterns = [
        'C(C)(C)C',  # tert-butyl
        'c1ccccc1',  # phenyl
        'C1CCCCC1',  # cyclohexyl
        'C(C)C',     # isopropyl
    ]
    
    for pattern in bulky_patterns:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        bulky_count += len(matches)
    
    return bulky_count

def count_sp2_carbons(mol):
    """Count sp2 hybridized carbons."""
    sp2_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon
            if atom.GetHybridization() == Chem.HybridizationType.SP2:
                sp2_count += 1
    return sp2_count

def count_carbons(mol):
    """Count total carbon atoms."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

def predict_crystallinity_tendency(features):
    """
    Predict crystallinity tendency based on structural features.
    Returns a score from 0 (amorphous) to 1 (highly crystalline).
    """
    if features is None:
        return None
    
    score = 0.5  # Start with neutral score
    
    # Factors promoting crystallinity
    score += 0.1 * features['has_symmetry']
    score += 0.1 * features['repeat_unit_regularity']
    score += 0.05 * min(features['h_bonding_potential'] / 10, 1)  # Cap contribution
    score += 0.1 * features['aromatic_fraction']  # π-π stacking
    
    # Factors inhibiting crystallinity
    score -= 0.2 * min(features['branching_density'] * 10, 1)  # Heavy penalty for branching
    score -= 0.1 * min(features['bulky_groups'] / 5, 1)
    score -= 0.1 * features['rotatable_bond_fraction']  # Flexibility reduces crystallinity
    
    # Ensure score is between 0 and 1
    return max(0, min(1, score))

# Main analysis
def analyze_dataset():
    """Analyze the training dataset to understand crystallinity patterns."""
    
    # Load data
    train_df = pd.read_csv('data/raw/train.csv')
    
    # Classify polymers based on thermal properties
    train_df['polymer_type'] = 'unknown'
    train_df.loc[(train_df['Tg'].notna()) & (train_df['Tc'].isna()), 'polymer_type'] = 'amorphous'
    train_df.loc[(train_df['Tc'].notna()) & (train_df['Tg'].isna()), 'polymer_type'] = 'crystalline'
    train_df.loc[(train_df['Tg'].notna()) & (train_df['Tc'].notna()), 'polymer_type'] = 'semi_crystalline'
    
    # Extract features for polymers with known types
    known_polymers = train_df[train_df['polymer_type'] != 'unknown'].copy()
    
    print(f"Analyzing {len(known_polymers)} polymers with known crystallinity type...")
    
    # Extract features
    all_features = []
    for idx, row in known_polymers.iterrows():
        features = extract_polymer_features(row['SMILES'])
        if features:
            features['polymer_type'] = row['polymer_type']
            features['id'] = row['id']
            features['smiles'] = row['SMILES']
            all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Analyze feature differences between polymer types
    print("\nFeature averages by polymer type:")
    print("="*60)
    
    feature_cols = [col for col in features_df.columns if col not in ['polymer_type', 'id', 'smiles']]
    
    for polymer_type in ['amorphous', 'crystalline', 'semi_crystalline']:
        subset = features_df[features_df['polymer_type'] == polymer_type]
        if len(subset) > 0:
            print(f"\n{polymer_type.upper()} (n={len(subset)}):")
            for feature in ['has_symmetry', 'repeat_unit_regularity', 'branching_density', 
                          'aromatic_fraction', 'h_bonding_potential', 'rotatable_bond_fraction']:
                if feature in feature_cols:
                    avg = subset[feature].mean()
                    print(f"  {feature}: {avg:.3f}")
    
    # Calculate crystallinity scores
    features_df['crystallinity_score'] = features_df.apply(
        lambda row: predict_crystallinity_tendency(row[feature_cols].to_dict()), 
        axis=1
    )
    
    # Evaluate scoring
    print("\nCrystallinity score by polymer type:")
    print("="*60)
    for polymer_type in ['amorphous', 'crystalline', 'semi_crystalline']:
        subset = features_df[features_df['polymer_type'] == polymer_type]
        if len(subset) > 0:
            scores = subset['crystallinity_score']
            print(f"{polymer_type}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    
    # Save detailed results
    features_df.to_csv('polymer_crystallinity_features.csv', index=False)
    
    # Example predictions
    print("\nExample predictions:")
    print("="*60)
    
    for polymer_type in ['amorphous', 'crystalline']:
        examples = features_df[features_df['polymer_type'] == polymer_type].head(3)
        print(f"\n{polymer_type.upper()} examples:")
        for _, row in examples.iterrows():
            print(f"  ID: {row['id']}")
            print(f"  SMILES: {row['smiles']}")
            print(f"  Score: {row['crystallinity_score']:.3f}")
            print(f"  Key features: symmetry={row['has_symmetry']:.1f}, "
                  f"branching={row['branching_density']:.3f}, "
                  f"aromatic={row['aromatic_fraction']:.3f}")
            print()
    
    return features_df

if __name__ == "__main__":
    analyze_dataset()