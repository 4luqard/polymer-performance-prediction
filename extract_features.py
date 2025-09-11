#!/usr/bin/env python3

def has_tetrahedral_carbon(smiles):
    """
    Check if the SMILES string contains tetrahedral carbon stereochemistry.
    Tetrahedral carbons are indicated by '@' in SMILES notation.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        
    Returns:
        bool: True if tetrahedral carbon is present, False otherwise
    """
    return '@' in smiles
