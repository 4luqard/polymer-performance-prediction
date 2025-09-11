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

def num_tetrahedral_carbon(smiles):
    """
    Count the number of tetrahedral carbon stereocenters in a SMILES string.
    
    In SMILES notation:
    - '@' indicates a tetrahedral carbon stereocenter
    - '@@' (two consecutive '@') represents a single stereocenter with specific configuration
    - Multiple non-consecutive '@' symbols should be counted separately
    
    Args:
        smiles (str): SMILES string representation of the molecule
        
    Returns:
        int: Number of tetrahedral carbon stereocenters
    """
    if not smiles:
        return 0
    
    # Replace '@@' with a placeholder to count it as single stereocenter
    modified_smiles = smiles.replace('@@', '#')
    
    # Count remaining '@' symbols
    single_at_count = modified_smiles.count('@')
    
    # Count the placeholders (which represent '@@')
    double_at_count = modified_smiles.count('#')
    
    # Total count is single '@' plus '@@' occurrences
    return single_at_count + double_at_count
