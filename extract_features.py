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
    
    count = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == '@':
            # Check if it's part of '@@'
            if i + 1 < len(smiles) and smiles[i + 1] == '@':
                # Found '@@', count as one stereocenter
                count += 1
                i += 2  # Skip both '@' symbols
            else:
                # Found single '@', count as one stereocenter
                count += 1
                i += 1
        else:
            i += 1
    
    return count
