#!/usr/bin/env python3

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

def longest_chain_atom_count(smiles):
    """
    Find the longest chain of atoms in a SMILES string.
    
    This function parses the SMILES manually and finds the longest
    path through the molecular graph.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        
    Returns:
        int: Number of atoms in the longest chain
    """
    if not smiles:
        return 0
    
    # Simple but effective SMILES parser for atom extraction
    atoms = []
    bonds = []
    atom_stack = []  # Stack for handling branches
    ring_closures = {}  # Dict to track ring opening/closing
    last_atom_idx = -1
    
    i = 0
    while i < len(smiles):
        char = smiles[i]
        
        # Skip special markers and bond types
        if char in '*=#-+./@\\':
            i += 1
            continue
            
        # Handle bracketed atoms like [N+], [O-], etc.
        elif char == '[':
            j = i + 1
            while j < len(smiles) and smiles[j] != ']':
                j += 1
            # Extract element from bracket (skip charges, isotopes)
            bracket_content = smiles[i+1:j]
            element = ''
            for k, c in enumerate(bracket_content):
                if c.isupper():
                    element = c
                    if k + 1 < len(bracket_content) and bracket_content[k+1].islower():
                        element += bracket_content[k+1]
                    break
            if element:
                atom_idx = len(atoms)
                atoms.append(element)
                if last_atom_idx >= 0:
                    bonds.append((last_atom_idx, atom_idx))
                last_atom_idx = atom_idx
            i = j + 1
            
        # Handle branch start
        elif char == '(':
            atom_stack.append(last_atom_idx)
            i += 1
            
        # Handle branch end
        elif char == ')':
            if atom_stack:
                last_atom_idx = atom_stack.pop()
            i += 1
            
        # Handle ring closures (single digit)
        elif char.isdigit():
            ring_num = int(char)
            if ring_num in ring_closures:
                # Close the ring
                bonds.append((ring_closures[ring_num], last_atom_idx))
                del ring_closures[ring_num]
            else:
                # Open a ring
                ring_closures[ring_num] = last_atom_idx
            i += 1
            
        # Handle two-digit ring closures (%10, %11, etc.)
        elif char == '%':
            if i + 2 < len(smiles) and smiles[i+1:i+3].isdigit():
                ring_num = int(smiles[i+1:i+3])
                if ring_num in ring_closures:
                    bonds.append((ring_closures[ring_num], last_atom_idx))
                    del ring_closures[ring_num]
                else:
                    ring_closures[ring_num] = last_atom_idx
                i += 3
            else:
                i += 1
                
        # Handle regular atoms
        elif char.isupper():
            # Check for two-letter elements
            element = char
            if i + 1 < len(smiles) and smiles[i+1].islower() and smiles[i+1] not in 'cnops':
                element += smiles[i+1]
                i += 1
            
            atom_idx = len(atoms)
            atoms.append(element)
            if last_atom_idx >= 0:
                bonds.append((last_atom_idx, atom_idx))
            last_atom_idx = atom_idx
            i += 1
            
        # Handle aromatic atoms (lowercase)
        elif char in 'cnops':
            atom_idx = len(atoms)
            atoms.append(char.upper())
            if last_atom_idx >= 0:
                bonds.append((last_atom_idx, atom_idx))
            last_atom_idx = atom_idx
            i += 1
            
        else:
            i += 1
    
    if not atoms:
        return 0
    
    # Build adjacency list
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for a, b in bonds:
        if 0 <= a < n and 0 <= b < n:
            adj[a].add(b)
            adj[b].add(a)
    
    # Find longest path using DFS
    def dfs_longest_path(start, visited):
        """DFS to find longest path from start node."""
        visited = visited | {start}
        max_path = 0
        
        for neighbor in adj[start]:
            if neighbor not in visited:
                path_len = dfs_longest_path(neighbor, visited)
                max_path = max(max_path, path_len)
        
        return max_path + 1
    
    # Try from all nodes and find the maximum
    longest = 0
    
    # For efficiency, start with leaf nodes (degree 1) as they're likely endpoints
    leaf_nodes = [i for i in range(n) if len(adj[i]) <= 1]
    
    # If there are leaf nodes, check from them first
    if leaf_nodes:
        for start in leaf_nodes:
            path_len = dfs_longest_path(start, frozenset())
            longest = max(longest, path_len)
    
    # For graphs without clear leaves or to ensure completeness,
    # sample a few more starting points
    if longest == 0 or n <= 20:  # For small molecules, check all
        for start in range(min(n, 10)):  # Limit for large molecules
            path_len = dfs_longest_path(start, frozenset())
            longest = max(longest, path_len)
    
    return longest


