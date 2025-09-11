# %% [code]
#!/usr/bin/env python3

def num_fused_rings(smiles):
    """
    Count the number of fused rings in a SMILES string.
    
    Fused rings share at least two consecutive atoms (an edge).
    Rings in branches marked with '-' are considered separately.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        
    Returns:
        int: Total number of fused rings
    """
    if not smiles:
        return 0
    
    # Parse SMILES to identify all rings with branch context
    ring_positions = {}  # ring_number -> [(position, atom_index, in_branch)]
    atom_index = 0
    i = 0
    in_branch = False
    paren_stack = []
    
    while i < len(smiles):
        char = smiles[i]
        
        # Check for branch marker (-c or -C followed by ring in parentheses)
        if char == '-' and i + 1 < len(smiles):
            next_char = smiles[i + 1]
            if next_char in 'cC' and i + 2 < len(smiles):
                # Look ahead for ring digit after potential branch
                j = i + 2
                while j < len(smiles) and j < i + 10:
                    if smiles[j].isdigit():
                        # Found a ring after -c, this indicates a branch
                        in_branch = True
                        break
                    elif smiles[j] == '(':
                        break
                    j += 1
        
        # Handle parentheses
        if char == '(':
            # Check if this is a branch opening  
            if i > 0 and smiles[i-1] == '-':
                in_branch = True
            paren_stack.append(in_branch)
            i += 1
            continue
        elif char == ')':
            if paren_stack:
                paren_stack.pop()
            in_branch = paren_stack[-1] if paren_stack else False
            i += 1
            continue
        
        # Skip special chars that don't represent atoms
        if char in '[]+-=#@./*':
            i += 1
            continue
        
        # Check for ring markers (single digit 1-9)
        if char.isdigit():
            ring_num = int(char)
            if ring_num not in ring_positions:
                ring_positions[ring_num] = []
            ring_positions[ring_num].append((i, atom_index, in_branch))
            i += 1
            continue
        
        # Check for two-digit ring numbers (%10-%99)
        if char == '%' and i + 2 < len(smiles):
            if smiles[i+1:i+3].isdigit():
                ring_num = int(smiles[i+1:i+3])
                if ring_num not in ring_positions:
                    ring_positions[ring_num] = []
                ring_positions[ring_num].append((i, atom_index, in_branch))
                i += 3
                continue
        
        # Count atoms (element symbols)
        if char.isalpha():
            # Handle two-letter elements
            if char in 'BCNOSPF' and i + 1 < len(smiles):
                next_char = smiles[i + 1]
                if char + next_char in ['Cl', 'Br', 'Si', 'Se', 'As', 'Al', 'Mg', 'Ca', 'Fe', 'Cu', 'Zn']:
                    atom_index += 1
                    i += 2
                    continue
            
            atom_index += 1
            i += 1
            continue
        
        i += 1
    
    # Create list of rings from positions
    rings = []
    ring_id = 0
    
    for ring_num, positions in ring_positions.items():
        # Pair up opens and closes
        for j in range(0, len(positions), 2):
            if j + 1 < len(positions):
                start_pos, start_atom, start_branch = positions[j]
                end_pos, end_atom, end_branch = positions[j + 1]
                
                # Ring is in branch if either endpoint is in branch
                is_branched = start_branch or end_branch
                
                rings.append({
                    'id': ring_id,
                    'number': ring_num,
                    'start_atom': start_atom,
                    'end_atom': end_atom,
                    'atom_range': (min(start_atom, end_atom), max(start_atom, end_atom)),
                    'is_branched': is_branched
                })
                ring_id += 1
            elif len(positions) == 1:
                # Handle incomplete SMILES with unclosed rings
                # Treat single ring marker as a complete ring from that position to end
                start_pos, start_atom, start_branch = positions[0]
                # Assume ring closes at the last atom
                end_atom = atom_index - 1 if atom_index > 0 else start_atom
                
                rings.append({
                    'id': ring_id,
                    'number': ring_num,
                    'start_atom': start_atom,
                    'end_atom': end_atom,
                    'atom_range': (min(start_atom, end_atom), max(start_atom, end_atom)),
                    'is_branched': start_branch
                })
                ring_id += 1
    
    # Separate main chain and branched rings
    main_rings = [r for r in rings if not r['is_branched']]
    branch_rings = [r for r in rings if r['is_branched']]
    
    # Find fused rings in main chain
    fused_rings = set()
    
    for i in range(len(main_rings)):
        for j in range(i + 1, len(main_rings)):
            ring1 = main_rings[i]
            ring2 = main_rings[j]
            
            # Check if rings share at least 2 atoms
            start1, end1 = ring1['atom_range']
            start2, end2 = ring2['atom_range']
            
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_end - overlap_start >= 1:
                fused_rings.add(ring1['id'])
                fused_rings.add(ring2['id'])
    
    # Find fused rings in branches (they fuse among themselves, not with main chain)
    for i in range(len(branch_rings)):
        for j in range(i + 1, len(branch_rings)):
            ring1 = branch_rings[i]
            ring2 = branch_rings[j]
            
            start1, end1 = ring1['atom_range']
            start2, end2 = ring2['atom_range']
            
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_end - overlap_start >= 1:
                fused_rings.add(ring1['id'])
                fused_rings.add(ring2['id'])
    
    return len(fused_rings)

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
        
        # Skip special markers and bond types (but not atom markers)
        if char in '*=-+./@\\':
            i += 1
            continue
        
        # Handle triple bond specially to not skip following atom  
        elif char == '#':
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
    
    # Close any unclosed rings (for incomplete SMILES)
    # For each unclosed ring, we assume there's a virtual atom that would close it
    for ring_num, atom_idx in ring_closures.items():
        if 0 <= atom_idx < len(atoms):
            # Add a virtual atom to close the ring
            virtual_atom_idx = len(atoms)
            atoms.append('C')  # Assume it's a carbon
            if last_atom_idx >= 0:
                bonds.append((last_atom_idx, virtual_atom_idx))
            bonds.append((atom_idx, virtual_atom_idx))
    
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


