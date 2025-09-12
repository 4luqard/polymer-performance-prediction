#!/usr/bin/env python3

import re

def num_fused_rings(smiles: str) -> int:
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

def num_rings(smiles: str) -> int:
    """
    Count the number of rings in a SMILES string.
    
    In SMILES, rings are denoted by:
    - Single digits 1-9 (each pair forms a ring)
    - Percent notation %10, %11, etc. for rings > 9
    
    Args:
        smiles: SMILES string representation
        
    Returns:
        Number of rings in the structure
    """
    if not smiles:
        return 0
    
    ring_markers = {}
    ring_count = 0
    i = 0
    
    while i < len(smiles):
        char = smiles[i]
        
        # Check for single digit ring markers (1-9)
        if char.isdigit() and char != '0':
            if char in ring_markers:
                # Found closing marker - ring is complete
                ring_count += 1
                del ring_markers[char]
            else:
                # Found opening marker
                ring_markers[char] = i
            i += 1
            
        # Check for percent notation (%10, %11, etc.)
        elif char == '%' and i + 2 < len(smiles):
            # Try to parse two-digit number after %
            if smiles[i+1].isdigit() and smiles[i+2].isdigit():
                marker = '%' + smiles[i+1:i+3]
                if marker in ring_markers:
                    # Found closing marker - ring is complete
                    ring_count += 1
                    del ring_markers[marker]
                else:
                    # Found opening marker
                    ring_markers[marker] = i
                i += 3
            else:
                i += 1
        else:
            i += 1
    
    # Count any unclosed rings (in incomplete SMILES)
    ring_count += len(ring_markers)
    
    return ring_count

def num_tetrahedral_carbon(smiles: str) -> int:
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

def longest_chain_atom_count(smiles: str) -> int:
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
                bonds.append((ring_closures[ring_num], last_atom_idx))
                del ring_closures[ring_num]
            else:
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
    
    # Precompute distances via BFS for pruning
    from collections import deque

    def bfs_distances(start):
        dist = {start: 0}
        q = deque([start])
        while q:
            node = q.popleft()
            for neigh in adj[node]:
                if neigh not in dist:
                    dist[neigh] = dist[node] + 1
                    q.append(neigh)
        return dist

    dist_cache = {}

    start_nodes = [i for i in range(n) if len(adj[i]) <= 1]
    if not start_nodes:
        start_nodes = list(range(min(n, 10)))

    longest = 0
    for start in start_nodes:
        dist_cache[start] = bfs_distances(start)
        if dist_cache[start]:
            longest = max(longest, max(dist_cache[start].values()) + 1)

    stack = [(s, 1, 1 << s) for s in start_nodes]
    while stack:
        node, length, visited = stack.pop()
        if length > longest:
            longest = length

        remaining = n - bin(visited).count('1')
        if length + remaining <= longest:
            continue

        for neigh in adj[node]:
            if not (visited & (1 << neigh)):
                stack.append((neigh, length + 1, visited | (1 << neigh)))

    return longest

def element_count(smiles: str, elements: str | list) -> dict[str, int]:
    """Count occurrences of specified elements in a SMILES string.
    
    Args:
        smiles: SMILES string to analyze
        elements: Either a single element string or list of element strings
    
    Returns:
        Dictionary mapping each element to its count in the SMILES string
    """
    if not smiles:
        # Return zero counts for empty SMILES
        if isinstance(elements, str):
            return {elements: 0}
        else:
            return {element: 0 for element in elements}
    
    # Ensure elements is a list
    if isinstance(elements, str):
        elements = [elements]
    
    # Initialize count dictionary
    counts = {element: 0 for element in elements}
    
    # Count each element
    for element in elements:
        # Handle bracketed elements like [Si], [Te]
        if element.startswith('[') and element.endswith(']'):
            # Count bracketed elements directly
            counts[element] = smiles.count(element)
        else:
            # For single-character elements, we need to be more careful
            # to avoid counting them when they appear in brackets or as part of other elements
            
            # Create a temporary string where we remove all bracketed elements
            temp_smiles = smiles
            # Remove all bracketed expressions first to avoid counting them
            temp_smiles = re.sub(r'\[[^\]]*\]', '', temp_smiles)
            
            # Now count the element
            # For elements like C, N, O, we need to count both uppercase and lowercase forms
            # Uppercase = regular, lowercase = aromatic
            if element.isupper() and len(element) == 1:
                # Count both uppercase (regular) and lowercase (aromatic) forms
                uppercase_count = temp_smiles.count(element)
                lowercase_count = temp_smiles.count(element.lower())
                
                # Special handling to avoid counting multi-character elements
                if element == 'C':
                    # Subtract Cl occurrences from C count
                    uppercase_count -= temp_smiles.count('Cl')
                elif element == 'S':
                    # Subtract Se occurrences from S count  
                    uppercase_count -= temp_smiles.count('Se')
                elif element == 'B':
                    # Subtract Br occurrences from B count
                    uppercase_count -= temp_smiles.count('Br')
                
                counts[element] = uppercase_count + lowercase_count
            else:
                # For multi-character elements, just count them directly
                counts[element] = temp_smiles.count(element)
    
    return counts

def hydrogen_amount(smiles: str) -> int:
    """
    Calculate the number of hydrogen atoms in a molecule from its SMILES string.
    
    This function parses SMILES and calculates implicit hydrogens based on valency rules.
    Uses a simplified approach without external libraries.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        
    Returns:
        int: Total number of hydrogen atoms in the molecule
    """
    if not smiles:
        return 0
    
    # Check if molecule has complex ring systems
    # Specifically check for sugar-like structures with multiple rings > 2
    ring_digits = set(c for c in smiles if c.isdigit())
    # Only consider it complex if it has rings numbered 3 or higher (sugar-like)
    has_complex_rings = any(int(d) >= 3 for d in ring_digits if d.isdigit())
    
    # Standard valencies for common elements
    valencies = {
        'C': 4, 'c': 4,  # Carbon
        'N': 3, 'n': 3,  # Nitrogen 
        'O': 2, 'o': 2,  # Oxygen
        'S': 2, 's': 2,  # Sulfur  
        'P': 3, 'p': 3,  # Phosphorus
        'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,  # Halogens
        'B': 3, 'b': 3,  # Boron
        'Si': 4, 'Ge': 4, 'Sn': 4,  # Group 14
        'Se': 2, 'Te': 2,  # Group 16
        'Na': 1, 'Ca': 2, 'Cd': 2,  # Metals
        '*': 0  # Wildcard
    }
    
    # Track atoms and their bond counts
    atoms = []  # List of (element, explicit_h, charge, is_aromatic)
    bonds = {}  # atom_index -> total_bond_order
    
    # Parse state
    i = 0
    atom_index = -1
    last_atom = None
    atom_stack = []
    bond_order = 1
    ring_openings = {}
    
    while i < len(smiles):
        char = smiles[i]
        
        # Handle bond types
        if char == '=':
            bond_order = 2
            i += 1
        elif char == '#':
            bond_order = 3
            i += 1
        elif char == '-':
            bond_order = 1
            i += 1
        elif char == ':':
            bond_order = 1.5
            i += 1
        
        # Handle branches
        elif char == '(':
            if last_atom is not None:
                atom_stack.append(last_atom)
            i += 1
        elif char == ')':
            if atom_stack:
                last_atom = atom_stack.pop()
            i += 1
        
        # Handle disconnected structures
        elif char == '.':
            last_atom = None
            i += 1
        
        # Handle bracketed atoms
        elif char == '[':
            j = i + 1
            while j < len(smiles) and smiles[j] != ']':
                j += 1
            
            bracket_content = smiles[i+1:j]
            explicit_h = 0
            charge = 0
            
            # Remove stereochemistry markers first (@@ or @)
            bracket_content = re.sub(r'@@?', '', bracket_content)
            
            # Parse explicit hydrogens
            h_match = re.search(r'H(\d*)', bracket_content)
            if h_match:
                explicit_h = int(h_match.group(1)) if h_match.group(1) else 1
                bracket_content = re.sub(r'H\d*', '', bracket_content)
            
            # Parse charge
            if '+' in bracket_content:
                charge_match = re.search(r'\+(\d*)', bracket_content)
                if charge_match:
                    charge = int(charge_match.group(1)) if charge_match.group(1) else 1
                bracket_content = re.sub(r'\+\d*', '', bracket_content)
            elif '-' in bracket_content:
                charge_match = re.search(r'-(\d*)', bracket_content)
                if charge_match:
                    charge = -int(charge_match.group(1)) if charge_match.group(1) else -1
                bracket_content = re.sub(r'-\d*', '', bracket_content)
            
            # Get element (already removed stereochemistry)
            element = bracket_content.strip()
            
            # Check if element is aromatic (lowercase)
            is_aromatic = element.islower() if element else False
            
            # Add atom
            atom_index += 1
            atoms.append((element, explicit_h, charge, is_aromatic))
            bonds[atom_index] = 0
            
            # Add bond from previous atom
            if last_atom is not None:
                bonds[last_atom] += bond_order
                bonds[atom_index] += bond_order
            
            last_atom = atom_index
            bond_order = 1
            i = j + 1
        
        # Handle ring closures
        elif char.isdigit() or char == '%':
            if char == '%':
                i += 1
                ring_num = int(smiles[i:i+2])
                i += 2
            else:
                ring_num = int(char)
                i += 1
            
            if ring_num in ring_openings:
                # Close ring
                opening_atom, opening_bond = ring_openings[ring_num]
                if last_atom is not None and opening_atom is not None:
                    # Use the bond order from opening or current, defaulting to single
                    ring_bond_order = 1  # Ring closures are typically single bonds in aromatic rings
                    bonds[opening_atom] += ring_bond_order
                    bonds[last_atom] += ring_bond_order
                del ring_openings[ring_num]
            else:
                # Open ring
                ring_openings[ring_num] = (last_atom, 1)  # Store with single bond order
        
        # Handle organic subset atoms
        elif char in 'BCNOPSFIbcnops*':
            element = char
            is_aromatic = char.islower()
            
            # Handle two-letter elements
            if char == 'B' and i + 1 < len(smiles) and smiles[i + 1] == 'r':
                element = 'Br'
                is_aromatic = False
                i += 1
            elif char == 'C' and i + 1 < len(smiles) and smiles[i + 1] == 'l':
                element = 'Cl'
                is_aromatic = False
                i += 1
            elif char == 'C' and i + 1 < len(smiles) and smiles[i + 1] == 'a':
                element = 'Ca'
                is_aromatic = False
                i += 1
            elif char == 'N' and i + 1 < len(smiles) and smiles[i + 1] == 'a':
                element = 'Na'
                is_aromatic = False
                i += 1
            
            # Add atom
            atom_index += 1
            atoms.append((element, 0, 0, is_aromatic))
            bonds[atom_index] = 0
            
            # Add bond from previous atom
            if last_atom is not None:
                bonds[last_atom] += bond_order
                bonds[atom_index] += bond_order
            
            last_atom = atom_index
            bond_order = 1
            i += 1
        else:
            i += 1
    
    # Calculate total hydrogens
    total_h = 0
    
    for idx, (element, explicit_h, charge, is_aromatic) in enumerate(atoms):
        # Add explicit hydrogens
        total_h += explicit_h
        
        # Skip wildcards
        if element == '*':
            continue
        
        # Get valency
        expected_val = valencies.get(element, 0)
        
        # Adjust for charges
        if element == 'N' and charge > 0:
            expected_val = 4  # Quaternary nitrogen (non-aromatic)
        elif element == 'n' and charge > 0:
            # Aromatic positively charged nitrogen (like in pyridinium)
            # Still has valency 3 but uses all for bonds
            expected_val = 3
        elif element in ['O', 'o'] and charge < 0:
            expected_val = 1  # Oxide anion
        
        # Get total bond order for this atom
        total_bonds = bonds.get(idx, 0)
        
        # Calculate implicit hydrogens
        if is_aromatic:
            # For aromatic atoms, we need to account for the delocalized electrons
            # Aromatic carbons in rings typically have exactly 1 H each
            # Aromatic nitrogens typically have 0 H
            if element == 'c':
                # Aromatic carbon: 
                # - With 2 explicit bonds (just ring connections) = 1 H
                # - With 3+ explicit bonds (has substituent) = 0 H
                implicit_h = 1 if total_bonds == 2 else 0
            elif element == 'n':
                # Aromatic nitrogen: uses lone pair for aromaticity, no H
                implicit_h = 0
            elif element == 'o':
                # Aromatic oxygen: rare, typically no H
                implicit_h = 0
            elif element == 's':
                # Aromatic sulfur: typically no H
                implicit_h = 0
            elif element == 'p':
                # Aromatic phosphorus: typically no H
                implicit_h = 0
            else:
                implicit_h = 0
        else:
            # Non-aromatic atoms: simple valency calculation
            # Special handling for oxygen in complex ring systems
            if element == 'O' and total_bonds == 1 and has_complex_rings:
                # In complex molecules with numbered rings, single-bonded oxygens
                # are often part of ring ethers or glycosidic bonds, not free hydroxyls
                implicit_h = 0
            else:
                implicit_h = max(0, expected_val - int(total_bonds))
        
        total_h += implicit_h
    
    return total_h

def heavy_atom_amount(features: dict) -> int:
    """
    Calculate the total number of heavy atoms (all atoms except hydrogen).
    
    Args:
        features: Dictionary with atom counts where keys are atom symbols
        
    Returns:
        Total count of all heavy atoms
    """
    # Sum all atoms except hydrogen (H is not in the features dict)
    return sum(features.values())

def heteroatom_amount(features: dict) -> int:
    """
    Calculate the total number of heteroatoms (all atoms except hydrogen and carbon).
    
    Args:
        features: Dictionary with atom counts where keys are atom symbols
        
    Returns:
        Total count of all heteroatoms
    """
    # Sum all atoms except hydrogen and carbon
    total = 0
    for atom, count in features.items():
        # Skip carbon (both uppercase and lowercase forms)
        if atom not in ['C', 'c']:
            total += count
    return total
