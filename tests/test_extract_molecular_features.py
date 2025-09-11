#!/usr/bin/env python3

import pytest
import numpy as np
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from extract_features import *

def test_num_tetrahedral_carbon():
    # Test 1: Has 1 "@" so there is a single tetrahedral carbon
    assert num_tetrahedral_carbon('*Nc1ccc(23C[C@H]4C(C3)C2C4)cc1N*') == 1

    # Test 2: Has 2 "@" but when two '@'s are next to each other ('@@'),
    # that specifies the configuration of tetrahedral carbon so there is only one tetrahedral carbon
    assert num_tetrahedral_carbon('*Nc1ccc(23C[C@@H]4C(C3)C2C4)cc1N*') == 1

    # Test 3: Has 3 "@" but two '@'s are next to each other so there are only two tetrahedral carbons ("@@", '@')
    assert num_tetrahedral_carbon('*Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C*))cc1N*') == 2

    # Test4: When there are no '@'
    assert num_tetrahedral_carbon('*Nc1ccc(23C4C(C3)C2C4(C#C*))cc1N*') == 0

def test_longest_chain_atom_count():
    # Test 1: Has two branches ('(=O)') but they are short chains,
    # main chain 'C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-N-C-N-C-C-C-C-C-C-N-C-N' is the longest chain of atoms, it has 30 atoms
    # Note: the end can be considered 'N' or the '(=O)' branch, either way the atom count stays the same
    assert longest_chain_atom_count('*CCCCCCCCCCCC(*)CCCCCCNC(=O)NCCCCCCNC(=O)N*') == 30

    # Test 2: The longest chain goes like 'C-C-c-c-c-c-c-S-C-C-C-C-C-C-C-C-C-C-C-C' is a branching chain, it has 20 atoms
    assert longest_chain_atom_count('*C#Cc1ccc(*)c(SCCCCCCCCCCCC)c1') == 20

    # Test 3: Longest chain 'C-C-c-c-c-c-c-S-C-C-C-C-C-C-C-C-C-C-C-C' is a branching chain there is also a branch going out from this chain with 3 C atoms
    # but since they are not in the same chain, it has 20 atoms
    assert longest_chain_atom_count('C#Cc1ccc(*)c(SCCCCCCC(CCC)CCCCC)c1') == 20

    # Test 4: Longest chain 'C-C-c-c-c-c-c-S-C-C-C-C-C-C-C-C-C-C-C-C' is a branching chain there is also a branch going out from this chain with 9 C atoms
    # which is more than the rest of the branch since there are only 5 C atoms are left when this branch comes out
    # that means that the longest chain is where we start 'c-S-C-C-C-C-C-C-C' and then continue with the 9 C atoms in the branch, so the longest chain of atoms has 24 atoms
    assert longest_chain_atom_count('*C#Cc1ccc(*)c(SCCCCCCC(CCCCCCCCC)CCCCC)c1') == 24

    # Test 5: Longest chain: 'c-c-c-c-C-c-c-c-c-N-C-c-c-c-c-O-C-C-N-c-c-c-c-C-C-c-c-c-c-N-O-O' has 34 atoms in its longest chain
    assert longest_chain_atom_count('*c1ccc(Cc2ccc(N3C(=O)c4ccc(OCCN(CCOc5ccc6c(c5)C(=O)N(*)C6=O)c5ccc(C=Cc6ccc([N+](=O)[O-])cc6)cc5)cc4C3=O)cc2)cc1') == 34

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
