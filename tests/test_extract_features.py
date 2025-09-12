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

    # Test 4: When there are no '@'
    assert num_tetrahedral_carbon('*Nc1ccc(23C4C(C3)C2C4(C#C*))cc1N*') == 0

    # Test 5: Empty SMILES string
    assert num_tetrahedral_carbon('') == 0

    # Test 6: Incomplete SMILES strings
    assert num_tetrahedral_carbon('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 2

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

    # Test 6: Empty SMILES string
    assert longest_chain_atom_count('') == 0

    # Test 7: Incomplete SMILES strings
    assert longest_chain_atom_count('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 15

def test_num_fused_rings():
    # Test 1: Number of fused rings are 2 since before ring 1 is closed another ring starts without branching,
    # meaning they share some of the aromatic or regular atoms
    assert num_fused_rings('*c1ccc2cc(*)ccc2c1') == 2

    # Test 2: Number of fused rings are 8 since rings 1,2,3 are fused and rings 4,5,6 are fused together but they are a branch so they are not fused with rings 1,2,3
    # There are two of ring 1 and ring 4
    assert num_fused_rings('*c1ccc2c(c1)SC1=Nc3cc(-c4ccc5c(c4)N=C4Sc6cc(*)ccc6N=C4N5)ccc3NC1=N2') == 8

    # Test 3: Number of fused rings are 3 since rings 1,2 are gused while the ring 3 is a ring that has branched so it is not fused
    # There are two of ring 1
    assert num_fused_rings('*c1ccc2c(c1)C(CCCCCC)(CCCCCC)c1cc(-c3cc(CCCCCCCCCC)c(*)cc3CCCCCCCCCC)ccc1-2') == 3

    # Test 4: Number of fused rings are 7, there are actually 7 rings, ring 4 and 5 are named two times so there are two of them,
    # rings 1,2,3 and one of the 4s are fused together, and the other ring 4 and the two ring 5s are fused together but in a branch
    assert num_fused_rings('*=c1cc2ccc3cc(=c4c5ccccc5c(=*)c5ccccc45)cc4ccc(c1)c2c34') == 7

    # Test 5: Empty SMILES string
    assert num_fused_rings('') == 0

    # Test 6: Incomplete SMILES strings
    assert num_fused_rings('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 4

def test_num_rings():
    # Test 1: Number of are 8, althought the ring numbers only got to 6 there are 2 repeating pairs, one is ring 1 and the other is ring 4
    assert num_rings('*c1ccc2c(c1)SC1=Nc3cc(-c4ccc5c(c4)N=C4Sc6cc(*)ccc6N=C4N5)ccc3NC1=N2') == 8

    # Test 5: Empty SMILES string
    assert num_rings('') == 0

    # Test 6: Incomplete SMILES strings
    assert num_rings('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 4

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
