#!/usr/bin/env python3
import pytest
import numpy as np
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from extract_features import *

def test_num_tetrahedral_carbon():
    # Has 1 "@" so there is a single tetrahedral carbon
    assert num_tetrahedral_carbon('*Nc1ccc(23C[C@H]4C(C3)C2C4)cc1N*') == 1

    # Has 2 "@" but when two '@'s are next to each other ('@@'),
    # that specifies the configuration of tetrahedral carbon so there is only one tetrahedral carbon
    assert num_tetrahedral_carbon('*Nc1ccc(23C[C@@H]4C(C3)C2C4)cc1N*') == 1

    # Has 3 "@" but two '@'s are next to each other so there are only two tetrahedral carbons ("@@", '@')
    assert num_tetrahedral_carbon('*Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C*))cc1N*') == 2

    # When there are no '@'
    assert num_tetrahedral_carbon('*Nc1ccc(23C4C(C3)C2C4(C#C*))cc1N*') == 0

    # Empty SMILES string
    assert num_tetrahedral_carbon('') == 0

    # Incomplete SMILES strings
    assert num_tetrahedral_carbon('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 2

def test_longest_chain_atom_count():
    # Has two branches ('(=O)') but they are short chains,
    # main chain 'C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-C-N-C-N-C-C-C-C-C-C-N-C-N' is the longest chain of atoms, it has 30 atoms
    # Note: the end can be considered 'N' or the '(=O)' branch, either way the atom count stays the same
    assert longest_chain_atom_count('*CCCCCCCCCCCC(*)CCCCCCNC(=O)NCCCCCCNC(=O)N*') == 30

    # The longest chain goes like 'C-C-c-c-c-c-c-S-C-C-C-C-C-C-C-C-C-C-C-C' is a branching chain, it has 20 atoms
    assert longest_chain_atom_count('*C#Cc1ccc(*)c(SCCCCCCCCCCCC)c1') == 20

    # Longest chain 'C-C-c-c-c-c-c-S-C-C-C-C-C-C-C-C-C-C-C-C' is a branching chain there is also a branch going out from this chain with 3 C atoms
    # but since they are not in the same chain, it has 20 atoms
    assert longest_chain_atom_count('C#Cc1ccc(*)c(SCCCCCCC(CCC)CCCCC)c1') == 20

    # Longest chain 'C-C-c-c-c-c-c-S-C-C-C-C-C-C-C-C-C-C-C-C' is a branching chain there is also a branch going out from this chain with 9 C atoms
    # which is more than the rest of the branch since there are only 5 C atoms are left when this branch comes out
    # that means that the longest chain is where we start 'c-S-C-C-C-C-C-C-C' and then continue with the 9 C atoms in the branch, so the longest chain of atoms has 24 atoms
    assert longest_chain_atom_count('*C#Cc1ccc(*)c(SCCCCCCC(CCCCCCCCC)CCCCC)c1') == 24

    # Longest chain: 'c-c-c-c-C-c-c-c-c-N-C-c-c-c-c-O-C-C-N-c-c-c-c-C-C-c-c-c-c-N-O-O' has 34 atoms in its longest chain
    assert longest_chain_atom_count('*c1ccc(Cc2ccc(N3C(=O)c4ccc(OCCN(CCOc5ccc6c(c5)C(=O)N(*)C6=O)c5ccc(C=Cc6ccc([N+](=O)[O-])cc6)cc5)cc4C3=O)cc2)cc1') == 34

    # Empty SMILES string
    assert longest_chain_atom_count('') == 0

    # Incomplete SMILES strings
    # TODO: the test is wrong, it needs to be 13
    assert longest_chain_atom_count('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 15

def test_num_fused_rings():
    # Number of fused rings are 2 since before ring 1 is closed another ring starts without branching,
    # meaning they share some of the aromatic or regular atoms
    assert num_fused_rings('*c1ccc2cc(*)ccc2c1') == 2

    # Number of fused rings are 8 since rings 1,2,3 are fused and rings 4,5,6 are fused together but they are a branch so they are not fused with rings 1,2,3
    # There are two of ring 1 and ring 4
    assert num_fused_rings('*c1ccc2c(c1)SC1=Nc3cc(-c4ccc5c(c4)N=C4Sc6cc(*)ccc6N=C4N5)ccc3NC1=N2') == 8

    # Number of fused rings are 3 since rings 1,2 are gused while the ring 3 is a ring that has branched so it is not fused
    # There are two of ring 1
    assert num_fused_rings('*c1ccc2c(c1)C(CCCCCC)(CCCCCC)c1cc(-c3cc(CCCCCCCCCC)c(*)cc3CCCCCCCCCC)ccc1-2') == 3

    # Number of fused rings are 7, there are actually 7 rings, ring 4 and 5 are named two times so there are two of them,
    # rings 1,2,3 and one of the 4s are fused together, and the other ring 4 and the two ring 5s are fused together but in a branch
    assert num_fused_rings('*=c1cc2ccc3cc(=c4c5ccccc5c(=*)c5ccccc45)cc4ccc(c1)c2c34') == 7

    # Empty SMILES string
    assert num_fused_rings('') == 0

    # Incomplete SMILES strings
    assert num_fused_rings('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 4

def test_num_rings():
    # Number of are 8, althought the ring numbers only got to 6 there are 2 repeating pairs, one is ring 1 and the other is ring 4
    assert num_rings('*c1ccc2c(c1)SC1=Nc3cc(-c4ccc5c(c4)N=C4Sc6cc(*)ccc6N=C4N5)ccc3NC1=N2') == 8

    # Empty SMILES string
    assert num_rings('') == 0

    # Incomplete SMILES strings
    assert num_rings('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C') == 4


def test_element_count():
    # Carbon count
    assert element_count('*CC(*)(C)C(=O)OCc1cc(Cl)ccc1Cl', 'C') == {'C': 11}

    # Sulfur and Silicon count
    assert element_count('*c1ccc(Oc2ccc(S(=O)(=O)c3ccc(Oc4ccc(N5C(=O)c6ccc([Si](C)(C)c7ccc8c(c7)C(=O)N(*)C8=O)cc6C5=O)cc4)cc3)cc2)cc1', ['S', '[Si]']) == {'S': 1,'[Si]': 1}

    # Empty SMILES string
    assert element_count('', ['C', 'S', '[Te]']) == {'C': 0, 'S': 0, '[Te]': 0}

    # Incomplete SMILES strings
    assert element_count('Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C', 'N') == {'N': 1}

def test_hydrogen_amount():
    # Vanilin
    assert hydrogen_amount('O=Cc1ccc(O)c(OC)c1') == 8

    # Bergenin
    assert hydrogen_amount('OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC(*))c(O)cc3C(=O)O2') == 15

    # Flavopereirin
    assert hydrogen_amount('*CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1') == 14

    # Thiamine
    assert hydrogen_amount('OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N') == 17

    # Empty SMILES strings
    assert hydrogen_amount('') == 0

def test_heavy_atom_amount():
    # Sum of every atom except hydrogen
    assert heavy_atom_amount({'C': 1, 'N': 4, 'O': 3}) == 8

    # Empty dict
    assert heavy_atom_amount({}) == 0

    # Empty SMILES string
    assert heavy_atom_amount({'C': 0, 'N': 0, 'O': 0}) == 0

def test_heteroatom_amount():
    # Sum of every atom except hydrogen and carbon
    assert heteroatom_amount({'C': 1, 'N': 4, 'O': 3}) == 7

    # Empty dict
    assert heteroatom_amount({}) == 0

    # Empty SMILES string
    assert heteroatom_amount({'C': 0, 'N': 0, 'O': 0}) == 0

def test_molecular_weight():
    # Sum of every atom's atomic weight
    assert np.isclose(molecular_weight({'C': 1, 'N': 4, 'O': 3, 'H': 9}), 125.108, atol=1e-2)

    # Empty dict
    assert molecular_weight({}) == 0

    # Empty SMILES string
    assert molecular_weight({'C': 0, 'N': 0, 'O': 0}) == 0

def test_vdw_volume():
    # Sum of every atom's van der Waals volume in centimeters cuber per mole
    assert np.isclose(vdw_volume({'C': 1, 'N': 4, 'O': 3, 'H': 9}), 115.815, atol=1e-2)

    # Empty dict
    assert vdw_volume({}) == 0

    # Empty SMILES string
    assert vdw_volume({'C': 0, 'N': 0, 'O': 0}) == 0

def test_density_estimate():
    # weight over volume, unit
    assert np.isclose(density_estimate({'molecular_weight': 125.108, 'vdw_volume': 115.815}), 1.080, atol=1e-2)

    # Empty dict
    assert density_estimate({}) == 0

    # Empty SMILES string
    assert density_estimate({'molecular_weight': 0, 'vdw_volume': 0}) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
