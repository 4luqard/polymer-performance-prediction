#!/usr/bin/env python3

import pytest
import numpy as np
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_processing import extract_molecular_features

@pytest.fixture
def smiles():
    return {
        'has_multi_spiro': '*Nc1ccc([C@H]2[C@@H]3C[C@H]4C[C@@H](C3)C[C@@H]2C4)cc1N*',
        'has_single_spiro': '*Nc1ccc(23C[C@H]4C(C3)C2C4)cc1N*',
    }

def test_has_multi_spiro(smiles):
    features = extract_molecular_features(smiles['has_multi_spiro'], False)
    assert features['has_spiro'] == True

def test_has_single_spiro(smiles):
    features = extract_molecular_features(smiles['has_single_spiro'], False)
    assert features['has_spiro'] == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
