#!/usr/bin/env python3

import pytest
import numpy as np
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from extract_features import *

def test_has_tetrahedral_carbon():
    assert has_tetrahedral_carbon('*Nc1ccc(23C[C@H]4C(C3)C2C4)cc1N*') # Test 1: Has '@' so there is tethrahedral carbon

def test_num_tetrahedral_carbon():
    assert num_tetrahedral_carbon('*Nc1ccc(23C[C@H]4C(C3)C2C4)cc1N*') == 1 # Test 1: Has 1 "@" so there is a single tetrahedral carbon
    assert num_tetrahedral_carbon('*Nc1ccc(23C[C@@H]4C(C3)C2C4)cc1N*') == 1 # Test 2: Has 2 "@" but when two '@'s are next to each other ('@@') than that specifies the configuration of tetrahedral carbon so there is only one tetrahedral carbon
    assert num_tetrahedral_carbon('*Nc1ccc(2[C@H]3C[C@@H]4C(C3)C2C4(C#C*))cc1N*') == 2 # Test 3: Has 3 "@" but two '@'s are next to each other so there are only two tetrahedral carbons ("@@", '@')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
