#!/usr/bin/env python3
import pytest
import sys
import os
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_integration():
    """Test that DeepChem tokenizer integration works"""
    from transformer_model import SMILESTokenizer
    
    # Test with DeepChem
    print("Testing DeepChem tokenizer...")
    tokenizer = SMILESTokenizer(max_length=100, use_deepchem=True)
    
    test_smiles = ['CC(C)C', 'c1ccccc1', 'CC(=O)O']
    tokens = tokenizer.tokenize(test_smiles)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Token shape: {tokens.shape}")
    print(f"Sample tokens: {tokens[0][:20]}")
    
    assert tokens.shape == (3, 100), f"Expected shape (3, 100), got {tokens.shape}"
    assert tokenizer.vocab_size > 100, f"Vocab size too small: {tokenizer.vocab_size}"
    
    # Test fallback
    print("\nTesting fallback tokenizer...")
    tokenizer_fallback = SMILESTokenizer(max_length=100, use_deepchem=False)
    tokens_fallback = tokenizer_fallback.tokenize(test_smiles)
    
    print(f"Fallback vocab size: {tokenizer_fallback.vocab_size}")
    print(f"Fallback token shape: {tokens_fallback.shape}")
    print(f"Fallback sample tokens: {tokens_fallback[0][:20]}")
    
    assert tokens_fallback.shape == (3, 100), f"Expected shape (3, 100), got {tokens_fallback.shape}"
    
    print("\nIntegration test passed!")
    return True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])