import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_model import SMILESTokenizer, TransformerModel


class TestDeepChemTokenizer:
    """Test DeepChem tokenizer integration in transformer_model.py"""
    
    def test_tokenizer_initialization(self):
        """Test that DeepChem tokenizer initializes correctly"""
        tokenizer = SMILESTokenizer(max_length=150, use_deepchem=True)
        
        # Check tokenizer was created
        assert tokenizer is not None
        assert tokenizer.max_length == 150
        
        # Check it has required attributes
        assert hasattr(tokenizer, 'tokenize')
        assert hasattr(tokenizer, 'vocab_size')
        assert tokenizer.vocab_size > 0
    
    def test_tokenizer_no_transformers_import(self):
        """Verify no direct transformers/tokenizers imports in transformer_model.py"""
        with open('transformer_model.py', 'r') as f:
            content = f.read()
        
        # Check for direct imports at module level
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('import '):
                assert 'transformers' not in line, f"Found transformers import: {line}"
                assert 'tokenizers' not in line, f"Found tokenizers import: {line}"
            if line.strip().startswith('from ') and not 'deepchem' in line:
                assert 'transformers' not in line, f"Found transformers import: {line}"
                assert 'tokenizers' not in line, f"Found tokenizers import: {line}"
    
    def test_tokenizer_uses_deepchem(self):
        """Test that tokenizer uses DeepChem's SmilesTokenizer"""
        with open('transformer_model.py', 'r') as f:
            content = f.read()
        
        # Check for DeepChem import
        assert 'from deepchem.feat.smiles_tokenizer import SmilesTokenizer' in content
        # Check no AutoTokenizer from transformers
        assert 'from transformers import AutoTokenizer' not in content
    
    def test_tokenize_smiles(self):
        """Test tokenizing SMILES strings"""
        tokenizer = SMILESTokenizer(max_length=50, use_deepchem=True)
        
        # Test single SMILES
        smiles = "CC(C)C(=O)O"
        tokens = tokenizer.tokenize(smiles)
        
        assert len(tokens) == 1
        assert len(tokens[0]) == 50  # Should be padded to max_length
        assert all(isinstance(t, (int, np.integer)) for t in tokens[0])
    
    def test_tokenize_multiple_smiles(self):
        """Test tokenizing multiple SMILES strings"""
        tokenizer = SMILESTokenizer(max_length=50, use_deepchem=True)
        
        smiles_list = ["CC(C)C(=O)O", "C1=CC=CC=C1", "CCO"]
        tokens = tokenizer.tokenize(smiles_list)
        
        assert len(tokens) == 3
        for token_seq in tokens:
            assert len(token_seq) == 50
            assert all(isinstance(t, (int, np.integer)) for t in token_seq)
    
    def test_fallback_to_character_level(self):
        """Test fallback to character-level tokenization"""
        tokenizer = SMILESTokenizer(max_length=50, use_deepchem=False)
        
        assert not tokenizer.use_deepchem
        assert hasattr(tokenizer, 'char_to_idx')
        assert tokenizer.vocab_size > 0
        
        # Test tokenization still works
        tokens = tokenizer.tokenize("CCO")
        assert len(tokens) == 1
        assert len(tokens[0]) == 50
    
    def test_optimizer_is_adam(self):
        """Test that optimizer is Adam not AdamW"""
        with open('transformer_model.py', 'r') as f:
            content = f.read()
        
        # Check for Adam optimizer
        assert 'keras.optimizers.Adam(' in content
        assert 'keras.optimizers.AdamW(' not in content
    
    def test_linear_activations(self):
        """Test that activations are linear in feed-forward layers"""
        with open('transformer_model.py', 'r') as f:
            content = f.read()
        
        # Count linear activations vs gelu
        linear_count = content.count('activation="linear"')
        assert linear_count >= 3, f"Expected at least 3 linear activations, found {linear_count}"
    
    def test_final_loss_output(self):
        """Test that final epoch losses are printed"""
        with open('transformer_model.py', 'r') as f:
            content = f.read()
        
        # Check for final loss output code
        assert 'Final Epoch - Training Loss:' in content
        assert 'final_train_loss' in content
        assert 'final_val_loss' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])