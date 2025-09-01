#!/usr/bin/env python3
"""Final validation that changes work correctly"""
import sys
sys.path.insert(0, '/workspace/kaggle/neurips-open-polymer-prediction-2025')

def test_final_validation():
    """Validate the tokenizer integration"""
    from transformer_model import SMILESTokenizer, TransformerModel
    import numpy as np
    
    # Test tokenizer
    tokenizer = SMILESTokenizer(max_length=100, use_deepchem=True)
    test_smiles = ['CC(C)C', 'c1ccccc1']
    tokens = tokenizer.tokenize(test_smiles)
    
    assert tokens.shape[0] == 2, "Wrong batch size"
    assert tokens.shape[1] == 100, "Wrong sequence length"
    assert tokenizer.use_deepchem == True, "DeepChem not enabled"
    print(f"✓ Tokenizer working with vocab_size={tokenizer.vocab_size}")
    
    # Test model initialization
    model = TransformerModel(random_state=42)
    assert model.tokenizer.use_deepchem == True, "Model not using DeepChem tokenizer"
    print(f"✓ Model initialized with DeepChem tokenizer")
    
    # Test prediction shape
    dummy_X = np.array(['CC(C)C'] * 10)
    dummy_y = np.random.rand(10, 5)
    
    model.fit(dummy_X[:5], dummy_y[:5])
    preds = model.predict(dummy_X[5:])
    
    assert preds.shape == (5, 5), f"Wrong prediction shape: {preds.shape}"
    print(f"✓ Model predictions have correct shape")
    
    print("\n✅ All validation tests passed!")
    return True

if __name__ == "__main__":
    test_final_validation()