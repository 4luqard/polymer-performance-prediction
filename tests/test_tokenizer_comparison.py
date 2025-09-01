import pytest
import sys
import os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_tokenizer_comparison():
    """Compare current SMILESTokenizer with DeepChem's SmilesTokenizer"""
    
    # Import tokenizers
    import sys
    sys.path.insert(0, '/workspace/kaggle/neurips-open-polymer-prediction-2025')
    from transformer_model import SMILESTokenizer as CurrentTokenizer
    from deepchem.feat import SmilesTokenizer as DeepChemTokenizer
    
    # Sample SMILES for testing
    test_smiles = [
        'CC(C)C',
        'C1=CC=CC=C1',
        'CC(=O)O',
        'C1CCCCC1',
        'CC(C)(C)C',
        'c1ccccc1',
        'O=C(O)C',
        'CC1=CC=CC=C1',
        'C1=CC=C(C=C1)O',
        'CC(C)CC(C)C'
    ]
    
    # Initialize tokenizers
    current_tokenizer = CurrentTokenizer(max_length=100)
    # DeepChem tokenizer needs a vocab file or pretrained model
    from transformers import AutoTokenizer
    deepchem_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    
    # Tokenize with current tokenizer
    current_tokens = current_tokenizer.tokenize(test_smiles)
    
    # Tokenize with DeepChem tokenizer
    deepchem_tokens = []
    for smiles in test_smiles:
        tokens = deepchem_tokenizer.encode(smiles, add_special_tokens=False)
        # Pad or truncate to match length
        if len(tokens) < 100:
            tokens = tokens + [0] * (100 - len(tokens))
        else:
            tokens = tokens[:100]
        deepchem_tokens.append(tokens)
    deepchem_tokens = np.array(deepchem_tokens)
    
    # Calculate correlations
    correlations = []
    for i in range(len(test_smiles)):
        if np.std(current_tokens[i]) > 0 and np.std(deepchem_tokens[i]) > 0:
            corr, _ = spearmanr(current_tokens[i], deepchem_tokens[i])
            if not np.isnan(corr):
                correlations.append(corr)
    
    avg_correlation = np.mean(correlations) if correlations else 0
    
    # Calculate cosine similarity
    similarities = []
    for i in range(len(test_smiles)):
        sim = cosine_similarity([current_tokens[i]], [deepchem_tokens[i]])[0][0]
        similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    
    print(f"Tokenizer Comparison Results:")
    print(f"Average Spearman Correlation: {avg_correlation:.4f}")
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")
    print(f"\nSample tokenization differences:")
    print(f"SMILES: {test_smiles[0]}")
    print(f"Current tokens (first 20): {current_tokens[0][:20]}")
    print(f"DeepChem tokens (first 20): {deepchem_tokens[0][:20]}")
    
    return avg_correlation, avg_similarity

if __name__ == "__main__":
    test_tokenizer_comparison()