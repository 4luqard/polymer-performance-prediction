#!/usr/bin/env python3
"""
Script to download and save the pretrained tokenizer for offline use.
Run this before uploading to Kaggle.
"""

import os
import json
from transformers import AutoTokenizer

def download_tokenizer():
    """Download and save tokenizer files locally"""
    
    # Create directory for tokenizer files
    tokenizer_dir = "tokenizer_files"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    print("Downloading tokenizer from HuggingFace...")
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        
        # Save tokenizer locally
        tokenizer.save_pretrained(tokenizer_dir)
        
        print(f"Tokenizer saved to {tokenizer_dir}/")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Test tokenizer
        test_smiles = "CC(C)C(=O)O"
        tokens = tokenizer.encode(test_smiles)
        print(f"\nTest encoding of '{test_smiles}': {tokens}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        return False

if __name__ == "__main__":
    success = download_tokenizer()
    if success:
        print("\nTokenizer downloaded successfully!")
        print("Upload the 'tokenizer_files' directory to Kaggle dataset.")
    else:
        print("\nFailed to download tokenizer.")