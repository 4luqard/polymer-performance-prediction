import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import hdbscan
import warnings
warnings.filterwarnings('ignore')

def test_tokenizer_clustering():
    """Compare tokenizers using HDBSCAN clustering"""
    
    # Import tokenizers
    import sys
    sys.path.insert(0, '/workspace/kaggle/neurips-open-polymer-prediction-2025')
    from transformer_model import SMILESTokenizer as CurrentTokenizer
    from transformers import AutoTokenizer
    
    # Load real data for more comprehensive testing
    data_path = '/workspace/kaggle/neurips-open-polymer-prediction-2025/data/raw/train.csv'
    df = pd.read_csv(data_path)
    
    # Sample 100 SMILES for testing
    sample_smiles = df['SMILES'].dropna().sample(n=min(100, len(df)), random_state=42).tolist()
    
    # Initialize tokenizers
    current_tokenizer = CurrentTokenizer(max_length=100)
    deepchem_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    
    # Tokenize with current tokenizer
    current_tokens = current_tokenizer.tokenize(sample_smiles)
    
    # Tokenize with DeepChem tokenizer
    deepchem_tokens = []
    for smiles in sample_smiles:
        tokens = deepchem_tokenizer.encode(smiles, add_special_tokens=False)
        # Pad or truncate to match length
        if len(tokens) < 100:
            tokens = tokens + [0] * (100 - len(tokens))
        else:
            tokens = tokens[:100]
        deepchem_tokens.append(tokens)
    deepchem_tokens = np.array(deepchem_tokens)
    
    # Reduce dimensionality for clustering
    pca = PCA(n_components=10, random_state=42)
    current_pca = pca.fit_transform(current_tokens)
    deepchem_pca = pca.fit_transform(deepchem_tokens)
    
    # Perform HDBSCAN clustering
    clusterer_current = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    current_clusters = clusterer_current.fit_predict(current_pca)
    
    clusterer_deepchem = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    deepchem_clusters = clusterer_deepchem.fit_predict(deepchem_pca)
    
    # Calculate clustering metrics
    n_clusters_current = len(set(current_clusters)) - (1 if -1 in current_clusters else 0)
    n_clusters_deepchem = len(set(deepchem_clusters)) - (1 if -1 in deepchem_clusters else 0)
    
    noise_ratio_current = (current_clusters == -1).sum() / len(current_clusters)
    noise_ratio_deepchem = (deepchem_clusters == -1).sum() / len(deepchem_clusters)
    
    # Compare cluster assignments
    from sklearn.metrics import adjusted_rand_score
    if n_clusters_current > 0 and n_clusters_deepchem > 0:
        ari = adjusted_rand_score(current_clusters, deepchem_clusters)
    else:
        ari = 0.0
    
    print(f"\nClustering Analysis Results:")
    print(f"Current Tokenizer:")
    print(f"  - Number of clusters: {n_clusters_current}")
    print(f"  - Noise ratio: {noise_ratio_current:.3f}")
    print(f"\nDeepChem Tokenizer:")
    print(f"  - Number of clusters: {n_clusters_deepchem}")
    print(f"  - Noise ratio: {noise_ratio_deepchem:.3f}")
    print(f"\nAdjusted Rand Index (similarity of clustering): {ari:.4f}")
    
    # Calculate average token length for each tokenizer
    current_lengths = [(tokens != 0).sum() for tokens in current_tokens]
    deepchem_lengths = [(tokens != 0).sum() for tokens in deepchem_tokens]
    
    print(f"\nToken Length Statistics:")
    print(f"Current Tokenizer: mean={np.mean(current_lengths):.1f}, std={np.std(current_lengths):.1f}")
    print(f"DeepChem Tokenizer: mean={np.mean(deepchem_lengths):.1f}, std={np.std(deepchem_lengths):.1f}")
    
    return ari, n_clusters_current, n_clusters_deepchem

if __name__ == "__main__":
    test_tokenizer_clustering()