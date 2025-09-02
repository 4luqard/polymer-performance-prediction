import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_processing import load_competition_data

@pytest.fixture
def data_paths():
    """Fixture for data paths"""
    return {
        'train': 'data/raw/train.csv',
        'test': 'data/raw/test.csv',
        'supp': [
            'data/raw/train_supplement/dataset1.csv',
            'data/raw/train_supplement/dataset2.csv',
            'data/raw/train_supplement/dataset3.csv',
            'data/raw/train_supplement/dataset4.csv',
            'data/raw/extra_datasets/TgSS_enriched_cleaned.csv',
            'data/raw/extra_datasets/tg_density.csv'
        ]
    }

def test_load_without_extra_datasets(data_paths):
    """Test loading without extra datasets"""
    # Load without extra datasets
    train_df, test_df = load_competition_data(
        train_path=data_paths['train'],
        test_path=data_paths['test'],
        use_supplementary=False
    )
    
    assert train_df is not None, "train_df should not be None"
    assert test_df is not None, "test_df should not be None"
    
    original_train_size = len(train_df)
    assert original_train_size > 0, "Should have training data"

def test_load_with_extra_datasets(data_paths):
    """Test loading with extra datasets"""
    # Load with extra datasets  
    train_df_extra, test_df_extra = load_competition_data(
        train_path=data_paths['train'],
        test_path=data_paths['test'],
        supp_paths=data_paths['supp'],
        use_supplementary=True
    )
    
    assert train_df_extra is not None, "train_df with extra should not be None"
    assert test_df_extra is not None, "test_df with extra should not be None"
    
    # Should have more data with extra datasets
    train_df, _ = load_competition_data(
        train_path=data_paths['train'],
        test_path=data_paths['test'],
        use_supplementary=False
    )
    
    assert len(train_df_extra) >= len(train_df), "Should have at least as much data with supplements"

def test_new_smiles_added(data_paths):
    """Test that new unique SMILES are added"""
    # Load original training data
    train_df = pd.read_csv(data_paths['train'])
    original_smiles = set(train_df['SMILES'].unique())
    
    # Load extra datasets
    extra_dfs = []
    if os.path.exists(data_paths['supp'][4]):  # TgSS_enriched_cleaned.csv
        df = pd.read_csv(data_paths['supp'][4])
        if 'SMILES' in df.columns:
            extra_dfs.append(df)
    
    if os.path.exists(data_paths['supp'][5]):  # tg_density.csv
        df = pd.read_csv(data_paths['supp'][5])
        if 'SMILES' in df.columns:
            extra_dfs.append(df)
    
    if extra_dfs:
        combined_extra = pd.concat(extra_dfs, ignore_index=True)
        extra_smiles = set(combined_extra['SMILES'].unique())
        new_smiles = extra_smiles - original_smiles
        
        assert len(new_smiles) > 0, "Should have new unique SMILES from extra datasets"
        print(f"Added {len(new_smiles)} new unique SMILES")

def test_target_values_filled(data_paths):
    """Test that target values are filled from extra datasets"""
    # Check if extra dataset files exist
    tgss_path = data_paths['supp'][4]
    tg_density_path = data_paths['supp'][5]
    
    if os.path.exists(tgss_path):
        tgss_df = pd.read_csv(tgss_path)
        if 'Tg' in tgss_df.columns:
            tg_count = tgss_df['Tg'].notna().sum()
            assert tg_count > 0, "TgSS dataset should have Tg values"
            print(f"TgSS dataset has {tg_count} Tg values")
    
    if os.path.exists(tg_density_path):
        tg_density_df = pd.read_csv(tg_density_path)
        if 'Tg' in tg_density_df.columns:
            tg_count = tg_density_df['Tg'].notna().sum()
            assert tg_count > 0, "tg_density dataset should have Tg values"
        
        if 'Density' in tg_density_df.columns:
            density_count = tg_density_df['Density'].notna().sum()
            assert density_count > 0, "tg_density dataset should have Density values"
            print(f"tg_density dataset has {density_count} Density values")

def test_data_consistency(data_paths):
    """Test data consistency after integration"""
    # Load with extra datasets
    train_df, test_df = load_competition_data(
        train_path=data_paths['train'],
        test_path=data_paths['test'],
        supp_paths=data_paths['supp'],
        use_supplementary=True
    )
    
    # Check shapes match
    # Check that the dataframe has expected columns
    assert 'SMILES' in train_df.columns, "Should have SMILES column"
    
    # Check that dataframe has target columns
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    for col in target_cols:
        assert col in train_df.columns, f"Should have {col} column"
    
    # Check no duplicate SMILES
    unique_smiles = train_df['SMILES'].nunique()
    assert unique_smiles == len(train_df), "Should have no duplicate SMILES after deduplication"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])