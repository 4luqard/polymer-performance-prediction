import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_processing import load_competition_data

def test_integration():
    """Test complete integration of new datasets"""
    
    print("Testing data integration with new datasets...")
    
    # Define paths
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    supp_paths = [
        'data/raw/train_supplement/dataset1.csv',
        'data/raw/train_supplement/dataset2.csv',
        'data/raw/train_supplement/dataset3.csv',
        'data/raw/train_supplement/dataset4.csv',
        'data/raw/extra_datasets/TgSS_enriched_cleaned.csv',
        'data/raw/extra_datasets/tg_density.csv'
    ]
    
    # Load without extra datasets
    print("\n1. Loading WITHOUT extra datasets...")
    train_df_without, test_df = load_competition_data(
        train_path, test_path, 
        supp_paths=supp_paths[:4],  # Only original 4 supplements
        use_supplementary=True
    )
    print(f"Train shape without extra: {train_df_without.shape}")
    
    # Load with extra datasets
    print("\n2. Loading WITH extra datasets...")
    train_df_with, test_df = load_competition_data(
        train_path, test_path, 
        supp_paths=supp_paths,  # All 6 datasets
        use_supplementary=True
    )
    print(f"Train shape with extra: {train_df_with.shape}")
    
    # Verify extra data was added
    rows_added = len(train_df_with) - len(train_df_without)
    print(f"\nRows added by extra datasets: {rows_added}")
    
    # Due to duplicate SMILES removal, we expect only unique new molecules
    # Based on analysis: TgSS adds 0 new unique SMILES, tg_density adds 86
    expected_unique_rows = 86
    
    print(f"Expected unique rows from extra datasets: {expected_unique_rows}")
    
    assert rows_added == expected_unique_rows, f"Expected {expected_unique_rows} unique rows, got {rows_added}"
    
    # Check column consistency
    print("\n3. Checking column consistency...")
    print(f"Columns: {train_df_with.columns.tolist()}")
    
    # Check targets
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    for target in targets:
        assert target in train_df_with.columns, f"Missing target column: {target}"
    
    # Check Tg values from new datasets
    print("\n4. Checking Tg coverage...")
    tg_coverage_without = train_df_without['Tg'].notna().sum()
    tg_coverage_with = train_df_with['Tg'].notna().sum()
    tg_added = tg_coverage_with - tg_coverage_without
    
    print(f"Tg values without extra: {tg_coverage_without}")
    print(f"Tg values with extra: {tg_coverage_with}")
    print(f"Tg values added: {tg_added}")
    
    # The new datasets add Tg values for both new and existing SMILES
    # TgSS adds ~7000 Tg values for existing SMILES, tg_density adds ~190
    assert tg_added > 1000, f"Should add significant Tg values, got {tg_added}"
    
    # Check Density values from tg_density.csv
    print("\n5. Checking Density coverage...")
    density_coverage_without = train_df_without['Density'].notna().sum()
    density_coverage_with = train_df_with['Density'].notna().sum()
    density_added = density_coverage_with - density_coverage_without
    
    print(f"Density values without extra: {density_coverage_without}")
    print(f"Density values with extra: {density_coverage_with}")
    print(f"Density values added: {density_added}")
    
    # From the unique tg_density rows, most should have Density values
    # Analysis shows 177 out of 190 new tg_density rows have Density
    min_expected_density = 70  # Conservative estimate for unique rows with Density
    assert density_added >= min_expected_density, f"Expected at least {min_expected_density} Density values"
    
    print("\n✅ All integration tests passed!")
    return True

if __name__ == "__main__":
    test_integration()