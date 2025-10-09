# -*- coding: utf-8 -*-
"""
Create Mini Dataset for Quick Model Testing
Extract latest 1000 records with selected features
"""

import pandas as pd
import numpy as np

def create_mini_dataset():
    """
    Extract mini dataset from NCCU_CRM_cleaned.xlsx
    - Latest 1000 records by serial number
    - Features: overdue days (F column), paid installments features (AE~AH columns)
    """
    print("Loading NCCU_CRM_cleaned.xlsx...")

    # Read the full dataset
    df = pd.read_excel('Train/Source/NCCU_CRM_cleaned.xlsx')

    print(f"Total records in file: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Sort by serial number descending to get latest records
    df_sorted = df.sort_values(by='\nserial number', ascending=False)

    # Select latest 1000 records
    df_mini = df_sorted.head(1000).copy()

    print(f"\nSelected latest 1000 records")
    print(f"Serial number range: {df_mini['\nserial number'].min()} - {df_mini['\nserial number'].max()}")

    # Select specific columns
    # F column: overdue days (index 5)
    # AE~AH columns: paid installments related (indices 30-33)
    selected_columns = [
        '\nserial number',  # Keep for reference
        'overdue days',  # F column (index 5)
        'paid installments',  # AE column (index 30)
        'number of overdue before the first month',  # AF column (index 31)
        'number of overdue in the first half of the first month',  # AG column (index 32)
        'number of overdue in the second half of the first month',  # AH column (index 33)
        'status'  # Target variable (assuming this is the default indicator)
    ]

    # Check if all columns exist
    missing_cols = [col for col in selected_columns if col not in df_mini.columns]
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")
        selected_columns = [col for col in selected_columns if col in df_mini.columns]

    df_mini_selected = df_mini[selected_columns].copy()

    # Display basic info
    print(f"\nMini dataset shape: {df_mini_selected.shape}")
    print(f"\nSelected features:")
    for i, col in enumerate(selected_columns, 1):
        print(f"  {i}. {col}")

    print(f"\nData types:")
    print(df_mini_selected.dtypes)

    print(f"\nMissing values:")
    print(df_mini_selected.isnull().sum())

    print(f"\nBasic statistics:")
    print(df_mini_selected.describe())

    # Check target variable distribution
    if 'status' in df_mini_selected.columns:
        print(f"\nTarget variable 'status' distribution:")
        print(df_mini_selected['status'].value_counts())

    # Save mini dataset
    output_path = 'Train/Source/mini_dataset_1000.csv'
    df_mini_selected.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Mini dataset saved to: {output_path}")

    return df_mini_selected

if __name__ == "__main__":
    mini_df = create_mini_dataset()
    print("\n✅ Mini dataset creation completed!")