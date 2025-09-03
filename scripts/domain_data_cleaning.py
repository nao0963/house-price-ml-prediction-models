"""
Domain-specific data cleaning script for Ames Housing dataset.
Handles one-time data cleaning operations that require domain knowledge.

This script should be run once on raw data to create a domain-cleaned dataset
that can then be processed by the statistical pipeline.

Key operations:
- Domain-specific missing value interpretation
- Extreme outlier removal (data quality issues)
- Basic data type corrections
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path


def handle_domain_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on domain knowledge.
    These are one-time interpretations that don't need to be in the pipeline.
    """
    df = df.copy()
    print("Handling domain-specific missing values...")
    
    # Columns with >80% missing values - these typically mean "None" or "Not Available"
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    for col in high_missing_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            print(f"  Filled {col} missing values with 'None'")
    
    # Basement columns - missing likely means no basement
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            print(f"  Filled {col} missing values with 'None'")
    
    # Garage columns - missing likely means no garage
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            print(f"  Filled {col} missing values with 'None'")
    
    # GarageYrBlt - fill with 0 to indicate no garage
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
        print("  Filled GarageYrBlt missing values with 0")
    
    # FireplaceQu - missing means no fireplace
    if 'FireplaceQu' in df.columns:
        df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
        print("  Filled FireplaceQu missing values with 'None'")
    
    # MasVnrType and MasVnrArea - missing likely means no masonry veneer
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
        print("  Filled MasVnrType missing values with 'None'")
    
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        print("  Filled MasVnrArea missing values with 0")
    
    # Electrical - fill with mode (most common value)
    if 'Electrical' in df.columns:
        mode_electrical = df['Electrical'].mode()[0]
        df['Electrical'] = df['Electrical'].fillna(mode_electrical)
        print(f"  Filled Electrical missing values with mode: {mode_electrical}")
    
    return df


def remove_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extreme outliers based on domain knowledge.
    These are data quality issues that should be permanently removed.
    """
    df = df.copy()
    initial_shape = df.shape[0]
    print("Removing extreme domain-specific outliers...")
    
    # 1. Unrealistic lot sizes (too large)
    if 'LotArea' in df.columns:
        unrealistic_lots = df[df['LotArea'] > 100000]  # > 2.3 acres is very unusual
        if len(unrealistic_lots) > 0:
            print(f"  Removing {len(unrealistic_lots)} houses with unrealistically large lots (>100,000 sq ft)")
            df = df[df['LotArea'] <= 100000]
    
    # 2. Extreme price outliers (based on domain knowledge)
    if 'SalePrice' in df.columns:
        price_outliers = df[(df['SalePrice'] < 50000) | (df['SalePrice'] > 800000)]
        if len(price_outliers) > 0:
            print(f"  Removing {len(price_outliers)} houses with extreme prices (<$50k or >$800k)")
            df = df[(df['SalePrice'] >= 50000) & (df['SalePrice'] <= 800000)]
    
    # 3. Houses with disproportionate living area to price ratio
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        # Large houses with very low prices (likely partial sales or data errors)
        disproportionate = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
        if len(disproportionate) > 0:
            print(f"  Removing {len(disproportionate)} houses with disproportionate size/price ratio")
            df = df.drop(disproportionate.index)
    
    final_shape = df.shape[0]
    total_removed = initial_shape - final_shape
    print(f"  Total extreme outliers removed: {total_removed}")
    print(f"  Dataset shape: {initial_shape} → {final_shape}")
    
    return df


def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data type optimization for memory efficiency.
    """
    df = df.copy()
    print("Optimizing data types...")
    
    # Convert numeric columns to category if they represent categories
    categorical_numeric = {
        'MSSubClass': 'category',
        'OverallQual': 'category', 
        'OverallCond': 'category',
        'MoSold': 'category'
    }
    
    converted_cols = []
    for col, dtype in categorical_numeric.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
            converted_cols.append(col)
    
    if converted_cols:
        print(f"  Converted numeric to category: {', '.join(converted_cols)}")
    
    # Convert string columns to category for memory efficiency
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        df[object_cols] = df[object_cols].astype('category')
        print(f"  Converted {len(object_cols)} string columns to category")
    
    return df


def clean_raw_data(input_path: str, output_dir: str = None) -> pd.DataFrame:
    """
    Main function to clean raw data with domain knowledge.
    
    Args:
        input_path: Path to raw CSV file
        output_dir: Directory to save cleaned data (optional)
        
    Returns:
        Cleaned DataFrame
    """
    print("="*60)
    print("DOMAIN-SPECIFIC DATA CLEANING")
    print("="*60)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded raw data: {df.shape}")
    print(f"Initial missing values: {df.isnull().sum().sum()}")
    
    # Apply cleaning steps
    df = handle_domain_missing_values(df)
    df = remove_extreme_outliers(df)
    df = optimize_data_types(df)
    
    # Final summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Final shape: {df.shape}")
    print(f"Final missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'domain_cleaned.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        # Save as pickle for type preservation
        pkl_path = os.path.join(output_dir, 'domain_cleaned.pkl')
        df.to_pickle(pkl_path)
        print(f"✓ Saved pickle: {pkl_path}")
    
    return df


def main():
    """Command line interface for domain data cleaning."""
    parser = argparse.ArgumentParser(description='Domain-specific data cleaning for Ames Housing dataset')
    parser.add_argument('input_path', help='Path to raw CSV file')
    parser.add_argument('--output_dir', '-o', default='../data/cleaned', 
                       help='Output directory (default: ../data/cleaned)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return
    
    # Run cleaning
    try:
        cleaned_df = clean_raw_data(args.input_path, args.output_dir)
        print(f"\n✓ Domain cleaning completed successfully!")
        print(f"✓ Cleaned dataset ready for statistical pipeline processing")
        
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        # Default paths for project structure
        input_path = "../data/split/train.csv"
        output_dir = "../data/cleaned"
        
        if os.path.exists(input_path):
            print("Running with default paths...")
            clean_raw_data(input_path, output_dir)
        else:
            print("Usage: python domain_data_cleaning.py <input_csv> [--output_dir <dir>]")
            print("Example: python domain_data_cleaning.py ../data/split/train.csv --output_dir ../data/cleaned")
    else:
        main()
