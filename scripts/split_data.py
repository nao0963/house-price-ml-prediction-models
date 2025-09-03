"""
Data splitting script for house price prediction model.

This script reads raw CSV data, performs stratified-like train/test split
based on target variable quantiles, and saves the splits with metadata
for reproducible machine learning workflows.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def setup_logging() -> None:
    """Configure logging with INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Split data into train/test sets with stratified-like sampling'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/raw.csv',
        help='Input CSV file path (default: data/raw/raw.csv)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='data/split',
        help='Output directory (default: data/split)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='SalePrice',
        help='Target column name (default: SalePrice)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--bins',
        type=int,
        default=10,
        help='Number of quantile bins for stratification (default: 10)'
    )
    
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable stratified-like splitting'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview split without saving files'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not (0 < args.test_size < 1):
        raise ValueError(f"test-size must be between 0 and 1, got {args.test_size}")
    
    if args.bins < 2:
        raise ValueError(f"bins must be at least 2, got {args.bins}")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")


def load_and_validate_data(input_path: str, target_col: str) -> pd.DataFrame:
    """Load CSV data and validate target column."""
    logging.info(f"Loading data from {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. "
                        f"Available columns: {list(df.columns)}")
    
    # Check target column for issues
    target_series = df[target_col]
    missing_count = target_series.isna().sum()
    if missing_count > 0:
        logging.warning(f"Target column has {missing_count} missing values")
    
    # Try to convert to numeric if needed
    if not pd.api.types.is_numeric_dtype(target_series):
        logging.warning(f"Target column is not numeric, attempting conversion")
        try:
            df[target_col] = pd.to_numeric(target_series, errors='coerce')
            new_missing = df[target_col].isna().sum() - missing_count
            if new_missing > 0:
                logging.warning(f"Conversion created {new_missing} additional missing values")
        except Exception as e:
            logging.warning(f"Failed to convert target to numeric: {e}")
    
    return df


def create_stratification_bins(target: pd.Series, n_bins: int) -> Optional[pd.Series]:
    """Create quantile-based bins for stratification."""
    try:
        # Remove missing values for binning
        valid_target = target.dropna()
        if len(valid_target) == 0:
            logging.warning("No valid target values for binning")
            return None
        
        bins = pd.qcut(valid_target, q=n_bins, duplicates='drop', labels=False)
        
        # Map bins back to original index
        bin_mapping = pd.Series(index=target.index, dtype='Int64')
        bin_mapping.loc[valid_target.index] = bins
        
        actual_bins = bin_mapping.nunique()
        logging.info(f"Created {actual_bins} quantile bins from {n_bins} requested bins")
        
        return bin_mapping
    
    except Exception as e:
        logging.warning(f"Failed to create stratification bins: {e}")
        return None


def perform_split(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float, 
    random_state: int, 
    use_stratify: bool, 
    n_bins: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Perform train/test split with optional stratification."""
    
    split_info = {
        'stratified': False,
        'n_bins_used': 0
    }
    
    stratify = None
    if use_stratify:
        bins = create_stratification_bins(df[target_col], n_bins)
        if bins is not None and bins.notna().sum() > 0:
            stratify = bins
            split_info['stratified'] = True
            split_info['n_bins_used'] = bins.nunique()
            logging.info(f"Using stratified split with {split_info['n_bins_used']} bins")
        else:
            logging.warning("Falling back to non-stratified split")
    else:
        logging.info("Using non-stratified split")
    
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logging.info(f"Split completed - Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df, split_info
    
    except Exception as e:
        raise RuntimeError(f"Failed to split data: {e}")


def calculate_target_summary(data: pd.Series, name: str) -> Dict[str, float]:
    """Calculate summary statistics for target variable."""
    valid_data = data.dropna()
    if len(valid_data) == 0:
        return {f'{name}_count': 0}
    
    return {
        f'{name}_count': len(valid_data),
        f'{name}_mean': float(valid_data.mean()),
        f'{name}_std': float(valid_data.std()),
        f'{name}_min': float(valid_data.min()),
        f'{name}_median': float(valid_data.median()),
        f'{name}_max': float(valid_data.max())
    }


def save_splits_and_metadata(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    split_info: Dict[str, Any],
    preview: bool = False
) -> None:
    """Save train/test splits and metadata."""
    
    outdir = Path(args.outdir)
    if not preview:
        outdir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {outdir}")
    
    # Prepare metadata
    metadata = {
        'input_path': args.input,
        'outdir': args.outdir,
        'target': args.target,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'n_bins_used': split_info['n_bins_used'],
        'train_size': len(train_df),
        'test_size_abs': len(test_df),
        'stratified': split_info['stratified']
    }
    
    # Add target summaries
    train_summary = calculate_target_summary(train_df[args.target], 'train')
    test_summary = calculate_target_summary(test_df[args.target], 'test')
    metadata['target_summary'] = {**train_summary, **test_summary}
    
    if preview:
        print("\n=== PREVIEW MODE - No files will be saved ===")
        print(f"Split summary:")
        print(f"  Train size: {metadata['train_size']} ({100*(1-args.test_size):.1f}%)")
        print(f"  Test size: {metadata['test_size_abs']} ({100*args.test_size:.1f}%)")
        print(f"  Stratified: {metadata['stratified']}")
        print(f"  Random state: {metadata['random_state']}")
        if metadata['stratified']:
            print(f"  Bins used: {metadata['n_bins_used']}")
        print("\nTarget summary:")
        print(f"  Train mean: {train_summary.get('train_mean', 'N/A')}")
        print(f"  Test mean: {test_summary.get('test_mean', 'N/A')}")
        return
    
    # Save CSV files
    train_path = outdir / 'train.csv'
    test_path = outdir / 'test.csv'
    meta_path = outdir / 'split_meta.json'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Output summary
    print(f"Data split completed successfully!")
    print(f"  Train: {train_path} ({metadata['train_size']} rows)")
    print(f"  Test: {test_path} ({metadata['test_size_abs']} rows)")
    print(f"  Metadata: {meta_path}")
    print(f"  Stratified: {metadata['stratified']}, Random state: {metadata['random_state']}")
    
    logging.info(f"Saved train data to {train_path}")
    logging.info(f"Saved test data to {test_path}")
    logging.info(f"Saved metadata to {meta_path}")


def main() -> None:
    """Main function to orchestrate the data splitting process."""
    setup_logging()
    
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Load and validate data
        df = load_and_validate_data(args.input, args.target)
        
        # Perform split
        use_stratify = not args.no_stratify
        train_df, test_df, split_info = perform_split(
            df, args.target, args.test_size, args.random_state, use_stratify, args.bins
        )
        
        # Save results
        save_splits_and_metadata(train_df, test_df, args, split_info, args.preview)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
