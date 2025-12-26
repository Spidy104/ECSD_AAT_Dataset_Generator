#!/usr/bin/env python3
"""
Example usage script for solar datasets.

Demonstrates how to:
- Load generated datasets
- Perform basic data analysis
- Prepare data for machine learning
- Visualize key patterns

Run: python example_usage.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(dataset_name="solar_india_dataset"):
    """Load a generated solar dataset."""
    base_path = Path(dataset_name)
    
    if not base_path.exists():
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Please run the generator script first:")
        if "india" in dataset_name:
            print("  python generate-indian-dataset.py")
        else:
            print("  python generate-dataset.py")
        return None
    
    # Load normalized data (ready for ML)
    normalized_file = base_path / "all_days_normalized.csv"
    raw_file = base_path / "all_days_raw.csv"
    
    if normalized_file.exists():
        df = pd.read_csv(normalized_file)
        print(f"✓ Loaded normalized dataset: {len(df)} samples")
        return df
    elif raw_file.exists():
        df = pd.read_csv(raw_file)
        print(f"✓ Loaded raw dataset: {len(df)} samples")
        return df
    else:
        print("Error: No dataset files found.")
        return None


def analyze_dataset(df):
    """Perform basic statistical analysis."""
    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for seasons
    if 'season' in df.columns:
        print(f"\nSeasons: {df['season'].unique()}")
        print("\nSamples per season:")
        print(df['season'].value_counts())
    
    # LDR statistics
    if 'ldr' in df.columns:
        print(f"\nLDR Statistics:")
        print(f"  Mean: {df['ldr'].mean():.4f}")
        print(f"  Std:  {df['ldr'].std():.4f}")
        print(f"  Min:  {df['ldr'].min():.4f}")
        print(f"  Max:  {df['ldr'].max():.4f}")
    elif 'ldr_norm' in df.columns:
        print(f"\nLDR (normalized) Statistics:")
        print(f"  Mean: {df['ldr_norm'].mean():.4f}")
        print(f"  Std:  {df['ldr_norm'].std():.4f}")
        print(f"  Min:  {df['ldr_norm'].min():.4f}")
        print(f"  Max:  {df['ldr_norm'].max():.4f}")
    
    # Battery statistics
    if 'battery_voltage' in df.columns:
        print(f"\nBattery Voltage Statistics:")
        print(f"  Mean: {df['battery_voltage'].mean():.4f} V")
        print(f"  Std:  {df['battery_voltage'].std():.4f} V")
        print(f"  Min:  {df['battery_voltage'].min():.4f} V")
        print(f"  Max:  {df['battery_voltage'].max():.4f} V")
    elif 'battery_norm' in df.columns:
        print(f"\nBattery (normalized) Statistics:")
        print(f"  Mean: {df['battery_norm'].mean():.4f}")
        print(f"  Std:  {df['battery_norm'].std():.4f}")
    
    # Missing values
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  None (except prediction targets)")
    
    print("="*70)


def prepare_for_ml(df):
    """Prepare dataset for machine learning."""
    print("\n" + "="*70)
    print("MACHINE LEARNING PREPARATION")
    print("="*70)
    
    # Identify feature columns (exclude metadata and target)
    exclude_cols = ['day', 'season', 'time_s', 'future_ldr', 'future_ldr_norm']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Handle target variable
    if 'future_ldr_norm' in df.columns:
        y = df['future_ldr_norm'].copy()
        target_name = 'future_ldr_norm'
    elif 'future_ldr' in df.columns:
        y = df['future_ldr'].copy()
        target_name = 'future_ldr'
    else:
        print("\nWarning: No target variable found!")
        return None, None
    
    # Remove samples with NaN targets
    valid_mask = ~y.isna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    print(f"\nTarget variable: {target_name}")
    print(f"Valid samples: {len(y_clean)} / {len(y)} ({100*len(y_clean)/len(y):.1f}%)")
    print(f"Feature matrix shape: {X_clean.shape}")
    
    # Check for any remaining NaN in features
    if X_clean.isnull().sum().sum() > 0:
        print("\nWarning: NaN values found in features:")
        print(X_clean.isnull().sum()[X_clean.isnull().sum() > 0])
    
    print("\n✓ Data ready for training!")
    print("  Example usage:")
    print("    from sklearn.model_selection import train_test_split")
    print("    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)")
    print("="*70)
    
    return X_clean, y_clean


def show_sample_data(df, n=5):
    """Display sample rows from the dataset."""
    print("\n" + "="*70)
    print(f"SAMPLE DATA (first {n} rows)")
    print("="*70)
    print(df.head(n).to_string())
    print("="*70)


def seasonal_comparison(df):
    """Compare statistics across seasons (if applicable)."""
    if 'season' not in df.columns:
        return
    
    print("\n" + "="*70)
    print("SEASONAL COMPARISON")
    print("="*70)
    
    for season in df['season'].unique():
        season_data = df[df['season'] == season]
        print(f"\n{season.upper()}:")
        
        if 'ldr' in df.columns:
            print(f"  Avg LDR: {season_data['ldr'].mean():.4f}")
        elif 'ldr_norm' in df.columns:
            print(f"  Avg LDR: {season_data['ldr_norm'].mean():.4f}")
        
        if 'battery_voltage' in df.columns:
            print(f"  Avg Battery: {season_data['battery_voltage'].mean():.4f} V")
        elif 'battery_norm' in df.columns:
            print(f"  Avg Battery: {season_data['battery_norm'].mean():.4f}")
        
        print(f"  Samples: {len(season_data)}")
    
    print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("SOLAR DATASET EXAMPLE USAGE")
    print("="*70)
    
    # Try to load Indian dataset first, fall back to minimal
    df = load_dataset("solar_india_dataset")
    if df is None:
        df = load_dataset("solar_minimal_dataset")
    
    if df is None:
        print("\nNo datasets found. Please generate a dataset first.")
        return
    
    # Perform analysis
    analyze_dataset(df)
    
    # Show seasonal comparison if applicable
    seasonal_comparison(df)
    
    # Show sample data
    show_sample_data(df, n=3)
    
    # Prepare for ML
    X, y = prepare_for_ml(df)
    
    if X is not None:
        print("\n✓ Dataset successfully loaded and prepared!")
        print("\nNext steps:")
        print("  1. Split data into train/test sets")
        print("  2. Train your model (e.g., neural network, random forest)")
        print("  3. Evaluate performance on test set")
        print("  4. Deploy to edge device (TinyML)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
