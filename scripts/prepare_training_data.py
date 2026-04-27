#!/usr/bin/env python3
"""
DECODE-RAPL v2 Training Data Preparation

Processes collected CSV files into delay-embedded training datasets.
Generates multiple datasets with different tau values for temporal scale comparison.

Usage:
    python prepare_training_data.py --data-dir ../data/all-combinations-temp-0 \
                                      --output-dir ../data/processed \
                                      --tau 1 4 8 \
                                      --skip-startup 100 \
                                      --seed 42
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_workload_from_filename(filename: str) -> Dict[str, int]:
    """Extract workload configuration from filename"""
    pattern = r'cpu(\d+)-sys(\d+)-io(\d+)-pipe(\d+)-vm(\d+)-cache(\d+)'
    match = re.search(pattern, filename)

    if not match:
        return None

    return {
        'cpu': int(match.group(1)),
        'sys': int(match.group(2)),
        'io': int(match.group(3)),
        'pipe': int(match.group(4)),
        'vm': int(match.group(5)),
        'cache': int(match.group(6))
    }


def create_delay_embedding(df: pd.DataFrame, tau: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create delay-embedded vectors with feature-grouped ordering

    Args:
        df: DataFrame with columns [user_percent, system_percent, iowait_percent, log_ctx_switches, package_power_watts]
        tau: Time delay in samples
        d: Number of delays (embedding dimension)

    Returns:
        X: Embedded features (n_samples, 100) where 100 = 4 features * 25 delays
        y: Power labels (n_samples,)

    Vector structure:
        [user%(t), user%(t-τ), ..., user%(t-24τ),       # positions 0-24
         sys%(t), sys%(t-τ), ..., sys%(t-24τ),          # positions 25-49
         iowait%(t), iowait%(t-τ), ..., iowait%(t-24τ), # positions 50-74
         log_ctx(t), log_ctx(t-τ), ..., log_ctx(t-24τ)] # positions 75-99
    """
    features = ['user_percent', 'system_percent', 'iowait_percent', 'log_ctx_switches']
    n_features = len(features)

    # Calculate valid length after delay embedding
    n_samples = len(df) - (d - 1) * tau

    if n_samples <= 0:
        raise ValueError(f"Time series too short for embedding. Need at least {(d-1)*tau + 1} samples, got {len(df)}")

    # Initialize arrays
    X = np.zeros((n_samples, d * n_features))

    # Create delay-embedded vectors (feature-grouped)
    for i, feat in enumerate(features):
        for j in range(d):
            # Calculate offset for this delay
            offset = j * tau
            # Extract the windowed data
            start_idx = offset
            end_idx = offset + n_samples
            X[:, i * d + j] = df[feat].iloc[start_idx:end_idx].values

    # Get corresponding power labels (aligned with the most recent timestamp)
    y = df['package_power_watts'].iloc[(d-1)*tau:].values

    return X, y


def process_csv_file(csv_path: Path, tau: int, d: int, skip_startup: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Process a single CSV file: filter startup, log transform, delay embed

    Returns:
        X: Embedded features
        y: Power labels
        stats: Dictionary with processing statistics
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Parse workload config
    workload_config = parse_workload_from_filename(csv_path.name)

    stats = {
        'filename': csv_path.name,
        'workload': workload_config,
        'original_samples': len(df),
    }

    # Skip startup transient
    if skip_startup > 0:
        df = df.iloc[skip_startup:].reset_index(drop=True)
        stats['after_startup_filter'] = len(df)

    # Apply log transform to context switches
    df['log_ctx_switches'] = np.log1p(df['ctx_switches_per_sec'])

    # Check if we have enough samples for embedding
    min_samples_needed = (d - 1) * tau + 1
    if len(df) < min_samples_needed:
        stats['error'] = f"Insufficient samples: {len(df)} < {min_samples_needed}"
        return None, None, stats

    # Create delay embeddings
    try:
        X, y = create_delay_embedding(df, tau=tau, d=d)
        stats['embedded_samples'] = len(X)
        stats['power_mean'] = float(y.mean())
        stats['power_std'] = float(y.std())
        stats['power_min'] = float(y.min())
        stats['power_max'] = float(y.max())
    except Exception as e:
        stats['error'] = str(e)
        return None, None, stats

    return X, y, stats


def process_all_csvs(data_dir: Path, tau: int, d: int, skip_startup: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Process all CSV files in directory for given tau value

    Returns:
        X_all: Concatenated embedded features (n_total_samples, 100)
        y_all: Concatenated power labels (n_total_samples,)
        stats_list: List of per-file statistics
    """
    # Find all CSV files
    csv_files = sorted(data_dir.glob("run_*_of_*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {data_dir}")

    print(f"\nProcessing {len(csv_files)} CSV files with tau={tau}...")

    X_list = []
    y_list = []
    stats_list = []

    errors = 0

    for csv_path in tqdm(csv_files, desc=f"tau={tau}"):
        X, y, stats = process_csv_file(csv_path, tau=tau, d=d, skip_startup=skip_startup)

        if X is not None:
            X_list.append(X)
            y_list.append(y)
        else:
            errors += 1

        stats_list.append(stats)

    if errors > 0:
        print(f"Warning: {errors} files failed processing")

    # Concatenate all data
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    print(f"Total samples for tau={tau}: {len(X_all):,}")

    return X_all, y_all, stats_list


def apply_minmax_normalization(splits: Dict) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Apply MinMaxScaler normalization to features

    Computes min/max from training set and applies to all splits.

    Args:
        splits: Dictionary with X_train, X_val, X_test, y_train, y_val, y_test

    Returns:
        Tuple of (normalized_splits, feature_min, feature_max)
    """
    # Compute min and max from training data only
    feature_min = splits['X_train'].min(axis=0)  # Shape: (100,)
    feature_max = splits['X_train'].max(axis=0)  # Shape: (100,)

    # Avoid division by zero (for constant features)
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0

    # Print scaler attributes (requested by user)
    print("\n" + "="*80)
    print("MINMAX SCALER ATTRIBUTES (Training Data)")
    print("="*80)
    print(f"feature_min shape: {feature_min.shape}")
    print(f"feature_max shape: {feature_max.shape}")
    print(f"\nFeature min values (first 10): {feature_min[:10]}")
    print(f"Feature max values (first 10): {feature_max[:10]}")
    print(f"\nFeature min values (last 10):  {feature_min[-10:]}")
    print(f"Feature max values (last 10):  {feature_max[-10:]}")
    print(f"\nMin of all mins: {feature_min.min():.4f}")
    print(f"Max of all maxs: {feature_max.max():.4f}")
    print(f"Range (max-min) mean: {feature_range.mean():.4f}")
    print(f"Range (max-min) std:  {feature_range.std():.4f}")
    print("="*80 + "\n")

    # Apply normalization: (x - min) / (max - min)
    normalized_splits = {}

    for split_name in ['X_train', 'X_val', 'X_test']:
        X = splits[split_name]
        X_normalized = (X - feature_min) / feature_range

        # Clip to [0, 1] range (handles any numerical issues)
        X_normalized = np.clip(X_normalized, 0.0, 1.0)

        normalized_splits[split_name] = X_normalized

        print(f"{split_name} normalization:")
        print(f"  Original range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"  Normalized range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
        print(f"  Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")
        print(f"  Values outside [0,1] before clip: {np.sum((X_normalized < 0) | (X_normalized > 1))}")

    # Copy over y values (power labels, not normalized)
    normalized_splits['y_train'] = splits['y_train']
    normalized_splits['y_val'] = splits['y_val']
    normalized_splits['y_test'] = splits['y_test']

    return normalized_splits, feature_min, feature_range


def shuffle_and_split(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> Dict:
    """
    Shuffle and split data into train/val/test sets

    Returns:
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    n_samples = len(X)

    # Set random seed
    rng = np.random.RandomState(seed)

    # Shuffle indices
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    return {
        'X_train': X_shuffled[:train_end],
        'y_train': y_shuffled[:train_end],
        'X_val': X_shuffled[train_end:val_end],
        'y_val': y_shuffled[train_end:val_end],
        'X_test': X_shuffled[val_end:],
        'y_test': y_shuffled[val_end:],
    }


def save_dataset(output_dir: Path, tau: int, splits: Dict, stats_list: List[Dict],
                 feature_min: np.ndarray, feature_range: np.ndarray, args: argparse.Namespace):
    """Save dataset splits and metadata including normalization parameters"""

    # Create output directory for this tau
    tau_dir = output_dir / f"tau{tau}"
    tau_dir.mkdir(parents=True, exist_ok=True)

    # Save train/val/test splits
    np.savez_compressed(
        tau_dir / 'train.npz',
        X=splits['X_train'],
        y=splits['y_train']
    )

    np.savez_compressed(
        tau_dir / 'val.npz',
        X=splits['X_val'],
        y=splits['y_val']
    )

    np.savez_compressed(
        tau_dir / 'test.npz',
        X=splits['X_test'],
        y=splits['y_test']
    )

    # Calculate and save statistics
    metadata = {
        'tau': tau,
        'd': args.d,
        'skip_startup_samples': args.skip_startup,
        'seed': args.seed,
        'n_features': 4,
        'input_dim': 100,
        'feature_order': [
            'user_percent (0-24)',
            'system_percent (25-49)',
            'iowait_percent (50-74)',
            'log_ctx_switches (75-99)'
        ],
        'temporal_lookback_samples': (args.d - 1) * tau,
        'temporal_lookback_ms': (args.d - 1) * tau * 16,  # 16ms sampling rate
        'train_samples': len(splits['X_train']),
        'val_samples': len(splits['X_val']),
        'test_samples': len(splits['X_test']),
        'train_power_mean': float(splits['y_train'].mean()),
        'train_power_std': float(splits['y_train'].std()),
        'val_power_mean': float(splits['y_val'].mean()),
        'val_power_std': float(splits['y_val'].std()),
        'test_power_mean': float(splits['y_test'].mean()),
        'test_power_std': float(splits['y_test'].std()),
        'normalization': {
            'type': 'minmax',
            'feature_min': feature_min.tolist(),
            'feature_range': feature_range.tolist(),
        },
        'per_file_stats': stats_list,
    }

    with open(tau_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved tau={tau} dataset to {tau_dir}")
    print(f"  Train: {len(splits['X_train']):,} samples")
    print(f"  Val:   {len(splits['X_val']):,} samples")
    print(f"  Test:  {len(splits['X_test']):,} samples")
    print(f"  Temporal lookback: {(args.d-1)*tau} samples = {(args.d-1)*tau*16}ms")
    print(f"  Normalization: MinMaxScaler [0, 1]")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare DECODE-RAPL v2 training data with delay embedding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python prepare_training_data.py \\
    --data-dir ../data/all-combinations-temp-0 \\
    --output-dir ../data/processed \\
    --tau 1 4 8 \\
    --skip-startup 100 \\
    --seed 42
        """
    )

    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for processed datasets')
    parser.add_argument('--tau', type=int, nargs='+', default=[1, 4, 8],
                        help='Tau values for delay embedding (default: 1 4 8)')
    parser.add_argument('--d', type=int, default=25,
                        help='Number of delays (default: 25)')
    parser.add_argument('--skip-startup', type=int, default=100,
                        help='Number of initial samples to skip (default: 100)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train split ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate arguments
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return

    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be < 1.0")
        return

    print("="*70)
    print("DECODE-RAPL v2 Training Data Preparation")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tau values: {args.tau}")
    print(f"Embedding dimension (d): {args.d}")
    print(f"Skip startup samples: {args.skip_startup}")
    print(f"Train/Val/Test split: {args.train_ratio:.1%}/{args.val_ratio:.1%}/{1-args.train_ratio-args.val_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print("="*70)

    # Process each tau value
    for tau in args.tau:
        print(f"\n{'='*70}")
        print(f"Processing tau={tau}")
        print(f"{'='*70}")

        # Process all CSVs
        X_all, y_all, stats_list = process_all_csvs(
            args.data_dir,
            tau=tau,
            d=args.d,
            skip_startup=args.skip_startup
        )

        # Shuffle and split
        print(f"\nShuffling and splitting (seed={args.seed})...")
        splits = shuffle_and_split(
            X_all, y_all,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

        # Apply MinMaxScaler normalization
        print(f"\nApplying MinMaxScaler normalization...")
        normalized_splits, feature_min, feature_range = apply_minmax_normalization(splits)

        # Save dataset with normalization params
        save_dataset(args.output_dir, tau, normalized_splits, stats_list,
                    feature_min, feature_range, args)

    print(f"\n{'='*70}")
    print("All datasets generated successfully!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
