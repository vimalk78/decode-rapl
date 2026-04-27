#!/usr/bin/env python3
"""
Preprocess raw power data for training
Normalizes features to be scale-independent for VM portability

Usage:
    python3 scripts/preprocess_data.py data/training_raw.csv data/training_normalized.csv
    python3 scripts/preprocess_data.py data/training_raw.csv --auto-output
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path


def detect_system_info(df=None):
    """
    Detect system configuration from environment or data

    Args:
        df: Optional dataframe to infer info from (for historical data)

    Returns:
        dict with num_cores, memory_total_gb, swap_total_gb
    """
    # Try to detect from current system
    num_cores = os.cpu_count()

    # Try to read memory info from /proc/meminfo
    memory_total_gb = None
    swap_total_gb = None

    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    # Value in kB
                    memory_kb = int(line.split()[1])
                    memory_total_gb = memory_kb / (1024 * 1024)
                elif line.startswith('SwapTotal:'):
                    swap_kb = int(line.split()[1])
                    swap_total_gb = swap_kb / (1024 * 1024)
    except FileNotFoundError:
        pass

    # If we can't detect from system, try to infer from data
    if memory_total_gb is None and df is not None:
        if 'memory_total_mb' in df.columns:
            memory_total_gb = df['memory_total_mb'].iloc[0] / 1024
        else:
            # Estimate from other memory columns
            if all(col in df.columns for col in ['memory_used_mb', 'memory_free_mb', 'memory_cached_mb']):
                total_mb = (df['memory_used_mb'] + df['memory_free_mb']).max()
                memory_total_gb = total_mb / 1024

    if swap_total_gb is None and df is not None:
        if 'swap_total_mb' in df.columns:
            swap_total_gb = df['swap_total_mb'].iloc[0] / 1024
        elif 'swap_used_mb' in df.columns and 'swap_free_mb' in df.columns:
            swap_total_gb = (df['swap_used_mb'] + df['swap_free_mb']).max() / 1024

    # Defaults if still not found
    if memory_total_gb is None:
        print("Warning: Could not detect memory size, using default 95GB", file=sys.stderr)
        memory_total_gb = 95.0

    if swap_total_gb is None:
        print("Warning: Could not detect swap size, using default 8GB", file=sys.stderr)
        swap_total_gb = 8.0

    return {
        'num_cores': num_cores,
        'memory_total_gb': memory_total_gb,
        'swap_total_gb': swap_total_gb
    }


def detect_available_targets(df):
    """
    Detect which RAPL power targets have actual data

    Args:
        df: Dataframe with raw data

    Returns:
        list of column names that have valid power data
    """
    available = []

    for target in ['rapl_package_power', 'rapl_core_power', 'rapl_dram_power']:
        if target in df.columns:
            # Remove NaN and check if has non-zero values
            valid_values = df[target].dropna()
            if len(valid_values) > 0 and valid_values.max() > 0.1:
                available.append(target)
                print(f"✓ Found target: {target} (mean: {valid_values.mean():.2f}W, "
                      f"range: {valid_values.min():.2f}-{valid_values.max():.2f}W)",
                      file=sys.stderr)
            else:
                print(f"✗ Skipping {target}: no valid data", file=sys.stderr)
        else:
            print(f"✗ Missing column: {target}", file=sys.stderr)

    if not available:
        raise ValueError("No valid RAPL power targets found in data!")

    return available


def preprocess_features(df, system_info=None):
    """
    Add normalized features to dataframe

    Args:
        df: Raw dataframe
        system_info: Dict with num_cores, memory_total_gb, swap_total_gb
                     If None, will auto-detect

    Returns:
        Dataframe with additional normalized features
    """
    if system_info is None:
        system_info = detect_system_info(df)

    num_cores = system_info['num_cores']
    memory_total_gb = system_info['memory_total_gb']
    swap_total_gb = system_info['swap_total_gb']

    print(f"\nSystem Configuration:", file=sys.stderr)
    print(f"  CPU Cores: {num_cores}", file=sys.stderr)
    print(f"  Memory: {memory_total_gb:.1f} GB", file=sys.stderr)
    print(f"  Swap: {swap_total_gb:.1f} GB", file=sys.stderr)
    print(file=sys.stderr)

    # Add system scale features
    df['num_cores'] = num_cores
    df['memory_total_gb'] = memory_total_gb
    df['swap_total_gb'] = swap_total_gb

    # Convert CPU percentages to absolute CPU time (seconds/second)
    # This makes features naturally scale with core count for VM portability
    for cpu_type in ['user', 'system', 'idle', 'iowait', 'irq', 'softirq']:
        percent_col = f'cpu_{cpu_type}_percent'
        sec_col = f'cpu_{cpu_type}_sec'
        if percent_col in df.columns:
            # cpu_X_sec = (cpu_X_percent / 100) * num_cores
            # e.g., 50% on 20-core = 10 seconds/sec, 50% on 4-core = 2 seconds/sec
            df[sec_col] = (df[percent_col] / 100.0) * num_cores

    # Normalize per-second metrics to per-core rates
    if 'interrupts_sec' in df.columns:
        df['interrupts_per_core'] = df['interrupts_sec'] / num_cores

    if 'context_switches_sec' in df.columns:
        df['context_switches_per_core'] = df['context_switches_sec'] / num_cores

    if 'page_faults_sec' in df.columns:
        df['page_faults_per_core'] = df['page_faults_sec'] / num_cores

    if 'running_processes' in df.columns:
        df['running_processes_per_core'] = df['running_processes'] / num_cores

    # Convert memory to ratios (0-1)
    memory_total_mb = memory_total_gb * 1024

    if 'memory_used_mb' in df.columns:
        df['memory_used_ratio'] = df['memory_used_mb'] / memory_total_mb

    if 'memory_cached_mb' in df.columns:
        df['memory_cached_ratio'] = df['memory_cached_mb'] / memory_total_mb

    if 'memory_free_mb' in df.columns:
        df['memory_free_ratio'] = df['memory_free_mb'] / memory_total_mb

    # Convert swap to ratio
    swap_total_mb = swap_total_gb * 1024

    if 'swap_used_mb' in df.columns:
        df['swap_used_ratio'] = df['swap_used_mb'] / swap_total_mb

    return df


def preprocess_file(input_file, output_file=None, system_info=None):
    """
    Preprocess a CSV file with raw power data

    Args:
        input_file: Path to raw CSV
        output_file: Path for preprocessed CSV (None = auto-generate)
        system_info: Optional dict with system config, None = auto-detect

    Returns:
        Path to output file
    """
    print(f"Loading data from: {input_file}", file=sys.stderr)
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} samples\n", file=sys.stderr)

    # Detect available targets
    print("Detecting RAPL targets:", file=sys.stderr)
    available_targets = detect_available_targets(df)
    print(f"\nUsing {len(available_targets)} power targets: {', '.join(available_targets)}\n", file=sys.stderr)

    # Preprocess features
    print("Adding normalized features...", file=sys.stderr)
    df = preprocess_features(df, system_info)

    # Generate output filename if not specified
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_normalized.csv"

    # Save preprocessed data
    print(f"\nSaving preprocessed data to: {output_file}", file=sys.stderr)
    df.to_csv(output_file, index=False)

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print("PREPROCESSING COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Input:  {input_file}", file=sys.stderr)
    print(f"Output: {output_file}", file=sys.stderr)
    print(f"Samples: {len(df):,}", file=sys.stderr)
    print(f"Features: {len(df.columns)} columns", file=sys.stderr)
    print(f"\nNew normalized features added:", file=sys.stderr)

    normalized_features = [
        'num_cores', 'memory_total_gb', 'swap_total_gb',
        'cpu_user_sec', 'cpu_system_sec', 'cpu_idle_sec',
        'cpu_iowait_sec', 'cpu_irq_sec', 'cpu_softirq_sec',
        'interrupts_per_core', 'context_switches_per_core',
        'page_faults_per_core', 'running_processes_per_core',
        'memory_used_ratio', 'memory_cached_ratio',
        'memory_free_ratio', 'swap_used_ratio'
    ]

    for feat in normalized_features:
        if feat in df.columns:
            print(f"  ✓ {feat}", file=sys.stderr)

    print(f"{'='*60}\n", file=sys.stderr)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw power data with normalized features",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input_file',
        help='Input CSV file with raw power data'
    )

    parser.add_argument(
        'output_file',
        nargs='?',
        default=None,
        help='Output CSV file for preprocessed data (default: auto-generate name)'
    )

    parser.add_argument(
        '--auto-output',
        action='store_true',
        help='Auto-generate output filename from input'
    )

    parser.add_argument(
        '--num-cores',
        type=int,
        help='Override CPU core count (default: auto-detect)'
    )

    parser.add_argument(
        '--memory-gb',
        type=float,
        help='Override memory size in GB (default: auto-detect)'
    )

    parser.add_argument(
        '--swap-gb',
        type=float,
        help='Override swap size in GB (default: auto-detect)'
    )

    args = parser.parse_args()

    # Build system info if overrides provided
    system_info = None
    if args.num_cores or args.memory_gb or args.swap_gb:
        system_info = detect_system_info()  # Get defaults first
        if args.num_cores:
            system_info['num_cores'] = args.num_cores
        if args.memory_gb:
            system_info['memory_total_gb'] = args.memory_gb
        if args.swap_gb:
            system_info['swap_total_gb'] = args.swap_gb

    # Process file
    output_file = args.output_file if not args.auto_output else None
    preprocess_file(args.input_file, output_file, system_info)


if __name__ == '__main__':
    main()
