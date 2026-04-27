#!/usr/bin/env python3
"""
Convert MS-TCN data to DECODE-RAPL format

MS-TCN format:
- timestamp, rapl_package_power, cpu_user_percent, cpu_system_percent, cpu_idle_percent, etc.

DECODE-RAPL format:
- timestamp, machine_id, cpu_usage, power

Where:
- cpu_usage = 100 - cpu_idle_percent (or cpu_user + cpu_system + cpu_iowait + cpu_irq + cpu_softirq)
- power = rapl_package_power (main CPU power)
- machine_id = derived from filename or set to 'machine_0'
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def convert_mstcn_to_decode(input_path: str, output_path: str, machine_id: str = 'machine_0'):
    """
    Convert MS-TCN CSV to DECODE-RAPL format

    Args:
        input_path: Path to MS-TCN CSV file
        output_path: Path to save DECODE-RAPL format CSV
        machine_id: Machine identifier for this data
    """
    print(f"Loading MS-TCN data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Calculate CPU usage (EXCLUDING iowait)
    # CRITICAL: iowait inflates CPU% but doesn't contribute to power consumption
    # because the CPU is halted waiting for I/O (low power state)
    #
    # Compute time = user + system + irq + softirq (actual CPU work)
    # Excludes: idle, iowait (CPU not doing work)

    # CHANGED: Exclude iowait to better correlate with actual power
    cpu_usage = (
        df.get('cpu_user_percent', 0) +
        df.get('cpu_system_percent', 0) +
        df.get('cpu_irq_percent', 0) +
        df.get('cpu_softirq_percent', 0)
        # NOTE: Explicitly NOT including cpu_iowait_percent
    )

    # Clip to [0, 100] range
    cpu_usage = cpu_usage.clip(0, 100)

    # Get power (use rapl_package_power as primary metric)
    if 'rapl_package_power' in df.columns:
        power = df['rapl_package_power']
    elif 'rapl_core_power' in df.columns:
        power = df['rapl_core_power']
    else:
        raise ValueError("No RAPL power column found in data")

    # Create DECODE-RAPL format DataFrame
    decode_df = pd.DataFrame({
        'timestamp': pd.to_datetime(df['timestamp'], unit='s'),
        'machine_id': machine_id,
        'cpu_usage': cpu_usage,
        'power': power
    })

    # Remove rows with missing power data
    initial_len = len(decode_df)
    decode_df = decode_df.dropna(subset=['power'])
    removed = initial_len - len(decode_df)
    if removed > 0:
        print(f"Removed {removed} rows with missing power data ({100*removed/initial_len:.1f}%)")

    # Remove rows with zero or negative power (likely measurement errors or startup)
    decode_df = decode_df[decode_df['power'] > 0]

    # Sort by timestamp
    decode_df = decode_df.sort_values('timestamp').reset_index(drop=True)

    # Statistics
    print(f"\nConverted data statistics:")
    print(f"  Samples: {len(decode_df):,}")
    print(f"  Duration: {(decode_df['timestamp'].iloc[-1] - decode_df['timestamp'].iloc[0]).total_seconds():.1f} seconds")
    print(f"  CPU usage: min={decode_df['cpu_usage'].min():.1f}%, max={decode_df['cpu_usage'].max():.1f}%, mean={decode_df['cpu_usage'].mean():.1f}%")
    print(f"  Power: min={decode_df['power'].min():.1f}W, max={decode_df['power'].max():.1f}W, mean={decode_df['power'].mean():.1f}W")

    # Check sampling rate
    time_diffs = decode_df['timestamp'].diff().dt.total_seconds() * 1000  # in ms
    median_sampling = time_diffs.median()
    print(f"  Median sampling rate: {median_sampling:.2f}ms")

    # Save
    decode_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return decode_df


def main():
    parser = argparse.ArgumentParser(description='Convert MS-TCN data to DECODE-RAPL format')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to MS-TCN CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save DECODE-RAPL format CSV')
    parser.add_argument('--machine-id', type=str, default='machine_0',
                       help='Machine identifier (default: machine_0)')

    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Convert
    convert_mstcn_to_decode(args.input, args.output, args.machine_id)


if __name__ == '__main__':
    main()
