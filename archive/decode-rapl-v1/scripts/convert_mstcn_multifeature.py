#!/usr/bin/env python3
"""
Convert MS-TCN data to DECODE-RAPL Multi-Feature format

MS-TCN format:
- timestamp, rapl_package_power, cpu_user_percent, cpu_system_percent, context_switches_sec, etc.

DECODE-RAPL Multi-Feature format:
- timestamp, machine_id, user_percent, system_percent, context_switches, power

This preserves the breakdown of CPU activity types to distinguish:
- Light work (high system%, low user%): syscalls, /proc reading
- Heavy work (low system%, high user%): compute, memory operations

Where:
- user_percent = cpu_user_percent (user-space CPU time)
- system_percent = cpu_system_percent + cpu_irq_percent + cpu_softirq_percent (kernel CPU time)
- context_switches = context_switches_sec (system call overhead indicator)
- power = rapl_package_power (main CPU power)
- machine_id = derived from filename or set to 'machine_0'

Note: iowait is excluded as CPU is halted (not doing work)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def convert_mstcn_to_decode_multifeature(input_path: str, output_path: str, machine_id: str = 'machine_0'):
    """
    Convert MS-TCN CSV to DECODE-RAPL Multi-Feature format

    Args:
        input_path: Path to MS-TCN CSV file
        output_path: Path to save DECODE-RAPL format CSV
        machine_id: Machine identifier for this data
    """
    print(f"Loading MS-TCN data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Extract individual CPU components
    # user_percent: User-space CPU time (application code)
    user_percent = df.get('cpu_user_percent', 0).clip(0, 100)

    # system_percent: Kernel/syscall CPU time (includes irq and softirq)
    system_percent = (
        df.get('cpu_system_percent', 0) +
        df.get('cpu_irq_percent', 0) +
        df.get('cpu_softirq_percent', 0)
    ).clip(0, 100)

    # context_switches: Rate of context switches (syscall overhead indicator)
    # Normalize to reasonable range for model input
    context_switches_raw = df.get('context_switches_sec', 0)

    # Get power (use rapl_package_power as primary metric)
    if 'rapl_package_power' in df.columns:
        power = df['rapl_package_power']
    elif 'rapl_core_power' in df.columns:
        power = df['rapl_core_power']
    else:
        raise ValueError("No RAPL power column found in data")

    # Create DECODE-RAPL Multi-Feature format DataFrame
    decode_df = pd.DataFrame({
        'timestamp': pd.to_datetime(df['timestamp'], unit='s'),
        'machine_id': machine_id,
        'user_percent': user_percent,
        'system_percent': system_percent,
        'context_switches': context_switches_raw,
        'power': power
    })

    # Also compute total CPU for comparison
    decode_df['cpu_total'] = (decode_df['user_percent'] + decode_df['system_percent']).clip(0, 100)

    # Remove rows with missing data
    initial_len = len(decode_df)
    decode_df = decode_df.dropna()
    removed = initial_len - len(decode_df)
    if removed > 0:
        print(f"Removed {removed} rows with missing data ({100*removed/initial_len:.1f}%)")

    # Remove rows with zero or negative power (likely measurement errors or startup)
    decode_df = decode_df[decode_df['power'] > 0]

    # Sort by timestamp
    decode_df = decode_df.sort_values('timestamp').reset_index(drop=True)

    # Statistics
    print(f"\nConverted data statistics:")
    print(f"  Samples: {len(decode_df):,}")
    print(f"  Duration: {(decode_df['timestamp'].iloc[-1] - decode_df['timestamp'].iloc[0]).total_seconds():.1f} seconds")
    print(f"\nCPU breakdown:")
    print(f"  User%:    min={decode_df['user_percent'].min():.1f}%, max={decode_df['user_percent'].max():.1f}%, mean={decode_df['user_percent'].mean():.1f}%")
    print(f"  System%:  min={decode_df['system_percent'].min():.1f}%, max={decode_df['system_percent'].max():.1f}%, mean={decode_df['system_percent'].mean():.1f}%")
    print(f"  Total%:   min={decode_df['cpu_total'].min():.1f}%, max={decode_df['cpu_total'].max():.1f}%, mean={decode_df['cpu_total'].mean():.1f}%")
    print(f"\nContext switches: min={decode_df['context_switches'].min():.0f}, max={decode_df['context_switches'].max():.0f}, mean={decode_df['context_switches'].mean():.0f}")
    print(f"\nPower: min={decode_df['power'].min():.1f}W, max={decode_df['power'].max():.1f}W, mean={decode_df['power'].mean():.1f}W")

    # Check sampling rate
    time_diffs = decode_df['timestamp'].diff().dt.total_seconds() * 1000  # in ms
    median_sampling = time_diffs.median()
    print(f"\nMedian sampling rate: {median_sampling:.2f}ms")

    # Analyze user/system ratio at different power levels
    print(f"\n=== User/System Ratio Analysis ===")
    for low, high in [(20, 30), (30, 40), (40, 50)]:
        mask = (decode_df['power'] >= low) & (decode_df['power'] < high)
        if mask.sum() > 100:
            avg_user = decode_df.loc[mask, 'user_percent'].mean()
            avg_sys = decode_df.loc[mask, 'system_percent'].mean()
            avg_ctx = decode_df.loc[mask, 'context_switches'].mean()
            ratio = avg_sys / (avg_user + avg_sys + 0.001)  # Avoid div by zero
            print(f"Power {low}-{high}W: user={avg_user:.1f}%, sys={avg_sys:.1f}%, ctx={avg_ctx:.0f}/s, sys_ratio={ratio:.2f}")

    # Save
    decode_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return decode_df


def main():
    parser = argparse.ArgumentParser(description='Convert MS-TCN data to DECODE-RAPL Multi-Feature format')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to MS-TCN CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save DECODE-RAPL Multi-Feature format CSV')
    parser.add_argument('--machine-id', type=str, default='machine_0',
                       help='Machine identifier (default: machine_0)')

    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Convert
    convert_mstcn_to_decode_multifeature(args.input, args.output, args.machine_id)


if __name__ == '__main__':
    main()
