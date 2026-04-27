#!/usr/bin/env python3
"""
Validate DECODE-RAPL v2 data collection

Analyzes collected CSV files to verify:
- Data quality (completeness, sampling rate)
- Syscall patterns (high system%, low power)
- Power variations across workload types
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_workload_from_filename(filename: str) -> Dict[str, int]:
    """Extract workload parameters from filename.

    Example: run_3_of_12-cpu0-sys4-io0-pipe0-vm0-cache0.csv
    """
    match = re.search(
        r'cpu(\d+)-sys(\d+)-io(\d+)-pipe(\d+)-vm(\d+)-cache(\d+)',
        filename
    )
    if not match:
        return {}

    return {
        'cpu': int(match.group(1)),
        'sys': int(match.group(2)),
        'io': int(match.group(3)),
        'pipe': int(match.group(4)),
        'vm': int(match.group(5)),
        'cache': int(match.group(6)),
    }


def analyze_csv(csv_path: Path) -> Tuple[Dict, pd.DataFrame]:
    """Analyze a single CSV file."""
    df = pd.read_csv(csv_path)

    workload = parse_workload_from_filename(csv_path.name)

    # Calculate statistics
    stats = {
        'filename': csv_path.name,
        'workload': workload,
        'num_samples': len(df),
        'duration_sec': df['timestamp_unix'].max() - df['timestamp_unix'].min(),
        'user_pct_mean': df['user_percent'].mean(),
        'user_pct_std': df['user_percent'].std(),
        'system_pct_mean': df['system_percent'].mean(),
        'system_pct_std': df['system_percent'].std(),
        'iowait_pct_mean': df['iowait_percent'].mean(),
        'iowait_pct_std': df['iowait_percent'].std(),
        'ctx_switches_mean': df['ctx_switches_per_sec'].mean(),
        'ctx_switches_std': df['ctx_switches_per_sec'].std(),
        'power_mean': df['package_power_watts'].mean(),
        'power_std': df['package_power_watts'].std(),
        'power_min': df['package_power_watts'].min(),
        'power_max': df['package_power_watts'].max(),
    }

    # Estimate sampling rate
    if len(df) > 1:
        timestamps = df['timestamp_unix'].values
        intervals = np.diff(timestamps)
        stats['sampling_interval_ms'] = np.median(intervals) * 1000
        stats['sampling_rate_hz'] = 1.0 / np.median(intervals)
    else:
        stats['sampling_interval_ms'] = 0
        stats['sampling_rate_hz'] = 0

    return stats, df


def validate_collection(data_dir: Path) -> None:
    """Validate entire collection."""
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))

    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    print("=" * 80)
    print()

    all_stats = []

    for csv_path in csv_files:
        stats, df = analyze_csv(Path(csv_path))
        all_stats.append(stats)

    # Print summary
    print(f"{'Workload':<40} {'Samples':>8} {'Duration':>8} {'Power (W)':>12} {'Sampling':>10}")
    print(f"{'(cpu-sys-io-pipe-vm-cache)':<40} {'':>8} {'(sec)':>8} {'(mean±std)':>12} {'(Hz)':>10}")
    print("-" * 80)

    for stats in all_stats:
        w = stats['workload']
        workload_str = f"cpu{w['cpu']}-sys{w['sys']}-io{w['io']}-pipe{w['pipe']}-vm{w['vm']}-cache{w['cache']}"
        power_str = f"{stats['power_mean']:.1f}±{stats['power_std']:.1f}"

        print(f"{workload_str:<40} {stats['num_samples']:>8d} {stats['duration_sec']:>8.1f} {power_str:>12} {stats['sampling_rate_hz']:>10.1f}")

    print()
    print("=" * 80)
    print()

    # Validate syscall patterns
    print("SYSCALL PATTERN VALIDATION:")
    print("-" * 80)

    syscall_runs = [s for s in all_stats if s['workload'].get('sys', 0) > 0 and s['workload'].get('cpu', 0) == 0]

    if syscall_runs:
        for stats in syscall_runs:
            w = stats['workload']
            workload_str = f"cpu{w['cpu']}-sys{w['sys']}-io{w['io']}-pipe{w['pipe']}-vm{w['vm']}-cache{w['cache']}"

            print(f"\nWorkload: {workload_str}")
            print(f"  System%: {stats['system_pct_mean']:.1f}% ± {stats['system_pct_std']:.1f}%")
            print(f"  User%:   {stats['user_pct_mean']:.1f}% ± {stats['user_pct_std']:.1f}%")
            print(f"  Power:   {stats['power_mean']:.1f}W ± {stats['power_std']:.1f}W")

            if stats['system_pct_mean'] > 10 and stats['power_mean'] < 40:
                print("  ✓ GOOD: High system%, low power pattern detected!")
            else:
                print(f"  ⚠ WARNING: Expected high system% (>10%) and low power (<40W)")
    else:
        print("No pure syscall workloads found (sys>0, cpu=0)")

    print()
    print("=" * 80)
    print()

    # Compare workload types
    print("POWER COMPARISON BY WORKLOAD TYPE:")
    print("-" * 80)

    # CPU-only runs
    cpu_runs = [s for s in all_stats if s['workload'].get('cpu', 0) > 0 and
                s['workload'].get('sys', 0) == 0 and
                s['workload'].get('io', 0) == 0]
    if cpu_runs:
        cpu_power = np.mean([s['power_mean'] for s in cpu_runs])
        print(f"CPU-only (compute):  {cpu_power:.1f}W")

    # Syscall-only runs
    sys_runs = [s for s in all_stats if s['workload'].get('sys', 0) > 0 and
                s['workload'].get('cpu', 0) == 0 and
                s['workload'].get('io', 0) == 0]
    if sys_runs:
        sys_power = np.mean([s['power_mean'] for s in sys_runs])
        print(f"Syscall-only:        {sys_power:.1f}W")

        if cpu_runs:
            diff = cpu_power - sys_power
            print(f"  → Difference: {diff:.1f}W (CPU uses {diff/sys_power*100:.1f}% more power)")

    # I/O-only runs
    io_runs = [s for s in all_stats if s['workload'].get('io', 0) > 0 and
               s['workload'].get('cpu', 0) == 0 and
               s['workload'].get('sys', 0) == 0]
    if io_runs:
        io_power = np.mean([s['power_mean'] for s in io_runs])
        print(f"I/O-only:            {io_power:.1f}W")

    print()
    print("=" * 80)
    print()

    # Overall statistics
    print("OVERALL STATISTICS:")
    print("-" * 80)
    all_power_mean = np.mean([s['power_mean'] for s in all_stats])
    all_power_std = np.std([s['power_mean'] for s in all_stats])
    all_power_min = min(s['power_min'] for s in all_stats)
    all_power_max = max(s['power_max'] for s in all_stats)

    print(f"Power range:     {all_power_min:.1f}W - {all_power_max:.1f}W")
    print(f"Mean power:      {all_power_mean:.1f}W ± {all_power_std:.1f}W")

    avg_samples = np.mean([s['num_samples'] for s in all_stats])
    avg_duration = np.mean([s['duration_sec'] for s in all_stats])
    avg_sampling_rate = np.mean([s['sampling_rate_hz'] for s in all_stats if s['sampling_rate_hz'] > 0])

    print(f"Avg samples:     {avg_samples:.0f}")
    print(f"Avg duration:    {avg_duration:.1f}s")
    print(f"Avg sampling:    {avg_sampling_rate:.1f} Hz ({1000/avg_sampling_rate:.1f}ms interval)")

    if avg_sampling_rate > 60 and avg_sampling_rate < 65:
        print("  ✓ GOOD: Sampling rate ~62.5 Hz (16ms interval) as expected")
    else:
        print(f"  ⚠ WARNING: Expected ~62.5 Hz sampling rate")


def main():
    parser = argparse.ArgumentParser(description='Validate DECODE-RAPL v2 collection')
    parser.add_argument('data_dir', type=Path, help='Directory containing CSV files')

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: Directory not found: {args.data_dir}")
        return

    validate_collection(args.data_dir)


if __name__ == '__main__':
    main()
