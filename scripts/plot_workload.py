#!/usr/bin/env python3
"""
Plot DECODE-RAPL v2 workload data with configuration metadata

Generates a 3-panel plot showing:
- CPU utilization (user%, system%, iowait%)
- Context switches
- Package power

Saves plot as PNG in the same directory as the CSV file.

Usage:
    python plot_workload.py <csv_file>

Example:
    python plot_workload.py ../data/collection6/run_5_of_13-cpu8-sys0-io0-pipe0-vm0-cache0.csv
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import numpy as np


def parse_workload_from_filename(filename: str) -> dict:
    """
    Extract workload configuration from filename

    Example: run_9_of_2025-cpu0-sys0-io0-pipe0-vm2-cache20.csv
    Returns: {'cpu': 0, 'sys': 0, 'io': 0, 'pipe': 0, 'vm': 2, 'cache': 20}
    """
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


def plot_workload(csv_path: Path):
    """
    Plot time series data from CSV file with workload configuration
    Saves plot as PNG in the same directory as the CSV file
    """
    # Parse filename
    workload_config = parse_workload_from_filename(csv_path.name)

    if workload_config is None:
        print(f"Warning: Could not parse workload config from filename: {csv_path.name}")
        print(f"Continuing with generic labels...")

    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Check required columns
    required_cols = ['timestamp_unix', 'user_percent', 'system_percent',
                     'iowait_percent', 'ctx_switches_per_sec', 'package_power_watts']
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    # Calculate time relative to start (in seconds)
    df['time_sec'] = df['timestamp_unix'] - df['timestamp_unix'].iloc[0]

    # Calculate statistics
    duration = df['time_sec'].iloc[-1]
    num_samples = len(df)
    mean_power = df['package_power_watts'].mean()
    std_power = df['package_power_watts'].std()
    mean_user = df['user_percent'].mean()
    mean_sys = df['system_percent'].mean()
    mean_iowait = df['iowait_percent'].mean()
    mean_ctx = df['ctx_switches_per_sec'].mean()

    # Determine workload type
    if workload_config is None:
        workload_type = "UNKNOWN WORKLOAD"
    elif sum(workload_config.values()) > 0:
        active = [k for k, v in workload_config.items() if v > 0]
        workload_type = ", ".join([f"{k.upper()}={workload_config[k]}" for k in active])
    else:
        workload_type = "IDLE BASELINE"

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'DECODE-RAPL v2 Workload Data: {csv_path.name}', fontsize=14, fontweight='bold')

    # Plot 1: CPU percentages
    ax1 = axes[0]
    ax1.plot(df['time_sec'], df['user_percent'], label='User %', color='green', linewidth=1)
    ax1.plot(df['time_sec'], df['system_percent'], label='System %', color='blue', linewidth=1)
    ax1.plot(df['time_sec'], df['iowait_percent'], label='IOWait %', color='orange', linewidth=1)
    ax1.set_ylabel('CPU Utilization (%)', fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title('CPU Utilization Over Time', fontsize=11)

    # Plot 2: Context switches
    ax2 = axes[1]
    ax2.plot(df['time_sec'], df['ctx_switches_per_sec'], label='Context Switches/sec',
             color='purple', linewidth=1)
    ax2.set_ylabel('Context Switches/sec', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_title('Context Switches Over Time', fontsize=11)

    # Plot 3: Package power
    ax3 = axes[2]
    ax3.plot(df['time_sec'], df['package_power_watts'], label='Package Power',
             color='red', linewidth=1.5)
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Power (Watts)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_title('Package Power Over Time', fontsize=11)

    # Add text box with workload configuration and statistics
    textstr = '\n'.join([
        'Workload Configuration:',
        f'  {workload_type}',
        '',
        'Statistics:',
        f'  Duration: {duration:.1f}s',
        f'  Samples: {num_samples}',
        f'  Power: {mean_power:.1f}W ± {std_power:.1f}W',
        f'  User%: {mean_user:.1f}%',
        f'  System%: {mean_sys:.1f}%',
        f'  IOWait%: {mean_iowait:.1f}%',
        f'  Ctx/sec: {mean_ctx:.0f}',
    ])

    # Place text box in top-right of figure
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.98, 0.98, textstr, transform=fig.transFigure, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    # Save plot
    output_path = csv_path.parent / f"{csv_path.stem}_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot DECODE-RAPL v2 workload data with configuration metadata. Saves plot as PNG.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_workload.py data/collection6/run_5_of_13-cpu8-sys0-io0-pipe0-vm0-cache0.csv
  python plot_workload.py data/idle.csv
        """
    )
    parser.add_argument('csv_file', type=Path, help='Path to CSV file to plot')

    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)

    plot_workload(args.csv_file)


if __name__ == '__main__':
    main()
