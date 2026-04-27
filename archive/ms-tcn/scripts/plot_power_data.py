#!/usr/bin/env python3
"""
Plot CPU utilization and power data from power_data_collector.py output.

Usage:
    python plot_power_data.py power_data.csv
    python plot_power_data.py power_data.csv --with-power
    python plot_power_data.py power_data.csv --with-power --output plot.png
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_cpu_metrics(csv_file: str, include_power: bool = False, output_file: str = None):
    """
    Plot CPU utilization metrics and optionally power consumption.

    Args:
        csv_file: Path to CSV file from power_data_collector.py
        include_power: Whether to include RAPL package power on right y-axis
        output_file: Optional output file path to save plot
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Check required columns
    required_cols = ['timestamp', 'cpu_user_percent', 'cpu_system_percent', 'cpu_idle_percent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Convert timestamp to relative time in seconds from start
    df['time_seconds'] = df['timestamp'] - df['timestamp'].iloc[0]

    # Downsample to 500 equally spaced samples for cleaner plotting
    max_samples = 500
    if len(df) > max_samples:
        # Use evenly spaced indices
        indices = [int(i * len(df) / max_samples) for i in range(max_samples)]
        df_plot = df.iloc[indices].reset_index(drop=True)
    else:
        df_plot = df

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot CPU metrics on primary y-axis
    ax1.plot(df_plot['time_seconds'], df_plot['cpu_user_percent'],
             label='User %', color='#2E86AB', linewidth=1.5)
    ax1.plot(df_plot['time_seconds'], df_plot['cpu_system_percent'],
             label='System %', color='#A23B72', linewidth=1.5)
    ax1.plot(df_plot['time_seconds'], df_plot['cpu_idle_percent'],
             label='Idle %', color='#F18F01', linewidth=1.5, alpha=0.7)

    # Configure primary y-axis
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('CPU Utilization (%)', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)

    # Add power on secondary y-axis if requested
    if include_power and 'rapl_package_power' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df_plot['time_seconds'], df_plot['rapl_package_power'],
                label='Package Power', color='#C73E1D', linewidth=2, alpha=0.8)
        ax2.set_ylabel('Power (Watts)', fontsize=11, color='#C73E1D')
        ax2.tick_params(axis='y', labelcolor='#C73E1D')
        ax2.legend(loc='upper right', fontsize=10)

        # Set reasonable power range
        power_min = df_plot['rapl_package_power'].min()
        power_max = df_plot['rapl_package_power'].max()
        power_range = power_max - power_min
        ax2.set_ylim(max(0, power_min - 0.1 * power_range),
                     power_max + 0.1 * power_range)

        title = 'CPU Utilization and Power Consumption'
    else:
        title = 'CPU Utilization'

    # Set title
    plt.title(title, fontsize=13, fontweight='bold', pad=15)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def print_summary_stats(csv_file: str):
    """Print summary statistics of the data."""
    df = pd.read_csv(csv_file)

    print("\n" + "="*60)
    print("Data Summary Statistics")
    print("="*60)

    # Duration
    duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Samples: {len(df)}")
    print(f"Sample rate: {len(df)/duration:.2f} Hz")

    print("\nCPU Utilization:")
    print(f"  User:   min={df['cpu_user_percent'].min():5.1f}%  "
          f"max={df['cpu_user_percent'].max():5.1f}%  "
          f"avg={df['cpu_user_percent'].mean():5.1f}%")
    print(f"  System: min={df['cpu_system_percent'].min():5.1f}%  "
          f"max={df['cpu_system_percent'].max():5.1f}%  "
          f"avg={df['cpu_system_percent'].mean():5.1f}%")
    print(f"  Idle:   min={df['cpu_idle_percent'].min():5.1f}%  "
          f"max={df['cpu_idle_percent'].max():5.1f}%  "
          f"avg={df['cpu_idle_percent'].mean():5.1f}%")

    if 'rapl_package_power' in df.columns:
        print("\nPower Consumption:")
        print(f"  Package: min={df['rapl_package_power'].min():5.1f}W  "
              f"max={df['rapl_package_power'].max():5.1f}W  "
              f"avg={df['rapl_package_power'].mean():5.1f}W")

        if 'rapl_core_power' in df.columns:
            print(f"  Core:    min={df['rapl_core_power'].min():5.1f}W  "
                  f"max={df['rapl_core_power'].max():5.1f}W  "
                  f"avg={df['rapl_core_power'].mean():5.1f}W")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CPU utilization and power data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Plot CPU utilization only:
    %(prog)s power_data.csv

  Plot CPU utilization with power:
    %(prog)s power_data.csv --with-power

  Save plot to file:
    %(prog)s power_data.csv --with-power --output cpu_power_plot.png

  Show statistics only:
    %(prog)s power_data.csv --stats-only
        """
    )

    parser.add_argument(
        "csv_file",
        type=str,
        help="CSV file from power_data_collector.py"
    )

    parser.add_argument(
        "--with-power",
        action="store_true",
        help="Include RAPL package power on right y-axis"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save plot to file instead of displaying"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't plot"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Print statistics
    print_summary_stats(args.csv_file)

    # Plot unless stats-only
    if not args.stats_only:
        plot_cpu_metrics(args.csv_file, args.with_power, args.output)


if __name__ == "__main__":
    main()
