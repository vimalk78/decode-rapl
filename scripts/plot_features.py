#!/usr/bin/env python3
"""
Plot all input features from prediction CSV for comparison with training data

Creates a multi-panel plot showing:
- User CPU %
- System CPU %
- I/O wait %
- Context switches per second
- Predicted vs Actual power (if available)

Usage:
    python scripts/plot_features.py predictions.csv
    python scripts/plot_features.py predictions.csv --output feature_comparison.png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_features(csv_path: Path, output_path: Path = None):
    """
    Plot all features from prediction CSV

    Args:
        csv_path: Path to CSV file with predictions/features
        output_path: Path to save plot (default: same dir as CSV)
    """
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Check for required columns
    required = ['user_percent', 'system_percent', 'iowait_percent']

    # Handle different column name variations
    if 'ctx_switches_per_sec' in df.columns:
        ctx_col = 'ctx_switches_per_sec'
    elif 'context_switches' in df.columns:
        ctx_col = 'context_switches'
    else:
        print("Error: No context switches column found")
        sys.exit(1)

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    # Check if we have power data
    has_predicted = 'predicted_power' in df.columns
    has_actual = 'actual_power' in df.columns or 'package_power_watts' in df.columns

    if has_actual:
        actual_col = 'actual_power' if 'actual_power' in df.columns else 'package_power_watts'

    # Determine number of subplots
    n_plots = 5 if (has_predicted or has_actual) else 4

    # Create figure
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots), sharex=True)
    fig.suptitle(f'Feature Analysis: {csv_path.name}', fontsize=14, fontweight='bold')

    # Time or sample numbers for x-axis
    if 'timestamp_unix' in df.columns:
        x_values = df['timestamp_unix'] - df['timestamp_unix'].iloc[0]
        x_label = 'Time (seconds)'
    elif 'timestamp' in df.columns:
        x_values = df['timestamp'] - df['timestamp'].iloc[0]
        x_label = 'Time (seconds)'
    else:
        x_values = np.arange(len(df))
        x_label = 'Sample #'

    # Plot 1: User CPU %
    ax = axes[0]
    ax.plot(x_values, df['user_percent'], linewidth=1, color='green', alpha=0.7)
    ax.set_ylabel('User CPU (%)', fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.set_title('User CPU Usage', fontsize=11)

    user_stats = f"μ={df['user_percent'].mean():.1f}% σ={df['user_percent'].std():.1f}% range=[{df['user_percent'].min():.1f}, {df['user_percent'].max():.1f}]"
    ax.text(0.02, 0.95, user_stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: System CPU %
    ax = axes[1]
    ax.plot(x_values, df['system_percent'], linewidth=1, color='blue', alpha=0.7)
    ax.set_ylabel('System CPU (%)', fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.set_title('System CPU Usage', fontsize=11)

    sys_stats = f"μ={df['system_percent'].mean():.1f}% σ={df['system_percent'].std():.1f}% range=[{df['system_percent'].min():.1f}, {df['system_percent'].max():.1f}]"
    ax.text(0.02, 0.95, sys_stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: I/O wait %
    ax = axes[2]
    ax.plot(x_values, df['iowait_percent'], linewidth=1, color='orange', alpha=0.7)
    ax.set_ylabel('I/O Wait (%)', fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.set_title('I/O Wait Percentage', fontsize=11)

    io_stats = f"μ={df['iowait_percent'].mean():.1f}% σ={df['iowait_percent'].std():.1f}% range=[{df['iowait_percent'].min():.1f}, {df['iowait_percent'].max():.1f}]"
    ax.text(0.02, 0.95, io_stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Context switches
    ax = axes[3]
    ax.plot(x_values, df[ctx_col], linewidth=1, color='purple', alpha=0.7)
    ax.set_ylabel('Context Switches/sec', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Context Switches per Second', fontsize=11)

    ctx_stats = f"μ={df[ctx_col].mean():.0f} σ={df[ctx_col].std():.0f} range=[{df[ctx_col].min():.0f}, {df[ctx_col].max():.0f}]"
    ax.text(0.02, 0.95, ctx_stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 5: Power (if available)
    if has_predicted or has_actual:
        ax = axes[4]

        if has_actual:
            ax.plot(x_values, df[actual_col], linewidth=1.5, color='blue',
                   label='Actual Power', alpha=0.7)
            actual_mean = df[actual_col].mean()
            actual_std = df[actual_col].std()

        if has_predicted:
            ax.plot(x_values, df['predicted_power'], linewidth=1.5, color='red',
                   linestyle='--', label='Predicted Power', alpha=0.7)
            pred_mean = df['predicted_power'].mean()
            pred_std = df['predicted_power'].std()

        ax.set_ylabel('Power (W)', fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title('Power Consumption', fontsize=11)

        # Stats box
        if has_actual and has_predicted:
            mae = np.mean(np.abs(df['predicted_power'] - df[actual_col]))
            rmse = np.sqrt(np.mean((df['predicted_power'] - df[actual_col])**2))
            power_stats = f"Actual: μ={actual_mean:.1f}W σ={actual_std:.1f}W\nPred: μ={pred_mean:.1f}W σ={pred_std:.1f}W\nMAE={mae:.2f}W RMSE={rmse:.2f}W"
        elif has_actual:
            power_stats = f"μ={actual_mean:.1f}W σ={actual_std:.1f}W range=[{df[actual_col].min():.1f}, {df[actual_col].max():.1f}]"
        else:
            power_stats = f"μ={pred_mean:.1f}W σ={pred_std:.1f}W range=[{df['predicted_power'].min():.1f}, {df['predicted_power'].max():.1f}]"

        ax.text(0.02, 0.95, power_stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[3].set_xlabel('Sample #', fontsize=11)

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_features.png"

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature plot saved to: {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS")
    print(f"{'='*70}")
    print(f"Samples: {len(df)}")
    print(f"\nUser CPU:     {df['user_percent'].mean():.1f}% ± {df['user_percent'].std():.1f}%")
    print(f"System CPU:   {df['system_percent'].mean():.1f}% ± {df['system_percent'].std():.1f}%")
    print(f"I/O Wait:     {df['iowait_percent'].mean():.1f}% ± {df['iowait_percent'].std():.1f}%")
    print(f"Context Sw:   {df[ctx_col].mean():.0f} ± {df[ctx_col].std():.0f} /sec")

    if has_actual:
        print(f"\nActual Power: {df[actual_col].mean():.2f}W ± {df[actual_col].std():.2f}W")
    if has_predicted:
        print(f"Pred Power:   {df['predicted_power'].mean():.2f}W ± {df['predicted_power'].std():.2f}W")

    print(f"{'='*70}\n")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot all input features from prediction CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_features.py predictions.csv
  python scripts/plot_features.py data/real_workload.csv --output comparison.png
        """
    )

    parser.add_argument('csv_file', type=Path,
                       help='CSV file with predictions and features')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output PNG file (default: <csv_stem>_features.png)')

    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)

    plot_features(args.csv_file, args.output)


if __name__ == '__main__':
    main()
