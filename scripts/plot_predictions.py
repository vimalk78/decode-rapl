#!/usr/bin/env python3
"""
Parse and plot live prediction output from power_predictor.py (v3)
Supports both terminal log format and CSV format

Usage:
    # From terminal log
    sudo python3 src/power_predictor.py --model models/tau8/tau8_multifeature_model.pth --live --scroll 2>&1 | tee prediction_log.txt
    python3 scripts/plot_predictions.py prediction_log.txt

    # From CSV file
    python3 src/power_predictor.py --model models/tau8/tau8_multifeature_model.pth --live --save predictions.csv
    python3 scripts/plot_predictions.py predictions.csv

    # Custom output filename
    python3 scripts/plot_predictions.py predictions.csv custom_plot.png

    # Or from stdin
    cat prediction_log.txt | python3 scripts/plot_predictions.py
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def parse_prediction_line(line):
    """
    Parse a line like:
    [14:50:00] #    38 (  8.7Hz) | CPU:  12.5% (U: 8.0% S: 3.5% IO: 1.0%) | Pred:  69.23W | Actual:  60.08W | Err: +9.15W (+15.24%) | MAPE:  9.03%

    Returns: dict with parsed data or None if line doesn't match
    """
    # Pattern to match the v3 log format
    pattern = r'\[(\d+:\d+:\d+)\]\s+#\s*(\d+)\s+\([^)]+\)\s+\|\s+CPU:\s+([\d.]+)%.*?\|\s+Pred:\s+([\d.]+)W\s+\|\s+Actual:\s+([\d.]+)W\s+\|\s+Err:\s+([+-][\d.]+)W\s+\(([+-][\d.]+)%\)'

    match = re.search(pattern, line)

    if not match:
        return None

    timestamp = match.group(1)
    seq_num = int(match.group(2))
    cpu_usage = float(match.group(3))
    predicted = float(match.group(4))
    actual = float(match.group(5))
    error_pct = float(match.group(7))

    return {
        'seq_num': seq_num,
        'timestamp': timestamp,
        'cpu_usage': cpu_usage,
        'predicted': predicted,
        'actual': actual,
        'error_pct': error_pct
    }


def detect_file_format(file_path):
    """Detect if file is CSV or log format"""
    if file_path == '-':
        # Can't easily detect stdin, assume log
        return 'log'

    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        # CSV format starts with "timestamp,sample,..."
        if first_line.startswith('timestamp,'):
            return 'csv'
        return 'log'


def parse_csv_file(file_path):
    """Parse CSV file with prediction data (v3 format)

    Expected columns:
    - timestamp, sample, user_percent, system_percent, iowait_percent, context_switches
    - predicted_power, actual_power (optional), error_pct (optional)

    Returns: list of prediction dicts
    """
    df = pd.read_csv(file_path)

    data = []

    # Check required columns
    required = ['sample', 'predicted_power']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return data

    has_actual = 'actual_power' in df.columns

    # Check if we have v3 multi-feature format
    has_multifeature = all(col in df.columns for col in ['user_percent', 'system_percent', 'iowait_percent'])

    for _, row in df.iterrows():
        # Calculate total CPU usage
        if has_multifeature:
            # V3 format: separate columns for user, system, iowait
            cpu_usage = row['user_percent'] + row['system_percent'] + row['iowait_percent']
        elif 'cpu_usage' in df.columns:
            # Fallback: single cpu_usage column
            cpu_usage = float(row['cpu_usage'])
        else:
            # No CPU data available
            cpu_usage = 0.0

        entry = {
            'seq_num': int(row['sample']),
            'timestamp': str(row.get('timestamp', '')),
            'cpu_usage': cpu_usage,
            'predicted': row['predicted_power'],
        }

        if has_actual:
            entry['actual'] = row['actual_power']
            # Use error_pct if available, otherwise calculate
            if 'error_pct' in df.columns:
                entry['error_pct'] = row['error_pct']
            else:
                actual = row['actual_power']
                predicted = row['predicted_power']
                entry['error_pct'] = ((predicted - actual) / actual * 100) if actual != 0 else 0

        data.append(entry)

    return data


def parse_log_file(file_path):
    """Parse log file and extract all predictions

    Returns: list of prediction dicts
    """
    data = []

    if file_path == '-':
        # Read from stdin
        lines = sys.stdin.readlines()
    else:
        with open(file_path, 'r') as f:
            lines = f.readlines()

    for line in lines:
        parsed = parse_prediction_line(line)
        if parsed:
            data.append(parsed)

    return data


def plot_predictions(data, output_file='results/plots/prediction_comparison.png'):
    """Create visualization of predictions vs actual

    Args:
        data: List of prediction dicts
        output_file: Path to save the plot
    """

    if not data:
        print("ERROR: No prediction data found")
        return

    # Extract arrays
    seq_nums = [d['seq_num'] for d in data]
    cpu_usage = np.array([d['cpu_usage'] for d in data])
    predicted = np.array([d['predicted'] for d in data])

    # Check if we have actual power
    has_actual = 'actual' in data[0]

    if has_actual:
        actual = np.array([d['actual'] for d in data])
        error = predicted - actual
        error_pct = np.array([d['error_pct'] for d in data])

        # Calculate statistics
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        mape = np.mean(np.abs(error_pct))
        mean_error = np.mean(error)
        max_error = np.max(np.abs(error))

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('DECODE-RAPL v3 Power Prediction Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Predicted vs Actual over time with CPU usage
        ax1_cpu = ax1.twinx()  # Secondary y-axis for CPU usage

        # Plot power
        ax1.plot(seq_nums, actual, 'b-', linewidth=2, label='Actual Power', alpha=0.7)
        ax1.plot(seq_nums, predicted, 'r--', linewidth=2, label='Predicted Power', alpha=0.7)
        ax1.fill_between(seq_nums, actual, predicted, alpha=0.2, color='gray', label='Error')

        # Plot CPU usage on secondary axis
        ax1_cpu.plot(seq_nums, cpu_usage, 'g-', linewidth=1, alpha=0.3, label='CPU Usage')

        ax1.set_xlabel('Sample #', fontsize=12)
        ax1.set_ylabel('Power (W)', fontsize=12, color='b')
        ax1_cpu.set_ylabel('CPU Usage (%)', fontsize=12, color='g')
        ax1.set_title('Predicted vs Actual Power Over Time', fontsize=14)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_cpu.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_cpu.tick_params(axis='y', labelcolor='g')

        # Add statistics box
        stats_text = f'MAE: {mae:.2f}W\nRMSE: {rmse:.2f}W\nMAPE: {mape:.2f}%\nMean Error: {mean_error:+.2f}W\nMax Error: {max_error:.2f}W'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Error over time
        ax2.plot(seq_nums, error, 'g-', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax2.axhline(y=mae, color='r', linestyle=':', linewidth=1, label=f'MAE = {mae:.2f}W')
        ax2.axhline(y=-mae, color='r', linestyle=':', linewidth=1)
        ax2.fill_between(seq_nums, 0, error, where=(error>=0), alpha=0.3, color='red', label='Over-prediction')
        ax2.fill_between(seq_nums, 0, error, where=(error<0), alpha=0.3, color='blue', label='Under-prediction')
        ax2.set_xlabel('Sample #', fontsize=12)
        ax2.set_ylabel('Error (W)', fontsize=12)
        ax2.set_title('Prediction Error Over Time', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scatter plot - Predicted vs Actual
        ax3.scatter(actual, predicted, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Calculate R²
        correlation_matrix = np.corrcoef(actual, predicted)
        r_squared = correlation_matrix[0, 1]**2

        ax3.set_xlabel('Actual Power (W)', fontsize=12)
        ax3.set_ylabel('Predicted Power (W)', fontsize=12)
        ax3.set_title(f'Predicted vs Actual Power (R² = {r_squared:.4f})', fontsize=14)
        ax3.legend(loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')

    else:
        # Only predictions available (no actual power)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle('DECODE-RAPL v3 Power Predictions', fontsize=16, fontweight='bold')

        # Plot 1: Predicted power over time with CPU usage
        ax1_cpu = ax1.twinx()

        ax1.plot(seq_nums, predicted, 'r-', linewidth=2, label='Predicted Power', alpha=0.7)
        ax1_cpu.plot(seq_nums, cpu_usage, 'g-', linewidth=1, alpha=0.3, label='CPU Usage')

        ax1.set_xlabel('Sample #', fontsize=12)
        ax1.set_ylabel('Power (W)', fontsize=12, color='r')
        ax1_cpu.set_ylabel('CPU Usage (%)', fontsize=12, color='g')
        ax1.set_title('Predicted Power Over Time', fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_cpu.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='r')
        ax1_cpu.tick_params(axis='y', labelcolor='g')

        # Plot 2: Power vs CPU usage scatter
        ax2.scatter(cpu_usage, predicted, alpha=0.5, s=20)
        ax2.set_xlabel('CPU Usage (%)', fontsize=12)
        ax2.set_ylabel('Predicted Power (W)', fontsize=12)
        ax2.set_title('Power vs CPU Usage', fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    print(f"")

    if has_actual:
        print(f"Actual Power:")
        print(f"  Mean:  {actual.mean():.2f}W")
        print(f"  Range: {actual.min():.2f}W - {actual.max():.2f}W")
        print(f"")
        print(f"Predicted Power:")
        print(f"  Mean:  {predicted.mean():.2f}W")
        print(f"  Range: {predicted.min():.2f}W - {predicted.max():.2f}W")
        print(f"")
        print(f"Error Metrics:")
        print(f"  MAE:        {mae:.2f}W")
        print(f"  RMSE:       {rmse:.2f}W")
        print(f"  MAPE:       {mape:.2f}%")
        print(f"  Mean Error: {mean_error:+.2f}W ({'over' if mean_error > 0 else 'under'}-prediction)")
        print(f"  Max Error:  {max_error:.2f}W")
        print(f"  R²:         {r_squared:.4f}")
        print(f"")

        # Analyze bias
        if abs(mean_error) > 5:
            print(f"⚠️  WARNING: Systematic bias detected!")
            print(f"   Model consistently {'over' if mean_error > 0 else 'under'}-predicts by {abs(mean_error):.2f}W")
            print(f"   This could be due to normalization offset or training data bias")
        else:
            print(f"✓ No significant systematic bias")

        # Check if target accuracy achieved
        if mape < 5.0:
            print(f"✓ Target accuracy (<5% MAPE) achieved!")
        else:
            print(f"✗ Target accuracy not achieved. Current: {mape:.2f}%, Target: <5%")
    else:
        print(f"Predicted Power:")
        print(f"  Mean:  {predicted.mean():.2f}W")
        print(f"  Range: {predicted.min():.2f}W - {predicted.max():.2f}W")
        print(f"")
        print(f"⚠️  No actual power data available for comparison")

    print(f"")
    print(f"CPU Usage:")
    print(f"  Mean:  {cpu_usage.mean():.1f}%")
    print(f"  Range: {cpu_usage.min():.1f}% - {cpu_usage.max():.1f}%")

    print(f"{'='*60}\n")

    return fig


def main():
    """Main function"""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '-'  # Read from stdin

    print(f"Parsing prediction data from: {input_file if input_file != '-' else 'stdin'}")

    # Detect format and parse data
    file_format = detect_file_format(input_file)

    if file_format == 'csv':
        print("Detected CSV format")
        data = parse_csv_file(input_file)
    else:
        print("Detected log format")
        data = parse_log_file(input_file)

    if not data:
        print("ERROR: No prediction data found in input")
        print("\nExpected formats:")
        print("  Log: [HH:MM:SS] #   N (X.XHz) | CPU: XX.X% (U:XX.X% S:XX.X% IO:XX.X%) | Pred: XX.XXW | Actual: XX.XXW | Err: +XX.XXW (+XX.XX%) | MAPE: XX.XX%")
        print("  CSV: timestamp,sample,user_percent,system_percent,iowait_percent,context_switches,predicted_power,actual_power,error_pct")
        sys.exit(1)

    print(f"Found {len(data)} samples")

    # Determine output filename
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'results/plots/prediction_comparison.png'

    # Create plot
    print(f"\n{'='*60}")
    print(f"Creating plot...")
    print(f"{'='*60}")
    plot_predictions(data, output_file=output_file)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
