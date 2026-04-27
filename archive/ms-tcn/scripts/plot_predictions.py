#!/usr/bin/env python3
"""
Parse and plot live prediction output from power_predictor.py
Supports both terminal log format and CSV format
Generates separate plots for package and DRAM power

Usage:
    # From terminal log (creates *_package.png and *_dram.png)
    sudo python3 src/power_predictor.py --model models/best_model.pth --live --scroll --duration 300 2>&1 | tee prediction_log.txt
    python3 scripts/plot_predictions.py prediction_log.txt

    # From CSV file (multi-output model with package + dram)
    python3 src/power_predictor.py --model models/best_model.pth --live --save predictions.csv --duration 300
    python3 scripts/plot_predictions.py predictions.csv

    # Custom output filename base (creates custom_name_package.png and custom_name_dram.png)
    python3 scripts/plot_predictions.py predictions.csv custom_name

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
    [10:49:14] #    73 (  1.0Hz) | package:  37.45W (act: 25.72W err:+11.73W  +45.6%) dram:  8.12W (act: 7.95W err:+0.17W  +2.1%)

    Returns: dict with 'package' and 'dram' data, each containing seq_num, timestamp, predicted, actual, error, error_pct
    """
    # Extract common parts
    header_pattern = r'\[(\d+:\d+:\d+)\]\s+#\s+(\d+)\s+\([^)]+\)\s+\|'
    header_match = re.search(header_pattern, line)

    if not header_match:
        return None

    timestamp = header_match.group(1)
    seq_num = int(header_match.group(2))

    result = {}

    # Pattern to match each power domain (package or dram)
    domain_pattern = r'(\w+):\s+([\d.]+)W\s+\(act:\s+([\d.]+)W\s+err:([+-][\d.]+)W\s+([+-][\d.]+)%\)'

    for match in re.finditer(domain_pattern, line):
        domain = match.group(1)
        predicted = float(match.group(2))
        actual = float(match.group(3))
        error = float(match.group(4))
        error_pct = float(match.group(5))

        result[domain] = {
            'seq_num': seq_num,
            'timestamp': timestamp,
            'predicted': predicted,
            'actual': actual,
            'error': error,
            'error_pct': error_pct
        }

    return result if result else None

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
    """Parse CSV file with prediction data

    Expected columns:
    - timestamp, sample, predicted_rapl_package_power, actual_rapl_package_power
    - predicted_rapl_dram_power, actual_rapl_dram_power

    Returns: dict with 'package' and 'dram' lists
    """
    df = pd.read_csv(file_path)

    data = {'package': [], 'dram': []}

    # Check which columns are available
    has_package = 'predicted_rapl_package_power' in df.columns
    has_dram = 'predicted_rapl_dram_power' in df.columns

    for _, row in df.iterrows():
        # Skip warmup samples (actual power is 0)
        if has_package and row.get('actual_rapl_package_power', 0.0) == 0.0:
            continue

        # Parse package power
        if has_package:
            predicted = row['predicted_rapl_package_power']
            actual = row['actual_rapl_package_power']
            error = predicted - actual
            error_pct = (error / actual * 100) if actual != 0 else 0

            data['package'].append({
                'seq_num': int(row['sample']),
                'timestamp': str(row['timestamp']),
                'predicted': predicted,
                'actual': actual,
                'error': error,
                'error_pct': error_pct
            })

        # Parse DRAM power
        if has_dram:
            predicted = row['predicted_rapl_dram_power']
            actual = row['actual_rapl_dram_power']
            error = predicted - actual
            error_pct = (error / actual * 100) if actual != 0 else 0

            data['dram'].append({
                'seq_num': int(row['sample']),
                'timestamp': str(row['timestamp']),
                'predicted': predicted,
                'actual': actual,
                'error': error,
                'error_pct': error_pct
            })

    return data

def parse_log_file(file_path):
    """Parse log file and extract all predictions

    Returns: dict with 'package' and 'dram' lists
    """
    data = {'package': [], 'dram': []}

    if file_path == '-':
        # Read from stdin
        lines = sys.stdin.readlines()
    else:
        with open(file_path, 'r') as f:
            lines = f.readlines()

    for line in lines:
        parsed = parse_prediction_line(line)
        if parsed:
            # parsed is a dict with 'package' and/or 'dram' keys
            for domain, domain_data in parsed.items():
                data[domain].append(domain_data)

    return data

def plot_predictions(data, domain='package', output_file='results/plots/prediction_comparison.png'):
    """Create visualization of predictions vs actual for a specific power domain

    Args:
        data: List of prediction dicts for the domain
        domain: Power domain name ('package' or 'dram')
        output_file: Path to save the plot
    """

    if not data:
        print(f"ERROR: No prediction data found for {domain} domain")
        return

    # Extract arrays
    seq_nums = [d['seq_num'] for d in data]
    predicted = np.array([d['predicted'] for d in data])
    actual = np.array([d['actual'] for d in data])
    error = np.array([d['error'] for d in data])

    # Calculate statistics
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mean_error = np.mean(error)
    max_error = np.max(np.abs(error))

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    domain_title = domain.upper() if domain else 'Package'
    fig.suptitle(f'MS-TCN {domain_title} Power Prediction Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Predicted vs Actual over time
    ax1.plot(seq_nums, actual, 'b-', linewidth=2, label='Actual Power', alpha=0.7)
    ax1.plot(seq_nums, predicted, 'r--', linewidth=2, label='Predicted Power', alpha=0.7)
    ax1.fill_between(seq_nums, actual, predicted, alpha=0.2, color='gray', label='Error')
    ax1.set_xlabel('Sample #', fontsize=12)
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Predicted vs Actual Power Over Time', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f'MAE: {mae:.2f}W\nRMSE: {rmse:.2f}W\nMean Error: {mean_error:+.2f}W\nMax Error: {max_error:.2f}W'
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

    # Check if data is empty for all domains
    has_package = len(data.get('package', [])) > 0
    has_dram = len(data.get('dram', [])) > 0

    if not has_package and not has_dram:
        print("ERROR: No prediction data found in input")
        print("\nExpected formats:")
        print("  Log: [HH:MM:SS] #   N (1.0Hz) | package: XX.XXW (act: YY.YYW err:+ZZ.ZZW  +PP.P%) dram: XX.XXW (...)")
        print("  CSV: timestamp,sample,predicted_rapl_package_power,actual_rapl_package_power,predicted_rapl_dram_power,...")
        sys.exit(1)

    print(f"Found {len(data['package'])} package samples, {len(data['dram'])} dram samples")

    # Determine base output filename
    if len(sys.argv) > 2:
        output_base = sys.argv[2]
        # Remove extension if present
        output_base = output_base.rsplit('.', 1)[0]
    else:
        output_base = 'results/plots/prediction_comparison'

    # Create plots for each available domain
    output_files = []

    if has_package:
        output_file_package = f"{output_base}_package.png"
        print(f"\n{'='*60}")
        print(f"Creating plot for PACKAGE power...")
        print(f"{'='*60}")
        plot_predictions(data['package'], domain='package', output_file=output_file_package)
        output_files.append(output_file_package)

    if has_dram:
        output_file_dram = f"{output_base}_dram.png"
        print(f"\n{'='*60}")
        print(f"Creating plot for DRAM power...")
        print(f"{'='*60}")
        plot_predictions(data['dram'], domain='dram', output_file=output_file_dram)
        output_files.append(output_file_dram)

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(output_files)} plots:")
    for f in output_files:
        print(f"  - {f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
