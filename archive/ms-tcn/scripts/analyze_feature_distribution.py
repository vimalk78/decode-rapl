#!/usr/bin/env python3
"""
Feature Distribution Analyzer

Compares live idle features vs training idle features to identify
if the model sees out-of-distribution features during inference.

This helps diagnose why the model predicts ~40W when training data
had plenty of ~27W samples.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_training_idle_features(training_csv, idle_threshold=30.0):
    """Load and extract idle samples from training data."""

    print(f"Loading training data from {training_csv}...")
    df = pd.read_csv(training_csv)

    # Extract idle samples (power < threshold)
    idle_mask = df['rapl_package_power'] < idle_threshold
    idle_df = df[idle_mask]

    print(f"Found {len(idle_df)} idle samples (power < {idle_threshold}W)")
    print(f"  Power range: {idle_df['rapl_package_power'].min():.1f}W - {idle_df['rapl_package_power'].max():.1f}W")
    print(f"  Power mean: {idle_df['rapl_package_power'].mean():.1f}W")

    return df, idle_df


def capture_live_idle_features():
    """Capture current system features (requires running on target machine)."""

    print("\nCapturing live system features...")
    print("NOTE: This requires psutil. Install with: pip install psutil")

    try:
        import psutil
        import time
    except ImportError:
        print("ERROR: psutil not installed")
        return None

    # Collect 10 samples over 10 seconds
    samples = []
    for i in range(10):
        cpu_times = psutil.cpu_times_percent(interval=1.0)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load = psutil.getloadavg()

        sample = {
            'cpu_user_percent': cpu_times.user,
            'cpu_system_percent': cpu_times.system,
            'cpu_idle_percent': cpu_times.idle,
            'cpu_iowait_percent': getattr(cpu_times, 'iowait', 0.0),
            'cpu_irq_percent': getattr(cpu_times, 'irq', 0.0),
            'cpu_softirq_percent': getattr(cpu_times, 'softirq', 0.0),
            'memory_used_mb': mem.used / (1024 * 1024),
            'memory_cached_mb': getattr(mem, 'cached', 0) / (1024 * 1024),
            'memory_buffers_mb': getattr(mem, 'buffers', 0) / (1024 * 1024),
            'memory_free_mb': mem.free / (1024 * 1024),
            'swap_used_mb': swap.used / (1024 * 1024),
            'load_1min': load[0],
            'load_5min': load[1],
            'load_15min': load[2],
        }
        samples.append(sample)
        print(f"  Sample {i+1}/10: CPU={cpu_times.user:.1f}% user, {cpu_times.idle:.1f}% idle")

    live_df = pd.DataFrame(samples)
    print(f"Captured {len(live_df)} live samples")

    return live_df


def compare_distributions(training_idle_df, live_df, feature_cols):
    """Compare feature distributions between training idle and live."""

    print("\n" + "="*80)
    print("Feature Distribution Comparison: Training Idle vs Live")
    print("="*80)

    # For each feature, compute statistics
    results = []

    for col in feature_cols:
        if col not in training_idle_df.columns or col not in live_df.columns:
            print(f"Skipping {col} (not in both datasets)")
            continue

        train_vals = training_idle_df[col].values
        live_vals = live_df[col].values

        train_mean = np.mean(train_vals)
        train_std = np.std(train_vals)
        train_min = np.min(train_vals)
        train_max = np.max(train_vals)

        live_mean = np.mean(live_vals)
        live_std = np.std(live_vals)
        live_min = np.min(live_vals)
        live_max = np.max(live_vals)

        # Check if live is outside training range
        out_of_range = (live_min < train_min) or (live_max > train_max)

        # Check if live mean is significantly different (>2 std devs)
        if train_std > 0:
            mean_diff_sigma = abs(live_mean - train_mean) / train_std
        else:
            mean_diff_sigma = 0.0

        results.append({
            'feature': col,
            'train_mean': train_mean,
            'train_std': train_std,
            'train_range': f"{train_min:.2f} to {train_max:.2f}",
            'live_mean': live_mean,
            'live_std': live_std,
            'live_range': f"{live_min:.2f} to {live_max:.2f}",
            'mean_diff_sigma': mean_diff_sigma,
            'out_of_range': out_of_range,
        })

    # Print results
    print(f"\n{'Feature':<25} {'Train Mean':<12} {'Live Mean':<12} {'Δ (σ)':<10} {'Status'}")
    print("-" * 80)

    for r in results:
        status = ""
        if r['out_of_range']:
            status = "⚠️  OUT-OF-RANGE"
        elif r['mean_diff_sigma'] > 2.0:
            status = f"⚠️  SHIFTED ({r['mean_diff_sigma']:.1f}σ)"
        else:
            status = "✓ OK"

        print(f"{r['feature']:<25} {r['train_mean']:>11.2f} {r['live_mean']:>11.2f} {r['mean_diff_sigma']:>9.1f} {status}")

    print("\n" + "="*80)
    print("Detailed Feature Statistics")
    print("="*80)

    for r in results:
        print(f"\n{r['feature']}:")
        print(f"  Training idle: μ={r['train_mean']:.2f}, σ={r['train_std']:.2f}, range={r['train_range']}")
        print(f"  Live:          μ={r['live_mean']:.2f}, σ={r['live_std']:.2f}, range={r['live_range']}")
        if r['out_of_range']:
            print(f"  ⚠️  WARNING: Live values outside training range!")
        if r['mean_diff_sigma'] > 2.0:
            print(f"  ⚠️  WARNING: Live mean differs by {r['mean_diff_sigma']:.1f} standard deviations!")

    return results


def compare_normalized_distributions(training_df, training_idle_df, live_df, feature_cols):
    """Compare normalized feature distributions."""

    print("\n" + "="*80)
    print("Normalized Feature Distribution Comparison")
    print("="*80)
    print("\nThis shows what the MODEL actually sees after StandardScaler normalization")

    # Fit scaler on ALL training data (not just idle)
    scaler = StandardScaler()
    train_features = training_df[feature_cols].values
    scaler.fit(train_features)

    print(f"\nScaler fitted on {len(training_df)} training samples")

    # Transform idle training data
    train_idle_normalized = scaler.transform(training_idle_df[feature_cols].values)

    # Transform live data
    live_normalized = scaler.transform(live_df[feature_cols].values)

    print(f"\n{'Feature':<25} {'Train Idle':<20} {'Live':<20} {'Status'}")
    print("-" * 80)

    issues = []

    for i, col in enumerate(feature_cols):
        train_norm_mean = np.mean(train_idle_normalized[:, i])
        train_norm_std = np.std(train_idle_normalized[:, i])

        live_norm_mean = np.mean(live_normalized[:, i])
        live_norm_std = np.std(live_normalized[:, i])

        # Check if normalized values are extreme (>3 std from 0)
        extreme = abs(live_norm_mean) > 3.0

        status = ""
        if extreme:
            status = f"⚠️  EXTREME ({live_norm_mean:.1f}σ)"
            issues.append(col)
        elif abs(live_norm_mean - train_norm_mean) > 1.0:
            status = f"⚠️  SHIFTED"
            issues.append(col)
        else:
            status = "✓ OK"

        train_str = f"μ={train_norm_mean:>5.2f}, σ={train_norm_std:>4.2f}"
        live_str = f"μ={live_norm_mean:>5.2f}, σ={live_norm_std:>4.2f}"

        print(f"{col:<25} {train_str:<20} {live_str:<20} {status}")

    if issues:
        print("\n" + "="*80)
        print("⚠️  POTENTIAL ISSUE DETECTED")
        print("="*80)
        print(f"\nThese features have unusual normalized values during live idle:")
        for feature in issues:
            print(f"  - {feature}")
        print("\nThis could cause the model to predict incorrectly!")
    else:
        print("\n✓ All normalized features look reasonable")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature distribution: training idle vs live"
    )
    parser.add_argument('--training-data', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--live', action='store_true',
                       help='Capture live system features (requires psutil)')
    parser.add_argument('--live-csv', type=str,
                       help='Or provide pre-captured live features CSV')
    parser.add_argument('--idle-threshold', type=float, default=30.0,
                       help='Power threshold for idle samples (default: 30W)')

    args = parser.parse_args()

    # Load training data
    training_df, training_idle_df = load_training_idle_features(
        args.training_data, args.idle_threshold
    )

    # Get feature columns (exclude timestamp and target columns)
    exclude_cols = ['timestamp', 'rapl_package_power', 'rapl_core_power', 'rapl_dram_power']
    feature_cols = [col for col in training_df.columns if col not in exclude_cols]

    print(f"\nFeatures to analyze: {len(feature_cols)}")
    for col in feature_cols:
        print(f"  - {col}")

    # Get live data
    if args.live:
        live_df = capture_live_idle_features()
        if live_df is None:
            sys.exit(1)
    elif args.live_csv:
        print(f"\nLoading live data from {args.live_csv}...")
        live_df = pd.read_csv(args.live_csv)
        print(f"Loaded {len(live_df)} samples")
    else:
        print("\nERROR: Must specify --live or --live-csv")
        sys.exit(1)

    # Compare raw distributions
    results = compare_distributions(training_idle_df, live_df, feature_cols)

    # Compare normalized distributions
    issues = compare_normalized_distributions(
        training_df, training_idle_df, live_df, feature_cols
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if issues:
        print("\n⚠️  Found potential distribution mismatch issues!")
        print(f"   {len(issues)} features have unusual normalized values during live idle.")
        print("\nThis may explain why the model predicts ~40W instead of ~27W:")
        print("  - The model sees feature combinations it never saw during training")
        print("  - Even though raw values look similar, normalization amplifies differences")
        print("\nRecommendation:")
        print("  1. Check if feature collection differs between training and live")
        print("  2. Verify scaler parameters match between training and inference")
        print("  3. Consider retraining with more diverse idle states")
    else:
        print("\n✓ Feature distributions look similar between training and live")
        print("  This is NOT the cause of the 40W prediction issue.")
        print("  Need to investigate other factors (model architecture, temporal, etc.)")


if __name__ == '__main__':
    main()
