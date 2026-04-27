#!/usr/bin/env python3
"""
Compare feature distributions between training idle and live idle
"""

import pandas as pd
import numpy as np
import sys

# 15 features used in current model
FEATURES_15 = [
    'cpu_user_percent', 'cpu_system_percent', 'cpu_idle_percent',
    'cpu_iowait_percent', 'cpu_irq_percent', 'cpu_softirq_percent',
    'interrupts_sec', 'context_switches_sec',
    'running_processes', 'page_faults_sec',
    'memory_used_mb', 'memory_cached_mb', 'memory_buffers_mb',
    'memory_free_mb', 'swap_used_mb'
]

def analyze_idle_features(training_file, live_file):
    """Compare feature distributions"""

    # Load data
    train_df = pd.read_csv(training_file)
    live_df = pd.read_csv(live_file)

    print("=" * 80)
    print("FEATURE COMPARISON: Training Idle vs Live Idle (15 features)")
    print("=" * 80)
    print()

    print(f"Training idle samples: {len(train_df)}")
    print(f"Live idle samples: {len(live_df)}")
    print()

    # Compare each feature
    print(f"{'Feature':<25} {'Train Mean':<12} {'Live Mean':<12} {'Diff %':<10} {'Status'}")
    print("-" * 80)

    large_diffs = []

    for feature in FEATURES_15:
        if feature not in train_df.columns:
            print(f"{feature:<25} MISSING IN TRAINING DATA")
            continue
        if feature not in live_df.columns:
            print(f"{feature:<25} MISSING IN LIVE DATA")
            continue

        train_mean = train_df[feature].mean()
        live_mean = live_df[feature].mean()

        # Calculate percentage difference
        if abs(train_mean) > 0.01:
            diff_pct = ((live_mean - train_mean) / abs(train_mean)) * 100
        else:
            diff_pct = 0

        # Flag large differences
        status = ""
        if abs(diff_pct) > 50:
            status = "🚨 LARGE"
            large_diffs.append((feature, diff_pct, train_mean, live_mean))
        elif abs(diff_pct) > 20:
            status = "⚠️  MEDIUM"
            large_diffs.append((feature, diff_pct, train_mean, live_mean))
        elif abs(diff_pct) > 10:
            status = "⚡ SMALL"

        print(f"{feature:<25} {train_mean:>11.2f}  {live_mean:>11.2f}  {diff_pct:>+9.1f}%  {status}")

    # Show power comparison
    print()
    print("=" * 80)
    if 'rapl_package_power' in train_df.columns and 'rapl_package_power' in live_df.columns:
        train_power = train_df['rapl_package_power'].mean()
        live_power = live_df['rapl_package_power'].mean()
        print(f"Actual Power:")
        print(f"  Training idle: {train_power:.2f}W")
        print(f"  Live idle:     {live_power:.2f}W")
        print(f"  Difference:    {live_power - train_power:+.2f}W")

    # Summary
    print()
    print("=" * 80)
    print("FEATURES WITH SIGNIFICANT DIFFERENCES:")
    print("=" * 80)

    if large_diffs:
        for feature, diff_pct, train_val, live_val in large_diffs:
            print(f"\n{feature}:")
            print(f"  Training: {train_val:.2f}")
            print(f"  Live:     {live_val:.2f}")
            print(f"  Change:   {diff_pct:+.1f}%")
    else:
        print("No significant differences found")

    print()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 compare_idle_features.py <training_idle.csv> <live_idle.csv>")
        sys.exit(1)

    analyze_idle_features(sys.argv[1], sys.argv[2])
