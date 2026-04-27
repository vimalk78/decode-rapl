#!/usr/bin/env python3
"""
Analyze how idle samples appear in training data
Check if there are any long sustained idle sequences (64+ samples)
"""

import pandas as pd
import numpy as np
import sys

def analyze_idle_sequences(csv_file, sequence_length=64):
    """Find and analyze idle sequence patterns"""

    df = pd.read_csv(csv_file)

    # Define idle as cpu_idle > 95%
    is_idle = df['cpu_idle_percent'] > 95.0

    # Find consecutive idle runs
    idle_runs = []
    current_run = 0

    for idle in is_idle:
        if idle:
            current_run += 1
        else:
            if current_run > 0:
                idle_runs.append(current_run)
            current_run = 0

    if current_run > 0:
        idle_runs.append(current_run)

    print("=" * 80)
    print(f"IDLE SEQUENCE ANALYSIS: {csv_file}")
    print("=" * 80)
    print()
    print(f"Total samples: {len(df):,}")
    print(f"Idle samples (cpu_idle > 95%): {is_idle.sum():,} ({is_idle.sum()/len(df)*100:.1f}%)")
    print()
    print(f"Consecutive idle runs found: {len(idle_runs)}")
    print()

    if idle_runs:
        idle_runs = np.array(idle_runs)
        print(f"Idle run lengths:")
        print(f"  Min:    {idle_runs.min()} samples")
        print(f"  Max:    {idle_runs.max()} samples")
        print(f"  Mean:   {idle_runs.mean():.1f} samples")
        print(f"  Median: {np.median(idle_runs):.0f} samples")
        print()

        # Check for sequence-length runs
        long_runs = idle_runs[idle_runs >= sequence_length]
        print(f"Runs >= {sequence_length} samples (full sequence): {len(long_runs)}")

        if len(long_runs) > 0:
            print(f"  These {len(long_runs)} runs contain {long_runs.sum():,} idle samples")
            print(f"  Longest sustained idle: {long_runs.max()} samples ({long_runs.max():.1f} seconds)")
        else:
            print(f"  ⚠️  WARNING: NO idle runs >= {sequence_length} samples!")
            print(f"  This means the model never saw 'fully idle' sequences during training.")
            print(f"  All training idle samples had recent workload in their 64-sample window.")

        print()
        print("Distribution of idle run lengths:")
        bins = [1, 5, 10, 20, 30, 50, 64, 100, 200, 500, max(idle_runs.max(), 1000)]
        for i in range(len(bins)-1):
            count = ((idle_runs >= bins[i]) & (idle_runs < bins[i+1])).sum()
            if count > 0:
                print(f"  {bins[i]:>4}-{bins[i+1]:<4} samples: {count:>4} runs")

        # Show power during different idle run lengths
        print()
        print("Average power by idle run length:")

        # Reconstruct which samples belong to which run
        run_idx = 0
        sample_to_run_length = np.zeros(len(df))
        current_run_len = 0

        for i, idle in enumerate(is_idle):
            if idle:
                current_run_len += 1
            else:
                if current_run_len > 0:
                    # Mark previous samples with their run length
                    for j in range(i - current_run_len, i):
                        sample_to_run_length[j] = current_run_len
                current_run_len = 0

        # Handle last run
        if current_run_len > 0:
            for j in range(len(df) - current_run_len, len(df)):
                sample_to_run_length[j] = current_run_len

        # Calculate average power for different run lengths
        if 'rapl_package_power' in df.columns:
            for bin_start, bin_end in [(1, 10), (10, 30), (30, 64), (64, 200), (200, 1000)]:
                mask = (sample_to_run_length >= bin_start) & (sample_to_run_length < bin_end)
                if mask.sum() > 0:
                    avg_power = df.loc[mask, 'rapl_package_power'].mean()
                    print(f"  Runs {bin_start:>3}-{bin_end:<4} samples: {avg_power:.2f}W (n={mask.sum():>5})")

    print()
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_idle_sequences.py <data.csv>")
        sys.exit(1)

    analyze_idle_sequences(sys.argv[1])
