#!/usr/bin/env python3
"""
Example: Real-time Power Prediction in a VM

This script demonstrates how to use the trained LSTM model for real-time
power prediction inside a VM. It collects CPU usage via psutil and predicts
power consumption every second.

Usage:
    python example_vm_inference.py --model power_lstm_model.pth --duration 60
"""

import argparse
import time
import sys
from collections import deque
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Run: pip install psutil")
    sys.exit(1)

try:
    import pickle
except ImportError:
    pass

# Import from power_lstm module
try:
    from power_lstm import PowerPredictor
except ImportError:
    print("Error: power_lstm.py not found in current directory")
    sys.exit(1)


def get_vm_vcpus():
    """Get number of vCPUs in current VM"""
    return psutil.cpu_count(logical=True)


def collect_cpu_usage(duration: int = 1) -> float:
    """
    Collect CPU usage percentage over duration

    Args:
        duration: Measurement duration in seconds

    Returns:
        Average CPU usage (%)
    """
    return psutil.cpu_percent(interval=duration)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time power prediction in VM"
    )

    parser.add_argument(
        '--model',
        type=str,
        default='power_lstm_model.pth',
        help='Path to trained model'
    )

    parser.add_argument(
        '--preprocessor',
        type=str,
        default='preprocessor.pkl',
        help='Path to preprocessor pickle file'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Monitoring duration in seconds'
    )

    parser.add_argument(
        '--seq_length',
        type=int,
        default=60,
        help='Input sequence length (must match training)'
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train the model first: python power_lstm.py --mode train")
        return 1

    # Load preprocessor
    try:
        with open(args.preprocessor, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Loaded preprocessor from {args.preprocessor}")
    except FileNotFoundError:
        print(f"Error: Preprocessor not found at {args.preprocessor}")
        print("Train the model first to generate preprocessor.pkl")
        return 1

    # Initialize predictor
    print(f"Loading model from {args.model}...")
    predictor = PowerPredictor(
        args.model,
        seq_length=args.seq_length,
        device='cpu'  # Force CPU for VM deployment
    )
    predictor.set_power_scaler(preprocessor.power_scaler)

    # Get VM vCPUs
    vcpus = get_vm_vcpus()
    print(f"\nVM Configuration:")
    print(f"  vCPUs: {vcpus}")
    print(f"  Platform: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")

    # Initialize CPU history buffer
    cpu_history = deque(maxlen=args.seq_length)

    print(f"\nCollecting initial {args.seq_length} seconds of CPU usage...")
    for i in range(args.seq_length):
        cpu = collect_cpu_usage(1)
        cpu_history.append(cpu)
        print(f"  [{i+1}/{args.seq_length}] CPU: {cpu:.1f}%", end='\r')

    print("\n\nStarting real-time power prediction...")
    print(f"Monitoring for {args.duration} seconds\n")
    print(f"{'Time':>8s} | {'CPU (%)':>8s} | {'Predicted Power (W)':>20s}")
    print("-" * 50)

    start_time = time.time()
    sample_count = 0

    try:
        while time.time() - start_time < args.duration:
            # Collect current CPU usage
            cpu = collect_cpu_usage(1)
            cpu_history.append(cpu)

            # Predict power
            predicted_power = predictor.predict(list(cpu_history), vcpus)

            # Display
            elapsed = int(time.time() - start_time)
            print(f"{elapsed:8d} | {cpu:8.1f} | {predicted_power:20.2f}")

            sample_count += 1

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")

    # Summary
    print("\n" + "=" * 50)
    print(f"Monitoring complete: {sample_count} samples collected")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())