#!/usr/bin/env python3
"""
Run Go data collector, then run inference on collected data

This script:
1. Runs the Go collector (my_data_collector) to collect CPU metrics + power
2. User runs their workload manually in another terminal
3. Waits for collection duration (or Ctrl+C) and CSV output
4. Runs power_predictor.py in CSV mode on the collected data
5. Produces final prediction CSV compatible with plot_predictions.py

Usage:
    # Run collector (user starts workload manually):
    python scripts/run_workload_and_predict.py --model models/tau8/tau8_multifeature_model.pth \\
        --duration 60 --output predictions.csv

    # Run inference on pre-collected data:
    python scripts/run_workload_and_predict.py --model models/tau8/tau8_multifeature_model.pth \\
        --collector-csv data/my_workload.csv --output predictions.csv
"""

import argparse
import subprocess
import sys
import time
import signal
from pathlib import Path


def run_collector(duration: int, output_csv: str) -> bool:
    """
    Run the Go data collector

    Args:
        duration: Collection duration in seconds
        output_csv: Path to save CSV output

    Returns:
        True if collection succeeded, False otherwise
    """
    collector_path = Path("collector/my_data_collector")

    if not collector_path.exists():
        print(f"ERROR: Go collector not found at {collector_path}")
        print("Please build it first: cd collector && go build my_data_collector.go")
        return False

    # Start collector
    collector_cmd = [
        "sudo",
        str(collector_path),
        "--outfile", output_csv
    ]

    print(f"\n{'='*80}")
    print("Running Go data collector...")
    print(f"{'='*80}")
    print(f"Duration: {duration}s")
    print(f"Output: {output_csv}")
    print(f"Command: {' '.join(collector_cmd)}")
    print()
    print("RUN YOUR WORKLOAD NOW in another terminal!")
    print("Press Ctrl+C to stop collection early")
    print()

    # Start collector
    try:
        collector_proc = subprocess.Popen(collector_cmd)
        print(f"✓ Collector started (PID: {collector_proc.pid})")
        print(f"\nCollecting data for {duration} seconds...\n")

        # Wait for duration
        time.sleep(duration)

        # Send SIGINT to collector to trigger graceful shutdown
        print("\nStopping collector...")
        collector_proc.send_signal(signal.SIGINT)

        # Wait for collector to finish writing CSV
        collector_proc.wait(timeout=10)
        print(f"✓ Collection completed successfully")

        return True

    except subprocess.TimeoutExpired:
        print(f"\n✗ Collector did not exit cleanly")
        collector_proc.kill()
        return False
    except KeyboardInterrupt:
        print(f"\n\nStopping collection early (Ctrl+C)...")
        collector_proc.send_signal(signal.SIGINT)
        collector_proc.wait(timeout=10)
        return True
    except Exception as e:
        print(f"\n✗ Collection failed: {e}")
        try:
            collector_proc.kill()
        except:
            pass
        return False


def run_inference(model_path: str, csv_file: str, output_csv: str) -> bool:
    """
    Run power_predictor.py in CSV mode on collected data

    Args:
        model_path: Path to trained model checkpoint
        csv_file: Input CSV from Go collector
        output_csv: Output CSV with predictions

    Returns:
        True if inference succeeded, False otherwise
    """
    predictor_script = Path("src/power_predictor.py")

    if not predictor_script.exists():
        print(f"ERROR: Predictor script not found at {predictor_script}")
        return False

    cmd = [
        "python",
        str(predictor_script),
        "--model", model_path,
        "--csv", csv_file,
        "--save", output_csv
    ]

    print(f"\n{'='*80}")
    print("Running inference on collected data...")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Input CSV: {csv_file}")
    print(f"Output: {output_csv}")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Inference completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Inference failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Go collector and predict power consumption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Collect data and predict (run workload in another terminal):
    python %(prog)s --model models/tau8/tau8_multifeature_model.pth \\
        --duration 60 --output predictions.csv

  Run inference on pre-collected data:
    python %(prog)s --model models/tau8/tau8_multifeature_model.pth \\
        --collector-csv data/workload_2025.csv --output predictions.csv
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--collector-csv', type=str,
                       help='Use existing CSV from Go collector (skip collection phase)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Collection duration in seconds (default: 60)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file for predictions (default: predictions.csv)')
    parser.add_argument('--keep-collector-csv', action='store_true',
                       help='Keep intermediate collector CSV file')

    args = parser.parse_args()

    # Determine collector CSV path
    if args.collector_csv:
        collector_csv = args.collector_csv
        skip_collection = True
        print(f"Using existing collector CSV: {collector_csv}")
    else:
        # Generate temporary CSV filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        collector_csv = f"temp_collector_{timestamp}.csv"
        skip_collection = False

    # Step 1: Run collector (if needed)
    if not skip_collection:
        success = run_collector(
            duration=args.duration,
            output_csv=collector_csv
        )

        if not success:
            print("\n✗ Collection phase failed")
            sys.exit(1)

    # Verify collector CSV exists
    if not Path(collector_csv).exists():
        print(f"\n✗ Collector CSV not found: {collector_csv}")
        sys.exit(1)

    # Step 2: Run inference
    success = run_inference(
        model_path=args.model,
        csv_file=collector_csv,
        output_csv=args.output
    )

    if not success:
        print("\n✗ Inference phase failed")
        sys.exit(1)

    # Clean up temporary collector CSV if requested
    if not skip_collection and not args.keep_collector_csv:
        try:
            Path(collector_csv).unlink()
            print(f"\n✓ Cleaned up temporary file: {collector_csv}")
        except Exception as e:
            print(f"\nWarning: Could not delete temporary file {collector_csv}: {e}")

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print(f"Predictions saved to: {args.output}")
    print(f"\nTo visualize results, run:")
    print(f"  python scripts/plot_predictions.py {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
