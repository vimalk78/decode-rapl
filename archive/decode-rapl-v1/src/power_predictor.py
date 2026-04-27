#!/usr/bin/env python3
"""
Power Predictor - DECODE-RAPL Model Inference

Shows predicted vs actual power consumption with color-coded accuracy.

Usage:
    # Live monitoring
    sudo python power_predictor.py --model checkpoints/tau8/best_model.pth --live --scroll

    # With custom frequency
    sudo python power_predictor.py --model checkpoints/tau8/best_model.pth --live --scroll --frequency 1.0 --interval 0.016

    # CSV replay
    python power_predictor.py --model checkpoints/tau8/best_model.pth --csv test_data.csv
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Import inference module
sys.path.append('.')
from src.inference import RAPLPredictor


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class CPUReader:
    """Read CPU usage from /proc/stat."""

    def __init__(self, multifeature: bool = False):
        """
        Args:
            multifeature: If True, read user%, system%, context_switches separately
        """
        self.multifeature = multifeature
        self.prev_stat = None
        self.prev_ctx_switches = None
        self.prev_time = None

    def read_cpu_usage(self):
        """
        Read current CPU usage.

        Returns:
            If multifeature=False: CPU usage percentage (float, 0-100)
            If multifeature=True: Dict with keys 'user_percent', 'system_percent', 'context_switches'
        """
        current_time = time.time()

        # Read /proc/stat
        with open("/proc/stat") as f:
            lines = f.readlines()

            # First line is aggregate CPU
            cpu_line = lines[0]
            parts = cpu_line.split()

            cpu_times = {
                "user": int(parts[1]),
                "nice": int(parts[2]),
                "system": int(parts[3]),
                "idle": int(parts[4]),
                "iowait": int(parts[5]),
                "irq": int(parts[6]),
                "softirq": int(parts[7]),
            }

            # Find context switches line (starts with "ctxt")
            ctx_switches = None
            for line in lines:
                if line.startswith("ctxt"):
                    ctx_switches = int(line.split()[1])
                    break

        if self.prev_stat and self.prev_time:
            # Calculate deltas
            user_delta = (cpu_times["user"] + cpu_times["nice"]) - (self.prev_stat["user"] + self.prev_stat["nice"])
            system_delta = (cpu_times["system"] + cpu_times["irq"] + cpu_times["softirq"]) - (self.prev_stat["system"] + self.prev_stat["irq"] + self.prev_stat["softirq"])
            total_delta = sum(cpu_times[k] - self.prev_stat[k] for k in cpu_times.keys())
            time_delta = current_time - self.prev_time

            if self.multifeature:
                # Return user%, system%, context_switches separately
                if total_delta > 0:
                    user_percent = 100.0 * user_delta / total_delta
                    system_percent = 100.0 * system_delta / total_delta
                else:
                    user_percent = 0.0
                    system_percent = 0.0

                # Calculate context switches per second
                if ctx_switches is not None and self.prev_ctx_switches is not None and time_delta > 0:
                    ctx_switches_per_sec = (ctx_switches - self.prev_ctx_switches) / time_delta
                else:
                    ctx_switches_per_sec = 0.0

                result = {
                    'user_percent': max(0.0, min(100.0, user_percent)),
                    'system_percent': max(0.0, min(100.0, system_percent)),
                    'context_switches': max(0.0, ctx_switches_per_sec)
                }
            else:
                # Return total CPU% (backward compatible)
                idle_delta = cpu_times["idle"] - self.prev_stat["idle"]
                if total_delta > 0:
                    cpu_usage = 100.0 * (total_delta - idle_delta) / total_delta
                else:
                    cpu_usage = 0.0
                result = cpu_usage
        else:
            # First reading
            if self.multifeature:
                result = {
                    'user_percent': 0.0,
                    'system_percent': 0.0,
                    'context_switches': 0.0
                }
            else:
                result = 0.0

        self.prev_stat = cpu_times
        self.prev_ctx_switches = ctx_switches
        self.prev_time = current_time

        return result


class RAPLReader:
    """Read RAPL power measurements."""

    def __init__(self):
        self.rapl_base = Path("/sys/class/powercap/intel-rapl")
        self.package_path = None
        self.prev_reading = None
        self.prev_time = None
        self._discover_package()

    def _discover_package(self):
        """Discover package-0 RAPL domain."""
        if not self.rapl_base.exists():
            return

        for domain_path in sorted(self.rapl_base.glob("intel-rapl:*")):
            name_file = domain_path / "name"
            if not name_file.exists():
                continue

            domain_name = name_file.read_text().strip()
            if domain_name == "package-0":
                self.package_path = domain_path / "energy_uj"
                break

    def is_available(self) -> bool:
        """Check if RAPL is available."""
        return self.package_path is not None

    def read_power(self) -> Optional[float]:
        """
        Read package power consumption in Watts.

        Returns:
            Power in Watts, or None if not available
        """
        if not self.package_path:
            return None

        current_time = time.time()

        try:
            current_reading = int(self.package_path.read_text().strip())
        except (OSError, ValueError):
            return None

        if self.prev_reading is not None and self.prev_time is not None:
            time_delta = current_time - self.prev_time

            if time_delta > 0:
                energy_delta = current_reading - self.prev_reading

                # Handle counter rollover
                if energy_delta < 0:
                    energy_delta += 2**32

                # Convert uJ to J, then to Watts
                power = (energy_delta / 1_000_000) / time_delta
            else:
                power = 0.0
        else:
            power = None  # First reading

        self.prev_reading = current_reading
        self.prev_time = current_time

        return power


class PowerPredictorApp:
    """Power prediction application."""

    def __init__(self, model_path: str):
        self.predictor = RAPLPredictor(
            checkpoint_path=model_path,
            vm_mode=False
        )
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"✓ Model loaded: {model_path}")
        print(f"  τ (tau): {self.predictor.config['embedding']['tau']} samples")
        print(f"  Window size: {self.predictor.config['preprocessing']['window_size']}")
        print(f"  Buffer size: {self.predictor.buffer_size}")

        # Check if model is multi-feature
        self.is_multifeature = self.predictor.is_multifeature
        if self.is_multifeature:
            print(f"  Mode: Multi-feature ({', '.join(self.predictor.feature_columns)})")
        else:
            print(f"  Mode: Single-feature (CPU%)")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n\nStopping...")
        self.running = False

    def live_monitor(self, interval: float = 0.016,
                    save_output: Optional[str] = None,
                    scroll_mode: bool = False,
                    prediction_frequency: Optional[float] = None):
        """
        Live monitoring mode with real-time predictions.

        Args:
            interval: Sampling interval for data collection (seconds)
            save_output: Path to save predictions CSV
            scroll_mode: Use scrolling output instead of clearing screen
            prediction_frequency: Frequency of predictions in Hz (e.g., 1.0 for once per second)
        """
        # Calculate prediction interval
        prediction_interval = None
        if prediction_frequency is not None:
            prediction_interval = 1.0 / prediction_frequency
            print(f"✓ Prediction frequency: {prediction_frequency:.2f} Hz (once every {prediction_interval:.3f}s)")
            print(f"✓ Data collection: {1.0/interval:.1f} Hz (continuous)")
        else:
            print(f"✓ Prediction and data collection: {1.0/interval:.1f} Hz")

        # Initialize readers
        cpu_reader = CPUReader(multifeature=self.is_multifeature)
        rapl_reader = RAPLReader()

        has_rapl = rapl_reader.is_available()
        if has_rapl:
            print(f"✓ RAPL available - will show actual power for comparison")
        else:
            print(f"⚠ RAPL not available - showing predictions only")

        # Prediction log
        predictions_log = []

        print(f"\n" + "="*80)
        mode_str = "Scrolling" if scroll_mode else "Updating"
        print(f"Live Power Monitoring - {mode_str} Mode (Press Ctrl+C to stop)")
        print("="*80 + "\n")

        print("Warming up buffer...")

        # Discard first reading (returns zeros)
        cpu_reader.read_cpu_usage()
        if has_rapl:
            rapl_reader.read_power()
        time.sleep(interval)

        # Fill predictor buffer
        required_samples = self.predictor.buffer_size + (self.predictor.embedder.d - 1) * self.predictor.embedder.tau
        for i in range(required_samples):
            cpu_usage = cpu_reader.read_cpu_usage()
            self.predictor.update_usage(cpu_usage)
            time.sleep(interval)
            print(f"\rBuffer: {i+1}/{required_samples}", end="", flush=True)

        print("\n\nStarting predictions...\n")

        sample_count = 0
        prediction_count = 0
        start_time = time.time()
        last_prediction_time = start_time

        # Track MAPE
        errors = []

        try:
            while self.running:
                # Read CPU usage
                cpu_usage = cpu_reader.read_cpu_usage()

                # Read RAPL if available
                if has_rapl:
                    actual_power = rapl_reader.read_power()
                else:
                    actual_power = None

                # Update predictor
                self.predictor.update_usage(cpu_usage)
                sample_count += 1

                # Determine if we should predict
                current_time = time.time()
                should_predict = False

                if prediction_interval is None:
                    should_predict = True
                else:
                    if (current_time - last_prediction_time) >= prediction_interval:
                        should_predict = True
                        last_prediction_time = current_time

                if should_predict:
                    # Predict
                    power_pred = self.predictor.predict()

                    if power_pred is not None:
                        elapsed = time.time() - start_time
                        prediction_count += 1

                        # Calculate error
                        if actual_power is not None and actual_power > 0:
                            error_pct = abs(power_pred - actual_power) / actual_power * 100
                            errors.append(error_pct)
                        else:
                            error_pct = None

                        # Display
                        if scroll_mode:
                            self._print_prediction_scroll(
                                prediction_count, elapsed, cpu_usage,
                                power_pred, actual_power, error_pct, errors
                            )
                        else:
                            print("\033[2J\033[H", end="")
                            self._print_prediction_full(
                                sample_count, prediction_count, elapsed,
                                cpu_usage, power_pred, actual_power, error_pct, errors
                            )

                        # Log prediction
                        if save_output:
                            log_entry = {
                                'timestamp': current_time,
                                'sample': prediction_count,
                                'cpu_usage': cpu_usage,
                                'predicted_power': power_pred,
                            }
                            if actual_power is not None:
                                log_entry['actual_power'] = actual_power
                                log_entry['error_pct'] = error_pct
                            predictions_log.append(log_entry)

                time.sleep(interval)

        except KeyboardInterrupt:
            pass

        elapsed = time.time() - start_time
        print(f"\n\n{Colors.BOLD}Monitoring stopped{Colors.RESET}")
        print(f"Total samples collected: {sample_count}")
        print(f"Total predictions made: {prediction_count}")
        print(f"Duration: {elapsed:.1f}s")

        if errors:
            mape = np.mean(errors)
            print(f"Average MAPE: {Colors.GREEN if mape < 5 else Colors.YELLOW if mape < 10 else Colors.RED}{mape:.2f}%{Colors.RESET}")

        # Save predictions
        if save_output and predictions_log:
            df = pd.DataFrame(predictions_log)
            df.to_csv(save_output, index=False)
            print(f"✓ Predictions saved to {save_output}")

    def _print_prediction_scroll(self, count: int, elapsed: float,
                                cpu_usage, power_pred: float,
                                actual_power: Optional[float],
                                error_pct: Optional[float],
                                errors: list):
        """Print compact one-line prediction."""
        timestamp = time.strftime("%H:%M:%S")
        rate = count / elapsed if elapsed > 0 else 0

        # Build output
        output = f"[{timestamp}] #{count:>6} ({rate:>5.1f}Hz) | "

        # Format CPU usage
        if isinstance(cpu_usage, dict):
            # Multi-feature: show user/system breakdown
            total_cpu = cpu_usage['user_percent'] + cpu_usage['system_percent']
            output += f"CPU: {total_cpu:>5.1f}% (U:{cpu_usage['user_percent']:>4.1f}% S:{cpu_usage['system_percent']:>4.1f}%) | "
        else:
            # Single feature
            output += f"CPU: {cpu_usage:>5.1f}% | "

        if actual_power is not None and error_pct is not None:
            # Color code based on error
            if error_pct < 5:
                color = Colors.GREEN
            elif error_pct < 10:
                color = Colors.YELLOW
            else:
                color = Colors.RED

            output += f"Pred: {color}{power_pred:>6.2f}W{Colors.RESET} | "
            output += f"Actual: {actual_power:>6.2f}W | "
            output += f"Err: {color}{error_pct:>+6.2f}%{Colors.RESET}"

            # Running MAPE
            if errors:
                mape = np.mean(errors)
                mape_color = Colors.GREEN if mape < 5 else Colors.YELLOW if mape < 10 else Colors.RED
                output += f" | MAPE: {mape_color}{mape:>5.2f}%{Colors.RESET}"
        else:
            output += f"Pred: {power_pred:>6.2f}W"

        print(output)

    def _print_prediction_full(self, sample_count: int, prediction_count: int,
                              elapsed: float, cpu_usage,
                              power_pred: float, actual_power: Optional[float],
                              error_pct: Optional[float], errors: list):
        """Print full screen display."""
        print(f"{Colors.BOLD}Live Power Prediction - DECODE-RAPL{Colors.RESET}")
        print(f"Samples: {sample_count} | Predictions: {prediction_count} | Elapsed: {elapsed:.1f}s")
        print(f"Sample rate: {sample_count/elapsed:.1f} Hz | Prediction rate: {prediction_count/elapsed:.1f} Hz")
        print(f"Timestamp: {time.time():.2f}\n")

        print(f"{'Metric':<20} {'Value':<15}")
        print("-" * 35)

        # Format CPU usage
        if isinstance(cpu_usage, dict):
            # Multi-feature: show breakdown
            total_cpu = cpu_usage['user_percent'] + cpu_usage['system_percent']
            print(f"{'CPU Usage (Total)':<20} {total_cpu:>13.1f}%")
            print(f"{'  User%':<20} {cpu_usage['user_percent']:>13.1f}%")
            print(f"{'  System%':<20} {cpu_usage['system_percent']:>13.1f}%")
            print(f"{'Context Switches/s':<20} {cpu_usage['context_switches']:>13.0f}")
        else:
            # Single feature
            print(f"{'CPU Usage':<20} {cpu_usage:>13.1f}%")
        print()

        if actual_power is not None and error_pct is not None:
            # Color code
            if error_pct < 5:
                color = Colors.GREEN
            elif error_pct < 10:
                color = Colors.YELLOW
            else:
                color = Colors.RED

            error_w = power_pred - actual_power

            print(f"{'Predicted Power':<20} {color}{power_pred:>13.2f}W{Colors.RESET}")
            print(f"{'Actual Power':<20} {actual_power:>13.2f}W")
            print(f"{'Error':<20} {color}{error_w:>+13.2f}W{Colors.RESET}")
            print(f"{'Error %':<20} {color}{error_pct:>+13.2f}%{Colors.RESET}")

            if errors:
                mape = np.mean(errors)
                mape_color = Colors.GREEN if mape < 5 else Colors.YELLOW if mape < 10 else Colors.RED
                print(f"{'Running MAPE':<20} {mape_color}{mape:>13.2f}%{Colors.RESET}")
        else:
            print(f"{'Predicted Power':<20} {power_pred:>13.2f}W")
            print(f"{'Actual Power':<20} {'N/A':>15}")

    def predict_from_csv(self, csv_file: str, save_output: Optional[str] = None):
        """
        Run predictions on CSV data.

        Args:
            csv_file: Path to CSV with cpu_usage and power columns
            save_output: Path to save predictions
        """
        print(f"\nLoading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples")

        # Check columns
        if 'cpu_usage' not in df.columns:
            print("Error: CSV must have 'cpu_usage' column")
            sys.exit(1)

        has_actual = 'power' in df.columns

        # Run predictions
        print("Running inference...")
        predictions = []
        actuals = []

        # Reset predictor
        self.predictor.reset()

        # Need to fill buffer first
        required_samples = self.predictor.buffer_size + (self.predictor.embedder.d - 1) * self.predictor.embedder.tau

        for i in range(len(df)):
            cpu_usage = df['cpu_usage'].iloc[i]
            self.predictor.update_usage(cpu_usage)

            # Start predicting after buffer is full
            if i >= required_samples:
                power_pred = self.predictor.predict()

                if power_pred is not None:
                    predictions.append(power_pred)

                    if has_actual:
                        actuals.append(df['power'].iloc[i])

            if (i + 1) % 10000 == 0:
                print(f"Processed {i+1}/{len(df)} samples...")

        predictions = np.array(predictions)
        print(f"\nGenerated {len(predictions)} predictions")

        # Calculate metrics
        if has_actual and len(actuals) > 0:
            actuals = np.array(actuals)

            mae = np.mean(np.abs(predictions - actuals))
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

            print(f"\n{'='*80}")
            print("Performance Metrics")
            print(f"{'='*80}")
            print(f"MAE:  {mae:.3f}W")
            print(f"MAPE: {mape:.2f}%")
            print(f"RMSE: {rmse:.3f}W")
            print(f"Mean Predicted: {np.mean(predictions):.2f}W")
            print(f"Mean Actual: {np.mean(actuals):.2f}W")

        # Save predictions
        if save_output:
            result_df = pd.DataFrame({
                'predicted_power': predictions
            })

            if has_actual and len(actuals) > 0:
                result_df['actual_power'] = actuals
                result_df['error'] = predictions - actuals
                result_df['error_pct'] = (predictions - actuals) / actuals * 100

            result_df.to_csv(save_output, index=False)
            print(f"\n✓ Predictions saved to {save_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Power prediction inference with DECODE-RAPL model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live monitoring with real-time predictions:
    sudo %(prog)s --model checkpoints/tau8/best_model.pth --live --scroll
    sudo %(prog)s --model checkpoints/tau8/best_model.pth --live --scroll --frequency 1.0 --interval 0.016
    sudo %(prog)s --model checkpoints/tau8/best_model.pth --live --save predictions.csv

  Run predictions on CSV file:
    %(prog)s --model checkpoints/tau8/best_model.pth --csv data/real_data_2hr_clean.csv
    %(prog)s --model checkpoints/tau8/best_model.pth --csv test_data.csv --save predictions.csv
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--live', action='store_true',
                       help='Live monitoring mode')
    parser.add_argument('--csv', type=str,
                       help='CSV file with cpu_usage (and optionally power) for prediction')
    parser.add_argument('--scroll', action='store_true',
                       help='Scrolling output mode (live mode only)')
    parser.add_argument('--interval', type=float, default=0.1,
                       help='Sampling interval for live mode (seconds, default: 0.1)')
    parser.add_argument('--frequency', type=float,
                       help='Prediction frequency in Hz (e.g., 1.0 for once per second)')
    parser.add_argument('--save', type=str,
                       help='Save predictions to CSV file')

    args = parser.parse_args()

    # Create predictor
    app = PowerPredictorApp(args.model)

    # Live monitoring mode
    if args.live:
        app.live_monitor(
            interval=args.interval,
            save_output=args.save,
            scroll_mode=args.scroll,
            prediction_frequency=args.frequency
        )

    # CSV mode
    elif args.csv:
        app.predict_from_csv(args.csv, save_output=args.save)

    else:
        print("Error: Must specify either --live or --csv")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
