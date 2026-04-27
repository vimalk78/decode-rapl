#!/usr/bin/env python3
"""
Power Predictor - MS-TCN Model Inference

Shows predicted vs actual power consumption with color-coded accuracy.

Usage:
    # Live monitoring
    python power_predictor.py --model model.pth --live

    # CSV replay
    python power_predictor.py --model model.pth --csv test_data.csv
    python power_predictor.py --model model.pth --csv data.csv --save predictions.csv
"""

import argparse
import os
import pickle
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Import model architecture from train_model.py
try:
    from train_model import MSTCN
except ImportError:
    print("Error: train_model.py not found. Make sure it's in the same directory.")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class SystemMetricsReader:
    """Read live system metrics for prediction."""

    def __init__(self):
        self.prev_stat = None
        self.prev_vmstat = None
        self.prev_time = None

        # System configuration (cached)
        self.num_cores = os.cpu_count()
        self.memory_total_gb = None
        self.swap_total_gb = None
        self._detect_system_config()

    def read_metrics(self) -> Dict[str, float]:
        """Read current system metrics."""
        current_time = time.time()
        metrics = {}

        # CPU metrics
        cpu_times, ctxt, intr = self._read_proc_stat_full()

        if self.prev_stat and self.prev_time:
            time_delta = current_time - self.prev_time
            total_delta = sum(cpu_times[k] - self.prev_stat[k] for k in cpu_times.keys())

            if total_delta > 0:
                metrics["cpu_user_percent"] = 100.0 * (cpu_times["user"] - self.prev_stat["user"]) / total_delta
                metrics["cpu_system_percent"] = 100.0 * (cpu_times["system"] - self.prev_stat["system"]) / total_delta
                metrics["cpu_idle_percent"] = 100.0 * (cpu_times["idle"] - self.prev_stat["idle"]) / total_delta
                metrics["cpu_iowait_percent"] = 100.0 * (cpu_times["iowait"] - self.prev_stat["iowait"]) / total_delta
                metrics["cpu_irq_percent"] = 100.0 * (cpu_times["irq"] - self.prev_stat["irq"]) / total_delta
                metrics["cpu_softirq_percent"] = 100.0 * (cpu_times["softirq"] - self.prev_stat["softirq"]) / total_delta

            # CPU time in seconds/second (absolute work metrics)
            # /proc/stat uses jiffies (USER_HZ = 100 typically, so 1 jiffy = 0.01s)
            USER_HZ = 100.0  # Clock ticks per second
            if time_delta > 0:
                metrics["cpu_user_sec"] = (cpu_times["user"] - self.prev_stat["user"]) / USER_HZ / time_delta
                metrics["cpu_system_sec"] = (cpu_times["system"] - self.prev_stat["system"]) / USER_HZ / time_delta
                metrics["cpu_idle_sec"] = (cpu_times["idle"] - self.prev_stat["idle"]) / USER_HZ / time_delta
                metrics["cpu_iowait_sec"] = (cpu_times["iowait"] - self.prev_stat["iowait"]) / USER_HZ / time_delta
                metrics["cpu_irq_sec"] = (cpu_times["irq"] - self.prev_stat["irq"]) / USER_HZ / time_delta
                metrics["cpu_softirq_sec"] = (cpu_times["softirq"] - self.prev_stat["softirq"]) / USER_HZ / time_delta

                metrics["context_switches_sec"] = (ctxt - self.prev_stat.get("ctxt", ctxt)) / time_delta
                metrics["interrupts_sec"] = (intr - self.prev_stat.get("intr", intr)) / time_delta
        else:
            # First reading - use zeros
            for key in ["cpu_user_percent", "cpu_system_percent", "cpu_idle_percent",
                       "cpu_iowait_percent", "cpu_irq_percent", "cpu_softirq_percent",
                       "cpu_user_sec", "cpu_system_sec", "cpu_idle_sec",
                       "cpu_iowait_sec", "cpu_irq_sec", "cpu_softirq_sec",
                       "context_switches_sec", "interrupts_sec"]:
                metrics[key] = 0.0

        cpu_times["ctxt"] = ctxt
        cpu_times["intr"] = intr
        self.prev_stat = cpu_times

        # Memory metrics
        meminfo = self._read_meminfo()
        metrics.update(meminfo)
        metrics["memory_used_mb"] = meminfo["memory_total_mb"] - meminfo["memory_available_mb"]
        metrics["swap_used_mb"] = meminfo["swap_total_mb"] - meminfo["swap_free_mb"]

        # VM statistics
        vmstat = self._read_vmstat()
        if self.prev_vmstat and self.prev_time:
            time_delta = current_time - self.prev_time
            pgfault_delta = vmstat.get("pgfault", 0) - self.prev_vmstat.get("pgfault", 0)
            metrics["page_faults_sec"] = pgfault_delta / time_delta if time_delta > 0 else 0.0
        else:
            metrics["page_faults_sec"] = 0.0

        self.prev_vmstat = vmstat

        # Load average
        loadavg = self._read_loadavg()
        metrics.update(loadavg)
        metrics["blocked_processes"] = metrics["total_processes"] - metrics["running_processes"]

        self.prev_time = current_time

        return metrics

    def _read_proc_stat_full(self) -> Tuple[Dict[str, int], int, int]:
        """Read /proc/stat for CPU times, context switches, and interrupts."""
        cpu_times = {}
        ctxt = 0
        intr = 0

        with open("/proc/stat") as f:
            for line in f:
                parts = line.split()
                if parts[0] == "cpu":
                    cpu_times = {
                        "user": int(parts[1]),
                        "nice": int(parts[2]),
                        "system": int(parts[3]),
                        "idle": int(parts[4]),
                        "iowait": int(parts[5]),
                        "irq": int(parts[6]),
                        "softirq": int(parts[7]),
                        "steal": int(parts[8]) if len(parts) > 8 else 0,
                    }
                elif parts[0] == "ctxt":
                    ctxt = int(parts[1])
                elif parts[0] == "intr":
                    intr = int(parts[1])

        return cpu_times, ctxt, intr

    def _read_meminfo(self) -> Dict[str, float]:
        """Read /proc/meminfo and return values in MB."""
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1])  # Value in kB
                    meminfo[key] = value / 1024  # Convert to MB

        return {
            "memory_total_mb": meminfo.get("MemTotal", 0),
            "memory_free_mb": meminfo.get("MemFree", 0),
            "memory_available_mb": meminfo.get("MemAvailable", 0),
            "memory_buffers_mb": meminfo.get("Buffers", 0),
            "memory_cached_mb": meminfo.get("Cached", 0),
            "swap_total_mb": meminfo.get("SwapTotal", 0),
            "swap_free_mb": meminfo.get("SwapFree", 0),
        }

    def _read_vmstat(self) -> Dict[str, int]:
        """Read /proc/vmstat for page fault statistics."""
        vmstat = {}
        with open("/proc/vmstat") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    vmstat[parts[0]] = int(parts[1])
        return vmstat

    def _read_loadavg(self) -> Dict[str, float]:
        """Read /proc/loadavg."""
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            running, total = parts[3].split("/")
            return {
                "load_1min": float(parts[0]),
                "load_5min": float(parts[1]),
                "load_15min": float(parts[2]),
                "running_processes": int(running),
                "total_processes": int(total),
            }

    def _detect_system_config(self):
        """Detect system configuration (memory, swap sizes)."""
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        memory_kb = int(line.split()[1])
                        self.memory_total_gb = memory_kb / (1024 * 1024)
                    elif line.startswith('SwapTotal:'):
                        swap_kb = int(line.split()[1])
                        self.swap_total_gb = swap_kb / (1024 * 1024)
        except FileNotFoundError:
            # Defaults if can't read
            self.memory_total_gb = 95.0
            self.swap_total_gb = 8.0

    def normalize_features(self, raw_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize raw metrics to match training features.

        For CPU time model: Uses absolute CPU time (seconds/second) and memory ratios.
        This ensures VM portability as CPU time naturally scales with core count.

        Args:
            raw_metrics: Dictionary of raw metrics from read_metrics()

        Returns:
            Dictionary with normalized features matching training FEATURE_COLUMNS
        """
        normalized = {}

        # CPU time (seconds/second) - naturally scales with core count
        # 4-core at 50% = 2 sec/sec, 20-core at 50% = 10 sec/sec
        for key in ['cpu_user_sec', 'cpu_system_sec', 'cpu_idle_sec',
                    'cpu_iowait_sec', 'cpu_irq_sec', 'cpu_softirq_sec']:
            normalized[key] = raw_metrics.get(key, 0.0)

        # System activity (absolute counts per second)
        normalized['interrupts_sec'] = raw_metrics.get('interrupts_sec', 0.0)
        normalized['context_switches_sec'] = raw_metrics.get('context_switches_sec', 0.0)
        normalized['page_faults_sec'] = raw_metrics.get('page_faults_sec', 0.0)
        normalized['running_processes'] = raw_metrics.get('running_processes', 0.0)

        # Memory ratios (0-1, scale-independent)
        memory_total_mb = self.memory_total_gb * 1024
        normalized['memory_used_ratio'] = raw_metrics.get('memory_used_mb', 0.0) / memory_total_mb
        normalized['memory_cached_ratio'] = raw_metrics.get('memory_cached_mb', 0.0) / memory_total_mb
        normalized['memory_free_ratio'] = raw_metrics.get('memory_free_mb', memory_total_mb) / memory_total_mb

        # Swap ratio
        swap_total_mb = self.swap_total_gb * 1024
        normalized['swap_used_ratio'] = raw_metrics.get('swap_used_mb', 0.0) / swap_total_mb

        return normalized


class RAPLReader:
    """Read RAPL power measurements."""

    def __init__(self):
        self.rapl_base = Path("/sys/class/powercap/intel-rapl")
        self.domains = {}
        self.prev_readings = {}
        self.prev_time = None
        self._discover_domains()

    def _discover_domains(self):
        """Discover available RAPL domains."""
        if not self.rapl_base.exists():
            return

        for domain_path in sorted(self.rapl_base.glob("intel-rapl:*")):
            name_file = domain_path / "name"
            if not name_file.exists():
                continue

            domain_name = name_file.read_text().strip()

            if domain_name == "package-0":
                self.domains["package"] = domain_path / "energy_uj"

                for subdomain_path in sorted(domain_path.glob("intel-rapl:*:*")):
                    sub_name_file = subdomain_path / "name"
                    if sub_name_file.exists():
                        sub_name = sub_name_file.read_text().strip()
                        if sub_name == "core":
                            self.domains["core"] = subdomain_path / "energy_uj"
                        elif sub_name == "dram":
                            self.domains["dram"] = subdomain_path / "energy_uj"

    def is_available(self) -> bool:
        """Check if RAPL is available."""
        return len(self.domains) > 0

    def read_power(self) -> Dict[str, float]:
        """Read power consumption in Watts."""
        if not self.domains:
            return {}

        current_time = time.time()
        current_readings = {}

        for domain, path in self.domains.items():
            try:
                current_readings[domain] = int(path.read_text().strip())
            except (OSError, ValueError):
                current_readings[domain] = 0

        power = {}
        if self.prev_readings and self.prev_time:
            time_delta = current_time - self.prev_time
            if time_delta > 0:
                for domain, energy in current_readings.items():
                    if domain in self.prev_readings:
                        energy_delta = energy - self.prev_readings[domain]
                        if energy_delta < 0:
                            energy_delta += 2**32
                        power[f"rapl_{domain}_power"] = (energy_delta / 1_000_000) / time_delta
        else:
            for domain in current_readings.keys():
                power[f"rapl_{domain}_power"] = 0.0

        self.prev_readings = current_readings
        self.prev_time = current_time

        return power


class PowerPredictor:
    """Power prediction inference engine."""

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor_params = None
        self.feature_scaler = None
        self.target_scaler = None
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._load_model(model_path)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n\nStopping...")
        self.running = False

    def _load_model(self, model_path: str):
        """Load trained model and preprocessing parameters."""

        if not Path(model_path).exists():
            print(f"Error: Model file '{model_path}' not found")
            sys.exit(1)

        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'preprocessor' not in checkpoint:
            print("Error: Model file missing preprocessor parameters")
            sys.exit(1)

        self.preprocessor_params = checkpoint['preprocessor']

        # Setup scalers
        # Feature scaler can be StandardScaler or ColumnTransformer (pickled)
        if 'feature_scaler' in self.preprocessor_params:
            # New format: pickled scaler object
            self.feature_scaler = pickle.loads(self.preprocessor_params['feature_scaler'])
        else:
            # Old format: mean/scale attributes (for backward compatibility)
            self.feature_scaler = StandardScaler()
            self.feature_scaler.mean_ = self.preprocessor_params['feature_scaler_mean']
            self.feature_scaler.scale_ = self.preprocessor_params['feature_scaler_scale']

        self.target_scaler = StandardScaler()
        self.target_scaler.mean_ = self.preprocessor_params['target_scaler_mean']
        self.target_scaler.scale_ = self.preprocessor_params['target_scaler_scale']

        # Create model - detect architecture from checkpoint
        num_features = len(self.preprocessor_params['feature_columns'])
        num_targets = len(self.preprocessor_params['target_columns'])

        # Detect hidden_dim from temporal blocks (reliable for any hidden_dim value)
        hidden_dim = checkpoint['model_state_dict']['temporal_blocks.0.conv1.weight'].shape[0]

        # Find compatible num_heads for the hidden_dim
        preferred_heads = [8, 7, 6, 5, 4, 3, 2, 1]
        num_heads = next((h for h in preferred_heads if hidden_dim % h == 0), 1)

        self.model = MSTCN(num_features=num_features, num_targets=num_targets,
                          hidden_dim=hidden_dim, num_heads=num_heads)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded successfully")
        print(f"  Features: {num_features}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Targets: {self.preprocessor_params['target_columns']}")
        print(f"  Sequence length: {self.preprocessor_params['sequence_length']}")

    def live_monitor(self, interval: float = 0.016, save_output: Optional[str] = None,
                     scroll_mode: bool = False, prediction_frequency: Optional[float] = None):
        """Live monitoring mode with real-time predictions.

        Args:
            interval: Sampling interval for data collection (seconds)
            save_output: Path to save predictions CSV
            scroll_mode: Use scrolling output instead of clearing screen
            prediction_frequency: Frequency of predictions in Hz (e.g., 1.0 for once per second).
                                 If None, predicts every sample. Data collection always at full rate.
        """

        seq_len = self.preprocessor_params['sequence_length']
        feature_cols = self.preprocessor_params['feature_columns']
        target_cols = self.preprocessor_params['target_columns']

        # Calculate prediction interval
        prediction_interval = None
        if prediction_frequency is not None:
            prediction_interval = 1.0 / prediction_frequency
            print(f"✓ Prediction frequency: {prediction_frequency:.2f} Hz (once every {prediction_interval:.3f}s)")
            print(f"✓ Data collection: {1.0/interval:.1f} Hz (continuous)")
        else:
            print(f"✓ Prediction and data collection: {1.0/interval:.1f} Hz")

        # Initialize readers
        metrics_reader = SystemMetricsReader()
        rapl_reader = RAPLReader()

        has_rapl = rapl_reader.is_available()
        if has_rapl:
            print(f"✓ RAPL available - will show actual power for comparison")
        else:
            print(f"⚠ RAPL not available - showing predictions only")

        # Sequence buffer
        sequence_buffer = deque(maxlen=seq_len)

        # Prediction log
        predictions_log = []

        # Raw data log for debugging
        raw_data_log = []

        print(f"\n" + "="*80)
        mode_str = "Scrolling" if scroll_mode else "Updating"
        print(f"Live Power Monitoring - {mode_str} Mode (Press Ctrl+C to stop)")
        print("="*80 + "\n")

        print("Warming up sequence buffer...")

        # Discard first reading (it returns zeros for delta-based metrics)
        metrics_reader.read_metrics()
        time.sleep(interval)

        # Fill buffer with valid readings
        for i in range(seq_len):
            metrics = metrics_reader.read_metrics()

            # Normalize raw metrics to match training features
            normalized_metrics = metrics_reader.normalize_features(metrics)

            # Extract normalized features in correct order
            features = [normalized_metrics.get(col, 0.0) for col in feature_cols]
            sequence_buffer.append(features)

            time.sleep(interval)
            print(f"\rBuffer: {i+1}/{seq_len}", end="", flush=True)

        print("\n\nStarting predictions...\n")

        sample_count = 0
        prediction_count = 0
        start_time = time.time()
        last_prediction_time = start_time

        try:
            while self.running:
                # Read current metrics
                metrics = metrics_reader.read_metrics()

                # Read RAPL if available
                if has_rapl:
                    rapl_power = rapl_reader.read_power()
                    actual_values = [rapl_power.get(col, 0.0) for col in target_cols]
                else:
                    actual_values = None

                # Normalize raw metrics to match training features
                normalized_metrics = metrics_reader.normalize_features(metrics)

                # Extract normalized features in correct order
                features = [normalized_metrics.get(col, 0.0) for col in feature_cols]
                sequence_buffer.append(features)

                # Log raw data for debugging (first 100 samples)
                if len(raw_data_log) < 100:
                    row = {'timestamp': time.time()}
                    if has_rapl:
                        for i, col in enumerate(target_cols):
                            row[col] = actual_values[i] if actual_values else 0.0
                    for i, col in enumerate(feature_cols):
                        row[col] = features[i]
                    raw_data_log.append(row)

                sample_count += 1

                # Determine if we should run inference
                current_time = time.time()
                should_predict = False
                if prediction_interval is None:
                    # Predict every sample (original behavior)
                    should_predict = True
                else:
                    # Predict based on frequency
                    if (current_time - last_prediction_time) >= prediction_interval:
                        should_predict = True
                        last_prediction_time = current_time

                if should_predict:
                    # Create sequence
                    sequence = np.array(list(sequence_buffer))

                    # Normalize
                    sequence_normalized = self.feature_scaler.transform(sequence)

                    # Predict
                    with torch.no_grad():
                        seq_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
                        prediction_normalized = self.model(seq_tensor)

                        prediction = self.target_scaler.inverse_transform(
                            prediction_normalized.cpu().numpy()
                        )[0]

                        # DEBUG: Print raw normalized output and input features (first few predictions)
                        if prediction_count < 3:
                            print(f"\n[DEBUG] Prediction #{prediction_count + 1}")
                            print(f"  Raw features (last 3 samples):")
                            for j in range(-3, 0):
                                sample = sequence[j]
                                # Use new CPU time-based features
                                cpu_user_idx = feature_cols.index('cpu_user_sec') if 'cpu_user_sec' in feature_cols else feature_cols.index('cpu_user_percent')
                                interrupts_idx = feature_cols.index('interrupts_sec') if 'interrupts_sec' in feature_cols else feature_cols.index('interrupts_per_core')
                                running_idx = feature_cols.index('running_processes') if 'running_processes' in feature_cols else feature_cols.index('running_processes_per_core')

                                cpu_label = 'cpu_user_sec' if 'cpu_user_sec' in feature_cols else 'cpu_user%'
                                print(f"    [{j}] interrupts={sample[interrupts_idx]:.1f}, "
                                      f"running_procs={sample[running_idx]:.2f}, "
                                      f"{cpu_label}={sample[cpu_user_idx]:.2f}")
                            print(f"  Normalized output: {prediction_normalized.cpu().numpy()[0]}")
                            print(f"  Denormalized: {prediction[0]:.2f}W\n")

                    elapsed = time.time() - start_time
                    prediction_count += 1

                    if scroll_mode:
                        # Scrolling mode - compact one-line output
                        self._print_prediction_scroll(prediction_count, elapsed, prediction, actual_values, target_cols)
                    else:
                        # Clear screen mode - full display
                        print("\033[2J\033[H", end="")
                        print(f"{Colors.BOLD}Live Power Prediction{Colors.RESET}")
                        print(f"Samples: {sample_count} | Predictions: {prediction_count} | Elapsed: {elapsed:.1f}s")
                        print(f"Sample rate: {sample_count/elapsed:.1f} Hz | Prediction rate: {prediction_count/elapsed:.1f} Hz")
                        print(f"Timestamp: {time.time():.2f}\n")
                        self._print_prediction_table(prediction, actual_values, target_cols)

                    # Log prediction
                    if save_output:
                        log_entry = {
                            'timestamp': time.time(),
                            'sample': prediction_count
                        }
                        for i, col in enumerate(target_cols):
                            log_entry[f'predicted_{col}'] = prediction[i]
                            if actual_values:
                                log_entry[f'actual_{col}'] = actual_values[i]
                                # Add percentage error
                                actual_val = actual_values[i]
                                error_pct = ((prediction[i] - actual_val) / actual_val * 100) if actual_val != 0 else 0
                                log_entry[f'{col}_pct_error'] = error_pct
                        predictions_log.append(log_entry)

                time.sleep(interval)

        except KeyboardInterrupt:
            pass

        elapsed = time.time() - start_time
        print(f"\n\n{Colors.BOLD}Monitoring stopped{Colors.RESET}")
        print(f"Total samples collected: {sample_count}")
        print(f"Total predictions made: {prediction_count}")
        print(f"Duration: {elapsed:.1f}s")

        # Save predictions if requested
        if save_output and predictions_log:
            df = pd.DataFrame(predictions_log)
            df.to_csv(save_output, index=False)
            print(f"✓ Predictions saved to {save_output}")

        # Save raw data log for debugging
        if raw_data_log:
            debug_file = "debug_live_data.csv"
            df_debug = pd.DataFrame(raw_data_log)
            df_debug.to_csv(debug_file, index=False)
            print(f"✓ Raw live data saved to {debug_file} for debugging")

    def predict_from_csv(self, csv_file: str, show_realtime: bool = False,
                        save_output: str = None) -> Dict:
        """Run predictions on CSV data and compare to actual values."""

        # Load CSV
        print(f"\nLoading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples")

        # Extract features
        feature_cols = self.preprocessor_params['feature_columns']
        target_cols = self.preprocessor_params['target_columns']
        seq_len = self.preprocessor_params['sequence_length']

        # Check if all features are present
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"Error: Missing features in CSV: {missing_features}")
            sys.exit(1)

        # Check if targets are present
        missing_targets = [col for col in target_cols if col not in df.columns]
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
            print("Will only show predictions, not comparisons")
            has_actual = False
        else:
            has_actual = True

        # Prepare data
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X_normalized = self.feature_scaler.transform(X)

        if has_actual:
            y_actual = df[target_cols].values
            y_actual = np.nan_to_num(y_actual, nan=0.0)

        # Create sequences
        sequences = []
        actual_targets = []
        timestamps = []

        for i in range(len(X_normalized) - seq_len):
            seq = X_normalized[i:i + seq_len]
            sequences.append(seq)

            if has_actual:
                actual_targets.append(y_actual[i + seq_len - 1])

            if 'timestamp' in df.columns:
                timestamps.append(df['timestamp'].iloc[i + seq_len - 1])

        sequences = np.array(sequences)
        if has_actual:
            actual_targets = np.array(actual_targets)

        print(f"Created {len(sequences)} sequences for prediction\n")

        # Run predictions
        print("Running inference...")
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(sequences), 32):
                batch = sequences[i:i + 32]
                batch_tensor = torch.FloatTensor(batch).to(self.device)

                predictions = self.model(batch_tensor)
                all_predictions.append(predictions.cpu().numpy())

        all_predictions = np.vstack(all_predictions)

        # Denormalize predictions
        predictions_denorm = self.target_scaler.inverse_transform(all_predictions)

        # Show results
        if show_realtime:
            self._show_realtime_predictions(predictions_denorm, actual_targets if has_actual else None,
                                           target_cols, timestamps)
        else:
            self._show_summary(predictions_denorm, actual_targets if has_actual else None, target_cols)

        # Save predictions if requested
        if save_output:
            self._save_predictions(predictions_denorm, actual_targets if has_actual else None,
                                  target_cols, timestamps, save_output)

        # Return metrics
        if has_actual:
            metrics = self._calculate_metrics(predictions_denorm, actual_targets, target_cols)
            return metrics
        else:
            return {}

    def _show_realtime_predictions(self, predictions: np.ndarray, actuals: np.ndarray,
                                   target_cols: List[str], timestamps: List[float]):
        """Show predictions in real-time with delay."""

        print("\n" + "="*80)
        print("Real-time Prediction Display (Press Ctrl+C to stop)")
        print("="*80 + "\n")

        try:
            for i in range(len(predictions)):
                print("\033[2J\033[H", end="")

                print(f"Sample {i+1}/{len(predictions)}")
                if timestamps:
                    print(f"Timestamp: {timestamps[i]:.2f}\n")

                self._print_prediction_table(predictions[i],
                                            actuals[i] if actuals is not None else None,
                                            target_cols)

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

    def _show_summary(self, predictions: np.ndarray, actuals: np.ndarray, target_cols: List[str]):
        """Show summary statistics."""

        print("\n" + "="*80)
        print("Prediction Summary")
        print("="*80 + "\n")

        if actuals is not None:
            print("Sample Predictions (first 5):\n")
            for i in range(min(5, len(predictions))):
                print(f"Sample {i+1}:")
                self._print_prediction_table(predictions[i], actuals[i], target_cols)
                print()

            print("\n" + "="*80)
            print("Overall Performance Metrics")
            print("="*80 + "\n")

            metrics = self._calculate_metrics(predictions, actuals, target_cols)
            self._print_metrics_table(metrics, target_cols)

        else:
            print("Predictions (first 10):\n")
            for i in range(min(10, len(predictions))):
                print(f"Sample {i+1}: ", end="")
                for j, col in enumerate(target_cols):
                    print(f"{col}: {predictions[i, j]:.2f}W", end="  ")
                print()

    def _print_prediction_table(self, prediction: np.ndarray, actual: Optional[np.ndarray],
                                target_cols: List[str]):
        """Print a formatted table for one prediction."""

        if actual is not None:
            print(f"{'Domain':<15} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'Error %':<10}")
            print("-" * 61)
        else:
            print(f"{'Domain':<15} {'Predicted':<12}")
            print("-" * 27)

        for i, col in enumerate(target_cols):
            domain = col.replace('rapl_', '').replace('_power', '').title()
            pred_val = prediction[i]

            if actual is not None:
                actual_val = actual[i]
                error = pred_val - actual_val
                error_pct = (error / actual_val * 100) if actual_val != 0 else 0

                if abs(error_pct) < 5:
                    color = Colors.GREEN
                elif abs(error_pct) < 10:
                    color = Colors.YELLOW
                else:
                    color = Colors.RED

                print(f"{domain:<15} {color}{pred_val:>10.2f}W{Colors.RESET}  "
                      f"{actual_val:>10.2f}W  "
                      f"{color}{error:>+10.2f}W{Colors.RESET}  "
                      f"{color}{error_pct:>+8.1f}%{Colors.RESET}")
            else:
                print(f"{domain:<15} {pred_val:>10.2f}W")

    def _print_prediction_scroll(self, sample_count: int, elapsed: float,
                                 prediction: np.ndarray, actual: Optional[np.ndarray],
                                 target_cols: List[str]):
        """Print compact one-line prediction for scrolling mode."""

        timestamp = time.strftime("%H:%M:%S")
        rate = sample_count / elapsed if elapsed > 0 else 0

        # Build output line
        output = f"[{timestamp}] #{sample_count:>6} ({rate:>5.1f}Hz) | "

        for i, col in enumerate(target_cols):
            domain = col.replace('rapl_', '').replace('_power', '')
            pred_val = prediction[i]

            if actual is not None:
                actual_val = actual[i]
                error = pred_val - actual_val
                error_pct = (error / actual_val * 100) if actual_val != 0 else 0

                # Color code
                if abs(error_pct) < 5:
                    color = Colors.GREEN
                elif abs(error_pct) < 10:
                    color = Colors.YELLOW
                else:
                    color = Colors.RED

                output += f"{domain}: {color}{pred_val:>6.2f}W{Colors.RESET} "
                output += f"(act:{actual_val:>6.2f}W err:{color}{error:>+5.2f}W{Colors.RESET} "
                output += f"{color}{error_pct:>+6.1f}%{Colors.RESET}) "
            else:
                output += f"{domain}: {pred_val:>6.2f}W "

        print(output)

    def _calculate_metrics(self, predictions: np.ndarray, actuals: np.ndarray,
                          target_cols: List[str]) -> Dict:
        """Calculate performance metrics."""

        metrics = {}

        for i, col in enumerate(target_cols):
            pred = predictions[:, i]
            actual = actuals[:, i]

            mae = mean_absolute_error(actual, pred)
            mape = mean_absolute_percentage_error(actual, pred) * 100
            r2 = r2_score(actual, pred)

            metrics[col] = {
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'mean_pred': np.mean(pred),
                'mean_actual': np.mean(actual)
            }

        return metrics

    def _print_metrics_table(self, metrics: Dict, target_cols: List[str]):
        """Print formatted metrics table."""

        print(f"{'Domain':<15} {'MAE (W)':<12} {'MAPE (%)':<12} {'R² Score':<12} {'Avg Pred':<12} {'Avg Actual':<12}")
        print("-" * 75)

        for col in target_cols:
            domain = col.replace('rapl_', '').replace('_power', '').title()
            m = metrics[col]

            if m['r2'] > 0.95:
                r2_color = Colors.GREEN
            elif m['r2'] > 0.85:
                r2_color = Colors.YELLOW
            else:
                r2_color = Colors.RED

            print(f"{domain:<15} "
                  f"{m['mae']:>10.3f}  "
                  f"{m['mape']:>10.2f}  "
                  f"{r2_color}{m['r2']:>10.4f}{Colors.RESET}  "
                  f"{m['mean_pred']:>10.2f}W  "
                  f"{m['mean_actual']:>10.2f}W")

    def _save_predictions(self, predictions: np.ndarray, actuals: Optional[np.ndarray],
                         target_cols: List[str], timestamps: List[float], output_file: str):
        """Save predictions to CSV file."""

        data = {}

        if timestamps:
            data['timestamp'] = timestamps

        for i, col in enumerate(target_cols):
            data[f'predicted_{col}'] = predictions[:, i]
            if actuals is not None:
                data[f'actual_{col}'] = actuals[:, i]
                data[f'error_{col}'] = predictions[:, i] - actuals[:, i]
                # Add percentage error
                pct_errors = np.where(actuals[:, i] != 0,
                                     ((predictions[:, i] - actuals[:, i]) / actuals[:, i] * 100),
                                     0)
                data[f'{col}_pct_error'] = pct_errors

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        print(f"\n✓ Predictions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Power prediction inference with MS-TCN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live monitoring with real-time predictions:
    %(prog)s --model model.pth --live
    %(prog)s --model model.pth --live --scroll
    %(prog)s --model model.pth --live --save live_predictions.csv
    %(prog)s --model model.pth --live --scroll --frequency 1.0  # Predict once per second

  Run predictions on CSV file:
    %(prog)s --model model.pth --csv test_data.csv
    %(prog)s --model model.pth --csv test_data.csv --realtime
    %(prog)s --model model.pth --csv data.csv --save predictions.csv
        """
    )

    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--live', action='store_true', help='Live monitoring mode')
    parser.add_argument('--csv', type=str, help='CSV file with data for prediction')
    parser.add_argument('--realtime', action='store_true', help='Show real-time prediction updates (CSV mode)')
    parser.add_argument('--scroll', action='store_true', help='Scrolling output mode (live mode only)')
    parser.add_argument('--interval', type=float, default=0.1, help='Sampling interval for live mode (seconds)')
    parser.add_argument('--frequency', type=float, help='Prediction frequency in Hz (e.g., 1.0 for once per second). Data collection continues at full rate.')
    parser.add_argument('--save', type=str, help='Save predictions to CSV file')

    args = parser.parse_args()

    # Create predictor
    predictor = PowerPredictor(args.model)

    # Live monitoring mode
    if args.live:
        predictor.live_monitor(interval=args.interval, save_output=args.save, scroll_mode=args.scroll,
                              prediction_frequency=args.frequency)

    # CSV mode
    elif args.csv:
        start_time = time.time()
        metrics = predictor.predict_from_csv(args.csv,
                                            show_realtime=args.realtime,
                                            save_output=args.save)
        elapsed = time.time() - start_time

        if metrics:
            print(f"\n{'='*80}")
            print("Performance")
            print(f"{'='*80}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Samples: {len(metrics)}")

    else:
        print("Error: Must specify either --live or --csv")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
