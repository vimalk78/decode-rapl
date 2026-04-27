"""
DECODE-RAPL Data Collection
MSR batch reading for CPU usage and RAPL power on bare-metal machines
"""

import os
import struct
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MSRReader:
    """
    Model-Specific Register (MSR) reader for Intel CPUs

    Reads:
    - CPU usage from performance counters (IA32_FIXED_CTR1, IA32_FIXED_CTR2)
    - RAPL power from energy status (MSR_PKG_ENERGY_STATUS)
    """

    # MSR addresses
    MSR_FIXED_CTR1 = 0x30A  # Unhalted core cycles
    MSR_FIXED_CTR2 = 0x30B  # Reference cycles
    MSR_PKG_ENERGY_STATUS = 0x611  # Package energy status
    MSR_RAPL_POWER_UNIT = 0x606  # RAPL power unit

    def __init__(self, core: int = 0):
        """
        Args:
            core: CPU core to read from (default: 0)
        """
        self.core = core
        self.msr_device = f"/dev/cpu/{core}/msr"

        # Check if MSR device exists
        if not os.path.exists(self.msr_device):
            raise FileNotFoundError(
                f"MSR device {self.msr_device} not found. "
                f"Ensure MSR kernel module is loaded: sudo modprobe msr"
            )

        # Check if we have read access
        if not os.access(self.msr_device, os.R_OK):
            raise PermissionError(
                f"Cannot read {self.msr_device}. "
                f"Requires root privileges or CAP_SYS_RAWIO capability."
            )

        # Read RAPL energy unit
        self.energy_unit = self._get_rapl_energy_unit()
        logger.info(f"RAPL energy unit: {self.energy_unit} Joules per unit")

        # Initialize previous values for delta calculations
        self.prev_core_cycles = None
        self.prev_ref_cycles = None
        self.prev_energy = None
        self.prev_time = None

    def _read_msr(self, msr_addr: int) -> int:
        """
        Read 64-bit value from MSR

        Args:
            msr_addr: MSR address

        Returns:
            64-bit integer value
        """
        try:
            fd = os.open(self.msr_device, os.O_RDONLY)
            os.lseek(fd, msr_addr, os.SEEK_SET)
            data = os.read(fd, 8)
            os.close(fd)

            value = struct.unpack('Q', data)[0]
            return value

        except Exception as e:
            logger.error(f"Error reading MSR 0x{msr_addr:X}: {e}")
            return 0

    def _get_rapl_energy_unit(self) -> float:
        """Get RAPL energy unit for converting raw values to Joules"""
        power_unit_raw = self._read_msr(self.MSR_RAPL_POWER_UNIT)

        # Energy unit is bits 12:8
        energy_unit_raw = (power_unit_raw >> 8) & 0x1F

        # Energy = 1 / (2^ESU) Joules
        energy_unit = 1.0 / (2 ** energy_unit_raw)

        return energy_unit

    def read_cpu_usage(self) -> Optional[float]:
        """
        Read CPU usage percentage from performance counters

        Returns:
            CPU usage in percent (0-100), or None if invalid
        """
        core_cycles = self._read_msr(self.MSR_FIXED_CTR1)
        ref_cycles = self._read_msr(self.MSR_FIXED_CTR2)

        if self.prev_core_cycles is None:
            # First reading, store baseline
            self.prev_core_cycles = core_cycles
            self.prev_ref_cycles = ref_cycles
            return None

        # Calculate deltas
        delta_core = core_cycles - self.prev_core_cycles
        delta_ref = ref_cycles - self.prev_ref_cycles

        # Update previous values
        self.prev_core_cycles = core_cycles
        self.prev_ref_cycles = ref_cycles

        # Handle counter overflow (64-bit wrap)
        if delta_core < 0:
            delta_core += (1 << 64)
        if delta_ref < 0:
            delta_ref += (1 << 64)

        # Calculate usage
        if delta_ref > 0:
            usage = 100.0 * delta_core / delta_ref
            return min(100.0, max(0.0, usage))  # Clamp to [0, 100]

        return None

    def read_power(self) -> Optional[float]:
        """
        Read package power consumption from RAPL

        Returns:
            Power in Watts, or None if invalid
        """
        current_time = time.time()
        energy_raw = self._read_msr(self.MSR_PKG_ENERGY_STATUS)

        # Energy status is lower 32 bits
        energy_raw = energy_raw & 0xFFFFFFFF

        # Convert to Joules
        energy = energy_raw * self.energy_unit

        if self.prev_energy is None:
            # First reading, store baseline
            self.prev_energy = energy
            self.prev_time = current_time
            return None

        # Calculate energy delta
        delta_energy = energy - self.prev_energy
        delta_time = current_time - self.prev_time

        # Handle counter overflow (32-bit wrap)
        if delta_energy < 0:
            delta_energy += (1 << 32) * self.energy_unit

        # Update previous values
        self.prev_energy = energy
        self.prev_time = current_time

        # Calculate power (Watts = Joules / seconds)
        if delta_time > 0:
            power = delta_energy / delta_time
            return max(0.0, power)  # Ensure non-negative

        return None

    def read_all(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Read both CPU usage and power

        Returns:
            (cpu_usage, power) tuple
        """
        usage = self.read_cpu_usage()
        power = self.read_power()

        return usage, power


class DataCollector:
    """
    Data collector that samples CPU usage and RAPL power at high frequency

    Runs stress-ng workloads to generate varied CPU loads
    """

    def __init__(
        self,
        machine_id: str,
        sampling_rate_ms: int = 1,
        duration_hours: float = 10,
        output_csv: str = "data/rapl_data.csv",
        workload_config: Optional[dict] = None
    ):
        """
        Args:
            machine_id: Unique identifier for this machine
            sampling_rate_ms: Sampling rate in milliseconds
            duration_hours: Duration of data collection in hours
            output_csv: Output CSV file path
            workload_config: Workload configuration (load_min, load_max, etc.)
        """
        self.machine_id = machine_id
        self.sampling_rate_ms = sampling_rate_ms
        self.duration_hours = duration_hours
        self.output_csv = output_csv

        self.workload_config = workload_config or {
            'load_min': 5,
            'load_max': 90,
            'duration_min_sec': 1,
            'duration_max_sec': 10
        }

        # Initialize MSR reader
        try:
            self.msr_reader = MSRReader(core=0)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to initialize MSR reader: {e}")
            logger.error("Falling back to synthetic data mode...")
            self.msr_reader = None

        # Data buffer
        self.data = []

        # Control flag
        self.running = False

    def _start_workload(self):
        """Start stress-ng workload with random CPU load"""
        load = np.random.randint(
            self.workload_config['load_min'],
            self.workload_config['load_max']
        )

        duration = np.random.randint(
            self.workload_config['duration_min_sec'],
            self.workload_config['duration_max_sec']
        )

        # stress-ng command: stress CPU with given load
        cmd = [
            'stress-ng',
            '--cpu', str(os.cpu_count()),
            '--cpu-load', str(load),
            '--timeout', f'{duration}s',
            '--quiet'
        ]

        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"Started stress-ng: load={load}%, duration={duration}s")
        except FileNotFoundError:
            logger.warning("stress-ng not found. Install with: sudo apt-get install stress-ng")

    def _generate_synthetic_sample(self) -> Tuple[float, float]:
        """Generate synthetic CPU usage and power (fallback mode)"""
        # Random walk for CPU usage
        if not hasattr(self, '_synthetic_usage'):
            self._synthetic_usage = 50.0

        drift = np.random.uniform(-5, 5)
        self._synthetic_usage = np.clip(self._synthetic_usage + drift, 0, 100)

        # Nonlinear power model
        usage_norm = self._synthetic_usage / 100.0
        power = 50 + 30 * usage_norm + 50 * usage_norm ** 2 + 20 * usage_norm ** 3
        power += np.random.normal(0, 2)

        return self._synthetic_usage, power

    def collect(self):
        """Main data collection loop"""
        logger.info(f"Starting data collection for {self.duration_hours} hours...")
        logger.info(f"Machine ID: {self.machine_id}")
        logger.info(f"Sampling rate: {self.sampling_rate_ms}ms")
        logger.info(f"Output: {self.output_csv}")

        if self.msr_reader is None:
            logger.warning("Using synthetic data mode (MSR not available)")

        self.running = True

        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nStopping data collection...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        start_time = time.time()
        target_duration = self.duration_hours * 3600
        next_workload_change = time.time() + np.random.uniform(1, 10)

        iteration = 0
        valid_samples = 0

        while self.running and (time.time() - start_time) < target_duration:
            iter_start = time.time()

            # Periodically change workload
            if time.time() >= next_workload_change:
                self._start_workload()
                next_workload_change = time.time() + np.random.uniform(
                    self.workload_config['duration_min_sec'],
                    self.workload_config['duration_max_sec']
                )

            # Read measurements
            if self.msr_reader:
                cpu_usage, power = self.msr_reader.read_all()
            else:
                cpu_usage, power = self._generate_synthetic_sample()

            # Store data if valid
            if cpu_usage is not None and power is not None:
                self.data.append({
                    'timestamp': pd.Timestamp.now(),
                    'machine_id': self.machine_id,
                    'cpu_usage': cpu_usage,
                    'power': power
                })

                valid_samples += 1

                # Log progress every 1000 samples
                if valid_samples % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = valid_samples / elapsed
                    logger.info(
                        f"Samples: {valid_samples}, "
                        f"Time: {elapsed:.1f}s, "
                        f"Rate: {rate:.1f} samples/s"
                    )

            iteration += 1

            # Sleep to maintain sampling rate
            elapsed = time.time() - iter_start
            sleep_time = (self.sampling_rate_ms / 1000.0) - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif iteration % 100 == 0:
                logger.warning(f"Cannot maintain {self.sampling_rate_ms}ms sampling rate")

        # Save data
        self._save_data()

        logger.info(f"Data collection completed. Collected {valid_samples} samples.")

    def _save_data(self):
        """Save collected data to CSV"""
        if not self.data:
            logger.warning("No data to save")
            return

        df = pd.DataFrame(self.data)

        # Ensure output directory exists
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)

        # Append to existing file if it exists
        if Path(self.output_csv).exists():
            df.to_csv(self.output_csv, mode='a', header=False, index=False)
            logger.info(f"Appended {len(df)} samples to {self.output_csv}")
        else:
            df.to_csv(self.output_csv, index=False)
            logger.info(f"Saved {len(df)} samples to {self.output_csv}")

        logger.info(f"Data statistics:")
        logger.info(f"  CPU usage: mean={df['cpu_usage'].mean():.2f}%, "
                   f"std={df['cpu_usage'].std():.2f}%")
        logger.info(f"  Power: mean={df['power'].mean():.2f}W, "
                   f"std={df['power'].std():.2f}W")


def main():
    """Main entry point for data collection"""
    import argparse

    parser = argparse.ArgumentParser(description="DECODE-RAPL Data Collector")
    parser.add_argument('--machine-id', type=str, required=True,
                       help="Unique machine identifier (e.g., skylake1)")
    parser.add_argument('--duration', type=float, default=10,
                       help="Duration in hours (default: 10)")
    parser.add_argument('--sampling-rate', type=int, default=1,
                       help="Sampling rate in milliseconds (default: 1)")
    parser.add_argument('--output', type=str, default="data/rapl_data.csv",
                       help="Output CSV file path")
    parser.add_argument('--synthetic', action='store_true',
                       help="Force synthetic data mode (for testing)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DECODE-RAPL Data Collector")
    logger.info("=" * 60)

    collector = DataCollector(
        machine_id=args.machine_id,
        sampling_rate_ms=args.sampling_rate,
        duration_hours=args.duration,
        output_csv=args.output
    )

    if args.synthetic:
        logger.info("Forcing synthetic data mode")
        collector.msr_reader = None

    collector.collect()


if __name__ == "__main__":
    main()
