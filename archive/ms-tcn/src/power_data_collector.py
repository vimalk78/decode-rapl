#!/usr/bin/env python3
"""
CPU Power Prediction Data Collector

Collects synchronized RAPL power readings and system metrics for training
machine learning models to predict CPU power usage.

Usage:
    python power_data_collector.py --duration 3600 --output power_data.csv
    python power_data_collector.py --continuous --max-file-size 1GB
    python power_data_collector.py --interval 0.016 --duration 7200
"""

import argparse
import csv
import gzip
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


class RAPLReader:
    """Reads RAPL power measurements from sysfs interface."""

    def __init__(self):
        self.rapl_base = Path("/sys/class/powercap/intel-rapl")
        self.domains = {}
        self._discover_domains()
        self.prev_readings = {}
        self.prev_time = None

    def _discover_domains(self):
        """Discover available RAPL domains."""
        if not self.rapl_base.exists():
            raise RuntimeError("RAPL interface not found. Intel CPU with RAPL support required.")

        # Find package-0 and its subdomains
        for domain_path in sorted(self.rapl_base.glob("intel-rapl:*")):
            name_file = domain_path / "name"
            if not name_file.exists():
                continue

            domain_name = name_file.read_text().strip()

            if domain_name == "package-0":
                # Read max energy range for wraparound handling
                max_range_file = domain_path / "max_energy_range_uj"
                max_energy = int(max_range_file.read_text().strip()) if max_range_file.exists() else 2**32

                self.domains["package"] = {
                    "path": domain_path / "energy_uj",
                    "max": max_energy
                }

                # Look for subdomains (core, dram)
                for subdomain_path in sorted(domain_path.glob("intel-rapl:*:*")):
                    sub_name_file = subdomain_path / "name"
                    if sub_name_file.exists():
                        sub_name = sub_name_file.read_text().strip()

                        # Read max energy range for subdomain
                        sub_max_file = subdomain_path / "max_energy_range_uj"
                        sub_max = int(sub_max_file.read_text().strip()) if sub_max_file.exists() else 2**32

                        if sub_name == "core":
                            self.domains["core"] = {
                                "path": subdomain_path / "energy_uj",
                                "max": sub_max
                            }
                        elif sub_name == "dram":
                            self.domains["dram"] = {
                                "path": subdomain_path / "energy_uj",
                                "max": sub_max
                            }

        if not self.domains:
            raise RuntimeError("No RAPL domains found")

    def read_power(self) -> Dict[str, float]:
        """
        Read power consumption in Watts for all domains.
        Returns average power since last reading.
        """
        current_time = time.time()
        current_readings = {}

        # Read energy counters (in microjoules)
        for domain, config in self.domains.items():
            try:
                current_readings[domain] = int(config["path"].read_text().strip())
            except (OSError, ValueError) as e:
                print(f"Warning: Failed to read {domain}: {e}", file=sys.stderr)
                current_readings[domain] = 0

        # Calculate power (Watts) from energy difference
        power = {}
        if self.prev_readings and self.prev_time:
            time_delta = current_time - self.prev_time
            if time_delta > 0:
                for domain, energy in current_readings.items():
                    if domain in self.prev_readings:
                        # Handle counter wrap-around using actual max_energy_range_uj
                        energy_delta = energy - self.prev_readings[domain]
                        if energy_delta < 0:
                            energy_delta += self.domains[domain]["max"]

                        # Convert microjoules to Watts
                        power_watts = (energy_delta / 1_000_000) / time_delta

                        # Sanity check: filter out impossible power values
                        # Package: 0-500W, Core: 0-300W, DRAM: 0-100W
                        max_power = {"package": 500, "core": 300, "dram": 100}.get(domain, 500)

                        if 0 <= power_watts <= max_power:
                            power[f"rapl_{domain}_power"] = power_watts
                        else:
                            print(f"Warning: Invalid {domain} power {power_watts:.2f}W (filtered)", file=sys.stderr)
                            power[f"rapl_{domain}_power"] = 0.0
        else:
            # First reading, return zeros
            for domain in current_readings.keys():
                power[f"rapl_{domain}_power"] = 0.0

        self.prev_readings = current_readings
        self.prev_time = current_time

        return power


class SystemMetricsCollector:
    """Collects system metrics from /proc filesystem."""

    def __init__(self):
        self.prev_stat = None
        self.prev_vmstat = None
        self.prev_interrupts = None
        self.prev_time = None
        self.cpu_count = os.cpu_count() or 1

    def _read_proc_stat(self) -> Dict[str, int]:
        """Read /proc/stat and parse CPU times."""
        with open("/proc/stat") as f:
            line = f.readline()  # First line is aggregate CPU
            parts = line.split()
            return {
                "user": int(parts[1]),
                "nice": int(parts[2]),
                "system": int(parts[3]),
                "idle": int(parts[4]),
                "iowait": int(parts[5]),
                "irq": int(parts[6]),
                "softirq": int(parts[7]),
                "steal": int(parts[8]) if len(parts) > 8 else 0,
                "guest": int(parts[9]) if len(parts) > 9 else 0,
                "guest_nice": int(parts[10]) if len(parts) > 10 else 0,
            }

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

    def _read_diskstats(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Read /proc/diskstats and return read/write operations and sectors."""
        diskstats = {}
        with open("/proc/diskstats") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 14:
                    dev_name = parts[2]
                    # Only consider physical devices (no partitions)
                    if dev_name.startswith(("sd", "nvme", "vd")) and not any(c.isdigit() for c in dev_name[-1]):
                        reads = int(parts[3])
                        read_sectors = int(parts[5])
                        writes = int(parts[7])
                        write_sectors = int(parts[9])
                        diskstats[dev_name] = (reads, read_sectors, writes, write_sectors)

        return diskstats

    def _read_netdev(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Read /proc/net/dev and return rx/tx bytes and packets."""
        netdev = {}
        with open("/proc/net/dev") as f:
            # Skip header lines
            next(f)
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) >= 10:
                    iface = parts[0].rstrip(":")
                    # Skip loopback
                    if iface != "lo":
                        rx_bytes = int(parts[1])
                        rx_packets = int(parts[2])
                        tx_bytes = int(parts[9])
                        tx_packets = int(parts[10])
                        netdev[iface] = (rx_bytes, rx_packets, tx_bytes, tx_packets)

        return netdev

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

    def collect(self) -> Dict[str, float]:
        """Collect all system metrics."""
        current_time = time.time()
        metrics = {}

        # CPU utilization
        cpu_times, ctxt, intr = self._read_proc_stat_full()
        if self.prev_stat and self.prev_time:
            time_delta = current_time - self.prev_time

            # Calculate CPU percentages
            total_delta = sum(cpu_times[k] - self.prev_stat[k] for k in cpu_times.keys())
            if total_delta > 0:
                metrics["cpu_user_percent"] = 100.0 * (cpu_times["user"] - self.prev_stat["user"]) / total_delta
                metrics["cpu_system_percent"] = 100.0 * (cpu_times["system"] - self.prev_stat["system"]) / total_delta
                metrics["cpu_idle_percent"] = 100.0 * (cpu_times["idle"] - self.prev_stat["idle"]) / total_delta
                metrics["cpu_iowait_percent"] = 100.0 * (cpu_times["iowait"] - self.prev_stat["iowait"]) / total_delta
                metrics["cpu_irq_percent"] = 100.0 * (cpu_times["irq"] - self.prev_stat["irq"]) / total_delta
                metrics["cpu_softirq_percent"] = 100.0 * (cpu_times["softirq"] - self.prev_stat["softirq"]) / total_delta

            # CPU time in seconds/second (absolute work metrics for VM portability)
            # /proc/stat uses jiffies (USER_HZ = 100 typically, so 1 jiffy = 0.01s)
            USER_HZ = 100.0  # Clock ticks per second
            if time_delta > 0:
                metrics["cpu_user_sec"] = (cpu_times["user"] - self.prev_stat["user"]) / USER_HZ / time_delta
                metrics["cpu_system_sec"] = (cpu_times["system"] - self.prev_stat["system"]) / USER_HZ / time_delta
                metrics["cpu_idle_sec"] = (cpu_times["idle"] - self.prev_stat["idle"]) / USER_HZ / time_delta
                metrics["cpu_iowait_sec"] = (cpu_times["iowait"] - self.prev_stat["iowait"]) / USER_HZ / time_delta
                metrics["cpu_irq_sec"] = (cpu_times["irq"] - self.prev_stat["irq"]) / USER_HZ / time_delta
                metrics["cpu_softirq_sec"] = (cpu_times["softirq"] - self.prev_stat["softirq"]) / USER_HZ / time_delta

            # Context switches and interrupts per second
            if time_delta > 0:
                metrics["context_switches_sec"] = (ctxt - self.prev_stat.get("ctxt", ctxt)) / time_delta
                metrics["interrupts_sec"] = (intr - self.prev_stat.get("intr", intr)) / time_delta
        else:
            # First reading
            metrics["cpu_user_percent"] = 0.0
            metrics["cpu_system_percent"] = 0.0
            metrics["cpu_idle_percent"] = 0.0
            metrics["cpu_iowait_percent"] = 0.0
            metrics["cpu_irq_percent"] = 0.0
            metrics["cpu_softirq_percent"] = 0.0
            metrics["cpu_user_sec"] = 0.0
            metrics["cpu_system_sec"] = 0.0
            metrics["cpu_idle_sec"] = 0.0
            metrics["cpu_iowait_sec"] = 0.0
            metrics["cpu_irq_sec"] = 0.0
            metrics["cpu_softirq_sec"] = 0.0
            metrics["context_switches_sec"] = 0.0
            metrics["interrupts_sec"] = 0.0

        cpu_times["ctxt"] = ctxt
        cpu_times["intr"] = intr
        self.prev_stat = cpu_times

        # Memory metrics
        meminfo = self._read_meminfo()
        metrics.update(meminfo)
        metrics["memory_used_mb"] = meminfo["memory_total_mb"] - meminfo["memory_available_mb"]
        metrics["swap_used_mb"] = meminfo["swap_total_mb"] - meminfo["swap_free_mb"]

        # VM statistics (page faults)
        vmstat = self._read_vmstat()
        if self.prev_vmstat and self.prev_time:
            time_delta = current_time - self.prev_time
            pgfault_delta = vmstat.get("pgfault", 0) - self.prev_vmstat.get("pgfault", 0)
            metrics["page_faults_sec"] = pgfault_delta / time_delta if time_delta > 0 else 0.0
        else:
            metrics["page_faults_sec"] = 0.0

        self.prev_vmstat = vmstat

        # Load average and process info
        loadavg = self._read_loadavg()
        metrics.update(loadavg)

        # Calculate blocked processes
        metrics["blocked_processes"] = metrics["total_processes"] - metrics["running_processes"]

        self.prev_time = current_time

        return metrics


class PowerDataCollector:
    """Main data collector coordinating RAPL and system metrics."""

    def __init__(self, args):
        self.args = args
        self.rapl = RAPLReader()
        self.sysmetrics = SystemMetricsCollector()
        self.running = True
        self.sample_count = 0
        self.start_time = None
        self.csv_writer = None
        self.csv_file = None
        self.file_size = 0
        self.file_index = 0

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for clean shutdown."""
        print("\nReceived interrupt signal. Shutting down gracefully...", file=sys.stderr)
        self.running = False

    def _get_output_filename(self) -> str:
        """Generate output filename with optional rotation index."""
        if self.args.continuous and self.file_index > 0:
            base, ext = os.path.splitext(self.args.output)
            return f"{base}_{self.file_index:04d}{ext}"
        return self.args.output

    def _open_output_file(self):
        """Open output CSV file and write header."""
        filename = self._get_output_filename()

        if self.args.compress:
            self.csv_file = gzip.open(filename + ".gz", "wt", newline="")
        else:
            self.csv_file = open(filename, "w", newline="")

        # Define column order
        columns = [
            "timestamp",
            # Power metrics
            "rapl_package_power", "rapl_core_power", "rapl_dram_power",
            # CPU utilization (percentages)
            "cpu_user_percent", "cpu_system_percent", "cpu_idle_percent",
            "cpu_iowait_percent", "cpu_irq_percent", "cpu_softirq_percent",
            # CPU time (seconds/second - for VM portability)
            "cpu_user_sec", "cpu_system_sec", "cpu_idle_sec",
            "cpu_iowait_sec", "cpu_irq_sec", "cpu_softirq_sec",
            # System activity
            "context_switches_sec", "interrupts_sec",
            # Memory metrics
            "memory_used_mb", "memory_cached_mb", "memory_buffers_mb",
            "memory_free_mb", "swap_used_mb", "page_faults_sec",
            # System load
            "load_1min", "load_5min", "load_15min",
            "running_processes", "blocked_processes",
        ]

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=columns, extrasaction='ignore')
        self.csv_writer.writeheader()
        self.csv_file.flush()

        print(f"Writing data to: {filename}", file=sys.stderr)

    def _rotate_file_if_needed(self):
        """Rotate output file if size limit is reached."""
        if not self.args.continuous or not self.args.max_file_size:
            return

        # Check file size
        self.file_size = os.path.getsize(self._get_output_filename())
        max_bytes = self._parse_size(self.args.max_file_size)

        if self.file_size >= max_bytes:
            print(f"\nRotating file (size: {self.file_size / (1024**3):.2f} GB)", file=sys.stderr)
            self.csv_file.close()
            self.file_index += 1
            self._open_output_file()

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '1GB' to bytes."""
        size_str = size_str.upper().strip()
        if size_str.endswith("GB"):
            return int(float(size_str[:-2]) * 1024**3)
        elif size_str.endswith("MB"):
            return int(float(size_str[:-2]) * 1024**2)
        elif size_str.endswith("KB"):
            return int(float(size_str[:-2]) * 1024)
        else:
            return int(size_str)

    def _display_progress(self, elapsed: float):
        """Display progress information."""
        if not self.args.show_progress:
            return

        if self.sample_count % 100 == 0:  # Update every 100 samples
            rate = self.sample_count / elapsed if elapsed > 0 else 0
            print(f"\rSamples: {self.sample_count} | "
                  f"Rate: {rate:.1f} Hz | "
                  f"Elapsed: {elapsed:.1f}s",
                  end="", file=sys.stderr)

    def _log_metadata(self):
        """Log collection metadata."""
        print("=" * 60, file=sys.stderr)
        print("Power Data Collection Metadata", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Start time: {datetime.now().isoformat()}", file=sys.stderr)
        print(f"Sampling interval: {self.args.interval * 1000:.1f}ms ({1/self.args.interval:.1f} Hz)", file=sys.stderr)

        if not self.args.continuous:
            print(f"Duration: {self.args.duration}s", file=sys.stderr)

        # CPU info
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        print(f"CPU: {cpu_model}", file=sys.stderr)
                        break
        except Exception:
            pass

        # RAPL domains
        print(f"RAPL domains: {', '.join(self.rapl.domains.keys())}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    def run(self):
        """Main collection loop."""
        self._log_metadata()
        self._open_output_file()

        self.start_time = time.time()
        next_sample_time = self.start_time

        try:
            while self.running:
                # Collect data
                timestamp = time.time()

                # Combine all metrics
                data = {"timestamp": timestamp}
                data.update(self.rapl.read_power())
                data.update(self.sysmetrics.collect())

                # Write to CSV
                self.csv_writer.writerow(data)
                self.sample_count += 1

                # Periodic flush
                if self.sample_count % 100 == 0:
                    self.csv_file.flush()

                # Check for file rotation
                self._rotate_file_if_needed()

                # Display progress
                elapsed = timestamp - self.start_time
                self._display_progress(elapsed)

                # Check duration limit
                if not self.args.continuous and elapsed >= self.args.duration:
                    break

                # Sleep until next sample
                next_sample_time += self.args.interval
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We're falling behind
                    next_sample_time = time.time()

        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources and print summary."""
        if self.csv_file:
            self.csv_file.close()

        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 60, file=sys.stderr)
        print("Collection Summary", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Total samples: {self.sample_count}", file=sys.stderr)
        print(f"Duration: {elapsed:.2f}s", file=sys.stderr)
        print(f"Average rate: {self.sample_count / elapsed:.2f} Hz", file=sys.stderr)
        print(f"Output file: {self._get_output_filename()}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Collect CPU power and system metrics for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic collection for 1 hour:
    %(prog)s --duration 3600 --output power_data.csv

  Continuous collection with file rotation:
    %(prog)s --continuous --max-file-size 1GB --output power_data.csv

  Custom sampling rate for 2 hours:
    %(prog)s --interval 0.016 --duration 7200 --output power_data.csv

  Compressed output:
    %(prog)s --duration 3600 --output power_data.csv --compress
        """
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=3600,
        help="Collection duration in seconds (default: 3600)"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.016,
        help="Sampling interval in seconds (default: 0.016 = 62.5Hz)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="power_data.csv",
        help="Output CSV file path (default: power_data.csv)"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (ignore duration limit)"
    )

    parser.add_argument(
        "--max-file-size",
        type=str,
        help="Max file size before rotation, e.g., '1GB', '500MB' (only with --continuous)"
    )

    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output with gzip"
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show real-time progress (default: True)"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display"
    )

    args = parser.parse_args()

    if args.no_progress:
        args.show_progress = False

    # Validate arguments
    if args.continuous and args.max_file_size is None:
        print("Warning: --continuous mode without --max-file-size may create very large files",
              file=sys.stderr)

    try:
        collector = PowerDataCollector(args)
        collector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
