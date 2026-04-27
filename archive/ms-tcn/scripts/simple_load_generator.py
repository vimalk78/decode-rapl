#!/usr/bin/env python3
"""
Simple Load Generator - Pure Python CPU Load Control
No external dependencies (no stress-ng, no fio)

Creates controllable CPU loads with different workload types:
- Light work (low power): /proc scanning, file parsing, JSON
- Medium work (medium power): list/dict operations, string processing
- Heavy work (high power): math, crypto hashing, compression

Usage:
    python simple_load_generator.py --cpu 20 --workload light --duration 60
    python simple_load_generator.py --cpu 50 --workload heavy --duration 120
    python simple_load_generator.py --cpu 10 --workload proc --duration 300
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path


class CPULoadGenerator:
    """Generate CPU load at specific percentage."""

    def __init__(self, target_cpu_percent: float, workload_type: str = "medium"):
        self.target_cpu_percent = target_cpu_percent
        self.workload_type = workload_type
        self.running = True

    def _busy_loop(self, duration_ms: float):
        """Busy-wait for specified milliseconds."""
        end = time.time() + (duration_ms / 1000.0)
        while time.time() < end:
            pass  # Busy wait

    def _light_work(self, iterations: int):
        """
        Light CPU work - minimal power consumption.
        Simulates /proc reading, file parsing, JSON operations.
        """
        for _ in range(iterations):
            # String operations
            text = "sample text for parsing" * 10
            words = text.split()
            text.upper().lower()

            # Simple JSON operations
            data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            json_str = json.dumps(data)
            json.loads(json_str)

            # List operations
            items = list(range(100))
            sum(items)
            max(items)

    def _medium_work(self, iterations: int):
        """
        Medium CPU work - moderate power consumption.
        List comprehensions, dict operations, regex.
        """
        for _ in range(iterations):
            # List comprehensions
            squares = [x**2 for x in range(100)]
            filtered = [x for x in squares if x % 2 == 0]

            # Dictionary operations
            data = {i: i**2 for i in range(100)}
            values = list(data.values())
            keys = list(data.keys())

            # String processing with regex
            text = "test string 123 with numbers 456"
            re.findall(r'\d+', text)
            re.sub(r'\d+', 'X', text)

    def _heavy_work(self, iterations: int):
        """
        Heavy CPU work - high power consumption.
        Math operations, hashing, compression.
        """
        for _ in range(iterations):
            # Math operations
            result = 0
            for i in range(500):
                result += i ** 2
                result = result % 1000000

            # Hashing (CPU intensive)
            data = b"test data" * 100
            hashlib.sha256(data).hexdigest()
            hashlib.md5(data).hexdigest()

            # Compression (CPU + memory intensive)
            compressed = zlib.compress(data)
            zlib.decompress(compressed)

    def _proc_scanning_work(self, iterations: int):
        """
        /proc scanning work - light power, high syscall overhead.
        Mimics Prometheus exporter behavior.
        """
        for _ in range(iterations):
            proc_data = {}

            # Scan some /proc entries
            try:
                # Read process list
                pids = [d for d in os.listdir('/proc') if d.isdigit()][:50]

                for pid in pids:
                    try:
                        # Read stat file
                        with open(f'/proc/{pid}/stat', 'r') as f:
                            stat = f.read().split()
                            if len(stat) >= 15:
                                proc_data[pid] = {
                                    'utime': int(stat[13]),
                                    'stime': int(stat[14])
                                }

                        # Read cmdline
                        with open(f'/proc/{pid}/cmdline', 'r') as f:
                            cmdline = f.read().replace('\x00', ' ')
                            proc_data[pid]['cmdline'] = cmdline[:50]
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue
            except Exception:
                pass

            # Compute some metrics (light processing)
            total_procs = len(proc_data)
            if total_procs > 0:
                avg_utime = sum(p['utime'] for p in proc_data.values()) / total_procs

    def run_for_duration(self, duration_seconds: float, verbose: bool = False):
        """
        Run CPU load for specified duration.

        Uses work/sleep cycling to achieve target CPU percentage:
        - work_time: Do CPU-intensive work
        - sleep_time: Sleep to reduce CPU usage

        The ratio work_time / (work_time + sleep_time) = target_cpu_percent
        """
        start_time = time.time()
        end_time = start_time + duration_seconds

        # Calculate work/sleep cycle times (in milliseconds)
        cycle_duration_ms = 100  # 100ms cycle
        work_time_ms = cycle_duration_ms * (self.target_cpu_percent / 100.0)
        sleep_time_ms = cycle_duration_ms - work_time_ms

        # Select work function based on type
        work_functions = {
            'light': (self._light_work, 50),
            'medium': (self._medium_work, 30),
            'heavy': (self._heavy_work, 10),
            'proc': (self._proc_scanning_work, 5),
        }

        work_func, iterations_per_cycle = work_functions.get(
            self.workload_type,
            (self._medium_work, 30)
        )

        cycle_count = 0

        while time.time() < end_time and self.running:
            cycle_start = time.time()

            # Do work
            if work_time_ms > 0:
                work_func(iterations_per_cycle)

            # Sleep to control CPU usage
            if sleep_time_ms > 0:
                time.sleep(sleep_time_ms / 1000.0)

            cycle_count += 1

            if verbose and cycle_count % 10 == 0:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                print(f"\r[{self.workload_type}] Target: {self.target_cpu_percent:.1f}% CPU | "
                      f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s",
                      end='', file=sys.stderr)

        if verbose:
            print(file=sys.stderr)  # New line


def run_worker(target_cpu: float, workload_type: str, duration: float, worker_id: int, verbose: bool):
    """Worker process function."""
    generator = CPULoadGenerator(target_cpu, workload_type)
    if verbose and worker_id == 0:
        generator.run_for_duration(duration, verbose=True)
    else:
        generator.run_for_duration(duration, verbose=False)


def main():
    parser = argparse.ArgumentParser(
        description="Simple CPU Load Generator - Pure Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Light work (low power) at 20% CPU for 60 seconds:
    %(prog)s --cpu 20 --workload light --duration 60

  Heavy work (high power) at 50% CPU for 120 seconds:
    %(prog)s --cpu 50 --workload heavy --duration 120

  /proc scanning (light power) at 15% CPU for 300 seconds:
    %(prog)s --cpu 15 --workload proc --duration 300

  Medium work on all cores at 30% CPU:
    %(prog)s --cpu 30 --workload medium --duration 60 --cores 0

  Medium work on 4 cores at 75% CPU:
    %(prog)s --cpu 75 --workload medium --duration 120 --cores 4

Workload Types:
  light:  Low power consumption (string ops, JSON, simple parsing)
  medium: Moderate power (list comprehensions, dict ops, regex)
  heavy:  High power (math, hashing, compression)
  proc:   Light power with high syscall overhead (/proc scanning)
"""
    )

    parser.add_argument('--cpu', type=float, required=True,
                       help='Target CPU usage percentage (0-100)')
    parser.add_argument('--workload', type=str, default='medium',
                       choices=['light', 'medium', 'heavy', 'proc'],
                       help='Type of workload to generate')
    parser.add_argument('--duration', type=float, required=True,
                       help='Duration in seconds')
    parser.add_argument('--cores', type=int, default=1,
                       help='Number of CPU cores to use (0 = all cores, default: 1)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Validate
    if not (0 <= args.cpu <= 100):
        print("Error: CPU percentage must be between 0 and 100", file=sys.stderr)
        return 1

    if args.duration <= 0:
        print("Error: Duration must be positive", file=sys.stderr)
        return 1

    # Determine number of workers
    total_cores = multiprocessing.cpu_count()
    if args.cores == 0:
        num_workers = total_cores
    else:
        num_workers = min(args.cores, total_cores)

    print(f"Starting load generator:", file=sys.stderr)
    print(f"  Target CPU: {args.cpu}%", file=sys.stderr)
    print(f"  Workload type: {args.workload}", file=sys.stderr)
    print(f"  Duration: {args.duration}s", file=sys.stderr)
    print(f"  Workers: {num_workers} (of {total_cores} cores)", file=sys.stderr)
    print(file=sys.stderr)

    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=run_worker,
            args=(args.cpu, args.workload, args.duration, i, args.verbose)
        )
        p.start()
        processes.append(p)

    # Wait for all workers to complete
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating workers...", file=sys.stderr)
        for p in processes:
            p.terminate()
            p.join()
        return 1

    print("\nLoad generation complete.", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
