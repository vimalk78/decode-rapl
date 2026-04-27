#!/usr/bin/env python3
"""
Workload Generator for Power Model Training Data Collection

This script generates various CPU workload patterns for training power estimation models.
It runs independently from the data collection script, allowing flexible workload patterns.

Supported patterns:
- random: Random CPU loads
- sine: Sinusoidal CPU load pattern
- burst: Sudden spikes and drops
- brownian: Random walk with momentum
- mixed: Combination of different stress types
"""

import argparse
import subprocess
import time
import random
import math
import logging
import signal
import sys
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkloadGenerator:
    def __init__(self, duration: int, min_load: int = 0, max_load: int = 100,
                 change_interval: float = 5.0, workload_types: List[str] = None):
        self.duration = duration
        self.min_load = min_load
        self.max_load = max_load
        self.change_interval = change_interval
        self.workload_types = workload_types or ['cpu']
        self.current_proc = None
        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.current_proc:
            self.current_proc.terminate()
        sys.exit(0)

    def _get_cpu_count(self) -> int:
        """Get logical CPU count for stress testing"""
        import psutil
        return psutil.cpu_count(logical=True)

    def _start_stress(self, cpu_load: int, workload_type: str = 'cpu') -> Optional[subprocess.Popen]:
        """Start stress process with given CPU load"""
        if cpu_load <= 0:
            return None

        cpu_count = self._get_cpu_count()

        try:
            if workload_type == 'cpu':
                cmd = ['stress-ng', '--cpu', str(cpu_count), '--cpu-load', str(cpu_load), '--quiet']
            elif workload_type == 'matrix':
                # Matrix operations - more intensive
                cmd = ['stress-ng', '--matrix', str(cpu_count), '--matrix-size', '128', '--quiet']
            elif workload_type == 'cache':
                # Cache thrashing
                cmd = ['stress-ng', '--cache', str(cpu_count), '--cache-size', '1M', '--quiet']
            elif workload_type == 'stream':
                # Memory streaming
                cmd = ['stress-ng', '--stream', str(cpu_count), '--stream-L3-size', '4M', '--quiet']
            else:
                # Default to CPU stress
                cmd = ['stress-ng', '--cpu', str(cpu_count), '--cpu-load', str(cpu_load), '--quiet']

            return subprocess.Popen(cmd)

        except FileNotFoundError:
            logging.error("stress-ng not found. Please install: sudo apt-get install stress-ng")
            return None

    def _stop_stress(self):
        """Stop current stress process"""
        if self.current_proc:
            self.current_proc.terminate()
            time.sleep(0.5)  # Allow graceful termination
            if self.current_proc.poll() is None:
                self.current_proc.kill()
            self.current_proc = None

    def generate_random_pattern(self):
        """Generate random CPU load pattern"""
        logging.info(f"Starting random workload pattern (min={self.min_load}%, max={self.max_load}%)")

        start_time = time.time()
        while self.running and time.time() - start_time < self.duration:
            # Random load between min and max
            target_load = random.randint(self.min_load, self.max_load)

            # Random workload type
            workload_type = random.choice(self.workload_types)

            # Random duration for this load level
            load_duration = random.uniform(self.change_interval * 0.5, self.change_interval * 2.0)

            logging.info(f"Setting {workload_type} load to {target_load}% for {load_duration:.1f}s")

            self._stop_stress()
            if target_load > 0:
                self.current_proc = self._start_stress(target_load, workload_type)

            time.sleep(min(load_duration, self.duration - (time.time() - start_time)))

    def generate_sine_pattern(self, frequency: float = 0.1, amplitude: float = None):
        """Generate sinusoidal CPU load pattern"""
        if amplitude is None:
            amplitude = (self.max_load - self.min_load) / 2

        center = (self.max_load + self.min_load) / 2

        logging.info(f"Starting sine wave pattern (freq={frequency}, amp={amplitude}, center={center})")

        start_time = time.time()
        last_change = 0

        while self.running and time.time() - start_time < self.duration:
            elapsed = time.time() - start_time

            # Calculate sine wave load
            target_load = center + amplitude * math.sin(2 * math.pi * frequency * elapsed)
            target_load = max(self.min_load, min(self.max_load, int(target_load)))

            # Only change load if enough time has passed or load changed significantly
            if elapsed - last_change >= self.change_interval:
                logging.info(f"Sine wave: setting load to {target_load}% (t={elapsed:.1f}s)")

                self._stop_stress()
                if target_load > 0:
                    self.current_proc = self._start_stress(target_load)

                last_change = elapsed

            time.sleep(1.0)

    def generate_burst_pattern(self):
        """Generate burst pattern with sudden spikes and drops"""
        logging.info(f"Starting burst pattern (min={self.min_load}%, max={self.max_load}%)")

        start_time = time.time()
        while self.running and time.time() - start_time < self.duration:
            # Burst phase: high load
            burst_load = random.randint(int(self.max_load * 0.8), self.max_load)
            burst_duration = random.uniform(2.0, 8.0)

            logging.info(f"BURST: {burst_load}% for {burst_duration:.1f}s")
            self._stop_stress()
            self.current_proc = self._start_stress(burst_load)
            time.sleep(min(burst_duration, self.duration - (time.time() - start_time)))

            if not self.running or time.time() - start_time >= self.duration:
                break

            # Quiet phase: low load
            quiet_load = random.randint(self.min_load, int(self.max_load * 0.3))
            quiet_duration = random.uniform(5.0, 15.0)

            logging.info(f"QUIET: {quiet_load}% for {quiet_duration:.1f}s")
            self._stop_stress()
            if quiet_load > 0:
                self.current_proc = self._start_stress(quiet_load)
            time.sleep(min(quiet_duration, self.duration - (time.time() - start_time)))

    def generate_brownian_pattern(self, step_size: int = 10):
        """Generate random walk (Brownian motion) pattern"""
        logging.info(f"Starting Brownian motion pattern (step_size={step_size}%)")

        current_load = (self.min_load + self.max_load) // 2  # Start in middle
        start_time = time.time()

        while self.running and time.time() - start_time < self.duration:
            # Random step up or down
            step = random.randint(-step_size, step_size)
            current_load += step

            # Keep within bounds
            current_load = max(self.min_load, min(self.max_load, current_load))

            logging.info(f"Brownian: setting load to {current_load}%")

            self._stop_stress()
            if current_load > 0:
                self.current_proc = self._start_stress(current_load)

            time.sleep(self.change_interval)

    def generate_mixed_pattern(self):
        """Generate mixed pattern using different workload types"""
        logging.info("Starting mixed workload pattern with various stress types")

        start_time = time.time()
        while self.running and time.time() - start_time < self.duration:
            # Choose random parameters
            target_load = random.randint(self.min_load, self.max_load)
            workload_type = random.choice(self.workload_types)
            load_duration = random.uniform(3.0, 12.0)

            logging.info(f"Mixed: {workload_type} at {target_load}% for {load_duration:.1f}s")

            self._stop_stress()
            if target_load > 0:
                self.current_proc = self._start_stress(target_load, workload_type)

            time.sleep(min(load_duration, self.duration - (time.time() - start_time)))

    def run_pattern(self, pattern: str, **kwargs):
        """Run the specified pattern"""
        try:
            if pattern == 'random':
                self.generate_random_pattern()
            elif pattern == 'sine':
                frequency = kwargs.get('frequency', 0.1)
                amplitude = kwargs.get('amplitude', None)
                self.generate_sine_pattern(frequency, amplitude)
            elif pattern == 'burst':
                self.generate_burst_pattern()
            elif pattern == 'brownian':
                step_size = kwargs.get('step_size', 10)
                self.generate_brownian_pattern(step_size)
            elif pattern == 'mixed':
                self.generate_mixed_pattern()
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

        finally:
            self._stop_stress()
            logging.info("Workload generation completed")

def main():
    parser = argparse.ArgumentParser(description="Generate workload patterns for power model training")
    parser.add_argument('--pattern', type=str, required=True,
                        choices=['random', 'sine', 'burst', 'brownian', 'mixed'],
                        help='Workload pattern to generate')
    parser.add_argument('--duration', type=int, default=600,
                        help='Duration to run workload (seconds)')
    parser.add_argument('--min-load', type=int, default=0,
                        help='Minimum CPU load percentage (default: 0)')
    parser.add_argument('--max-load', type=int, default=100,
                        help='Maximum CPU load percentage (default: 100)')
    parser.add_argument('--change-interval', type=float, default=5.0,
                        help='Average time between load changes (seconds)')
    parser.add_argument('--workload-types', type=str, nargs='+',
                        choices=['cpu', 'matrix', 'cache', 'stream'],
                        default=['cpu'],
                        help='Types of workloads to use (default: cpu)')

    # Sine wave specific options
    parser.add_argument('--frequency', type=float, default=0.1,
                        help='Sine wave frequency (Hz) - for sine pattern')
    parser.add_argument('--amplitude', type=float, default=None,
                        help='Sine wave amplitude (load %) - for sine pattern')

    # Brownian motion specific options
    parser.add_argument('--step-size', type=int, default=10,
                        help='Step size for Brownian motion - for brownian pattern')

    args = parser.parse_args()

    # Create and run workload generator
    generator = WorkloadGenerator(
        duration=args.duration,
        min_load=args.min_load,
        max_load=args.max_load,
        change_interval=args.change_interval,
        workload_types=args.workload_types
    )

    # Pattern-specific parameters
    kwargs = {}
    if args.pattern == 'sine':
        kwargs['frequency'] = args.frequency
        if args.amplitude is not None:
            kwargs['amplitude'] = args.amplitude
    elif args.pattern == 'brownian':
        kwargs['step_size'] = args.step_size

    # Run the pattern
    generator.run_pattern(args.pattern, **kwargs)

if __name__ == '__main__':
    main()