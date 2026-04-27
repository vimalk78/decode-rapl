#!/usr/bin/env python3
"""
System Load Generator for CPU Power Prediction Training

Generates diverse workload patterns to create varied power signatures for training
the MS-TCN model for CPU power prediction.

Usage:
    python load_generator.py --sequence comprehensive --duration 3600
    python load_generator.py --pattern cpu-ramp --levels 25,50,75,100 --duration 600
    python load_generator.py --workload memory-bandwidth --size 8G --duration 300
    python load_generator.py --interactive
"""

import argparse
import json
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class WorkloadExecutor:
    """Base class for workload execution."""

    def __init__(self):
        self.processes = []
        self.running = True

    def cleanup(self):
        """Terminate all running processes."""
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        self.processes.clear()

    def stop(self):
        """Stop all workloads."""
        self.running = False
        self.cleanup()


class CPUWorkload(WorkloadExecutor):
    """Generate CPU-intensive workloads."""

    def __init__(self):
        super().__init__()
        self.cpu_count = multiprocessing.cpu_count()

    def stress_cpu(self, workers: int, duration: float, method: str = "matrixprod", cpu_load: int = None):
        """
        Stress CPU using stress-ng.

        Args:
            workers: Number of CPU workers
            duration: Duration in seconds
            method: CPU stress method (cpu, matrixprod, fft, etc.)
            cpu_load: CPU load percentage per worker (0-100), if specified
        """
        cmd = [
            "stress-ng",
            f"--{method}", str(workers),
            "--timeout", f"{int(duration)}s",
            "--metrics-brief"
        ]

        # Add CPU load percentage if specified
        if cpu_load is not None:
            cmd.extend(["--cpu-load", str(cpu_load)])

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(proc)
        return proc

    def variable_cpu_load(self, target_percent: float, duration: float, cores: int = 1):
        """
        Generate variable CPU load at specific utilization level.

        Uses stress-ng with --cpu-load option for smooth, consistent load.
        """
        # Use stress-ng with cpu-load option for smoother load distribution
        return self.stress_cpu(cores, duration, method="cpu", cpu_load=int(target_percent))

    def instruction_mix(self, instruction_type: str, workers: int, duration: float):
        """
        Generate specific instruction type workload.

        Args:
            instruction_type: aea, sha, fft, matrixprod, etc.
            workers: Number of workers
            duration: Duration in seconds
        """
        valid_types = {
            "aes": "aes",
            "sha": "sha",
            "fft": "fft",
            "matrixprod": "matrixprod",
            "float": "fp",
            "integer": "cpu"
        }

        method = valid_types.get(instruction_type, "cpu")
        return self.stress_cpu(workers, duration, method)

    def bursty_cpu(self, duty_cycle: float, burst_duration: float,
                   total_duration: float, workers: int):
        """
        Generate bursty CPU workload with variable duty cycle.

        Args:
            duty_cycle: Fraction of time CPU is active (0.0-1.0)
            burst_duration: Duration of each burst in seconds
            total_duration: Total duration in seconds
            workers: Number of CPU workers
        """
        end_time = time.time() + total_duration

        while time.time() < end_time and self.running:
            # Active period
            active_time = burst_duration * duty_cycle
            self.stress_cpu(workers, active_time, "cpu")
            time.sleep(active_time)
            self.cleanup()

            # Idle period
            idle_time = burst_duration * (1 - duty_cycle)
            remaining = end_time - time.time()
            if remaining > 0:
                time.sleep(min(idle_time, remaining))


class MemoryWorkload(WorkloadExecutor):
    """Generate memory-intensive workloads."""

    def stress_memory(self, size: str, workers: int, duration: float,
                     pattern: str = "sequential"):
        """
        Stress memory using stress-ng.

        Args:
            size: Memory size per worker (e.g., "1G", "512M")
            workers: Number of memory workers
            duration: Duration in seconds
            pattern: Access pattern (sequential, random)
        """
        cmd = [
            "stress-ng",
            "--vm", str(workers),
            "--vm-bytes", size,
            "--vm-method", pattern,
            "--timeout", f"{int(duration)}s",
            "--metrics-brief"
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(proc)
        return proc

    def memory_bandwidth(self, size: str, duration: float):
        """Saturate memory bandwidth."""
        workers = multiprocessing.cpu_count()
        return self.stress_memory(size, workers, duration, "write64")

    def cache_hierarchy_test(self, cache_level: str, duration: float):
        """
        Test specific cache level.

        Args:
            cache_level: L1, L2, L3, or DRAM
            duration: Duration in seconds
        """
        # Typical cache sizes (adjust based on CPU)
        cache_sizes = {
            "L1": "32K",
            "L2": "256K",
            "L3": "8M",
            "DRAM": "2G"
        }

        size = cache_sizes.get(cache_level.upper(), "1G")

        # Use fewer workers for cache tests to avoid eviction
        workers = 1 if cache_level.upper() in ["L1", "L2"] else 4

        return self.stress_memory(size, workers, duration, "read64")

    def random_access(self, size: str, duration: float, workers: int = 4):
        """Generate random memory access pattern."""
        return self.stress_memory(size, workers, duration, "all")


class IOWorkload(WorkloadExecutor):
    """Generate I/O-intensive workloads."""

    def __init__(self, test_dir: Optional[str] = None):
        super().__init__()
        self.test_dir = test_dir or tempfile.mkdtemp(prefix="load_gen_")

    def fio_workload(self, pattern: str, block_size: str, size: str,
                     duration: float, num_jobs: int = 4, rw_mix: int = 50):
        """
        Generate I/O workload using fio.

        Args:
            pattern: read, write, randread, randwrite, randrw
            block_size: Block size (e.g., "4k", "64k", "1M")
            size: Total file size (e.g., "1G")
            duration: Duration in seconds
            num_jobs: Number of parallel jobs
            rw_mix: Read/write mix percentage for mixed workloads
        """
        fio_config = f"""
[global]
directory={self.test_dir}
ioengine=libaio
direct=1
size={size}
runtime={int(duration)}
time_based=1
group_reporting=1

[job]
name=load_gen_io
rw={pattern}
bs={block_size}
numjobs={num_jobs}
iodepth=32
rwmixread={rw_mix}
"""

        # Ensure test directory exists (may have been cleaned up by earlier phases)
        os.makedirs(self.test_dir, exist_ok=True)

        config_file = Path(self.test_dir) / "fio_config.fio"
        config_file.write_text(fio_config)

        cmd = ["fio", str(config_file)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(proc)
        return proc

    def sequential_read(self, block_size: str, size: str, duration: float):
        """Sequential read workload."""
        return self.fio_workload("read", block_size, size, duration)

    def sequential_write(self, block_size: str, size: str, duration: float):
        """Sequential write workload."""
        return self.fio_workload("write", block_size, size, duration)

    def random_read(self, block_size: str, size: str, duration: float):
        """Random read workload."""
        return self.fio_workload("randread", block_size, size, duration)

    def random_write(self, block_size: str, size: str, duration: float):
        """Random write workload."""
        return self.fio_workload("randwrite", block_size, size, duration)

    def mixed_rw(self, block_size: str, size: str, duration: float, read_percent: int = 70):
        """Mixed read/write workload."""
        return self.fio_workload("randrw", block_size, size, duration, rw_mix=read_percent)

    def metadata_stress(self, duration: float):
        """Filesystem metadata operations."""
        cmd = [
            "stress-ng",
            "--dentry", "4",
            "--timeout", f"{int(duration)}s",
            "--metrics-brief"
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(proc)
        return proc

    def cleanup(self):
        """Clean up processes and test files."""
        super().cleanup()
        # Clean up test files
        if self.test_dir and os.path.exists(self.test_dir):
            try:
                import shutil
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass


class MixedWorkload(WorkloadExecutor):
    """Generate mixed workloads combining multiple stress types."""

    def __init__(self, test_dir: Optional[str] = None):
        super().__init__()
        self.cpu = CPUWorkload()
        self.memory = MemoryWorkload()
        self.io = IOWorkload(test_dir)

    def cpu_memory_stress(self, cpu_workers: int, mem_size: str,
                         mem_workers: int, duration: float):
        """Combined CPU and memory stress."""
        self.cpu.stress_cpu(cpu_workers, duration, "matrixprod")
        self.memory.stress_memory(mem_size, mem_workers, duration, "all")

    def cpu_io_stress(self, cpu_workers: int, io_pattern: str,
                     block_size: str, duration: float):
        """Combined CPU and I/O stress."""
        self.cpu.stress_cpu(cpu_workers, duration, "cpu")
        self.io.fio_workload(io_pattern, block_size, "1G", duration)

    def full_system_stress(self, duration: float):
        """Stress all system components simultaneously."""
        cpu_count = multiprocessing.cpu_count()

        # CPU stress (use 70% of cores)
        self.cpu.stress_cpu(int(cpu_count * 0.7), duration, "cpu")

        # Memory stress
        self.memory.stress_memory("512M", 4, duration, "all")

        # I/O stress
        self.io.fio_workload("randrw", "64k", "2G", duration, num_jobs=4)

    def web_server_pattern(self, duration: float):
        """Simulate web server workload pattern."""
        # Bursty CPU with I/O
        end_time = time.time() + duration

        while time.time() < end_time and self.running:
            # Burst of requests (CPU + I/O)
            burst_duration = random.uniform(5, 15)
            self.cpu.stress_cpu(4, burst_duration, "cpu")
            self.io.random_read("4k", "500M", burst_duration)
            time.sleep(burst_duration)
            self.cleanup()

            # Quiet period
            quiet_duration = random.uniform(2, 8)
            remaining = end_time - time.time()
            if remaining > 0:
                time.sleep(min(quiet_duration, remaining))

    def database_pattern(self, duration: float):
        """Simulate database workload pattern."""
        # Mixed random I/O with periodic CPU spikes
        self.io.mixed_rw("8k", "4G", duration, read_percent=80)

        # Periodic query processing (CPU spike)
        end_time = time.time() + duration
        while time.time() < end_time and self.running:
            time.sleep(random.uniform(10, 30))
            if time.time() < end_time:
                spike_duration = min(5, end_time - time.time())
                self.cpu.stress_cpu(8, spike_duration, "matrixprod")
                time.sleep(spike_duration)
                self.cpu.cleanup()

    def cleanup(self):
        """Clean up all workload executors."""
        self.cpu.cleanup()
        self.memory.cleanup()
        self.io.cleanup()
        super().cleanup()


class LoadGenerator:
    """Main load generator orchestrator."""

    def __init__(self, args):
        self.args = args
        self.running = True
        self.workload_log = []
        self.start_time = None
        self.mixed = MixedWorkload(args.test_dir)
        self._shutting_down = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for clean shutdown."""
        if self._shutting_down:
            return

        self._shutting_down = True
        print("\n[!] Received interrupt signal. Cleaning up...", file=sys.stderr)
        self.running = False
        self.mixed.cleanup()

    def _log_workload(self, workload_type: str, parameters: Dict, duration: float):
        """Log workload metadata."""
        entry = {
            "timestamp": time.time(),
            "workload_type": workload_type,
            "parameters": parameters,
            "duration": duration
        }
        self.workload_log.append(entry)

        if self.args.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {workload_type} "
                  f"({parameters}) for {duration:.1f}s", file=sys.stderr)

    def _wait_with_progress(self, duration: float, description: str = "Running"):
        """Wait with progress indicator."""
        end_time = time.time() + duration

        while time.time() < end_time and self.running:
            elapsed = time.time() - (end_time - duration)
            remaining = end_time - time.time()
            percent = (elapsed / duration) * 100

            print(f"\r[{description}] Progress: {percent:5.1f}% | "
                  f"Elapsed: {elapsed:6.1f}s | Remaining: {remaining:6.1f}s",
                  end="", file=sys.stderr)

            time.sleep(1)

        print(file=sys.stderr)  # New line after progress

    def run_sequence_comprehensive(self, total_duration: float):
        """Run comprehensive test sequence covering all workload types."""
        print("\n" + "="*60, file=sys.stderr)
        print("Running Comprehensive Workload Sequence", file=sys.stderr)
        print("="*60, file=sys.stderr)

        # Calculate duration for each phase (divide total time)
        phase_duration = total_duration / 15  # 15 different phases

        sequences = [
            # CPU workloads
            ("Idle", lambda: time.sleep(phase_duration)),
            ("CPU 25%", lambda: self._run_cpu_load(25, phase_duration)),
            ("CPU 50%", lambda: self._run_cpu_load(50, phase_duration)),
            ("CPU 75%", lambda: self._run_cpu_load(75, phase_duration)),
            ("CPU 100%", lambda: self._run_cpu_load(100, phase_duration)),

            # Instruction mix
            ("AES Workload", lambda: self._run_instruction_mix("aes", phase_duration)),
            ("SHA Workload", lambda: self._run_instruction_mix("sha", phase_duration)),

            # Memory workloads
            ("Memory L1", lambda: self._run_cache_test("L1", phase_duration)),
            ("Memory L3", lambda: self._run_cache_test("L3", phase_duration)),
            ("Memory Bandwidth", lambda: self._run_memory_bandwidth(phase_duration)),

            # I/O workloads
            ("Sequential Read", lambda: self._run_io_seq_read(phase_duration)),
            ("Random Read", lambda: self._run_io_random_read(phase_duration)),
            ("Mixed I/O", lambda: self._run_io_mixed(phase_duration)),

            # Mixed workloads
            ("CPU+Memory", lambda: self._run_cpu_memory_mixed(phase_duration)),
            ("Full System", lambda: self._run_full_system(phase_duration)),
        ]

        for name, workload_func in sequences:
            if not self.running:
                break

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Phase: {name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            workload_func()

    def run_sequence_cpu_focused(self, total_duration: float):
        """Run CPU-focused test sequence optimized for CPU power prediction."""
        print("\n" + "="*60, file=sys.stderr)
        print("Running CPU-Focused Workload Sequence", file=sys.stderr)
        print("(Optimized for CPU Power Prediction - No I/O workloads)", file=sys.stderr)
        print("="*60, file=sys.stderr)

        # Calculate duration for each phase (14 phases)
        phase_duration = total_duration / 14

        sequences = [
            # Idle baseline
            ("Idle", lambda: time.sleep(phase_duration)),

            # CPU load progression
            ("CPU 25%", lambda: self._run_cpu_load(25, phase_duration)),
            ("CPU 50%", lambda: self._run_cpu_load(50, phase_duration)),
            ("CPU 75%", lambda: self._run_cpu_load(75, phase_duration)),
            ("CPU 100%", lambda: self._run_cpu_load(100, phase_duration)),

            # Instruction mix (different power signatures)
            ("AES Instructions", lambda: self._run_instruction_mix("aes", phase_duration)),
            ("SHA Instructions", lambda: self._run_instruction_mix("sha", phase_duration)),
            ("Matrix Operations", lambda: self._run_instruction_mix("matrixprod", phase_duration)),

            # Cache hierarchy (different memory access patterns)
            ("L1 Cache Test", lambda: self._run_cache_test("L1", phase_duration)),
            ("L2 Cache Test", lambda: self._run_cache_test("L2", phase_duration)),
            ("L3 Cache Test", lambda: self._run_cache_test("L3", phase_duration)),

            # Memory intensive
            ("Memory Bandwidth", lambda: self._run_memory_bandwidth(phase_duration)),

            # Mixed workloads
            ("CPU+Memory Mix", lambda: self._run_cpu_memory_mixed(phase_duration)),
            ("Full System Stress", lambda: self._run_full_system(phase_duration)),
        ]

        for name, workload_func in sequences:
            if not self.running:
                break

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Phase: {name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            workload_func()

    def run_sequence_cpu_random(self, total_duration: float, min_load: int = 2, max_load: int = 95,
                                min_interval: float = 1.0, max_interval: float = 10.0):
        """
        Run random diverse workload sequence including CPU, I/O, memory, and idle states.

        This provides temporal and workload diversity to complement structured phases,
        generating realistic interrupt patterns and system states matching live usage.

        Args:
            total_duration: Total duration in seconds
            min_load: Minimum CPU load percentage (default: 2%)
            max_load: Maximum CPU load percentage (default: 95%)
            min_interval: Minimum duration for each load level in seconds (default: 1.0s)
            max_interval: Maximum duration for each load level in seconds (default: 10.0s)
        """
        print("\n" + "="*60, file=sys.stderr)
        print("Running Random Diverse Workload Sequence", file=sys.stderr)
        print(f"(Workload types: CPU, I/O, Memory, Mixed, Idle)", file=sys.stderr)
        print(f"(Intervals: {min_interval}-{max_interval}s)", file=sys.stderr)
        print("="*60, file=sys.stderr)

        start_time = time.time()
        iteration = 0

        # Workload type distribution (weighted probabilities)
        # 50% CPU-only, 15% I/O, 15% Mixed, 10% Memory, 10% True Idle
        workload_types = (
            ['cpu'] * 50 +
            ['io'] * 15 +
            ['mixed'] * 15 +
            ['memory'] * 10 +
            ['idle'] * 10
        )

        while self.running and (time.time() - start_time) < total_duration:
            # Random workload type
            workload_type = random.choice(workload_types)

            # Random duration for this workload
            load_duration = random.uniform(min_interval, max_interval)

            # Don't exceed total duration
            remaining = total_duration - (time.time() - start_time)
            if remaining <= 0:
                break
            load_duration = min(load_duration, remaining)

            iteration += 1
            elapsed = time.time() - start_time

            # Execute workload based on type
            if workload_type == 'cpu':
                # Random CPU load
                target_load = random.randint(min_load, max_load)
                print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: CPU {target_load}% for {load_duration:.1f}s",
                      file=sys.stderr)
                self._log_workload("random_cpu", {"load": target_load}, load_duration)
                self._run_cpu_load(target_load, load_duration)

            elif workload_type == 'io':
                # Random I/O workload - mix of stress-ng and REAL I/O patterns
                io_type = random.choice(['hdd', 'sock', 'real_file_index', 'real_sync'])
                if io_type == 'hdd':
                    # Disk I/O (stress-ng)
                    workers = random.randint(1, 4)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Disk I/O ({workers} workers) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._log_workload("random_io_hdd", {"workers": workers}, load_duration)
                    self._run_stress_ng_workload(['--hdd', str(workers)], load_duration, "Disk I/O")
                elif io_type == 'sock':
                    # Network socket stress
                    workers = random.randint(2, 8)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Network sockets ({workers} workers) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._log_workload("random_io_sock", {"workers": workers}, load_duration)
                    self._run_stress_ng_workload(['--sock', str(workers)], load_duration, "Network")
                elif io_type == 'real_file_index':
                    # REAL I/O: File indexing with checksums (creates actual iowait)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: File indexing (REAL I/O) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._run_real_io_file_indexing(load_duration)
                else:  # real_sync
                    # REAL I/O: Sync writes (creates actual iowait)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Sync writes (REAL I/O) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._run_real_io_sync_writes(load_duration)

            elif workload_type == 'mixed':
                # Mixed CPU + I/O or CPU + Memory
                mix_type = random.choice(['cpu_io', 'cpu_mem'])
                if mix_type == 'cpu_io':
                    # CPU + I/O
                    cpu_workers = random.randint(1, 4)
                    io_workers = random.randint(1, 2)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Mixed CPU+I/O ({cpu_workers}+{io_workers} workers) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._log_workload("random_mixed_cpu_io", {"cpu": cpu_workers, "io": io_workers}, load_duration)
                    self._run_stress_ng_workload(['--cpu', str(cpu_workers), '--io', str(io_workers)], load_duration, "CPU+I/O")
                else:
                    # CPU + Memory
                    cpu_workers = random.randint(1, 4)
                    vm_workers = random.randint(1, 2)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Mixed CPU+Memory ({cpu_workers}+{vm_workers} workers) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._log_workload("random_mixed_cpu_mem", {"cpu": cpu_workers, "vm": vm_workers}, load_duration)
                    self._run_stress_ng_workload(['--cpu', str(cpu_workers), '--vm', str(vm_workers), '--vm-bytes', '128M'], load_duration, "CPU+Memory")

            elif workload_type == 'memory':
                # Memory stress (page faults, cache misses)
                vm_workers = random.randint(1, 4)
                vm_bytes = random.choice(['64M', '128M', '256M', '512M'])
                print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Memory stress ({vm_workers} workers, {vm_bytes}) for {load_duration:.1f}s",
                      file=sys.stderr)
                self._log_workload("random_memory", {"workers": vm_workers, "bytes": vm_bytes}, load_duration)
                self._run_stress_ng_workload(['--vm', str(vm_workers), '--vm-bytes', vm_bytes], load_duration, "Memory")

            elif workload_type == 'idle':
                # True idle (no stress) - captures baseline interrupts
                print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: True idle for {load_duration:.1f}s",
                      file=sys.stderr)
                self._log_workload("random_idle", {}, load_duration)
                time.sleep(load_duration)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Random sequence complete: {iteration} iterations in {elapsed:.1f}s", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    def run_sequence_io_realistic(self, total_duration: float, min_load: int = 2, max_load: int = 95,
                                   min_interval: float = 1.0, max_interval: float = 10.0):
        """
        Run random workload sequence with REALISTIC I/O and LIGHT CPU patterns.

        This includes real application workloads:
        - /proc monitoring (CRITICAL: light CPU syscalls, low power) - 33% of real_io
        - File indexing (high CPU% in iowait, low power)
        - Sync writes (database-like patterns)
        - Archive operations (mixed CPU compute + I/O)
        - File processing (exporter-like patterns)

        Higher I/O probability than cpu-random to better train the model
        to distinguish HEAVY CPU work (stress-ng) from LIGHT CPU work (syscalls, parsing).

        Args:
            total_duration: Total duration in seconds
            min_load: Minimum CPU load percentage (default: 2%)
            max_load: Maximum CPU load percentage (default: 95%)
            min_interval: Minimum duration for each load level in seconds (default: 1.0s)
            max_interval: Maximum duration for each load level in seconds (default: 10.0s)
        """
        print("\n" + "="*60, file=sys.stderr)
        print("Running I/O-Realistic + Light CPU Workload Sequence", file=sys.stderr)
        print(f"(Workload types: 30% CPU, 30% REAL I/O + /proc monitoring, 20% Mixed, 10% Memory, 10% Idle)", file=sys.stderr)
        print(f"(Intervals: {min_interval}-{max_interval}s)", file=sys.stderr)
        print("="*60, file=sys.stderr)

        start_time = time.time()
        iteration = 0

        # Workload type distribution (weighted probabilities)
        # INCREASED I/O probability, added REAL I/O patterns
        workload_types = (
            ['cpu'] * 30 +           # 30% CPU-only (reduced from 50%)
            ['real_io'] * 30 +       # 30% REAL I/O (NEW - file ops, sync writes)
            ['mixed'] * 20 +         # 20% Mixed (increased from 15%)
            ['memory'] * 10 +        # 10% Memory
            ['idle'] * 10            # 10% True Idle
        )

        while self.running and (time.time() - start_time) < total_duration:
            # Random workload type
            workload_type = random.choice(workload_types)

            # Random duration for this workload
            load_duration = random.uniform(min_interval, max_interval)

            # Don't exceed total duration
            remaining = total_duration - (time.time() - start_time)
            if remaining <= 0:
                break
            load_duration = min(load_duration, remaining)

            iteration += 1
            elapsed = time.time() - start_time

            # Execute workload based on type
            if workload_type == 'cpu':
                # Random CPU load
                target_load = random.randint(min_load, max_load)
                print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: CPU {target_load}% for {load_duration:.1f}s",
                      file=sys.stderr)
                self._log_workload("random_cpu", {"load": target_load}, load_duration)
                self._run_cpu_load(target_load, load_duration)

            elif workload_type == 'real_io':
                # REAL workloads: Mix of I/O-bound and light CPU work
                # Use LONGER durations (30-90s) for sustained patterns
                workload_duration = random.uniform(30, 90)  # Longer than other workloads
                workload_duration = min(workload_duration, remaining)  # Don't exceed total duration

                # IMPORTANT: Include /proc monitoring (the CRITICAL light CPU pattern)
                io_type = random.choice(['file_indexing', 'sync_writes', 'archive_ops', 'file_processing', 'proc_monitoring', 'proc_monitoring'])
                # proc_monitoring appears twice for 33% probability (2/6) - this is the key pattern!

                if io_type == 'file_indexing':
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: File indexing for {workload_duration:.1f}s",
                          file=sys.stderr)
                    self._run_real_io_file_indexing(workload_duration)
                elif io_type == 'sync_writes':
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Sync writes for {workload_duration:.1f}s",
                          file=sys.stderr)
                    self._run_real_io_sync_writes(workload_duration)
                elif io_type == 'archive_ops':
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Archive ops for {workload_duration:.1f}s",
                          file=sys.stderr)
                    self._run_real_io_archive_ops(workload_duration)
                elif io_type == 'file_processing':
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: File processing for {workload_duration:.1f}s",
                          file=sys.stderr)
                    self._run_real_io_file_processing(workload_duration)
                else:  # proc_monitoring
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: /proc monitoring for {workload_duration:.1f}s",
                          file=sys.stderr)
                    self._run_proc_monitoring_pattern(workload_duration)

            elif workload_type == 'mixed':
                # Mixed CPU + I/O or CPU + Memory
                mix_type = random.choice(['cpu_io', 'cpu_mem'])
                if mix_type == 'cpu_io':
                    # CPU + I/O
                    cpu_workers = random.randint(1, 4)
                    io_workers = random.randint(1, 2)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Mixed CPU+I/O ({cpu_workers}+{io_workers} workers) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._log_workload("random_mixed_cpu_io", {"cpu": cpu_workers, "io": io_workers}, load_duration)
                    self._run_stress_ng_workload(['--cpu', str(cpu_workers), '--io', str(io_workers)], load_duration, "CPU+I/O")
                else:
                    # CPU + Memory
                    cpu_workers = random.randint(1, 4)
                    vm_workers = random.randint(1, 2)
                    print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Mixed CPU+Memory ({cpu_workers}+{vm_workers} workers) for {load_duration:.1f}s",
                          file=sys.stderr)
                    self._log_workload("random_mixed_cpu_mem", {"cpu": cpu_workers, "vm": vm_workers}, load_duration)
                    self._run_stress_ng_workload(['--cpu', str(cpu_workers), '--vm', str(vm_workers), '--vm-bytes', '128M'], load_duration, "CPU+Memory")

            elif workload_type == 'memory':
                # Memory stress (page faults, cache misses)
                vm_workers = random.randint(1, 4)
                vm_bytes = random.choice(['64M', '128M', '256M', '512M'])
                print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: Memory stress ({vm_workers} workers, {vm_bytes}) for {load_duration:.1f}s",
                      file=sys.stderr)
                self._log_workload("random_memory", {"workers": vm_workers, "bytes": vm_bytes}, load_duration)
                self._run_stress_ng_workload(['--vm', str(vm_workers), '--vm-bytes', vm_bytes], load_duration, "Memory")

            elif workload_type == 'idle':
                # True idle (no stress) - captures baseline interrupts
                print(f"\n[Iteration {iteration}] t={elapsed:.1f}s: True idle for {load_duration:.1f}s",
                      file=sys.stderr)
                self._log_workload("random_idle", {}, load_duration)
                time.sleep(load_duration)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"I/O-Realistic sequence complete: {iteration} iterations in {elapsed:.1f}s", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    def _run_stress_ng_workload(self, stress_args: List[str], duration: float, description: str):
        """
        Run stress-ng with custom arguments.

        Args:
            stress_args: List of stress-ng arguments (e.g., ['--hdd', '2', '--io', '1'])
            duration: Duration in seconds
            description: Description for progress display
        """
        cmd = ["stress-ng"] + stress_args + ["--timeout", f"{int(duration)}s", "--metrics-brief"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.mixed.processes.append(proc)
        self._wait_with_progress(duration, description)
        self.mixed.cleanup()

    def _run_cpu_load(self, percent: int, duration: float):
        """Run CPU load at specific percentage."""
        total_cores = multiprocessing.cpu_count()

        # Calculate how many cores to use and at what percentage
        # For overall percent utilization, we can either:
        # 1. Use all cores at percent% each, OR
        # 2. Use (percent/100 * total_cores) cores at 100%
        # We'll use approach 1 for more uniform load distribution
        cores = total_cores
        per_core_percent = percent

        self._log_workload("cpu_load", {"percent": percent, "cores": cores, "per_core_percent": per_core_percent}, duration)

        proc = self.mixed.cpu.variable_cpu_load(per_core_percent, duration, cores)
        self._wait_with_progress(duration, f"CPU {percent}%")
        self.mixed.cleanup()

    def _run_instruction_mix(self, instr_type: str, duration: float):
        """Run specific instruction type workload."""
        workers = multiprocessing.cpu_count()
        self._log_workload("instruction_mix", {"type": instr_type, "workers": workers}, duration)

        self.mixed.cpu.instruction_mix(instr_type, workers, duration)
        self._wait_with_progress(duration, f"{instr_type.upper()} Instructions")
        self.mixed.cleanup()

    def _run_cache_test(self, cache_level: str, duration: float):
        """Run cache hierarchy test."""
        self._log_workload("cache_test", {"level": cache_level}, duration)

        self.mixed.memory.cache_hierarchy_test(cache_level, duration)
        self._wait_with_progress(duration, f"{cache_level} Cache Test")
        self.mixed.cleanup()

    def _run_memory_bandwidth(self, duration: float):
        """Run memory bandwidth saturation test."""
        self._log_workload("memory_bandwidth", {"workers": multiprocessing.cpu_count()}, duration)

        self.mixed.memory.memory_bandwidth("1G", duration)
        self._wait_with_progress(duration, "Memory Bandwidth")
        self.mixed.cleanup()

    def _run_io_seq_read(self, duration: float):
        """Run sequential I/O read."""
        self._log_workload("io_sequential_read", {"block_size": "1M"}, duration)

        self.mixed.io.sequential_read("1M", "2G", duration)
        self._wait_with_progress(duration, "Sequential Read")
        self.mixed.cleanup()

    def _run_io_random_read(self, duration: float):
        """Run random I/O read."""
        self._log_workload("io_random_read", {"block_size": "4k"}, duration)

        self.mixed.io.random_read("4k", "2G", duration)
        self._wait_with_progress(duration, "Random Read")
        self.mixed.cleanup()

    def _run_io_mixed(self, duration: float):
        """Run mixed I/O workload."""
        self._log_workload("io_mixed", {"block_size": "64k", "read_percent": 70}, duration)

        self.mixed.io.mixed_rw("64k", "2G", duration, read_percent=70)
        self._wait_with_progress(duration, "Mixed I/O")
        self.mixed.cleanup()

    def _run_cpu_memory_mixed(self, duration: float):
        """Run combined CPU and memory workload."""
        cpu_workers = multiprocessing.cpu_count() // 2
        self._log_workload("cpu_memory_mixed",
                          {"cpu_workers": cpu_workers, "mem_size": "1G"}, duration)

        self.mixed.cpu_memory_stress(cpu_workers, "1G", 4, duration)
        self._wait_with_progress(duration, "CPU+Memory")
        self.mixed.cleanup()

    def _run_full_system(self, duration: float):
        """Run full system stress."""
        self._log_workload("full_system_stress", {}, duration)

        self.mixed.full_system_stress(duration)
        self._wait_with_progress(duration, "Full System")
        self.mixed.cleanup()

    def _run_real_io_file_indexing(self, duration: float):
        """
        Real I/O workload: File indexing with checksumming.

        This creates sustained I/O wait as processes read files from disk,
        compute checksums, and wait for disk reads. High CPU% but low power
        because CPU is halted during I/O wait.
        """
        self._log_workload("real_io_file_indexing", {}, duration)

        # Find files and compute checksums (creates iowait)
        # Use /usr as target (large, diverse files, won't fill disk)
        cmd = f"timeout {int(duration)}s find /usr -type f -exec sha256sum {{}} \\; > /dev/null 2>&1"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.mixed.processes.append(proc)
        self._wait_with_progress(duration, "File Indexing (Real I/O)")
        self.mixed.cleanup()

    def _run_real_io_archive_ops(self, duration: float):
        """
        Real I/O workload: Archive compression/decompression.

        Creates bursty CPU (compression) + I/O wait (disk writes).
        Realistic workload pattern with mixed compute and I/O.
        """
        self._log_workload("real_io_archive_ops", {}, duration)

        # Create temp directory for archive operations
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="load_gen_archive_")

        try:
            # Create some test files to compress
            test_data_dir = Path(temp_dir) / "test_data"
            test_data_dir.mkdir()

            # Generate test files (mix of compressible and random data)
            for i in range(10):
                test_file = test_data_dir / f"file_{i}.dat"
                # 10MB each, total 100MB
                subprocess.run(
                    f"dd if=/dev/urandom of={test_file} bs=1M count=10 2>/dev/null",
                    shell=True, timeout=30
                )

            # Repeatedly compress and decompress
            archive_file = Path(temp_dir) / "archive.tar.gz"
            end_time = time.time() + duration

            while time.time() < end_time and self.running:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break

                # Compress (CPU + I/O write)
                compress_time = min(remaining / 2, 15)
                cmd = f"timeout {int(compress_time)}s tar -czf {archive_file} -C {test_data_dir} . 2>/dev/null"
                proc = subprocess.Popen(cmd, shell=True)
                proc.wait()

                if time.time() >= end_time or not self.running:
                    break

                # Decompress (I/O read + CPU)
                decompress_time = min(end_time - time.time(), 10)
                if decompress_time > 0:
                    cmd = f"timeout {int(decompress_time)}s tar -xzf {archive_file} -C {temp_dir} 2>/dev/null"
                    proc = subprocess.Popen(cmd, shell=True)
                    proc.wait()

                # Remove extracted files
                subprocess.run(f"rm -rf {temp_dir}/file_*.dat 2>/dev/null", shell=True)

        finally:
            # Cleanup
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        print(file=sys.stderr)  # New line after progress

    def _run_real_io_sync_writes(self, duration: float):
        """
        Real I/O workload: Synchronous write operations.

        Forces disk writes with fsync, creating sustained I/O wait.
        Simulates database-like write patterns.
        """
        self._log_workload("real_io_sync_writes", {}, duration)

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="load_gen_syncio_")

        try:
            # Python script for sync writes
            sync_script = f"""
import os
import time
end_time = time.time() + {duration}
iteration = 0
while time.time() < end_time:
    filepath = '{temp_dir}/sync_file_{{:04d}}.dat'.format(iteration % 100)
    with open(filepath, 'wb') as f:
        # Write 1MB with sync
        f.write(os.urandom(1024 * 1024))
        f.flush()
        os.fsync(f.fileno())  # Force disk write (creates iowait)
    iteration += 1
    if iteration % 10 == 0:
        # Periodic cleanup
        for i in range(max(0, iteration - 50), iteration - 10):
            try:
                os.remove('{temp_dir}/sync_file_{{:04d}}.dat'.format(i % 100))
            except:
                pass
"""
            proc = subprocess.Popen(
                ["python3", "-c", sync_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.mixed.processes.append(proc)
            self._wait_with_progress(duration, "Sync Writes (Real I/O)")
            self.mixed.cleanup()

        finally:
            # Cleanup
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_proc_monitoring_pattern(self, duration: float):
        """
        /proc monitoring workload (Prometheus exporter pattern).

        Simulates reading /proc/[pid]/* for all running processes:
        - High system CPU% from syscalls (open/read/close thousands of files)
        - Light parsing in userspace (string splitting, number parsing)
        - NO disk I/O (procfs is in-memory, kernel-generated)
        - NO iowait (data generated on-demand by kernel)

        Power characteristics:
        - At 5-15% CPU: ~26-27W (much lower than stress-ng at same CPU%)
        - This is LIGHT CPU WORK (syscalls + parsing)
        - vs stress-ng HEAVY WORK (compute/memory stress)

        This is the CRITICAL pattern that caused model failure on real workload.
        Real exporter reads /proc periodically, creating CPU spikes with low power.
        """
        self._log_workload("proc_monitoring", {}, duration)

        # Python script that mimics Prometheus process exporter
        monitoring_script = f"""
import os
import time
import glob

end_time = time.time() + {duration}
iteration = 0

while time.time() < end_time:
    iteration += 1
    start = time.time()

    # Get all process PIDs (like walking /proc)
    proc_data = {{}}

    try:
        # Read /proc for all processes
        for pid_dir in glob.glob('/proc/[0-9]*'):
            try:
                pid = os.path.basename(pid_dir)

                # Read stat file (process state, CPU times)
                with open(f'{{pid_dir}}/stat', 'r') as f:
                    stat = f.read().split()
                    # Parse key fields (mimics exporter parsing)
                    if len(stat) >= 15:
                        proc_data[pid] = {{
                            'utime': int(stat[13]),
                            'stime': int(stat[14]),
                            'state': stat[2]
                        }}

                # Read status file (memory, threads)
                with open(f'{{pid_dir}}/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:') or line.startswith('Threads:'):
                            # Parse values (string operations)
                            parts = line.split()
                            if len(parts) >= 2:
                                proc_data[pid][parts[0].rstrip(':')] = parts[1]

                # Read cmdline
                try:
                    with open(f'{{pid_dir}}/cmdline', 'r') as f:
                        cmdline = f.read().replace('\\x00', ' ').strip()
                        proc_data[pid]['cmdline'] = cmdline[:100]  # Truncate
                except:
                    pass

            except (FileNotFoundError, ProcessLookupError, PermissionError):
                # Process may have exited or we don't have permission
                continue

    except Exception as e:
        pass

    # Compute summary metrics (mimics exporter aggregation)
    total_procs = len(proc_data)

    elapsed = time.time() - start

    # Sleep to simulate scrape interval (like Prometheus exporter)
    # Typical scrape: 1-5 seconds
    if time.time() < end_time:
        sleep_time = max(0.1, 2.0 - elapsed)  # Target 2s scrape interval
        time.sleep(min(sleep_time, end_time - time.time()))
"""
        proc = subprocess.Popen(
            ["python3", "-c", monitoring_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.mixed.processes.append(proc)
        self._wait_with_progress(duration, "/proc Monitoring (Light CPU)")
        self.mixed.cleanup()

    def _run_real_io_file_processing(self, duration: float):
        """
        Real I/O workload: File reading + processing.

        Mimics exporter/monitoring workload pattern:
        - Read many files from filesystem
        - Parse/process content (regex, JSON parsing)
        - Creates periodic CPU spikes + iowait, but low power

        This is the key pattern that failed in real workload testing:
        high CPU% (due to file I/O + processing) but low power
        (because CPU is halted during disk reads).
        """
        self._log_workload("real_io_file_processing", {}, duration)

        # Python script that mimics exporter workload
        processing_script = f"""
import os
import re
import time
import json

end_time = time.time() + {duration}
iteration = 0

# Target directory with many files (system files)
base_dirs = ['/usr/share/doc', '/var/log', '/etc']

while time.time() < end_time:
    iteration += 1

    # Pick a directory to scan
    base_dir = base_dirs[iteration % len(base_dirs)]

    files_processed = 0
    data = {{}}

    # Walk directory tree and read files
    for root, dirs, files in os.walk(base_dir):
        if time.time() >= end_time:
            break

        for filename in files[:50]:  # Limit files per iteration
            if time.time() >= end_time:
                break

            filepath = os.path.join(root, filename)

            try:
                # Read file (creates I/O wait)
                with open(filepath, 'r', errors='ignore') as f:
                    content = f.read(4096)  # Read first 4KB

                # Process content (CPU work)
                # Simulate parsing: count lines, search patterns
                lines = content.count('\\n')
                words = len(content.split())

                # Simulate regex processing
                matches = len(re.findall(r'[a-zA-Z0-9]+', content))

                data[filename] = {{
                    'lines': lines,
                    'words': words,
                    'matches': matches,
                    'size': len(content)
                }}

                files_processed += 1

                if files_processed >= 100:
                    break

            except (IOError, OSError, UnicodeDecodeError):
                continue

        if files_processed >= 100:
            break

    # Simulate metric export (small CPU burst)
    metrics_json = json.dumps(data)

    # Small sleep between iterations (like exporter polling interval)
    if time.time() < end_time:
        time.sleep(0.5)
"""
        proc = subprocess.Popen(
            ["python3", "-c", processing_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.mixed.processes.append(proc)
        self._wait_with_progress(duration, "File Processing (Real I/O)")
        self.mixed.cleanup()

    def run_cpu_ramp(self, levels: List[int], duration_per_level: float):
        """Ramp CPU load through specified levels."""
        print("\n" + "="*60, file=sys.stderr)
        print("Running CPU Ramp Pattern", file=sys.stderr)
        print(f"Levels: {levels}", file=sys.stderr)
        print("="*60, file=sys.stderr)

        for level in levels:
            if not self.running:
                break
            self._run_cpu_load(level, duration_per_level)

    def run_specific_workload(self, workload_type: str, duration: float):
        """Run a specific workload type."""
        workload_map = {
            "cpu-light": lambda: self._run_cpu_load(25, duration),
            "cpu-medium": lambda: self._run_cpu_load(50, duration),
            "cpu-heavy": lambda: self._run_cpu_load(100, duration),
            "memory-bandwidth": lambda: self._run_memory_bandwidth(duration),
            "memory-random": lambda: self.mixed.memory.random_access("2G", duration),
            "io-seq-read": lambda: self._run_io_seq_read(duration),
            "io-seq-write": lambda: self.mixed.io.sequential_write("1M", "2G", duration),
            "io-random-read": lambda: self._run_io_random_read(duration),
            "io-random-write": lambda: self.mixed.io.random_write("4k", "2G", duration),
            "mixed-cpu-mem": lambda: self._run_cpu_memory_mixed(duration),
            "mixed-cpu-io": lambda: self.mixed.cpu_io_stress(4, "randrw", "64k", duration),
            "full-system": lambda: self._run_full_system(duration),
            "web-server": lambda: self.mixed.web_server_pattern(duration),
            "database": lambda: self.mixed.database_pattern(duration),
        }

        if workload_type in workload_map:
            print(f"\n[*] Running workload: {workload_type} for {duration}s", file=sys.stderr)
            self._log_workload(workload_type, {}, duration)
            workload_map[workload_type]()
            self._wait_with_progress(duration, workload_type)
            self.mixed.cleanup()
        else:
            print(f"[!] Unknown workload type: {workload_type}", file=sys.stderr)
            print(f"[*] Available types: {', '.join(workload_map.keys())}", file=sys.stderr)

    def save_metadata(self, filename: str):
        """Save workload metadata to JSON file."""
        metadata = {
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_duration": time.time() - self.start_time if self.start_time else 0,
            "cpu_count": multiprocessing.cpu_count(),
            "workloads": self.workload_log
        }

        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[*] Workload metadata saved to: {filename}", file=sys.stderr)

    def run(self):
        """Main execution logic."""
        self.start_time = time.time()

        try:
            if self.args.sequence == "comprehensive":
                self.run_sequence_comprehensive(self.args.duration)

            elif self.args.sequence == "cpu-focused":
                self.run_sequence_cpu_focused(self.args.duration)

            elif self.args.sequence == "cpu-random":
                self.run_sequence_cpu_random(
                    self.args.duration,
                    min_load=self.args.random_min_load,
                    max_load=self.args.random_max_load,
                    min_interval=self.args.random_min_interval,
                    max_interval=self.args.random_max_interval
                )

            elif self.args.sequence == "io-realistic":
                self.run_sequence_io_realistic(
                    self.args.duration,
                    min_load=self.args.random_min_load,
                    max_load=self.args.random_max_load,
                    min_interval=self.args.random_min_interval,
                    max_interval=self.args.random_max_interval
                )

            elif self.args.pattern == "cpu-ramp":
                levels = [int(x) for x in self.args.levels.split(",")]
                duration_per_level = self.args.duration / len(levels)
                self.run_cpu_ramp(levels, duration_per_level)

            elif self.args.workload:
                self.run_specific_workload(self.args.workload, self.args.duration)

            else:
                print("[!] No workload specified. Use --sequence, --pattern, or --workload",
                      file=sys.stderr)
                return

        finally:
            self.mixed.cleanup()

            # Save metadata if requested
            if self.args.metadata_output:
                self.save_metadata(self.args.metadata_output)

            print("\n" + "="*60, file=sys.stderr)
            print("Load Generation Complete", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Total workloads executed: {len(self.workload_log)}", file=sys.stderr)
            print(f"Total duration: {time.time() - self.start_time:.1f}s", file=sys.stderr)
            print("="*60, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="System Load Generator for CPU Power Prediction Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  CPU-focused test sequence (recommended for CPU power prediction):
    %(prog)s --sequence cpu-focused --duration 3600

  Comprehensive test sequence (includes I/O workloads):
    %(prog)s --sequence comprehensive --duration 3600

  CPU ramp pattern:
    %(prog)s --pattern cpu-ramp --levels 25,50,75,100 --duration 600

  Random CPU workload (temporal diversity):
    %(prog)s --sequence cpu-random --duration 3600
    %(prog)s --sequence cpu-random --duration 1800 --random-min-load 5 --random-max-load 90

  Specific workload:
    %(prog)s --workload memory-bandwidth --duration 300
    %(prog)s --workload web-server --duration 600

  With metadata logging:
    %(prog)s --sequence comprehensive --duration 3600 --metadata-output workload_log.json

Available workload types:
  CPU: cpu-light, cpu-medium, cpu-heavy
  Memory: memory-bandwidth, memory-random
  I/O: io-seq-read, io-seq-write, io-random-read, io-random-write
  Mixed: mixed-cpu-mem, mixed-cpu-io, full-system, web-server, database
        """
    )

    parser.add_argument(
        "--sequence",
        choices=["comprehensive", "cpu-focused", "cpu-random", "io-realistic"],
        help="Run predefined test sequence (comprehensive=all workloads, cpu-focused=CPU/memory only, cpu-random=random mixed loads, io-realistic=realistic I/O workloads)"
    )

    parser.add_argument(
        "--pattern",
        choices=["cpu-ramp"],
        help="Run specific pattern"
    )

    parser.add_argument(
        "--levels",
        type=str,
        help="Comma-separated levels for patterns (e.g., '25,50,75,100')"
    )

    parser.add_argument(
        "--workload",
        type=str,
        help="Specific workload type to run"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=300,
        help="Total duration in seconds (default: 300)"
    )

    parser.add_argument(
        "--test-dir",
        type=str,
        help="Directory for I/O test files (default: temp dir)"
    )

    parser.add_argument(
        "--metadata-output",
        type=str,
        help="Save workload metadata to JSON file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    # Random workload parameters
    parser.add_argument(
        "--random-min-load",
        type=int,
        default=2,
        help="Minimum CPU load for random sequence (default: 2%%)"
    )

    parser.add_argument(
        "--random-max-load",
        type=int,
        default=95,
        help="Maximum CPU load for random sequence (default: 95%%)"
    )

    parser.add_argument(
        "--random-min-interval",
        type=float,
        default=1.0,
        help="Minimum interval duration for random sequence in seconds (default: 1.0s)"
    )

    parser.add_argument(
        "--random-max-interval",
        type=float,
        default=10.0,
        help="Maximum interval duration for random sequence in seconds (default: 10.0s)"
    )

    args = parser.parse_args()

    # Validation
    if not any([args.sequence, args.pattern, args.workload]):
        parser.print_help()
        print("\n[!] Error: Must specify --sequence, --pattern, or --workload", file=sys.stderr)
        sys.exit(1)

    if args.pattern == "cpu-ramp" and not args.levels:
        print("[!] Error: --pattern cpu-ramp requires --levels", file=sys.stderr)
        sys.exit(1)

    # Check for required tools
    # stress-ng is always required
    # fio is only required for comprehensive sequence or I/O workloads
    required_tools = ["stress-ng"]

    # Add fio requirement if using I/O workloads
    needs_fio = (
        args.sequence == "comprehensive" or
        (args.workload and "io" in args.workload.lower())
    )

    if needs_fio:
        required_tools.append("fio")

    missing_tools = []
    for tool in required_tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)

    if missing_tools:
        print(f"[!] Error: Missing required tools: {', '.join(missing_tools)}", file=sys.stderr)
        if "fio" in missing_tools and "stress-ng" in missing_tools:
            print("[*] Install with: sudo apt-get install stress-ng fio", file=sys.stderr)
        elif "fio" in missing_tools:
            print("[*] Install with: sudo apt-get install fio", file=sys.stderr)
        else:
            print("[*] Install with: sudo apt-get install stress-ng", file=sys.stderr)
        sys.exit(1)

    try:
        generator = LoadGenerator(args)
        generator.run()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
