# DECODE-RAPL v2 Workload Types

Understanding the workload patterns in the training data.

## Overview

The training data consists of 6 synthetic workload types plus idle baseline, generated using `stress-ng` stressors. These workloads create diverse CPU states that span the full power spectrum from idle to maximum consumption.

## Workload to CPU Metric Mapping

**Key insight:** The workload name (cpu, sys, io) refers to the stress-ng stressor type, NOT the CPU time metric!

| stress-ng Stressor | Primary Metric | Secondary Metrics | Power Characteristics |
|-------------------|----------------|-------------------|----------------------|
| `--cpu` (compute) | user% | low system%, low iowait% | **HIGH** power (70-90W) |
| `--syscall` | system% | low user%, high ctx_switches | **LOW** power (28-35W) |
| `--io` | iowait% | system% (during I/O calls) | **LOW-MEDIUM** (30-40W) |
| `--pipe` | ctx_switches | system% (pipe operations) | **MEDIUM** (40-50W) |
| `--vm` (memory) | user% | moderate ctx_switches | **MEDIUM-HIGH** (50-70W) |
| `--cache` | user% | low system%, high thrashing | **HIGH** (60-80W) |
| Idle baseline | all near 0% | baseline ctx_switches ~2K/s | **LOW** (23-27W) |

## Detailed Workload Descriptions

### 0. Idle Baseline (cpu0-sys0-io0-pipe0-vm0-cache0)

**What it does:** Nothing - no stress-ng workers running

**CPU Metrics:**
- user%: 0-1%
- system%: 0-1%
- iowait%: 0%
- ctx_switches: 2,000-3,000/sec (OS background tasks)

**Power:** 23-27W (base system power)

**Why important:**
- Establishes baseline power consumption
- Critical for model to learn "zero load = base power"
- Missing from v1 training data

**Example CSV:** `run_1_of_2025-cpu0-sys0-io0-pipe0-vm0-cache0.csv`

### 1. CPU Compute (`--cpu`)

**What it does:** Floating-point arithmetic operations (sqrt, sin, cos)

**stress-ng command:** `stress-ng --cpu N`

**CPU Metrics:**
- user%: 70-100% (pure user-space compute)
- system%: 0-5%
- iowait%: 0%
- ctx_switches: Normal (~3K/s)

**Power:** 70-90W (**highest power** for given CPU%)

**Why important:**
- Pure compute workload
- Highest power per CPU% ratio
- Represents CPU-intensive applications (encoding, ML inference)

**Example:** `run_500_of_2025-cpu16-sys0-io0-pipe0-vm0-cache0.csv`

### 2. Syscalls (`--syscall`)

**What it does:** Rapidly executes lightweight system calls

**stress-ng command:** `stress-ng --syscall N --syscall-method fast10`

**CPU Metrics:**
- user%: 0-10%
- system%: 5-15% (note: low due to multi-core averaging)
- iowait%: 0%
- ctx_switches: 5,000-15,000/sec

**Power:** 28-35W (**lowest power for given CPU activity**)

**Why important:**
- **This is THE critical pattern missing from v1!**
- Shows "high activity, low power" (syscalls are fast, don't burn energy)
- Represents I/O-bound applications (web servers, databases)
- Model must learn: system% ≠ high power

**Note on system%:** Even pure syscall workloads show low system% (~5-15%) because:
- Fast syscalls execute in nanoseconds
- CPU is idle most of the time waiting
- Averaged across all cores, the percentage is diluted

**Example:** `run_800_of_2025-cpu0-sys16-io0-pipe0-vm0-cache0.csv`

### 3. I/O Operations (`--io`)

**What it does:** File I/O operations (write, read, sync)

**stress-ng command:** `stress-ng --io N`

**CPU Metrics:**
- user%: 0-10%
- system%: 3-10% (kernel I/O handling)
- iowait%: 10-40% (waiting for I/O)
- ctx_switches: 3,000-8,000/sec

**Power:** 30-45W (low to medium)

**Why important:**
- Creates iowait% patterns
- CPU is idle during I/O waits
- Represents disk-bound workloads

**Example:** `run_300_of_2025-cpu0-sys0-io8-pipe0-vm0-cache0.csv`

### 4. Pipe Operations (`--pipe`)

**What it does:** Inter-process communication via pipes

**stress-ng command:** `stress-ng --pipe N`

**CPU Metrics:**
- user%: 10-30%
- system%: 10-30% (pipe system calls)
- iowait%: 0%
- ctx_switches: 10,000-50,000/sec (**high!**)

**Power:** 40-55W (medium)

**Why important:**
- High context switch patterns
- Represents multi-process communication
- Tests model's ability to use ctx_switches feature

**Example:** `run_1200_of_2025-cpu0-sys0-io0-pipe8-vm0-cache0.csv`

### 5. Memory Operations (`--vm`)

**What it does:** Memory allocation, access, and modification

**stress-ng command:** `stress-ng --vm N --vm-bytes 1G`

**CPU Metrics:**
- user%: 40-80% (memory operations in user space)
- system%: 1-5%
- iowait%: 0-5%
- ctx_switches: 3,000-10,000/sec

**Power:** 50-75W (medium-high)

**Why important:**
- Memory-bound workload (different from compute)
- Shows that user% can mean different things
- cpu0-vm8: 40% user% @ 55W
- cpu8-vm0: 40% user% @ 75W
- Model must distinguish using other features

**Example:** `run_1500_of_2025-cpu0-sys0-io0-pipe0-vm8-cache0.csv`

### 6. Cache Thrashing (`--cache`)

**What it does:** L3 cache thrashing (forces cache misses)

**stress-ng command:** `stress-ng --cache N`

**CPU Metrics:**
- user%: 50-95% (cache access is user-space)
- system%: 0-3%
- iowait%: 0%
- ctx_switches: 3,000-8,000/sec

**Power:** 60-85W (high, but varies with cache efficiency)

**Why important:**
- Memory-hierarchy bound (not pure compute)
- Can exceed core count (cache thrashing doesn't need 1:1 core mapping)
- Represents applications with poor cache locality

**Example:** `run_2000_of_2025-cpu0-sys0-io0-pipe0-vm0-cache20.csv`

## Mixed Workloads

The combinatorial approach creates **mixed workloads** that combine multiple stressor types:

### Example: cpu4-sys4-io4-pipe0-vm0-cache0

- 4 CPU compute workers (high power)
- 4 syscall workers (low power)
- 4 I/O workers (low power)

**Result:** Moderate user%, moderate system%, some iowait%, mixed power (~50-60W)

**Why important:** Real applications are mixed! Web server = compute + syscalls + I/O.

## Understanding the Filename Format

Filename: `run_9_of_2025-cpu0-sys0-io0-pipe0-vm2-cache20.csv`

**Breakdown:**
- `run_9_of_2025`: 9th combination out of 2025 total
- `cpu0`: 0 `--cpu` workers
- `sys0`: 0 `--syscall` workers
- `io0`: 0 `--io` workers
- `pipe0`: 0 `--pipe` workers
- `vm2`: 2 `--vm` workers
- `cache20`: 20 `--cache` workers

**Common misconception:** "cpu0 means 0% CPU usage"
- ❌ Wrong: `cpu0` means zero CPU compute workers
- ✅ Correct: This file could still have 40% user% from the vm2+cache20 workers

## CPU% vs Workload Type Ambiguity

This is the core problem v1 failed to handle:

| Workload | CPU% | Power | Why Different? |
|----------|------|-------|----------------|
| cpu8-sys0-... | 40% user% | 75W | Compute-intensive (high energy) |
| vm4-cache4-... | 40% user% | 55W | Memory-bound (medium energy) |
| sys8-io4-... | 40% total | 32W | Syscall-heavy (low energy) |

**The model must use all 4 features to distinguish:**
- High user%, low system% → likely compute or memory
- Low user%, high system% → likely syscalls
- High iowait% → I/O bound
- High ctx_switches → pipe/IPC or startup

## Startup Transients (Filtered in Preprocessing)

**What happens:** When stress-ng spawns N workers, there's a ~1 second spike:
- Context switches: 200,000+/sec (process creation)
- Power: Ramps from idle (25W) → full load (80W)
- CPU%: Unstable during initialization

**Why we filter:** This startup pattern would contaminate training:
- It's workload-size dependent (bigger spike for more workers)
- Creates false correlation: "high ctx_switches → power ramp"
- Repeated in all 2025 CSVs

**Solution:** Skip first 100 samples (~1.6s) in preprocessing.

See the [Preprocessing Guide](preprocessing.md) for details.

## Visualizing Workloads

Use the plotting script to see patterns:

```bash
# Plot idle baseline
python scripts/plot_workload.py data/collection/run_1_of_2025-cpu0-sys0-io0-pipe0-vm0-cache0.csv

# Plot pure compute
python scripts/plot_workload.py data/collection/run_500_of_2025-cpu16-sys0-io0-pipe0-vm0-cache0.csv

# Plot pure syscall (the critical pattern!)
python scripts/plot_workload.py data/collection/run_800_of_2025-cpu0-sys16-io0-pipe0-vm0-cache0.csv

# Plot mixed workload
python scripts/plot_workload.py data/collection/run_1234_of_2025-cpu4-sys4-io4-pipe4-vm4-cache4.csv
```

Each plot shows:
- Top panel: user%, system%, iowait% over time
- Middle panel: context switches over time
- Bottom panel: package power over time
- Info box: workload config + statistics

## Power Spectrum Coverage

The combinatorial approach ensures full power spectrum:

- **Idle (23-27W)**: cpu0-sys0-io0-pipe0-vm0-cache0
- **Low (28-35W)**: Syscall-heavy workloads
- **Medium (40-60W)**: I/O, mixed, memory workloads
- **High (70-90W)**: Pure compute, heavy cache thrashing

This is critical for the model to learn the full mapping from CPU metrics → power.

## References

- stress-ng documentation: https://github.com/ColinIanKing/stress-ng
- Data collection guide: [data_collection.md](data_collection.md)
- Preprocessing guide: [preprocessing.md](preprocessing.md)
