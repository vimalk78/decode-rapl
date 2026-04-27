# Power Prediction Model Architecture

## Overview

This document describes how the MS-TCN (Multi-Scale Temporal Convolutional Network) model predicts CPU power consumption in a way that's portable across different system configurations (baremetal servers vs VMs with different core counts).

## Problem Statement

**Goal**: Build a model trained on baremetal (e.g., 20 cores) that can predict power for VMs with different configurations (e.g., 4 cores, 8GB RAM).

**Challenge**: Raw metrics like `interrupts_sec=2000` mean different things on a 20-core vs 4-core system.

## Solution: Normalized Features

### Feature Design

We normalize features to be **scale-independent**:

**Original Features (NOT portable):**
```
interrupts_sec: 2000          # High for 4 cores, low for 20 cores
memory_used_mb: 4096          # Doesn't account for total memory size
```

**Normalized Features (portable):**
```
interrupts_per_core: 100      # Same meaning regardless of core count
memory_used_ratio: 0.05       # 5% of total memory
num_cores: 20                 # Explicit system scale
```

### Feature Categories (17 total)

1. **CPU Percentages (6)** - Already scale-independent (0-100%)
   - `cpu_user_percent`, `cpu_system_percent`, `cpu_idle_percent`
   - `cpu_iowait_percent`, `cpu_irq_percent`, `cpu_softirq_percent`

2. **Per-Core Activity Rates (4)** - Normalized by dividing by `num_cores`
   - `interrupts_per_core` = interrupts_sec / num_cores
   - `context_switches_per_core` = context_switches_sec / num_cores
   - `page_faults_per_core` = page_faults_sec / num_cores
   - `running_processes_per_core` = running_processes / num_cores

3. **Memory Ratios (4)** - Normalized by total memory size (0-1)
   - `memory_used_ratio` = memory_used_mb / memory_total_mb
   - `memory_cached_ratio` = memory_cached_mb / memory_total_mb
   - `memory_free_ratio` = memory_free_mb / memory_total_mb
   - `swap_used_ratio` = swap_used_mb / swap_total_mb

4. **System Scale (3)** - Explicit system configuration
   - `num_cores` - Number of CPU cores (20 for baremetal, 4 for VM)
   - `memory_total_gb` - Total RAM in GB
   - `swap_total_gb` - Total swap in GB

## Power Decomposition

The model learns that total power has two components:

```
Total Power = Baseline Power + Active Power
```

### Baseline Power
- Idle power consumption of the system
- Function of: `num_cores`, `memory_total_gb`, thermal state
- Learned from ~112K idle samples in training (25% of dataset)
- Model learns: "More cores = higher baseline"

### Active Power
- Additional power from CPU/memory/IO activity
- Function of: `cpu_activity`, `interrupts_per_core`, `memory_pressure`, etc.
- Scales with activity intensity

## How The Model Learns

The neural network learns this decomposition **implicitly** through training:

**Training Pattern Discovery:**

```
# Idle samples teach baseline
Input:  num_cores=20, cpu_idle=100%, interrupts_per_core=90
Output: 26W  → Model learns: baseline(20 cores) ≈ 26W

# Active samples teach activity power
Input:  num_cores=20, cpu_idle=0%, interrupts_per_core=2000
Output: 70W  → Model learns: baseline(20) + activity(full load) ≈ 70W
```

The network's internal weights effectively implement:

```
Power ≈ W1·num_cores + W2·memory_gb + W3·(cpu_user% × interrupts_per_core) + ...
        └─ Baseline terms ─┘           └─ Activity terms ──────────────────┘
```

## Extrapolation to VMs

When predicting power for a 4-core VM:

**Model trained on 20-core baremetal:**
- Learned: `baseline(20 cores) = 26W`
- Learned: `per_core_idle_power ≈ 1W/core`

**Predicts for 4-core VM:**
```
Input:  num_cores=4, cpu_idle=100%, interrupts_per_core=90
Output: 10W  → Extrapolates: baseline(4 cores) ≈ 10W
```

**With 50% load:**
```
Input:  num_cores=4, cpu_idle=50%, interrupts_per_core=450
Output: 18W  → baseline(4) + activity(50% load) ≈ 10W + 8W
```

## Multi-Output: Package + DRAM Power

The model predicts two power domains simultaneously:

1. **Package Power** (CPU + uncore components)
   - Dominated by: CPU activity, core count
   - Typically: 26-70W range

2. **DRAM Power** (Memory subsystem)
   - Dominated by: Memory usage, page faults, memory bandwidth
   - Typically: 7-10W range

Multi-output training helps because:
- Shared features benefit both predictions
- Model learns CPU vs memory power attribution
- Better for VM scenarios with different memory allocations

## Training Process

**Step 1: Data Collection**
```bash
# Collect 2 hours of diverse workload data on baremetal
sudo python3 src/power_data_collector.py --duration 7200 --output training_raw.csv
python3 workload_generator.py --duration 7200  # Random CPU loads
```

**Step 2: Preprocessing**
```bash
# Add normalized features (must run on same machine as data collection)
python3 scripts/preprocess_data.py training_raw.csv training_normalized.csv
```

**Step 3: Training**
```bash
# Train multi-output model (package + dram)
python3 src/train_model.py training_normalized.csv \
  --batch-size 512 --epochs 30 --output models/model_17f.pth
```

**Step 4: Validation**
```bash
# Test baseline scaling with different core counts
taskset -c 0-3 stress-ng --cpu 4 --cpu-load 0    # 4-core idle
taskset -c 0-7 stress-ng --cpu 8 --cpu-load 0    # 8-core idle
# Compare predicted vs actual power scaling
```

## Key Assumptions

1. **Linear baseline scaling**: Assumes idle power scales roughly proportionally with core count
   - May not be perfect (uncore, package overhead)
   - Requires validation with taskset experiments

2. **Activity intensity transfers**: Assumes `interrupts_per_core=1000` means similar CPU intensity on 4-core vs 20-core
   - Generally valid for CPU-bound workloads
   - May differ for memory/IO-bound workloads

3. **Thermal independence**: Model doesn't explicitly model thermal state
   - Uses temporal context (64-sample sequences) to implicitly capture thermal lag
   - Training data showed 32W (warm) vs 26W (cold) idle states

## Model Architecture

- **Type**: MS-TCN (Multi-Scale Temporal Convolutional Network)
- **Sequence Length**: 64 samples (captures ~1 second of history at 60Hz)
- **Input**: 17 normalized features
- **Output**: 2 values (package power, dram power)
- **Parameters**: ~735K parameters
- **Training Data**: 450K samples over 2 hours

## Performance (Baremetal)

**Current Results (15-feature model, before normalization):**
- MAE: 3.77W (package power)
- R²: 0.9285
- Mean bias: +3.25W (likely thermal state tracking)

**Expected with 17-feature normalized model:**
- Similar or better accuracy
- Improved VM portability (to be validated)

## Limitations & Future Work

**Current Limitations:**
1. No explicit temperature/thermal modeling
2. Assumes linear power scaling with cores (needs validation)
3. Trained on single CPU architecture (Intel Xeon)
4. No GPU power modeling

**Future Improvements:**
1. Add CPU temperature features if available via `hwmon`
2. Validate and tune for non-linear power scaling
3. Train on diverse CPU architectures
4. Extend to GPU power prediction
5. Add per-VM attribution in multi-tenant scenarios

## Usage

**For Baremetal Prediction:**
```bash
sudo python3 src/power_predictor.py --model models/best_model.pth --live --duration 300
```

**For VM Prediction (future):**
```bash
# Inside VM - auto-detects 4 cores and normalizes accordingly
python3 src/power_predictor.py --model models/best_model.pth --live --duration 300
```

## References

- Training data: `data/training_diverse_2hr.csv` (raw) → `training_diverse_2hr_normalized.csv` (preprocessed)
- Model code: `src/train_model.py`
- Inference code: `src/power_predictor.py`
- Preprocessing: `scripts/preprocess_data.py`
- Analysis tools: `scripts/analyze_idle_sequences.py`, `scripts/compare_idle_features.py`

---

**Status**: Document created during model development. To be updated after training completes with actual 17-feature results.

**Last Updated**: 2025-10-09
