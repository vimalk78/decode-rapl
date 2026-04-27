# DECODE-RAPL v2 Data Preprocessing Guide

How to convert collected CSV files into delay-embedded training datasets.

## Overview

The preprocessing pipeline:
1. Filters startup transient samples
2. Applies log transform to context switches
3. Generates delay-embedded vectors (100-dimensional)
4. Merges all CSVs into one giant dataset
5. Performs global random shuffle
6. Splits into train/val/test (80/10/10)
7. Saves as compressed NPZ files

This process is repeated for 3 tau values (1, 4, 8) to compare temporal scales.

## Why Filter Startup Transients?

### The Problem

When stress-ng launches workers, there's a ~1-second startup phase with:
- **Context switch spike**: 200,000+/sec as processes spawn
- **Power ramp-up**: 25W → 80W as workers initialize
- **Transient CPU%**: Unstable patterns during process creation

This startup pattern:
- Is **workload-size dependent** (more workers = bigger spike)
- Creates **false correlations** (model might learn "high ctx_switches → power ramp" instead of steady-state behavior)
- Appears in **all 2025 CSV files** (repeated thousands of times)
- **Masks real patterns** we want the model to learn

### The Solution

Skip the first **100 samples** (~1.6 seconds) from each CSV:
- Removes all startup transients
- Keeps only steady-state workload behavior
- Still have ~2,900 samples per CSV = plenty of data
- Clean training signal

## Delay Embedding

### Concept

Delay embedding reconstructs the attractor from partial observations using time delays (Takens' theorem).

For each timestamp t, create a vector containing current + recent history:
```
input = [feature(t), feature(t-τ), feature(t-2τ), ..., feature(t-(d-1)τ)]
```

**Parameters:**
- **d = 25**: Number of delays (embedding dimension)
- **τ = 1, 4, or 8**: Time delay in samples
- **n_features = 4**: user%, system%, iowait%, log(ctx_switches)

### Feature-Grouped Ordering

The 100-dimensional vector is organized by feature (not by time):

```
Position  0-24:  [user%(t), user%(t-τ), ..., user%(t-24τ)]
Position 25-49:  [sys%(t),  sys%(t-τ),  ..., sys%(t-24τ)]
Position 50-74:  [iowait%(t), iowait%(t-τ), ..., iowait%(t-24τ)]
Position 75-99:  [log_ctx(t), log_ctx(t-τ), ..., log_ctx(t-24τ)]
```

**Why feature-grouped?**
- Feature temporal history is contiguous
- Easier to interpret latent space (which features matter most)
- Encoder might learn feature-specific temporal patterns
- Better for visualization

### Temporal Lookback

The lookback window depends on tau:

| Tau | Lookback Samples | Lookback Time | Use Case |
|-----|------------------|---------------|----------|
| 1   | 24 samples       | 384ms         | Fast dynamics (context switches, bursts) |
| 4   | 96 samples       | 1536ms (~1.5s)| Medium dynamics (workload transitions) |
| 8   | 192 samples      | 3072ms (~3s)  | Slow dynamics (thermal effects, sustained load) |

**Note:** With 16ms sampling, tau=1 provides highest temporal resolution.

### Log Transform

Context switches are log-transformed to handle outliers:
```python
log_ctx_switches = log(1 + ctx_switches_per_sec)
```

**Why?**
- ctx_switches ranges from 1,000 to 200,000+ (high variance)
- Log compresses the range: ~7 to 12
- Reduces impact of startup spikes
- Matches Gemini recommendation

## Data Split Strategy

### Why Random Global Shuffle?

After delay embedding, each 100-dim vector is a complete feature representation. The temporal structure is already encoded in the vector, so shuffling doesn't break anything.

**Global shuffle ensures:**
- No temporal leakage between splits
- Model trains on diverse samples from all workloads
- Standard supervised learning setup: f(100-dim vector) → power

**Alternative approaches (NOT used):**
- ❌ Temporal split within CSV: Creates artificial time-series task
- ❌ CSV-level split: Would test unseen workloads, but loses 20% of data diversity

### Split Ratios

- **80% train** (~4.5M samples): Main training set
- **10% val** (~560K samples): Hyperparameter tuning, early stopping
- **10% test** (~560K samples): Final evaluation

All splits come from the same shuffled pool, ensuring consistent statistics.

## Usage

### Basic Usage

```bash
cd scripts/

python prepare_training_data.py \
  --data-dir ../data/all-combinations-temp-0 \
  --output-dir ../data/processed \
  --tau 1 4 8 \
  --skip-startup 100 \
  --seed 42
```

### Options

```
--data-dir DIR          Directory containing CSV files (required)
--output-dir DIR        Output directory for processed datasets (required)
--tau TAU [TAU ...]     Tau values for delay embedding (default: 1 4 8)
--d N                   Number of delays (default: 25)
--skip-startup N        Samples to skip from each CSV (default: 100)
--train-ratio RATIO     Train split ratio (default: 0.8)
--val-ratio RATIO       Validation split ratio (default: 0.1)
--seed N                Random seed for reproducibility (default: 42)
```

### Example: Single Tau

To process only tau=1 (for faster testing):

```bash
python prepare_training_data.py \
  --data-dir ../data/collection-test \
  --output-dir ../data/processed-test \
  --tau 1 \
  --seed 42
```

## Output Structure

The script generates separate datasets for each tau value:

```
data/processed/
├── tau1/
│   ├── train.npz           # 4.5M samples
│   ├── val.npz             # 560K samples
│   ├── test.npz            # 560K samples
│   └── metadata.json       # Statistics and parameters
├── tau4/
│   ├── train.npz           # ~4.4M samples (fewer due to longer lookback)
│   ├── val.npz
│   ├── test.npz
│   └── metadata.json
└── tau8/
    ├── train.npz           # ~4.2M samples
    ├── val.npz
    ├── test.npz
    └── metadata.json
```

### NPZ File Format

Each `.npz` file contains:
```python
data = np.load('train.npz')
X = data['X']  # Shape: (N, 100) - input vectors
y = data['y']  # Shape: (N,) - power labels
```

**Loading example:**
```python
import numpy as np

# Load training data
data = np.load('data/processed/tau1/train.npz')
X_train = data['X']  # (4500000, 100)
y_train = data['y']  # (4500000,)

print(f"Training samples: {len(X_train):,}")
print(f"Input shape: {X_train.shape}")
print(f"Power range: {y_train.min():.1f}W - {y_train.max():.1f}W")
```

### Metadata Format

`metadata.json` contains:
```json
{
  "tau": 1,
  "d": 25,
  "skip_startup_samples": 100,
  "seed": 42,
  "n_features": 4,
  "input_dim": 100,
  "feature_order": [
    "user_percent (0-24)",
    "system_percent (25-49)",
    "iowait_percent (50-74)",
    "log_ctx_switches (75-99)"
  ],
  "temporal_lookback_samples": 24,
  "temporal_lookback_ms": 384,
  "train_samples": 4500000,
  "val_samples": 560000,
  "test_samples": 560000,
  "train_power_mean": 45.2,
  "train_power_std": 12.8,
  "per_file_stats": [...]
}
```

## Processing Time

For ~2025 CSV files with ~3000 samples each:

| Tau | Samples/CSV | Total Samples | Processing Time (estimate) |
|-----|-------------|---------------|---------------------------|
| 1   | ~2,879      | ~5.8M         | ~5 minutes |
| 4   | ~2,807      | ~5.7M         | ~5 minutes |
| 8   | ~2,711      | ~5.5M         | ~5 minutes |

Total: ~15 minutes for all three tau values.

**Bottlenecks:**
- CSV loading: ~60% of time
- Delay embedding: ~30% of time
- Shuffling & saving: ~10% of time

## Verification

### Check Dataset Integrity

```python
import numpy as np
import json

# Load and verify
data = np.load('data/processed/tau1/train.npz')
X = data['X']
y = data['y']

print("Shape verification:")
print(f"  X: {X.shape} (should be (N, 100))")
print(f"  y: {y.shape} (should be (N,))")

print("\nValue ranges:")
print(f"  user% [0-24]: {X[:, 0:25].min():.1f} - {X[:, 0:25].max():.1f}")
print(f"  sys% [25-49]: {X[:, 25:50].min():.1f} - {X[:, 25:50].max():.1f}")
print(f"  iowait% [50-74]: {X[:, 50:75].min():.1f} - {X[:, 50:75].max():.1f}")
print(f"  log_ctx [75-99]: {X[:, 75:100].min():.1f} - {X[:, 75:100].max():.1f}")
print(f"  power: {y.min():.1f}W - {y.max():.1f}W")

print("\nFeature continuity check (should show temporal smoothness):")
sample = X[0, :]
print(f"  user% [t, t-1, t-2]: {sample[0:3]}")
print(f"  sys% [t, t-1, t-2]: {sample[25:28]}")
```

### Check Metadata

```bash
cat data/processed/tau1/metadata.json | python -m json.tool
```

### Visualize Sample Distribution

```python
import matplotlib.pyplot as plt

# Load all splits
train = np.load('data/processed/tau1/train.npz')
val = np.load('data/processed/tau1/val.npz')
test = np.load('data/processed/tau1/test.npz')

# Compare power distributions
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(train['y'], bins=50, alpha=0.7, label='Train')
plt.xlabel('Power (W)')
plt.ylabel('Count')
plt.title('Train Set Distribution')
plt.legend()

plt.subplot(132)
plt.hist(val['y'], bins=50, alpha=0.7, label='Val', color='orange')
plt.xlabel('Power (W)')
plt.title('Val Set Distribution')
plt.legend()

plt.subplot(133)
plt.hist(test['y'], bins=50, alpha=0.7, label='Test', color='green')
plt.xlabel('Power (W)')
plt.title('Test Set Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('split_distributions.png', dpi=150)
print("Distributions should be similar across all splits")
```

## Next Steps

After preprocessing completes, proceed to model training:

```bash
python src/train.py \
  --data-dir data/processed/tau1 \
  --config config/v2_default.yaml \
  --output-dir results/tau1
```

See training documentation (TBD) for details.
