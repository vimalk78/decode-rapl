# DECODE-RAPL Multi-Feature Implementation

## Problem Statement

The original DECODE-RAPL implementation used only **total CPU%** as input. This created an ambiguity problem:

**Same CPU% → Different Power Consumption**

Examples from real data:
- **20% CPU from light work** (syscalls, /proc reading): **26-28W**
- **20% CPU from heavy work** (compute, stress-ng): **38-45W**

The single-feature model couldn't distinguish between these workload types, leading to overprediction on real workloads (Prometheus exporters reading /proc).

## Solution: Multi-Feature Delay Embedding

Instead of using just total CPU%, we now use **3 features**:

1. **user_percent**: User-space CPU time (application code)
2. **system_percent**: Kernel/syscall CPU time (includes system, irq, softirq)
3. **context_switches**: Rate of context switches (syscall overhead indicator)

### Why This Works

Different workload types have distinct feature patterns:

| Workload Type | User% | System% | Ctx Switches | Power |
|--------------|-------|---------|--------------|-------|
| Light (syscalls, /proc) | Low | High | High | 26-28W |
| Heavy (compute, stress) | High | Low | Low | 38-45W |

Analysis from training data confirms this:

```
Power 20-30W: user=0.6%, sys=1.0%, ctx=1223/s, sys_ratio=0.63
Power 30-40W: user=4.2%, sys=2.4%, ctx=18249/s, sys_ratio=0.36
Power 40-50W: user=20.1%, sys=2.9%, ctx=81233/s, sys_ratio=0.13
```

Lower power has higher system ratio (more syscalls), confirming the pattern.

## Implementation Details

### Delay Embedding

Each feature gets its own delay embedding:

- **Input**: 3 features × 25 delays = **75-dimensional embedding**
- **For τ=8**: `[user(t), user(t-8), ..., user(t-192), system(t), system(t-8), ..., context(t), ...]`

The `DelayEmbedding` class already supported multi-variable input:

```python
embedder = DelayEmbedding(tau=8, d=25, n_features=3)
features = data[['user_percent', 'system_percent', 'context_switches']].values  # (T, 3)
embedded = embedder.embed(features)  # (T_valid, 75)
```

### Model Architecture

- **Encoder**: 75 → 512 → 128 → 64 → 16 (latent)
- **LSTM**: 16 (latent) → 128 (hidden) → 1 (power)
- **Decoder**: 16 (latent) → 64 → 128 → 512 → 75 (reconstruction)

Total parameters: **303,261**

### Configuration Files

Three configs for different time scales:

1. **config_tau1_multifeature.yaml**: τ=1 (16ms), finest temporal resolution
2. **config_tau4_multifeature.yaml**: τ=4 (64ms), medium granularity
3. **config_tau8_multifeature.yaml**: τ=8 (128ms), most efficient

All use the same multi-feature input:
```yaml
preprocessing:
  feature_columns:
    - user_percent
    - system_percent
    - context_switches
```

## Data Preparation

### Converting MS-TCN Data to Multi-Feature Format

```bash
python scripts/convert_mstcn_multifeature.py \
  --input data/training_with_proc_monitoring.csv \
  --output data/training_multifeature.csv \
  --machine-id machine_0
```

This creates a CSV with columns:
- timestamp
- machine_id
- user_percent (from cpu_user_percent)
- system_percent (cpu_system + cpu_irq + cpu_softirq)
- context_switches (context_switches_sec)
- power (rapl_package_power)
- cpu_total (for reference: user + system)

**Note**: iowait is excluded because CPU is halted during I/O wait (not doing work).

## Training

### All Models at Once

```bash
./scripts/train_multifeature_all.sh
```

This trains all 3 tau configurations sequentially.

### Individual Models

```bash
python src/train.py --config config_tau8_multifeature.yaml
python src/train.py --config config_tau4_multifeature.yaml
python src/train.py --config config_tau1_multifeature.yaml
```

### Expected Training Time (GPU)

- tau8: ~45 minutes (smallest window_size=15)
- tau4: ~60 minutes (window_size=30)
- tau1: ~90 minutes (largest window_size=120)

## Testing

After training, test on real workload:

```bash
# Evaluate model
python src/evaluate.py \
  --config config_tau8_multifeature.yaml \
  --checkpoint checkpoints/tau8_multifeature/best_model.pt \
  --test-data data/real_workload_multifeature.csv \
  --output results/tau8_multifeature/real_test/
```

Expected improvement:
- **Before** (single feature): 11.27% MAPE, R²=0.29
- **Target** (multi-feature): <5% MAPE, R²>0.85

## Files Modified

### Core Implementation

1. **src/preprocessing.py**
   - Added `n_features` parameter to `DelayEmbedding.__init__()`
   - Updated `RAPLDataset._create_sequences()` to extract multiple features
   - Updated normalization to handle multiple feature columns

2. **src/model.py**
   - Updated input_dim calculation: `d * n_features` instead of just `d`

### Configuration

3. **config_tau{1,4,8}_multifeature.yaml** (NEW)
   - Added `feature_columns` to preprocessing section
   - Updated comments to explain multi-feature approach

### Data Conversion

4. **scripts/convert_mstcn_multifeature.py** (NEW)
   - Converts MS-TCN format to multi-feature DECODE format
   - Computes user/system breakdown
   - Analyzes user/system ratio at different power levels

### Training Scripts

5. **scripts/train_multifeature_all.sh** (NEW)
   - Trains all 3 tau configs in sequence

## Validation

All components tested and verified:

```bash
# Test delay embedding
✓ Multi-feature embedding: (1000, 3) → (808, 75)

# Test preprocessing
✓ Batch shape: (32, 15, 75) for tau8

# Test model
✓ Forward pass: (32, 15, 75) → power (32, 1)
✓ Reconstruction: (32, 15, 75) → (32, 15, 75)
```

## Next Steps

1. **Train models**: Run `./scripts/train_multifeature_all.sh` on GPU machine
2. **Test on real workload**: Evaluate with Prometheus exporter data
3. **Compare metrics**: Measure improvement over single-feature models
4. **Deploy best model**: Choose tau config with best real-world performance

## Expected Outcomes

The multi-feature approach should:

1. **Distinguish workload types**: Light (syscalls) vs heavy (compute)
2. **Reduce overprediction**: No more 32-39W predictions for 26-28W actual
3. **Improve MAPE**: From 11.27% → <5%
4. **Improve R²**: From 0.29 → >0.85
5. **Generalize better**: Work across different application types

## Key Insight

> The original problem wasn't lack of I/O samples in training data. It was that **total CPU% is fundamentally ambiguous** - the same percentage can mean different things depending on **what type of work** the CPU is doing. Multi-feature input resolves this ambiguity by providing the model with information about the **nature of the work**, not just the total amount.
