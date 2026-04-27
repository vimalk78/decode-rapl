# Joint Package + DRAM Power Prediction Model Plan

**Status**: Future work - deferred for separate implementation
**Created**: 2025-10-09
**Reason for deferral**: Current multi-output training shows poor DRAM performance (R²=0.0003) due to missing features and StandardScaler issues

---

## Problem Analysis

### Current Results (17-feature model with equal loss weights)

**Package Power**:
- R²: 0.9035 (acceptable but worse than 15f model)
- MAE: 4.40W

**DRAM Power**:
- R²: 0.0003 (essentially no correlation!)
- MAE: 1.25W
- Model predicts near-constant values (~8-10W)

### Root Causes Identified

1. **StandardScaler destroys critical features**:
   - `num_cores=20` (constant across training) → becomes 0 or NaN after scaling
   - Model can't learn baseline power scaling without this feature
   - Double normalization: features normalized to 0-1, then StandardScaler normalizes again

2. **Missing memory bandwidth features**:
   - Current features measure memory **allocation** (usage ratios)
   - DRAM power driven by memory **bandwidth** (actual memory activity)
   - Missing: page I/O rates, memory access patterns

3. **Shared feature extraction**:
   - All layers optimized for package power (dominant signal)
   - DRAM output head receives package-optimized representations
   - Attention mechanism operates on temporal dimension, not feature selection

---

## Solution Architecture

### 1. Selective StandardScaler (CRITICAL FIX)

**Problem**: Current code applies StandardScaler to ALL features indiscriminately (line 358-359 in `train_model.py`)

**Solution**: Use `sklearn.compose.ColumnTransformer` for selective scaling

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Features that need scaling (unbounded)
SCALE_FEATURES = [
    'interrupts_per_core',
    'context_switches_per_core',
    'page_faults_per_core',
    'running_processes_per_core',
    # Memory bandwidth (once added)
    'pgpgin_per_core',
    'pgpgout_per_core',
    'pgmajfault_per_core'
]

# Features that should NOT be scaled
NO_SCALE_FEATURES = [
    # Already bounded (0-100%)
    'cpu_user_percent', 'cpu_system_percent', 'cpu_idle_percent',
    'cpu_iowait_percent', 'cpu_irq_percent', 'cpu_softirq_percent',

    # Already bounded (0-1 ratios)
    'memory_used_ratio', 'memory_cached_ratio',
    'memory_free_ratio', 'swap_used_ratio',

    # System scale - MUST preserve for baseline learning!
    'num_cores',           # e.g., 20 for baremetal
    'memory_total_gb',     # e.g., 95 GB
    'swap_total_gb'        # e.g., 8 GB
]

# Create column transformer
self.feature_scaler = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), SCALE_FEATURES),
        ('passthrough', 'passthrough', NO_SCALE_FEATURES)
    ],
    remainder='drop'  # Ensure no features are missed
)
```

**Why this matters**:
- Preserves `num_cores` for baseline power learning (idle power ∝ cores)
- Keeps bounded features in their natural range (0-100%, 0-1)
- Only scales unbounded rate features that need normalization

---

### 2. Memory Bandwidth Features

**Current features measure allocation, not activity**:
- `memory_used_ratio`: How much RAM is allocated (doesn't correlate with DRAM power)
- DRAM power driven by memory **access rate** (reads/writes/page faults)

**Required new features from `/proc/vmstat`**:

```python
# Add to power_data_collector.py
def read_vmstat_bandwidth():
    """Read memory bandwidth metrics from /proc/vmstat"""
    with open('/proc/vmstat') as f:
        data = {}
        for line in f:
            if line.startswith('pgpgin'):
                data['pgpgin'] = int(line.split()[1])
            elif line.startswith('pgpgout'):
                data['pgpgout'] = int(line.split()[1])
            elif line.startswith('pgmajfault'):
                data['pgmajfault'] = int(line.split()[1])
    return data

# Convert to per-second rates (like interrupts_sec)
pgpgin_sec = (current['pgpgin'] - prev['pgpgin']) / time_delta
pgpgout_sec = (current['pgpgout'] - prev['pgpgout']) / time_delta
pgmajfault_sec = (current['pgmajfault'] - prev['pgmajfault']) / time_delta

# Normalize to per-core (for VM portability)
pgpgin_per_core = pgpgin_sec / num_cores
pgpgout_per_core = pgpgout_sec / num_cores
pgmajfault_per_core = pgmajfault_sec / num_cores
```

**Expected benefit**:
- `pgpgin_per_core`: Pages read from disk → DRAM read activity
- `pgpgout_per_core`: Pages written to disk → DRAM write activity
- `pgmajfault_per_core`: Major faults requiring disk I/O → DRAM intensive operations

These features are **VM-accessible** (no PMU/hardware counters needed).

---

### 3. Architecture Considerations

**Current architecture is likely sufficient**:
- Multi-head attention on temporal dimension
- Shared FC layers: 128 → 256 → 128
- Separate output heads: Linear(128, 1) per domain

**Why it should work**:
- Output heads learn domain-specific mappings from shared representation
- 128-dim bottleneck is rich enough for feature extraction
- Attention helps select relevant time steps

**If performance still poor, consider**:

#### Option A: Attention on Feature Dimension
Add feature-wise attention before output heads:

```python
# In MSTCN.__init__
self.feature_attention = nn.ModuleList([
    nn.Linear(128, 128) for _ in range(num_targets)
])

# In forward(), before output heads
feature_attended = []
for i, attn in enumerate(self.feature_attention):
    weights = torch.sigmoid(attn(x))  # Learn feature importance per domain
    feature_attended.append(x * weights)

outputs = [head(f) for head, f in zip(self.output_heads, feature_attended)]
```

#### Option B: Domain-Specific Branches
Split early layers by domain:

```python
# Replace single fc1/fc2 with domain-specific branches
self.package_fc = nn.Sequential(
    nn.Linear(hidden_dim, 128),
    nn.BatchNorm1d(128),
    nn.ReLU()
)

self.dram_fc = nn.Sequential(
    nn.Linear(hidden_dim, 128),
    nn.BatchNorm1d(128),
    nn.ReLU()
)
```

**Recommendation**: Try Option A first (simpler, less parameters).

---

### 4. Loss Function Considerations

**Current**: Equal weights [0.5, 0.5] with `MultiDomainHuberLoss`

**Option A: Keep equal weights**
- Let model learn both domains naturally
- With proper features, DRAM should improve

**Option B: Gradient balancing**
Instead of static loss weights, balance gradients dynamically:

```python
# From "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
def compute_grad_norm_loss(losses, weights, shared_params):
    """Balance gradients across tasks"""
    grad_norms = []
    for loss in losses:
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
        grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
        grad_norms.append(grad_norm)

    # Target: equal gradient magnitudes
    mean_grad = sum(grad_norms) / len(grad_norms)
    loss_weights = [mean_grad / (g + 1e-8) for g in grad_norms]

    return loss_weights
```

**Option C: Uncertainty weighting**
```python
# Learn loss weights via uncertainty estimation
self.log_vars = nn.Parameter(torch.zeros(num_targets))

# In loss calculation
weighted_losses = [
    torch.exp(-log_var) * loss + log_var
    for loss, log_var in zip(losses, self.log_vars)
]
```

**Recommendation**: Start with equal weights after fixing features. Add complexity only if needed.

---

### 5. Updated Feature List (20 total)

```python
FEATURE_COLUMNS = [
    # CPU percentages (0-100%, no scaling needed)
    'cpu_user_percent', 'cpu_system_percent', 'cpu_idle_percent',
    'cpu_iowait_percent', 'cpu_irq_percent', 'cpu_softirq_percent',

    # Per-core activity rates (scale these)
    'interrupts_per_core',
    'context_switches_per_core',
    'page_faults_per_core',
    'running_processes_per_core',

    # Memory ratios (0-1, no scaling needed)
    'memory_used_ratio',
    'memory_cached_ratio',
    'memory_free_ratio',
    'swap_used_ratio',

    # Memory bandwidth (scale these - NEW!)
    'pgpgin_per_core',
    'pgpgout_per_core',
    'pgmajfault_per_core',

    # System scale (DO NOT scale - critical!)
    'num_cores',
    'memory_total_gb',
    'swap_total_gb'
]

AVAILABLE_RAPL_TARGETS = ['rapl_package_power', 'rapl_dram_power']
```

---

## Implementation Steps

### Phase 1: Feature Engineering

1. **Update `src/power_data_collector.py`**:
   - Add `read_vmstat_bandwidth()` function
   - Track previous vmstat values for per-second rates
   - Add to main metrics collection

2. **Update `scripts/preprocess_data.py`**:
   - Add bandwidth feature normalization (per-core division)
   - Preserve system scale features

3. **Collect new training data**:
   - 2 hours diverse workload on baremetal
   - Ensure memory-intensive workloads (cache thrashing, swapping)
   - Verify bandwidth metrics show variation

### Phase 2: Model Training Updates

4. **Update `src/train_model.py`**:
   - Implement `ColumnTransformer` for selective scaling
   - Update `FEATURE_COLUMNS` to 20 features
   - Keep multi-output architecture
   - Keep equal loss weights `[0.5, 0.5]`

5. **Train and validate**:
   - Train on new data with bandwidth features
   - Monitor per-domain metrics during training
   - Target: Package R²>0.95, DRAM R²>0.80

### Phase 3: Architecture Enhancements (if needed)

6. **If DRAM still poor** (R²<0.70):
   - Try feature-wise attention (Option A above)
   - Try gradient balancing
   - Consider domain-specific branches

---

## Expected Results

**With selective StandardScaler + bandwidth features**:
- **Package power**: R²>0.95, MAE<3W (better than current 15f model)
- **DRAM power**: R²>0.80, MAE<0.5W (huge improvement from 0.0003)
- **VM portability**: Model works on 4-core VM (baseline scales correctly)

**Why this should work**:
1. `num_cores` preserved → learns baseline power ∝ cores
2. Bandwidth features → DRAM has actual signal to learn from
3. Bounded features preserved → natural scale maintained
4. Existing architecture → attention can select relevant features

---

## Validation Plan

### 1. Feature Correlation Analysis
Before training, verify bandwidth features correlate with DRAM:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_data.csv')

# Check correlation
corr = df[['rapl_dram_power', 'pgpgin_per_core', 'pgpgout_per_core',
           'pgmajfault_per_core', 'memory_used_ratio']].corr()
print(corr['rapl_dram_power'])

# Expect: bandwidth features > 0.5 correlation, memory_used_ratio < 0.3
```

### 2. StandardScaler Verification
After preprocessing, check constant features are preserved:

```python
# Load preprocessed data
X, y = preprocessor.prepare_features_targets(...)

# Check num_cores column
num_cores_idx = FEATURE_COLUMNS.index('num_cores')
print(f"num_cores range: {X[:, num_cores_idx].min()} - {X[:, num_cores_idx].max()}")
# Should be: 20.0 - 20.0 (or scaled consistently, not 0)

# Check it's not all zeros
assert X[:, num_cores_idx].std() > 1e-6 or X[:, num_cores_idx].mean() > 1.0
```

### 3. VM Portability Testing
After training, test on different core counts using taskset:

```bash
# 4-core idle
taskset -c 0-3 bash -c "while true; do :; done" &
sudo python src/power_predictor.py --model models/joint_model.pth --live --duration 60

# Expected: ~10W package (vs ~26W on 20-core)

# 8-core idle
taskset -c 0-7 bash -c "while true; do :; done" &
sudo python src/power_predictor.py --model models/joint_model.pth --live --duration 60

# Expected: ~14W package (linear scaling)
```

---

## Risk Mitigation

### Risk 1: Bandwidth features don't help DRAM
**Mitigation**:
- Add more memory-specific features (cache misses from `perf stat`)
- Consider separate DRAM model with simpler architecture

### Risk 2: StandardScaler still causes issues
**Mitigation**:
- Skip StandardScaler entirely for all features
- Use manual normalization (min-max to 0-1 for unbounded features)

### Risk 3: VM extrapolation fails (non-linear baseline)
**Mitigation**:
- Train on multi-core data (collect with different taskset configurations)
- Add `num_cores²` feature to capture non-linear effects

---

## Alternative: Separate Models

If joint model still underperforms:

**Package-only model**:
- 17 features (current normalized set)
- Single output
- Target: R²>0.95

**DRAM-only model**:
- Memory-focused features (bandwidth + allocation)
- Simpler architecture (fewer layers)
- Train independently with DRAM-specific loss

**Benefits**:
- Each model optimized for its domain
- Easier debugging
- Package model can be deployed immediately

**Drawback**:
- Need to run two models for full power accounting
- More inference overhead

---

## References

- Current 17f model results: `results/plots/prediction_comparison_17f_package.png`, `*_dram.png`
- Training summary: `results/training_summary_diverse_2hr.json`
- Current architecture: `src/train_model.py` lines 179-250
- StandardScaler issue: `src/train_model.py` line 358-359

---

**Next Action**: Implement package-only model with selective StandardScaler to validate VM portability first, then return to joint model with proper features.
