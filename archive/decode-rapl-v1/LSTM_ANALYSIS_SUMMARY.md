# LSTM and Latent Space Analysis Summary

## Questions to Answer

1. **What are the LSTM layers doing?**
2. **Should we increase latent space size?**
3. **Why did multi-feature fail?**

## Model Architecture Analysis

### Current Setup (tau8_multifeature)

```
Input: (batch, 15, 75)  ← 15 timesteps, 75-dim delay embedding (25 delays × 3 features)
   ↓
Encoder: 75 → 512 → 128 → 64 → 16 (latent)
   ↓
LSTM: 16 (latent input) → 128 (hidden) → 16 (output to FC)
   ↓
FC: 16 → 1 (power prediction)
```

**Total parameters**: ~303k

### Key Observations

1. **Latent Bottleneck**: 75-dim input compressed to 16-dim latent
   - Compression ratio: 75:16 = 4.7:1
   - This is a SEVERE bottleneck for 3-feature input!

2. **LSTM Hidden Size**: 128 units
   - Takes 16-dim latent input
   - Outputs processed to 1-dim power
   - But LSTM output dimension doesn't match latent (16 → 128 → back to FC input)

3. **Delay Embedding Already Captures Temporal Info**:
   - Input has 15 timesteps of 75-dim vectors
   - Each vector contains 25 delays (temporal history)
   - **LSTM may be redundant** - delay embedding already encodes time!

## Why Multi-Feature Likely Failed

### Hypothesis 1: Latent Space Too Small ⚠️ **MOST LIKELY**

**Single-feature model:**
- Input: 25-dim (25 delays × 1 feature)
- Latent: 16-dim
- Compression: 25:16 = 1.6:1 ✅ Reasonable

**Multi-feature model:**
- Input: 75-dim (25 delays × 3 features)
- Latent: 16-dim
- Compression: 75:16 = 4.7:1 ❌ Too aggressive!

**The 16-dim latent space cannot capture all information from 3 features.**

When you have:
- `user_percent` history (25 values)
- `system_percent` history (25 values)
- `context_switches` history (25 values)

Cramming all this into just 16 dimensions loses critical information, especially when context switches have a completely different scale (1k vs 695k) due to VM overhead.

### Hypothesis 2: LSTM is Redundant

Delay embedding already provides:
```
h(t) = [y(t), y(t-τ), y(t-2τ), ..., y(t-24τ)]
```

This captures temporal dependencies. Adding LSTM on top may:
- Add noise rather than signal
- Increase overfitting risk
- Not learn anything useful beyond what delay embedding provides

### Hypothesis 3: Context Switch Normalization Broke Everything

Training data context switches: 372 - 695,328/sec
Test data (VM) context switches: ~1,000/sec → normalized to 0.0009 (near zero!)

The model learned relationships based on a skewed distribution, then at test time sees values that appear as "zero" in normalized space.

## Recommendations

### Priority 1: Increase Latent Space for Multi-Feature ⚡

**Current**: 16-dim latent
**Recommended**: 48-64 dim latent for 3 features

Reasoning:
- Single feature works with 16-dim (25:16 ratio)
- 3 features should use ~48-dim (75:48 = 1.6:1 ratio, same as single-feature)
- This gives each feature ~16 dimensions to encode its temporal patterns

### Priority 2: Test if LSTM Helps or Hurts 🔍

**Ablation test needed:**
1. Full model: Encoder → LSTM → Power
2. No LSTM: Encoder → Power (direct)
3. Compare R² scores

**Prediction**: LSTM provides <5% improvement, possibly hurts.

**Why?** Delay embedding + windowing already captures temporal patterns. LSTM may just add parameters without adding information.

### Priority 3: Fix Context Switch Normalization 🔧

**Options:**
1. Log-scale context switches: `log(1 + ctx_switches)`
2. Clip outliers: Cap at 99th percentile
3. RobustScaler instead of MinMaxScaler
4. **Or just drop context switches entirely** if they're VM-polluted

## Concrete Action Plan

### Test 1: Increase Latent Dimension

Edit `config_tau8_multifeature.yaml`:
```yaml
model:
  latent_dim: 48  # Changed from 16
```

Retrain and compare:
- Before: R²=0.29, MAPE=12%
- After: R²=? (should improve significantly if bottleneck was the issue)

### Test 2: Remove LSTM Layer

Create simplified model:
```python
# In model.py, create alternative PowerHead
class DirectPowerHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, z_sequence):
        # Use last timestep directly
        return self.fc(z_sequence[:, -1, :])
```

Compare:
- With LSTM: Current performance
- Without LSTM: Simpler, possibly same or better

### Test 3: Fix Context Switches

In preprocessing, add before normalization:
```python
# Clip context switches to reasonable range
ctx_switches = np.clip(ctx_switches, 0, 50000)
# Or log-scale
ctx_switches_log = np.log1p(ctx_switches)
```

## Expected Outcomes

**If latent space is the bottleneck:**
- Increasing to 48-64 dim should improve R² from 0.29 to 0.7-0.9
- Multi-feature will finally work as intended

**If LSTM is redundant:**
- Removing LSTM won't hurt R²
- Model becomes simpler and faster
- Fewer parameters = less overfitting

**If context switches are the problem:**
- Better normalization will reduce systematic bias
- MAPE should drop from 12% to <8%

## Bottom Line

**The multi-feature model failed because:**

1. ❌ **16-dim latent is too small** for 75-dim input (4.7:1 compression)
2. ❌ **Context switch normalization** was destroyed by 695k/sec outliers from VMs
3. ⚠️  **LSTM may be unnecessary** given delay embedding already captures time

**To fix:**

1. **Increase latent_dim to 48** (quick test, likely big improvement)
2. **Log-scale or clip context switches** before training
3. **Consider removing LSTM** entirely (simpler is better)

The single-feature model works (R²=0.97) because it doesn't have these problems - it has a reasonable compression ratio and no VM-polluted features.
