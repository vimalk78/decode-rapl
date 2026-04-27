# DECODE-RAPL v2 Architecture

Technical details of the v2 model architecture and design decisions.

## Overview

DECODE-RAPL v2 is a simplified autoencoder-based architecture that predicts CPU package power from delay-embedded CPU metrics. Key improvements over v1:

- **No LSTM** - delay embedding already provides temporal encoding
- **Wider latent space** - 64 dimensions (vs 16 in v1)
- **Direct MLP power head** - simpler than LSTM-based prediction
- **Single vector processing** - no sliding window dimension

## Model Components

### 1. Encoder (100 → 64)

**Purpose:** Compress delay-embedded input to latent representation

**Architecture:**
```
Input (batch, 100)
   ↓
Linear(100 → 512) + ReLU + Dropout(0.2)
   ↓
Linear(512 → 128) + ReLU + Dropout(0.2)
   ↓
Linear(128 → 64)
   ↓
Latent (batch, 64)
```

**Parameters:** ~65K

**Design rationale:**
- Gentle compression ratio (100:64 = 1.56:1) gives features "room" to be represented
- Wider bottleneck than v1 (64 vs 16) to avoid information loss
- Two hidden layers provide sufficient capacity for non-linear mapping

### 2. Decoder (64 → 100)

**Purpose:** Reconstruct input for autoencoder quality assessment

**Architecture:**
```
Latent (batch, 64)
   ↓
Linear(64 → 128) + ReLU + Dropout(0.2)
   ↓
Linear(128 → 512) + ReLU + Dropout(0.2)
   ↓
Linear(512 → 100)
   ↓
Reconstructed (batch, 100)
```

**Parameters:** ~65K

**Design rationale:**
- Symmetric with encoder (mirror architecture)
- Reconstruction loss ensures latent space preserves input information
- Not used during inference (only encoder + power head)

### 3. Power Head (64 → 1) - NEW in v2

**Purpose:** Predict power directly from latent space

**Architecture:**
```
Latent (batch, 64)
   ↓
Linear(64 → 128) + ReLU + Dropout(0.2)
   ↓
Linear(128 → 64) + ReLU + Dropout(0.2)
   ↓
Linear(64 → 1)
   ↓
Power (batch, 1)
```

**Parameters:** ~12K

**Design rationale:**
- **Replaces LSTM from v1** - simpler and more direct
- MLP is sufficient because temporal encoding is already in the input vector
- No activation on final layer (regression task, not classification)
- Two hidden layers allow non-linear power mapping

### 4. Full Model

**Total parameters:** ~267K (much smaller than v1's LSTM-based model)

**Forward pass:**
```
Input (batch, 100)
   ↓
Encoder → Latent (batch, 64)
   ↓                     ↓
Decoder              Power Head
   ↓                     ↓
Reconstructed      Power Prediction
(batch, 100)          (batch, 1)
```

**Inference mode (power prediction only):**
```
Input (batch, 100)
   ↓
Encoder → Latent (batch, 64)
   ↓
Power Head
   ↓
Power (batch, 1)
```

## Input Format

### Delay-Embedded Vector (100 dimensions)

The input is a single delay-embedded vector with **feature-grouped ordering**:

```
Position  0-24:  [user%(t), user%(t-τ), ..., user%(t-24τ)]
Position 25-49:  [sys%(t), sys%(t-τ), ..., sys%(t-24τ)]
Position 50-74:  [iowait%(t), iowait%(t-τ), ..., iowait%(t-24τ)]
Position 75-99:  [log_ctx(t), log_ctx(t-τ), ..., log_ctx(t-24τ)]
```

**Why feature-grouped?**
- Temporal history for each feature is contiguous
- Easier to interpret which features contribute to latent dimensions
- Encoder can learn feature-specific temporal patterns

**Parameters (tau=1):**
- d=25 delays
- τ=1 sample (16ms)
- 4 features
- Total lookback: 24 samples = 384ms

## Loss Function

### Combined Loss

```python
total_loss = power_weight * power_mse + reconstruction_weight * reconstruction_mse
```

**Default weights:**
- `power_weight = 1.0` (main objective)
- `reconstruction_weight = 0.1` (reduced from v1's 0.5)

**Components:**

1. **Power MSE Loss:**
   ```
   power_loss = MSE(predicted_power, actual_power)
   ```
   - Main training objective
   - Measures prediction accuracy
   - Direct supervision signal

2. **Reconstruction MSE Loss:**
   ```
   reconstruction_loss = MSE(reconstructed_input, original_input)
   ```
   - Ensures latent space preserves input information
   - Prevents encoder from collapsing to trivial solution
   - Acts as regularization

**Why reconstruction loss is reduced in v2:**
- Power prediction is the primary goal
- With 64-dim latent space (vs 16), reconstruction is easier
- Lower weight prevents over-emphasis on autoencoder quality at expense of power accuracy

## Key Differences from v1

| Aspect | v1 | v2 | Rationale |
|--------|----|----|-----------|
| **Input shape** | (batch, window=15, emb=75) | (batch, 100) | Simpler - delay embedding handles temporal structure |
| **Latent dim** | 16 | 64 | Wider bottleneck reduces information loss |
| **Temporal** | LSTM (hidden=128) | None | Delay embedding replaces LSTM |
| **Power head** | LSTM + Linear | MLP (128→64→1) | Simpler, more direct mapping |
| **Window size** | 15 sequences | 1 vector | Single vector is sufficient |
| **Parameters** | ~500K | ~267K | Simpler architecture |
| **Reconstruction weight** | 0.5 | 0.1 | Focus more on power prediction |

## Design Rationale

### Why Remove LSTM?

**Problem with v1:** LSTM processes sequences of delay-embedded vectors, creating **double temporal encoding**:
- Delay embedding: t, t-1, t-2, ..., t-24 (temporal structure in vector)
- LSTM: processes sequence of these vectors (temporal structure in sequence)

**v2 Solution:** Process **single delay-embedded vectors**
- Takens' theorem: delay embedding reconstructs attractor (captures temporal dynamics)
- LSTM is redundant - the 100-dim vector already contains temporal history
- MLP is sufficient to map this vector to power

**Benefits:**
- Simpler architecture
- Faster inference (no recurrent computation)
- Easier to interpret (static mapping vs stateful LSTM)

### Why Widen Latent Space (16 → 64)?

**Problem with v1:** 75-dim input → 16-dim latent is aggressive compression (4.7:1 ratio)
- Risk of information bottleneck
- Hard for 4 new features to share 16 dimensions

**v2 Solution:** 100-dim input → 64-dim latent (1.56:1 ratio)
- Gentler compression
- Each feature has ~16 dimensions "on average" (64 ÷ 4)
- Reduces risk of losing important patterns

### Why Feature-Grouped Ordering?

**Alternative:** Time-grouped (all features at t, then all at t-1, etc.)
```
[user(t), sys(t), iowait(t), log_ctx(t), user(t-1), sys(t-1), ...]
```

**Chosen:** Feature-grouped (all timesteps for user%, then for sys%, etc.)
```
[user(t), user(t-1), ..., sys(t), sys(t-1), ..., iowait(t), ...]
```

**Benefits:**
- Feature temporal history is contiguous (easier for encoder to learn feature-specific patterns)
- Latent space interpretation: some dimensions may encode user% dynamics, others sys% dynamics
- Better for visualization (can analyze which delays matter per feature)

## Latent Space Properties

The 64-dim latent space should encode:

1. **Workload characteristics:**
   - Compute-heavy (high user%, high power)
   - Syscall-heavy (high system%, low power)
   - I/O-bound (high iowait%, low power)
   - Mixed patterns

2. **Temporal dynamics:**
   - Bursts vs sustained load
   - Transition patterns
   - Recent vs distant history importance

3. **Power-relevant features:**
   - Dimensions correlated with power (the "power knob")
   - Dimensions that distinguish workload types

**Validation (planned):**
- t-SNE/UMAP visualization: should show smooth power gradient
- Gradient analysis: identify "power knob" dimensions
- Clustering: workload types should form distinct clusters

## Inference Pipeline

**Real-time power prediction:**

```python
# Collect CPU metrics (16ms intervals)
metrics = collect_metrics()  # user%, sys%, iowait%, ctx_switches

# Maintain sliding buffer (100 samples for tau=1, d=25)
buffer.append(metrics)

# Create delay-embedded vector when buffer is full
if len(buffer) >= 100:
    x = create_delay_embedding(buffer, tau=1, d=25)  # (1, 100)

    # Predict power (encoder + power head)
    with torch.no_grad():
        power = model.predict_power(x)  # (1, 1)
```

**Latency:** <1ms on CPU (no recurrence, small model)

## Training Strategy

**Optimizer:** Adam (lr=0.001, weight_decay=0.0001)

**Learning rate schedule:** ReduceLROnPlateau
- Factor: 0.5
- Patience: 5 epochs
- Min LR: 1e-5

**Early stopping:** Patience 15 epochs on validation loss

**Batch size:** 256 (fits in memory, good gradient estimates)

**Data augmentation:** None (not applicable for time series regression)

**Regularization:**
- Dropout: 0.2 in all hidden layers
- Weight decay: 0.0001 (L2 regularization)
- Reconstruction loss: Acts as auxiliary task regularization

## Comparison with v1 Results

| Metric | v1 (tau=1, single-feature) | v2 (tau=1, 4-feature) | Target |
|--------|----------------------------|----------------------|--------|
| Test R² | 0.85 (on stress-ng) | TBD | >0.95 |
| Test MAE | ~5W | TBD | <3W |
| Prometheus R² | 0.12 (failed!) | TBD | >0.90 |
| Prometheus MAE | ~15W | TBD | <5W |

**Expected improvements:**
- Better generalization (4 features disambiguate workload types)
- Lower error on syscall workloads (model can learn "high system% ≠ high power")
- Successful transfer to Prometheus (training data includes syscall patterns)

## Future Enhancements

**Possible improvements:**
1. **Attention mechanism** - learn which delays matter most per feature
2. **Multi-task learning** - predict power + workload type jointly
3. **Uncertainty estimation** - Bayesian layers or ensembles
4. **Online learning** - fine-tune on production data

**Not recommended:**
- Adding LSTM back (defeats purpose of delay embedding)
- Deeper networks (current capacity is sufficient)
- Narrower latent space (64 is already conservative)

## References

- Takens' Embedding Theorem: Time-delay embedding reconstructs attractor
- v1 Architecture: `decode-rapl-v1/src/model.py`
- Gemini Recommendations: `decode-rapl-v1/gemini/GEMINI_RESPONSE.md`
- v1 Analysis: `decode-rapl-v1/LSTM_ANALYSIS_SUMMARY.md`
