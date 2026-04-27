# Latent Space Analysis: DECODE-RAPL

## Overview

DECODE-RAPL learns a **16-dimensional latent space** that compresses delay-embedded CPU usage patterns into a compact representation optimized for power prediction. This document analyzes what this latent space represents and why it's effective.

## 1. Dimensional Structure

### Input → Latent Compression

```
Raw CPU Usage (1D)
    ↓ Delay Embedding (τ=8, d=25)
25D Embedded Space
    ↓ Encoder (512 → 128 → 64 → 16)
16D Latent Space
    ↓ LSTM (120 timesteps)
Power Prediction (1D)
```

**Compression Pipeline:**
- **Input dimension:** 25 (delay embedding with d=25 from single CPU usage variable)
- **Latent dimension:** 16 (config.yaml:29)
- **Compression ratio:** 25 → 16 (36% reduction)
- **Encoder architecture:** 25 → 512 → 128 → 64 → 16 (non-linear, 3 ReLU layers)
- **Decoder architecture:** 16 → 64 → 128 → 512 → 25 (symmetric)

**Key insight:** Mild compression rather than aggressive dimensionality reduction. The model preserves most information content while learning a more structured representation.

## 2. What the Latent Space Represents

The 16-dimensional latent space learns a compact representation of **CPU usage dynamics** that captures:

### 2.1 Disentangled Temporal Patterns

- **Input:** 25 historical snapshots of CPU usage (via delay embedding)
- **Latent:** 16 dimensions capturing essential temporal dynamics
- **Function:** Learns "basis patterns" of CPU behavior
  - Ramp-ups (load increasing)
  - Plateaus (steady-state)
  - Drops (load decreasing)
  - Oscillations (periodic workloads)

### 2.2 Power-Relevant Features (Task-Specific)

**Loss function priorities:**
- Power prediction loss: **1.0** (primary objective)
- Reconstruction loss: **0.05** (minimal)
- Adversarial loss: **0.0** (disabled)

**Implication:** The latent space is optimized to **discard dynamics that don't affect power**, making it a "lossy by design" autoencoder.

**Examples of what gets discarded:**
- Brief CPU spikes that don't affect thermal mass
- High-frequency noise in usage readings
- Transients shorter than thermal time constants

**Examples of what gets preserved:**
- Sustained load changes (affect temperature → frequency → power)
- Ramp rates (thermal dynamics)
- Recent history (thermal state accumulation)

### 2.3 (Weakly) Machine-Invariant Representation

- **Original design:** Adversarial discriminator to encourage machine-invariance
- **Final configuration:** Adversarial loss disabled (weight = 0.0)
- **Observation:** 16-dim space generalizes naturally without explicit adversarial forcing
- **Hypothesis:** Power-relevant dynamics are inherently similar across machines, so task-specific optimization alone creates portable representations

## 3. Latent Space Properties (Empirical)

### 3.1 Information Preservation

**Evidence from tau8 results (3.56% MAPE):**
- Despite 36% compression (25→16), near-perfect predictions achieved
- **Conclusion:** 16 dimensions are **sufficient** to capture power-relevant CPU dynamics

**Theoretical justification (Takens' Embedding Theorem):**
- Need ≥ 2n+1 dimensions to reconstruct n-dimensional dynamical system
- If CPU power dynamics are ~6-7 dimensional, 16 latent dims provide comfortable margin
- Delay embedding (d=25) first unfolds the system, then encoder compresses to essential features

### 3.2 Smoothness and Manifold Structure

**Evidence:**
- LSTM operates on latent sequences, requiring smooth temporal transitions
- Low prediction error (mean=-0.16W, std=2.39W) suggests smooth manifold
- **Conclusion:** Nearby points in latent space correspond to similar power states

**Interpretation:**
- Latent space forms a **smooth 16-dimensional manifold** embedded in R^16
- Trajectories in this space correspond to workload executions
- LSTM learns dynamics on this manifold (latent state evolution → power)

### 3.3 Temporal Coherence

**Architecture design:**
- Encoder operates **independently on each timestep**: (batch, seq_len, 25) → (batch, seq_len, 16)
- Creates a **latent trajectory** rather than single latent vector
- LSTM models temporal dependencies in 16-dim latent trajectory space

**Evidence of coherence:**
- Tight error distribution (std=2.39W across entire test set)
- Successful prediction from 120-timestep windows (1.92 seconds of history)
- **Conclusion:** Latent space preserves temporal causality (future states predictable from past)

## 4. Comparison to Other Latent Spaces

### DECODE-RAPL vs VAE (Variational Autoencoder)

| Aspect | DECODE-RAPL | VAE |
|--------|-------------|-----|
| Encoder output | Deterministic point | Probability distribution (μ, σ) |
| Loss terms | Reconstruction + Power MSE | Reconstruction + KL divergence |
| Optimization goal | Prediction accuracy | Generative quality + regularization |
| Latent space structure | Functional mapping | Probabilistic (Gaussian prior) |

**Key difference:** DECODE-RAPL learns a **functional mapping** optimized for power prediction, not a generative model.

### DECODE-RAPL vs PCA (Principal Component Analysis)

| Aspect | DECODE-RAPL | PCA |
|--------|-------------|-----|
| Compression | Non-linear (3 ReLU layers) | Linear projection |
| Manifold | Can learn curved manifolds | Linear subspace only |
| Optimization | Task-specific (power prediction) | Variance maximization |
| Temporal modeling | Sequence-aware (via LSTM) | Static (no temporal structure) |

**Key difference:** PCA would give 25→16 linear projection. DECODE-RAPL learns **non-linear compression** tailored to power prediction.

### DECODE-RAPL vs MS-TCN

| Aspect | DECODE-RAPL | MS-TCN |
|--------|-------------|---------|
| Latent space | Explicit 16-dim intermediate representation | No explicit latent space |
| Architecture | 25 → 16 latent → LSTM → power | 14 features → TCN → power |
| Preprocessing | Delay embedding (1 var → 25 dims) | Multi-variate features (14 vars) |
| Interpretability | Latent trajectories can be visualized | Hidden states not interpretable |

**Key difference:** DECODE-RAPL creates an **interpretable intermediate representation** that could be analyzed (e.g., visualize latent trajectories for different workloads).

## 5. Physical Interpretation

### Hypothesized Dimension Roles

Based on the model achieving 3.56% MAPE with 16 dimensions, we hypothesize the latent space encodes:

1. **Current computational intensity** (1-2 dims)
   - Immediate power draw from current CPU utilization
   - Proportional to active cores × frequency

2. **Thermal state** (2-3 dims)
   - Recent history affecting temperature
   - Temperature affects turbo boost limits
   - Thermal mass creates power lag (seconds-scale)

3. **Workload pattern characteristics** (3-5 dims)
   - Bursty vs sustained execution
   - I/O-bound vs compute-bound
   - Memory-intensive vs CPU-intensive
   - Single-thread vs multi-thread

4. **Transition dynamics** (2-3 dims)
   - Ramp-up rate (load increasing)
   - Ramp-down rate (load decreasing)
   - Acceleration (second derivative)

5. **Residual/interaction patterns** (4-5 dims)
   - Non-linear interactions between features
   - Higher-order temporal patterns
   - Frequency-scaling dynamics (DVFS)

### Why 16 Dimensions Are Sufficient

**CPU power is primarily determined by:**
- Frequency (f)
- Voltage (V, coupled to f via DVFS)
- Utilization (U)
- Temperature (T, affects f via thermal throttling)

**But these are not independent:**
- V = V(f) - DVFS coupling
- f = f(U, T) - Turbo boost depends on load and temperature
- T = T(history of power) - Thermal mass integration

**Dimensionality estimate:**
- Current state: [f, V, U, T] ≈ 3-4 independent dims (due to coupling)
- 1st derivative: [df/dt, dU/dt, dT/dt] ≈ 3 dims
- 2nd derivative: [d²U/dt², d²T/dt²] ≈ 2 dims
- Higher-order terms and interactions: ≈ 7 dims

**Total: ~15 dimensions**

The model uses **16 dimensions**, providing just enough capacity without overfitting.

## 6. What Makes This Latent Space Effective

### 6.1 Task-Specific Co-Optimization

Unlike generic autoencoders (optimized for reconstruction only), this latent space is **co-optimized with the power predictor**.

**Training objective:**
```
L_total = 1.0 × L_power + 0.05 × L_reconstruction + 0.0 × L_adversarial
```

**Consequence:** The encoder "knows" its output will be fed to an LSTM for power prediction, creating a latent space with **temporal structure that LSTMs can exploit**.

### 6.2 Two-Stage Dimensionality Processing

**Stage 1: Unfold (Delay Embedding)**
- 1D CPU usage → 25D delay-embedded space
- Takens' theorem guarantees full dynamics reconstruction
- Expands the representation to reveal temporal structure

**Stage 2: Compress (Encoder)**
- 25D embedded space → 16D latent space
- Non-linear compression extracts power-relevant features
- Discards noise and power-irrelevant dynamics

**Benefit:** Two-stage process (unfold → compress) is more effective than direct 1D → 16D mapping.

### 6.3 Sequence-to-Sequence Operation

**Architecture choice:**
```python
# Encoder processes each timestep independently
z = self.encoder(x)  # (batch, seq_len, 25) → (batch, seq_len, 16)

# LSTM processes latent sequence
power = self.lstm(z)  # (batch, seq_len, 16) → (batch, 1)
```

**Creates:** A **latent trajectory** (sequence of 16-dim points) rather than a single latent vector.

**Benefit:** LSTM learns temporal dependencies in this **16-dim latent trajectory space**, which has better structure than the raw 25-dim delay embedding.

## 7. Potential Improvements and Trade-offs

### Could We Use Fewer Dimensions?

**Hypothesis:** Reduce to 12-14 dimensions with minimal accuracy loss.

**Expected behavior:**
- 12-14 dims: Slight degradation (maybe 3.8-4.0% MAPE)
- 8-10 dims: Noticeable degradation (>4.5% MAPE)
- <8 dims: Significant degradation (>5% MAPE, fails target)

**Trade-off:** Smaller latent space = faster inference, but risk underfitting.

### Could We Use More Dimensions?

**Hypothesis:** Beyond 20 dims = diminishing returns.

**Expected behavior:**
- 18-20 dims: Minimal improvement (<0.1% MAPE reduction)
- >24 dims: Risk of overfitting, longer training time

**Evidence:** Already at 3.56% MAPE (near-optimal for this task).

**Trade-off:** Larger latent space = more capacity, but slower and risk overfitting.

### Could We Add Structure?

**Disentangled representations (e.g., β-VAE style):**
- Separate dimensions for frequency vs thermal state vs workload pattern
- Might improve **interpretability** (could analyze each factor independently)
- **Unlikely to improve prediction accuracy** (task doesn't require disentanglement)

**Hierarchical latent spaces:**
- Coarse-grained (4 dims) + fine-grained (12 dims)
- Could enable multi-scale temporal modeling
- **Added complexity** without clear benefit for this application

## 8. Summary

The DECODE-RAPL latent space is a **16-dimensional learned representation** that:

1. **Compresses** delay-embedded CPU usage (25 dims → 16 dims) while preserving power-relevant dynamics
2. **Encodes temporal patterns** predictive of power consumption (ramps, plateaus, thermal state)
3. **Operates as a smooth manifold** enabling LSTM-based sequence modeling
4. **Is sufficient** to achieve 3.56% MAPE (meets <5% target with margin)
5. **Balances capacity vs efficiency** - enough to capture complexity, not so large as to overfit
6. **Is task-optimized** - discards power-irrelevant dynamics, retains power-relevant features

### In One Sentence

The latent space learns a **compact, temporally-coherent representation** of CPU usage dynamics that extracts the 16 most predictive features for power consumption from 25 delay-embedded measurements.

### Key Insight

The effectiveness comes from **three-stage processing**:
1. Delay embedding unfolds 1D → 25D (reveals temporal structure)
2. Non-linear encoder compresses 25D → 16D (extracts power-relevant features)
3. LSTM models 16D latent trajectories → power (learns temporal dependencies)

This architecture separates concerns: delay embedding handles temporal unfolding (fixed, theory-based), encoder handles feature extraction (learned), and LSTM handles sequence modeling (learned).

---

## References

- **Takens' Embedding Theorem:** Takens, F. (1981). "Detecting strange attractors in turbulence."
- **Delay Embedding for Power Modeling:** See `docs/delay_embedding_theory.md`
- **Model Architecture:** See `src/model.py`
- **Training Configuration:** See `config.yaml`
- **Training Results:** See `results/tau8/plots/`
