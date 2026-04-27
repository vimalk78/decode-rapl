# Deep Delay Autoencoder Approach for VM-Portable Power Prediction

## Overview

This document captures a future research direction for achieving VM portability using deep learning to discover latent dynamical system representations of CPU power consumption.

## Motivation

### Current Challenges
- **Extrapolation Problem**: Models trained on 20-core systems cannot predict power for 4-core VMs
- **Feature Engineering Limitations**: Hand-crafted features (CPU time, percentages) don't capture underlying power dynamics
- **Architecture Specificity**: RAPL power behavior differs across Intel architectures (Skylake, Haswell, etc.)
- **Non-linear Scaling**: Power doesn't scale linearly with core count due to:
  - Idle power baselines
  - Frequency scaling behavior
  - Thermal characteristics
  - Turbo boost dynamics

### Proposed Solution
Model RAPL power as a **dynamical system** and use deep learning to discover its latent space representation automatically, then predict power from this learned representation.

## Technical Approach

### Architecture
```
Time Series Data (System Metrics)
    ↓
Delay Embeddings (sliding window)
    ↓
Deep Autoencoder
    ↓
Latent Space (learned dynamics)
    ↓
LSTM Predictor
    ↓
Power Prediction
```

### Key Components

1. **Delay Embeddings**
   - Use time-delayed versions of system metrics
   - Currently using 64-sample sequences - can reuse or extend
   - Creates phase space reconstruction of system state

2. **Deep Autoencoder**
   - Encoder: Compress delay embeddings to low-dimensional latent space
   - Decoder: Reconstruct original embeddings from latent space
   - Learns fundamental structure of power dynamics

3. **Latent Space Representation**
   - Captures architecture-specific power behavior (Skylake, Haswell, etc.)
   - Represents underlying "physics" of power consumption
   - Should generalize across different core counts if dynamics are truly learned

4. **LSTM Predictor**
   - Takes latent variables as input
   - Predicts future power consumption
   - Models temporal dependencies in latent space

### Loss Function
```
Total Loss = α × Reconstruction Loss + β × Prediction Loss

where:
  Reconstruction Loss = MSE(decoded_embeddings, original_embeddings)
  Prediction Loss = MSE(predicted_power, actual_power)
  α, β = hyperparameters to balance objectives
```

### Training Strategy
- **Joint Training**: Train autoencoder and predictor simultaneously
- **Two-phase Training** (alternative):
  1. Pre-train autoencoder on reconstruction
  2. Fine-tune full model on power prediction
- **Multi-architecture Training**: Train on Skylake + Haswell data to learn architecture-agnostic latent dynamics

## Theoretical Foundation

Based on paper: **"Deep Delay Autoencoders Discover Dynamical Systems with Latent Variables"**
- Paper: https://arxiv.org/pdf/2201.05136
- Authors demonstrate that delay autoencoders can discover governing equations of dynamical systems
- We don't need full equation discovery - just latent space for prediction

### Key Insight
RAPL power consumption is a dynamical system:
- **State**: CPU utilization, memory usage, system activity
- **Dynamics**: How power evolves based on state changes
- **Parameters**: Architecture-specific (TDP, frequency curves, thermal limits)

If we learn the latent dynamics, model should generalize to:
- Different core counts (4-core VM vs 20-core baremetal)
- Different architectures (with appropriate training data)
- Different workload patterns (if dynamics are truly captured)

## VM Portability Advantage

### Why This Could Work
1. **Learned Representations**: Latent space captures fundamental power dynamics, not surface-level correlations
2. **Scale Independence**: If latent variables represent "work intensity" and "thermal state", these exist regardless of core count
3. **Architectural Modeling**: Explicitly model that RAPL behavior is architecture-specific
4. **Generalization**: Model learns "how power changes" not "what power is", enabling extrapolation

### Comparison to Current Approaches

| Approach | VM Portability | Architecture Portability | Requires VM Data |
|----------|----------------|-------------------------|------------------|
| CPU Percentages | ❌ No | ✓ Maybe | No |
| CPU Time Features | ❌ Extrapolation | ✓ Maybe | No |
| Potentiometer Scaling | ❌ Failed | ❌ No | No |
| **Deep Delay Autoencoder** | **✓ Possible** | **✓ Yes** | **Initially No** |

## Implementation Plan (Future)

### Phase 1: Proof of Concept (Single Architecture)
1. Implement delay embedding layer
2. Design autoencoder architecture (encoder/decoder dimensions)
3. Add LSTM predictor head
4. Implement combined loss function
5. Train on baremetal Skylake data
6. Test on Skylake VM → validate latent space generalization

### Phase 2: Multi-Architecture (If Phase 1 succeeds)
1. Collect power data from different Intel architectures
2. Add architecture embedding/conditioning
3. Train unified model across architectures
4. Test cross-architecture VM portability

### Phase 3: Equation Discovery (Optional)
1. Apply SINDy (Sparse Identification of Nonlinear Dynamics) to latent space
2. Discover symbolic equations governing power dynamics
3. Use for interpretability and physical validation

## Data Requirements

### Minimum (Phase 1)
- Existing baremetal training data (450K samples, 20-core)
- VM test data (4-core, 8-core, 16-core)

### Ideal (Phase 2)
- Multiple architectures: Skylake, Haswell, Cascade Lake
- Multiple core counts per architecture
- Diverse workload patterns

## Hyperparameters to Explore

- **Delay embedding window**: 32, 64, 128 samples
- **Latent dimension**: 4, 8, 16, 32
- **Autoencoder depth**: 2-5 layers
- **LSTM hidden size**: 64, 128, 256
- **Loss balance**: α=0.5/β=0.5, α=0.3/β=0.7, etc.

## Success Metrics

1. **Reconstruction Quality**: Can autoencoder reconstruct delay embeddings?
2. **Baremetal Accuracy**: Match current MAE=1.5W, R²=0.985
3. **VM Accuracy**: Achieve <3W MAE on 4-core VM (untrained)
4. **Cross-Architecture**: <5W MAE on different Intel architecture

## References

1. Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2022). "Deep Delay Autoencoders Discover Dynamical Systems with Latent Variables". arXiv:2201.05136
2. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems". PNAS.
3. Takens, F. (1981). "Detecting strange attractors in turbulence". Dynamical Systems and Turbulence.

## Status

**Current State**: Idea documented, not yet implemented

**Prerequisites**: CPU time feature approach tested first to establish baseline

**Decision Point**: Implement this if CPU time approach fails VM testing

## Lessons from CPU Time Approach (Updated 2025-10-11)

### What We Learned

After implementing and testing CPU time features (14f model with real CPU time from jiffies):

**1. Feature Engineering Alone Insufficient**
- Fixed distribution mismatch: training now uses real CPU time, not derived from percentages
- Model achieved MAE=4.06W, R²=0.9739 on stress-ng workload (comparable to 17f baseline)
- **Critical finding**: Model still fails on diverse workloads (VM, browser, compiler)
  - Predicted 38-40W for 25-26W idle host with VM running
  - R²=0.7210 with VM processes vs R²=0.9739 on pure stressor

**2. Training Data Diversity Problem**
- All training data from stress-ng workloads only
- Model learned "stress-ng pattern → power" not "CPU work → power physics"
- Different workloads produce different feature patterns:
  - stress-ng: High interrupts (>50K/sec), controlled memory, predictable cycles
  - VM/browser: Irregular bursts, lower interrupts (~1-10K/sec), JIT/GC, cache thrashing
- **Key insight**: Cannot enumerate all feature combinations - need model that learns causation

**3. MS-TCN Limitations Exposed**
- Architecture learned correlations without physical constraints:
  - No thermal lag modeling → overshoots on CPU spikes
  - No settling dynamics → doesn't track gradual power rise
  - No bounds → can predict negative power or physically impossible values
- Multi-scale convolution + attention = pattern matching, not physics modeling
- Pure statistical learning can't generalize to out-of-distribution inputs

### Implications for Deep Delay Autoencoder

**Advantages Over MS-TCN:**
- ✓ Could discover latent dynamics that represent fundamental power behavior
- ✓ Latent space might capture "thermal state" and "work intensity" independent of specific patterns
- ✓ Autoencoder reconstruction loss encourages learning compact representation of dynamics

**Still Needs Addressing:**
- ⚠️ Training data diversity still critical - needs VM, compiler, browser, database workloads
- ⚠️ Pure data-driven approach may still memorize patterns without physics
- ⚠️ No guarantee latent space discovers physically meaningful variables

**Recommended Enhancements:**
1. **Hybrid approach**: Combine autoencoder with physics-informed constraints
   - Add loss terms for known physics (thermal lag, power bounds, package≥core)
   - Constrain latent space to physically interpretable dimensions

2. **Diverse training strategy**:
   - Phase 1: stress-ng (establish baseline)
   - Phase 2: Add VM workloads, browsers, compilers
   - Phase 3: Add idle periods, low-activity scenarios

3. **Latent space regularization**:
   - Encourage smooth latent trajectories (penalize sudden jumps)
   - Add physics-based priors (e.g., thermal time constant ~1-2 seconds)

### Updated Success Metrics

Based on actual performance of current models:

1. **Reconstruction Quality**: MSE < 0.1 on delay embeddings
2. **Baremetal Accuracy (stress-ng)**: MAE ≤ 4W, R² ≥ 0.97 (current baseline)
3. **Diverse Workload Accuracy**: MAE < 5W on VM/browser/compiler (untrained patterns)
4. **VM Portability**: MAE < 5W on 4-core VM when trained on 20-core
5. **Physical Consistency**: No overshoots >10W, predictions always ≥ 0

### Decision Point

After CPU time testing, **both approaches need physics constraints**:
- Pure autoencoder: Better than MS-TCN but may still fail on distribution shift
- **Recommended**: Physics-Informed Neural Networks (PINN) as next step
  - Encode known power laws directly in loss function
  - Then consider hybrid PINN + autoencoder for residuals

---

*Document created: 2025-10-11*
*Author: Vimal Kumar + Claude*
*Updated: 2025-10-11 (post CPU time testing)*
