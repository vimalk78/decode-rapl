# Physics-Informed Neural Networks (PINN) for Power Prediction

## Overview

This document outlines an approach to power prediction that encodes known physical laws and constraints directly into the neural network architecture and loss function. Unlike pure statistical models (MS-TCN) that learn correlations, PINNs combine data-driven learning with physics-based reasoning.

## Motivation

### Why Current Approaches Fail

From CPU time testing (14f model) and MS-TCN analysis:

**Pure Statistical Learning Problems:**
1. **Memorizes patterns, not physics**: Learned "stress-ng → power" not "CPU work → power"
2. **Out-of-distribution failure**: MAE=4W on stress-ng, but 13W error on idle with VM
3. **No physical constraints**:
   - Overshoots on CPU spikes (predicts 55W when actual rises to 50W)
   - No thermal lag modeling
   - Can predict negative power or package < core
4. **Training data dependency**: Requires exhaustive enumeration of workload patterns

**Key Insight from Architecture Analysis:**
- Multi-scale convolution detects transitions but doesn't model physical response
- Attention learns relevance weights but not settling dynamics
- No layer enforces P ≥ 0, thermal time constants, or power scaling laws

### What Physics Can Provide

**Known Power Laws:**
1. **Dynamic power**: P_dyn ∝ C × V² × f (capacitance, voltage, frequency)
2. **Static power**: P_static = V × I_leak (leakage current)
3. **Thermal dynamics**: dT/dt = (P_in - P_out) / (m × c) with time constant τ ~ 1-2s
4. **Package composition**: P_package ≥ P_cores + P_uncore + P_dram
5. **Physical bounds**: P ≥ 0, P ≤ TDP

**Why This Helps:**
- Generalization: Physics is invariant across workloads (stress-ng, VM, browser)
- Extrapolation: Enables prediction on unseen patterns (20-core → 4-core VM)
- Interpretability: Model learns physically meaningful parameters
- Data efficiency: Less training data needed when physics provides structure

## Technical Approach

### Architecture Options

#### Option 1: Physics-Guided Neural Network
```
System Metrics
    ↓
Feature Encoder (Neural)
    ↓
Physics Layer (Differentiable)
    │  • Power = f(frequency, voltage, activity)
    │  • Thermal RC model: P(t) = P_in × (1 - e^(-t/τ))
    │  • Package = Σ(cores, uncore, dram)
    ↓
Residual Correction (Neural)
    │  • Learn deviations from ideal physics
    │  • Capture manufacturing variation
    │  • Model measurement noise
    ↓
Power Prediction
```

#### Option 2: Constrained Loss Function
```
Standard Neural Network (MS-TCN or simpler)
    ↓
Prediction: P_pred
    ↓
Multi-term Loss:
    L_total = L_data + λ_physics × L_physics + λ_bounds × L_bounds

where:
    L_data = MSE(P_pred, P_actual)
    L_physics = Physics violation penalties
    L_bounds = Constraint violation penalties
```

#### Option 3: Hybrid Neural ODE
```
System Metrics → Neural Network → dP/dt (learned dynamics)
    ↓
Integrate ODE: P(t+1) = P(t) + ∫dP/dt·dt
    ↓
Apply Physics Constraints:
    • Thermal settling: P(t+1) ≤ P(t) + ΔP_max
    • Package bounds: P_package ≥ Σ(components)
    • Non-negativity: P ≥ 0
```

### Physics Constraints to Encode

#### 1. Power Bounds
```python
L_bounds = relu(−P_pred)² + relu(P_pred − TDP)²
```
Ensures 0 ≤ P ≤ TDP

#### 2. Thermal Time Constant
```python
L_thermal = |P(t+1) − P(t) − (P_target − P(t)) × (1 − exp(−Δt/τ))|²
```
Enforces exponential settling with time constant τ ~ 1-2 seconds

#### 3. Package Composition
```python
L_package = relu(P_cores + P_uncore + P_dram − P_package)²
```
Ensures package ≥ sum of components

#### 4. Dynamic Power Scaling
```python
# Frequency-voltage relationship
L_dvfs = |P_dyn − α × V² × f|²

# Activity proportionality
L_activity = |P_cpu − (P_base + β × CPU_active × f)|²
```

#### 5. Temporal Smoothness
```python
L_smooth = |P(t+1) − P(t)|² for |CPU(t+1) − CPU(t)| < threshold
```
Prevents sudden jumps when CPU activity changes slowly

### Complete Loss Function

```python
L_total = (
    # Data fidelity
    w_data × MSE(P_pred, P_actual)

    # Physics constraints
    + w_bounds × L_bounds
    + w_thermal × L_thermal
    + w_package × L_package
    + w_dvfs × L_dvfs
    + w_smooth × L_smooth

    # Optional regularization
    + w_reg × (L1 or L2 regularization)
)
```

**Hyperparameters to tune**: w_data, w_bounds, w_thermal, w_package, w_dvfs, w_smooth

### Recommended Starting Point: Option 2 (Constrained Loss)

**Rationale:**
1. Reuse existing MS-TCN architecture (proven feature extraction)
2. Add physics constraints gradually through loss terms
3. Easier to implement than full Neural ODE
4. Can tune constraint weights independently

**Implementation:**
```python
class PhysicsInformedMSTCN(nn.Module):
    def __init__(self):
        self.mstcn = MSTCN(...)  # Existing model
        self.tau = nn.Parameter(torch.tensor(1.5))  # Learnable thermal time constant

    def forward(self, x):
        P_pred = self.mstcn(x)
        return P_pred

    def physics_loss(self, P_pred, P_prev, dt):
        # Thermal settling constraint
        L_thermal = thermal_constraint(P_pred, P_prev, self.tau, dt)

        # Bounds
        L_bounds = bounds_constraint(P_pred, TDP=125)

        # Smoothness
        L_smooth = smoothness_constraint(P_pred)

        return L_thermal + L_bounds + L_smooth
```

## Implementation Plan

### Phase 1: Minimal Physics Constraints (1-2 weeks)
**Goal**: Validate that physics constraints improve out-of-distribution generalization

1. **Start with bounds only**:
   - Add L_bounds to existing MS-TCN loss
   - Validate: Model never predicts P < 0 or P > TDP
   - Baseline: Test on stress-ng (should match current MAE=4W)

2. **Add thermal settling**:
   - Implement L_thermal with learnable τ
   - Validate: Overshoots reduced on CPU spikes
   - Test: Does prediction track gradual power rise better?

3. **Test on VM workload**:
   - Collect data: Idle host with VM/browser running (no new training!)
   - Compare: Physics-constrained model vs baseline MS-TCN
   - Success metric: MAE < 8W (vs current 13W error)

### Phase 2: Core Physics Laws (2-3 weeks)
**Goal**: Encode frequency-power relationship and package composition

4. **Add DVFS constraints**:
   - Need frequency data: Read `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
   - Implement L_dvfs: P ∝ V² × f
   - Learn α parameter from data

5. **Add package composition**:
   - Implement L_package: P_package ≥ P_cores
   - Test: Does model learn reasonable core/uncore split?

6. **Validation**:
   - Train on diverse workloads (stress-ng + collect VM/browser data)
   - Test generalization: Hold out one workload type, measure MAE
   - Success: MAE < 5W on unseen workload patterns

### Phase 3: VM Portability (3-4 weeks)
**Goal**: Test if physics constraints enable 20-core → 4-core extrapolation

7. **Collect multi-core data**:
   - Baremetal: 20 cores (existing)
   - VM: 4-core, 8-core, 16-core (new)
   - Diverse workloads on each

8. **Train physics-informed model**:
   - Train on 20-core only
   - Hypothesis: Physics constraints enable extrapolation

9. **Test VM portability**:
   - Predict on 4-core VM (never seen during training)
   - Success metric: MAE < 5W
   - Compare: PINN vs baseline MS-TCN vs potentiometer scaling

### Phase 4: Interpretability (Optional)
**Goal**: Extract physically meaningful parameters

10. **Analyze learned parameters**:
    - Thermal time constant τ: Should be ~1-2 seconds
    - DVFS coefficient α: Compare to Intel datasheets
    - Package composition: Does core/uncore split match architecture?

11. **Sensitivity analysis**:
    - Which physics constraints reduce overshoot?
    - Which enable VM portability?
    - Can we remove some for efficiency?

## Data Requirements

### Phase 1 (Bounds + Thermal)
- ✓ Existing: 450K samples, baremetal 20-core, stress-ng
- ✓ Existing: Test data with VM (idle host, for validation)

### Phase 2 (DVFS + Package)
- ⚠️ Need: Frequency data added to collection
- ⚠️ Need: VM/browser/compiler workload data (10-20 min each)
- ⚠️ Need: Idle periods with varied core counts

### Phase 3 (VM Portability)
- ❌ Need: 4-core VM data (30-60 min diverse workloads)
- ❌ Need: 8-core VM data (30-60 min diverse workloads)
- ❌ Need: 16-core VM data (30-60 min diverse workloads)

## Expected Improvements

### Over Baseline MS-TCN

| Metric | MS-TCN (17f) | MS-TCN (14f CPU time) | PINN (Expected) |
|--------|--------------|------------------------|-----------------|
| stress-ng MAE | 4.25W | 4.06W | 3.5-4.5W (similar) |
| stress-ng R² | 0.9802 | 0.9739 | 0.975-0.985 |
| VM/diverse MAE | ~13W | ~13W | **< 6W** |
| VM/diverse R² | 0.7210 | ~0.72 | **> 0.90** |
| Overshoot | 10-15W | 10-15W | **< 5W** |
| Negative predictions | Possible | Possible | **Never** |
| 20c→4c VM MAE | Untested | Untested | **< 5W (goal)** |

### Why PINN Should Outperform

1. **Bounds constraints**: No negative predictions, no >TDP predictions
2. **Thermal settling**: Reduces overshoot on CPU spikes by 50-70%
3. **Physics generalization**: Works on VM/browser without retraining
4. **DVFS modeling**: Captures frequency-power relationship explicitly
5. **Data efficiency**: Needs less diverse training data (physics provides structure)

## Risks and Mitigations

### Risk 1: Physics Too Simplistic
**Problem**: Real power behavior more complex than P ∝ V² × f

**Mitigation**:
- Use residual correction (neural network learns deviations)
- Start with soft constraints (tunable weights, not hard constraints)
- Validate against RAPL measurements, refine physics model

### Risk 2: Constraint Conflicts
**Problem**: Multiple physics constraints may be inconsistent

**Mitigation**:
- Add constraints incrementally (test each individually)
- Use weighted loss (can reduce weight if conflict detected)
- Monitor constraint violations during training

### Risk 3: Optimization Difficulty
**Problem**: Multi-objective loss harder to train than pure MSE

**Mitigation**:
- Start with pre-trained MS-TCN (warm start)
- Use adaptive weight scheduling (increase physics weight gradually)
- Try constrained optimization methods (Lagrange multipliers)

### Risk 4: Overfitting to Physics
**Problem**: Model forced to follow physics even when wrong

**Mitigation**:
- Always include L_data term (never pure physics)
- Use validation set to tune constraint weights
- Allow physics violations if data strongly disagrees

## Alternative: Hybrid Physics + Autoencoder

If PINN Phase 1-2 succeeds but Phase 3 (VM portability) fails:

**Combine PINN with Deep Delay Autoencoder**:
```
Delay Embeddings → Autoencoder → Latent Space (dynamics)
                                      ↓
                              PINN Predictor (physics-constrained)
                                      ↓
                              Power Prediction
```

**Advantage**: Latent space captures architecture-specific dynamics, PINN ensures physics compliance

## Success Criteria

### Phase 1: Proof of Concept
- ✓ Model never predicts P < 0 or P > TDP
- ✓ Overshoot reduced by ≥30% vs baseline
- ✓ MAE on stress-ng matches baseline (≤4.5W)
- ✓ MAE on VM/idle improved by ≥30% (from 13W to <9W)

### Phase 2: Physics Integration
- ✓ Learned τ in range [0.5, 3.0] seconds (physically plausible)
- ✓ DVFS relationship visible in model behavior
- ✓ MAE < 5W on held-out workload type (unseen during training)

### Phase 3: VM Portability (Primary Goal)
- ✓ MAE < 5W on 4-core VM when trained on 20-core
- ✓ R² > 0.90 on VM prediction
- ✓ Physically consistent predictions (bounds, settling, package composition)

### Phase 4: Interpretability
- ✓ Extracted parameters match known physics
- ✓ Can explain why model generalizes better than MS-TCN
- ✓ Identify which constraints most important for VM portability

## References

### Physics-Informed Neural Networks
1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". *Journal of Computational Physics*.

2. Karniadakis, G. E., et al. (2021). "Physics-informed machine learning". *Nature Reviews Physics*.

### CPU Power Modeling
3. Rotem, E., et al. (2012). "Power-Management Architecture of the Intel Microarchitecture Code-Named Sandy Bridge". *IEEE Micro*.

4. David, H., et al. (2010). "RAPL: Memory power estimation and capping". *ISLPED*.

### Neural ODEs and Dynamics
5. Chen, R. T., et al. (2018). "Neural Ordinary Differential Equations". *NeurIPS*.

6. Champion, K., et al. (2019). "Data-driven discovery of coordinates and governing equations". *PNAS*.

## Related Documents

- `MODEL_ARCHITECTURE.md`: MS-TCN baseline architecture and limitations
- `deep_delay_autoencoder_approach.md`: Alternative dynamics-based approach
- `CPU_TIME_FEATURES.md`: Lessons from feature engineering approach (if exists)

## Status

**Current State**: Planning document

**Prerequisites**: None - can start immediately with existing data

**Recommended**: Implement Phase 1 before collecting more data

**Next Steps**:
1. Implement L_bounds and L_thermal constraints
2. Modify training script to support physics loss
3. Test on existing stress-ng + VM data

---

*Document created: 2025-10-11*
*Author: Vimal Kumar + Claude*
