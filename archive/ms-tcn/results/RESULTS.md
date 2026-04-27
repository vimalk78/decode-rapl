# MS-TCN Power Prediction - Results and Analysis

## Executive Summary

We trained a Multi-Scale Temporal Convolutional Network (MS-TCN) to predict CPU power consumption from system metrics. While the model achieved **R² = 0.9063** on validation data, live predictions revealed critical limitations related to training data distribution and workload diversity.

**Key Finding:** The model works well only when test workloads match the training data distribution. Training data was heavily skewed toward low power (98% < 30W), causing poor predictions on high-power or different workloads.

---

## Training Results

### Dataset
- **Duration:** 60 minutes (3,600 seconds)
- **Samples:** 224,975 samples collected at 62.5 Hz
- **Input features:** 19 system metrics from /proc
  - CPU: user%, system%, idle%, iowait%, irq%, softirq%
  - System: context switches/sec, interrupts/sec
  - Memory: used, cached, buffers, free, swap
  - Load: 1min, 5min, 15min averages
  - Processes: running, blocked counts
  - Page faults per second
- **Targets:** 2 power domains (package, core) from RAPL

### Model Configuration
- **Architecture:** MS-TCN with 740,302 parameters
- **Input window:** 64 timesteps (1.024 seconds at 62.5 Hz)
- **Loss function:** Huber loss (dual-head for package and core)
- **Optimizer:** AdamW (lr=1e-4, weight decay=1e-5)
- **Learning rate schedule:** Cosine annealing (1e-4 → 1e-6)
- **Early stopping:** Patience = 30 epochs

### Training Metrics

```json
{
  "best_val_loss": 0.0405,
  "final_train_loss": 0.0148,
  "final_val_loss": 0.0420,
  "final_val_mae": 1.55W,
  "final_val_r2": 0.9063
}
```

**Interpretation:**
- **R² = 0.9063:** Model explains 90.6% of power variance in validation data
- **MAE = 1.55W:** Average absolute error of 1.55 watts
- Training stopped at epoch ~48 due to no improvement for 30 consecutive epochs
- Best model saved at epoch 22 (validation loss = 0.0405)

---

## Training Data Analysis

![Training Data Analysis](plots/training_data_analysis.png)

### Critical Issue: Severe Power Distribution Imbalance

**Power distribution in 60-minute training data:**

| Power Range | Sample Count | Percentage |
|-------------|--------------|------------|
| 0-10W       | 62,531       | 27.8%      |
| 10-20W      | 41,154       | 18.3%      |
| 20-30W      | 116,607      | **51.8%**  |
| 30-40W      | 3,726        | 1.7%       |
| 40-50W      | 703          | 0.3%       |
| 50-60W      | 180          | 0.1%       |
| 60-80W      | 64           | 0.03%      |
| **> 40W**   | **952**      | **0.4%**   |

**Key observations:**

1. **98% of training data is below 30W** - Model learned predominantly low-power patterns
2. **Only 0.4% above 40W** - Extremely limited exposure to high-power workloads
3. **CPU reached 100% but power stayed at 20-30W** (see bottom plot):
   - Around minute 120-200: 100% CPU → 20-30W power
   - Around minute 500-600: 100% CPU → 20-30W power
4. **Brief spikes to 50-80W** but not sustained - Only transient high power

### Root Cause Analysis

**Why didn't high CPU utilization produce high power?**

Comparing training data collection with live testing revealed:

| Scenario | CPU Utilization | Power Draw | Instruction Mix |
|----------|----------------|------------|-----------------|
| Training data collection | 100% (stress-ng default) | 20-30W | Generic stress workload |
| Live test with stress-ng --cpu-method ackermann | 25% | 15-35W | Ackermann function (heavy recursion) |
| Live test with load_generator cpu-focused | Variable | Variable | Mixed workload types |

**Conclusion:** Different instruction mixes produce different power signatures. The training workload used low-power instructions even at 100% CPU.

---

## Prediction Results

### Test Setup

**Inference configuration:**
- Model: `best_model.pth` (epoch 22, R²=0.9063)
- Sampling rate: 62.5 Hz (`--interval 0.016`)
- Prediction rate: 0.1 Hz (`--frequency 0.1`, once per 10 seconds)
- Workloads: stress-ng with various cpu-methods and load_generator.py

**Inference overhead:**
- With `--frequency 0.1`: ~3% total CPU (50% spike during inference every 10s)
- Without `--frequency`: ~45% total CPU (700% on single core)

### Best Predictions

When workload characteristics matched training data, predictions were excellent:

| Sample | Predicted Package | Actual Package | Error | Predicted Core | Actual Core | Error |
|--------|-------------------|----------------|-------|----------------|-------------|-------|
| 24     | 29.5W             | 33.6W          | -12.3% | 25.7W          | 24.8W       | +3.7% |
| 27     | 25.1W             | 25.4W          | **-1.0%** | 21.3W          | 20.8W       | **+2.4%** |
| 28     | 6.1W              | 6.3W           | -4.1%  | 3.6W           | 2.5W        | +46.1% |

**Sample 27 (near-perfect prediction):**
- Package: predicted 25.1W vs actual 25.4W (within 0.3W, -1.0% error)
- Core: predicted 21.3W vs actual 20.8W (within 0.5W, +2.4% error)

This demonstrates the model **can** predict accurately when conditions are right.

### Poor Predictions

When workload or power level didn't match training data:

| Sample | Predicted Package | Actual Package | Error | Predicted Core | Actual Core | Error |
|--------|-------------------|----------------|-------|----------------|-------------|-------|
| 1      | 9.9W              | 44.1W          | **-77.5%** | 6.9W           | 38.2W       | **-81.9%** |
| 22     | 17.9W             | 38.3W          | **-53.1%** | 14.2W          | 28.4W       | **-49.9%** |
| 42     | 22.2W             | 39.3W          | **-43.5%** | 17.1W          | 26.6W       | **-35.8%** |
| 45     | 5.6W              | 23.7W          | **-76.5%** | 3.1W           | 18.1W       | **-82.8%** |

**Pattern in failures:**
- Actual power 35-45W: Model predicts 15-25W (underprediction)
- Actual power < 10W: Model sometimes overpredicts
- Errors often 40-80% when outside training distribution

### Prediction Accuracy by Power Range

Analyzing all predictions in live_prediction_0.1Hz.csv:

| Actual Power Range | Samples | Avg Error | Pattern |
|--------------------|---------|-----------|---------|
| < 10W              | 18      | ±30-100%  | Overpredicts low idle power |
| 10-25W             | 15      | ±10-40%   | **Best accuracy** |
| 25-35W             | 12      | -10-30%   | Slight underprediction |
| > 35W              | 6       | **-40-80%** | **Severe underprediction** |

**Conclusion:** Model is most accurate in the 10-30W range where 98% of training data resides.

---

## Key Insights

### 1. Training Data Distribution is Critical

**Problem:** Extreme imbalance in training data
- 98% of samples concentrated in 0-30W range
- Only 952 samples (0.4%) above 40W out of 225K total
- Model learned to predict "typical" power (mean ≈ 17W)

**Impact:** Model defaults to predicting power in the familiar 15-25W range even when actual power is 40-50W.

**Analogy:** Training a person to estimate temperature using data 98% from 15-25°C. When asked about 40°C, they'd guess ~20-25°C because that's what they know.

### 2. Workload-Specific Power Signatures

**Discovery:** Same CPU utilization produces different power depending on instruction mix.

**Evidence from training data:**
- 100% CPU utilization during training → 20-30W power
- 25% CPU utilization with ackermann method → 15-35W power (higher power per CPU%)

**Reason:** Different instructions have different power characteristics:
- Simple ALU operations: Low power
- AVX-512 vector operations: High power
- Memory-intensive: Higher uncore power
- Branch-heavy: Different cache/pipeline behavior

**Implication:** Model needs training data covering all workload types it will encounter in production.

### 3. Temporal Resolution Matters

**Finding:** Inference must use same sampling rate as training (62.5 Hz).

**Failure mode:**
- Training: 64 samples @ 62.5 Hz = 1.024 seconds
- Inference with 10 Hz: 64 samples @ 10 Hz = 6.4 seconds
- Model sees 6× slower temporal dynamics → predictions fail

**Solution:** Always use `--interval 0.016` (62.5 Hz) for data collection, regardless of prediction frequency.

### 4. Inference Overhead Solved

**Problem:** Initial implementation ran inference at 62.5 Hz → 700% CPU → 45W overhead.

**Solution:** `--frequency` parameter decouples data collection from prediction:
- Data collection: 62.5 Hz (maintains temporal resolution)
- Inference: 0.1-1 Hz (reduces overhead to 3-5% CPU)

**Result:** Predictor no longer interferes with workload being measured.

---

## Limitations and Failure Modes

### 1. Out-of-Distribution Predictions

**Symptom:** Large errors (>40%) when actual power differs significantly from training mean (17W).

**Root cause:** Model extrapolates poorly beyond training data range.

**Example:**
- Training: mostly 10-30W
- Test: 40-45W sustained
- Prediction: defaults to ~20W (within training range)
- Error: -50%

### 2. Workload Generalization

**Symptom:** Good predictions on stress-ng, poor on browser workloads (or vice versa).

**Root cause:** Model learns instruction mix signatures from training data.

**Example:**
- Training: stress-ng (ALU-heavy, low memory)
- Test: Chrome + Google Meet (SIMD video decode, high memory bandwidth)
- Result: Different power for same CPU% → prediction mismatch

### 3. Feature Limitations

**Current features:** Only 19 aggregate metrics from /proc
- Cannot distinguish AVX-512 vs scalar instructions
- Cannot see cache miss rates
- Cannot see memory bandwidth
- Cannot see thermal state

**Impact:** Workloads with same aggregate CPU/memory but different micro-architecture behavior look identical to the model.

### 4. Warm-up Period

**Observation:** Predictions take 4-8 iterations to stabilize after workload change.

**Reason:** 64-sample window (1 second) still contains old workload data.

**Example with --frequency 0.1:**
- t=0-90s: Idle → predictions accurate for idle
- t=100s: Load starts
- t=100-140s: Window still has 50%+ idle samples → poor predictions
- t=140s+: Window fully populated with load samples → predictions improve

---

## Recommendations

### 1. Improve Training Data Collection

**Collect balanced power distribution:**
- Ensure equal samples across power ranges (0-10W, 10-20W, ..., 60-90W)
- Use longer sustained high-power workloads (minutes, not seconds)
- Target: 15-20% of samples above 40W

**Use diverse workload types:**
```bash
# Example training data collection
for method in ackermann matrixprod fft float factorial crc16; do
    stress-ng --cpu 8 --cpu-method $method --timeout 600s &
    python3 power_data_collector.py --duration 600 --append training.csv
done
```

**Include real applications:**
- Browsers with video playback
- Compilers (gcc, clang)
- Databases (postgres under load)
- ML inference (PyTorch, TensorFlow)

### 2. Enhanced Features (If Available)

**Hardware performance counters (via perf):**
- Instructions per second
- Cache misses (L1, L2, L3)
- Branch mispredicts
- TLB misses
- Memory bandwidth

**Thermal sensors:**
- CPU junction temperature
- Package temperature
- Thermal throttling state

**Frequency information:**
- Current CPU frequency per core
- DVFS state
- Turbo boost active/inactive

**Impact:** Better differentiation between workload types with same CPU%.

### 3. Model Architecture Improvements

**Consider:**
- Separate models for different workload classes
- Multi-task learning with workload classification
- Uncertainty estimation (predict confidence intervals)
- Ensemble of models trained on different data subsets

### 4. Deployment Strategy

**For VM environments:**
- Train separate model on VM-specific metrics
- Use host-side power attribution for validation
- Accept higher error rates (VMs have abstraction overhead)

**For production monitoring:**
- Continuous retraining with recent data
- Anomaly detection for out-of-distribution inputs
- Fallback to heuristic models when confidence is low

---

## Conclusion

The MS-TCN model demonstrates **proof of concept** for ML-based power prediction:
- Achieves R² = 0.90+ on validation data
- Predicts within 1-3W when workload matches training
- Inference overhead reduced to 3% with `--frequency` parameter

However, **production deployment requires addressing data quality**:
- Current training data is 98% concentrated in 0-30W range
- Model cannot generalize to high-power or different workloads
- Need balanced, diverse training data covering full power range

**Next steps:**
1. Re-collect training data with diverse workloads and balanced power distribution
2. Consider adding hardware performance counter features
3. Validate on production workloads before deployment
4. Implement continuous retraining and out-of-distribution detection

The architecture and methodology are sound. The bottleneck is training data quality, not model capacity.

---

## Files

- **Training summary:** `training_summary.json`
- **Training plots:** `plots/training_history.png`
- **Data analysis:** `plots/training_data_analysis.png`
- **Latest predictions:** `predictions/live_prediction_0.1Hz.csv`
- **Model checkpoint:** `../models/best_model.pth` (epoch 22, R²=0.9063)
