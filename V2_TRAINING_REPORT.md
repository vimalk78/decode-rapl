# DECODE-RAPL v2 Training Report

**Date**: October 21, 2025
**Model Version**: v2 (Autoencoder + Power Head)
**Tau Variant**: τ=1 (16ms sampling interval)

---

## 1. Data Collection

### Dataset Overview
- **Total Samples**: 5,865,318
  - Train: 4,692,254 (80%)
  - Validation: 586,531 (10%)
  - Test: 586,533 (10%)
- **Workload Files**: 2,028 stress-ng runs
- **Temporal Window**: 384ms lookback (24 samples × 16ms)

### Features (Delay Embedding)
- **Input Dimension**: 100 (4 features × 25 delays)
- **Features**:
  1. User CPU % (indices 0-24)
  2. System CPU % (indices 25-49)
  3. I/O Wait % (indices 50-74)
  4. log(context switches) (indices 75-99)

### Workload Coverage
- CPU stress: 0-100% (intervals: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
- System calls: 0-100%
- I/O workers: 0-100
- Pipe operations: 0-40
- VM operations: 0-2
- Cache stress: 0-20%

### Power Statistics
- **Train**: Mean=80.94W, Std=14.88W
- **Val**: Mean=80.97W, Std=14.86W
- **Test**: Mean=80.96W, Std=14.83W
- **Range**: ~24W (idle) to ~102W (max stress)

---

## 2. Model Architecture (v2)

### Design
```
Input (100) → Encoder → Latent (64) → Decoder → Reconstruction (100)
                           ↓
                      Power Head → Power (1)
```

### Layer Configuration
- **Encoder**: 100 → 512 → 128 → 64
- **Decoder**: 64 → 128 → 512 → 100
- **Power Head**: 64 → 128 → 64 → 1
- **Parameters**: 267,941 (~268K)
- **Samples-to-Params Ratio**: 21:1 (optimal range: 10-100)

### Loss Function
```
Total Loss = Power Loss + λ × Reconstruction Loss
```

---

## 3. Training Configuration

### Common Settings
- **Batch Size**: 256
- **Max Epochs**: 100
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Patience varies (10-15 epochs)
- **Device**: CUDA (GTX 1650)
- **Training Time**: ~26s/epoch (surprisingly fast!)

### Training Speed Analysis
- **Batches per Epoch**: 18,330 (4.69M samples / 256 batch size)
- **Samples per Second**: ~180,000
- **Total Training Time**: ~5.7 minutes (13 epochs with early stopping)

**Initial Concern**: 26 seconds per epoch seemed suspiciously fast for GTX 1650.
**Investigation**: Verified dataset loading showed 18,330 batches/epoch.
**Conclusion**: Speed is legitimate - GTX 1650 performing well on this workload:
  - Relatively small model (268K parameters)
  - Simple forward pass (no complex operations)
  - Good GPU utilization with batch size 256
  - Efficient PyTorch data loading

---

## 4. Training Runs

### Run 1: Baseline (v2_tau1_original)
**Config**:
- Learning Rate: 0.001
- Weight Decay: 0.0001
- Dropout: 0.2
- Reconstruction Weight (λ): 0.1
- Early Stopping Patience: 15

**Training Progress**:
| Epoch | Val Loss | Notes |
|-------|----------|-------|
| 0 | 13.47 | **BEST** ✓ |
| 1 | 12.38 | ↑ |
| 7 | 19.07 | ↑ (peak) |
| 15 | ~18 | Early stop |

**Results**:
- MAE: 2.61W
- R²: 0.9460
- RMSE: 3.45W
- MAPE: 3.28%
- Best Epoch: **0** (initialization!)

**Issue**: Best model at epoch 0 indicates immediate overfitting.

---

### Run 2: Reduced Reconstruction Weight
**Config Changes**:
- Learning Rate: 0.001 → **0.0005**
- Weight Decay: 0.0001 → **0.0003**
- Dropout: 0.2 → **0.25**
- Reconstruction Weight: 0.1 → **0.03**
- Early Stopping Patience: 15 → **10**

**Training Progress**:
| Epoch | Val Loss | Notes |
|-------|----------|-------|
| 0 | 23.68 | |
| 1 | 12.17 | ↓ |
| 2 | 11.41 | **BEST** ✓ |
| 3 | 12.12 | ↑ |
| 12 | 16.74 | Early stop |

**Results**:
- MAE: 2.35W (-10%)
- R²: 0.9541 (+0.8%)
- RMSE: 3.18W
- MAPE: 2.94%
- Best Epoch: **2**

**Improvement**: Delayed overfitting to epoch 2, better metrics overall.

---

### Run 3: Further Regularization
**Config Changes** (from Run 2):
- Dropout: 0.25 → **0.3**
- Reconstruction Weight: 0.03 → **0.01**

**Training Progress**:
| Epoch | Val Loss | Notes |
|-------|----------|-------|
| 0 | 16.15 | |
| 1 | 11.81 | ↓ |
| 2 | 11.02 | **BEST** ✓ |
| 3 | 11.88 | ↑ |
| 9 | 21.12 | ↑ (peak) |
| 12 | 17.80 | Early stop |

**Results**:
- MAE: 2.40W
- R²: 0.9522
- RMSE: 3.25W
- MAPE: 3.01%
- Best Epoch: **2**

**Observation**: Marginal difference from Run 2.

---

## 5. Results Comparison

| Metric | Run 1 (λ=0.1) | Run 2 (λ=0.03) | Run 3 (λ=0.01) | Target |
|--------|---------------|----------------|----------------|--------|
| **MAE** | 2.61W | **2.35W** ✓ | 2.40W | <3W |
| **R²** | 0.946 | **0.954** ✓ | 0.952 | >0.95 |
| **MAPE** | 3.28% | **2.94%** ✓ | 3.01% | <10% |
| **RMSE** | 3.45W | **3.18W** ✓ | 3.25W | - |
| **Best Epoch** | 0 | 2 | 2 | - |
| **Best Val Loss** | 13.47 | **11.41** ✓ | 11.02 | - |
| **Final Train Loss** | 14.61 | 25.56 | 20.63 | - |
| **Final Val Loss** | ~19 | 16.74 | 17.80 | - |

**Best Overall**: Run 2 (reconstruction weight = 0.03)

---

## 6. Key Findings

### Consistent Overfitting Pattern
All three runs exhibit the same behavior:
1. **Early best model** (epoch 0-2)
2. **Val loss increases** while train loss decreases
3. **Early stopping triggered** (~13 epochs)

### Training Dynamics
- **Train loss at epoch 12**:
  - Run 1: 14.61 (heavy reconstruction focus)
  - Run 2: 25.56 (power loss dominant)
  - Run 3: 20.63 (balanced)
- **Reconstruction loss remains high**: ~12-17 (poor reconstruction even at low weight)

### Validation Loss Breakdown (Run 3, Epoch 12)
- Power Loss: 20.45
- Reconstruction Loss: 17.50
- Combined (with λ=0.01): 20.45 + 0.01×17.50 = **20.63**

### Hyperparameter Sensitivity
| Change | Impact |
|--------|--------|
| Reconstruction weight: 0.1 → 0.03 | ✓ Best epoch: 0 → 2, MAE: 2.61 → 2.35W |
| Reconstruction weight: 0.03 → 0.01 | ≈ Minimal change |
| Dropout: 0.2 → 0.3 | ≈ Minimal impact |
| Learning rate: 0.001 → 0.0005 | ✓ More stable early training |

---

## 7. Conclusions

### Achievements
✅ All runs meet MAE target (<3W)
✅ Runs 2-3 meet R² target (>0.95)
✅ All runs meet MAPE target (<10%)
✅ Fast training (~6-7 minutes per run)

### Concerns
⚠️ **Persistent overfitting** regardless of hyperparameters
⚠️ **Best model always within first 2-3 epochs**
⚠️ **Val loss increases while train loss decreases**
⚠️ **Poor reconstruction** (loss ~12-17 throughout training)

### Hypotheses
1. **Train/val distribution mismatch** - Different workload characteristics?
2. **Architecture mismatch** - Reconstruction task may not help power prediction
3. **Model capacity** - 268K params may be too much, leading to memorization
4. **Data leakage** - Temporal correlation between train/val splits?

### Recommended Next Steps
1. **Investigate data splits** - Analyze train vs val workload distributions
2. **Try simpler architecture** - Remove decoder, use only Encoder → Power Head
3. **Baseline comparison** - Train simple MLP or linear regression
4. **Cross-validation** - Use k-fold or leave-one-workload-out validation
5. **Monitor power loss only** - Ignore reconstruction loss during validation

---

## Appendix: File Locations

### Configurations
- `config/v2_tau1_original.yaml` - Run 1 (baseline)
- `config/v2_tau1.yaml` - Runs 2-3 (current)

### Results
- `comparisons/tau1_original/` - Run 1 backup
- `results/v2_tau1_train2/plots/` - Run 2 plots
- `results/v2_tau1_train3/plots/` - Run 3 plots

### Logs
- `logs/logs/train_tau1.out` - Run 2 log
- `logs/logs/train_tau1_train3.out` - Run 3 log

### Data
- `data/processed/tau1/` - Processed NPZ files
- `data/processed/tau1/metadata.json` - Dataset statistics
