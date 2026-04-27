# MS-TCN Power Prediction Model

Multi-Scale Temporal Convolutional Network (MS-TCN) for CPU power prediction from system metrics.

## Overview

This project uses deep learning to predict CPU power consumption (package and core) from system-level metrics available in `/proc`. The model is trained on RAPL power measurements and can predict power on systems where RAPL is unavailable (e.g., VMs).

**Model Performance:**
- R² = 0.90+ on validation data
- MAE < 1.5W on workloads similar to training

## Project Structure

```
ms-tcn/
├── src/                              # Source code
│   ├── power_data_collector.py       # Collect RAPL + system metrics
│   ├── load_generator.py             # Generate training workloads
│   ├── train_model.py                # Train MS-TCN model
│   └── power_predictor.py            # Run inference
├── data/                             # Training data (CSV files)
├── models/                           # Saved model checkpoints (.pth)
├── results/                          # Experiment outputs
│   ├── predictions/                  # Prediction CSVs
│   ├── plots/                        # Visualizations
│   └── training_summary.json         # Training metrics
├── docs/                             # Documentation
│   ├── MODEL_ARCHITECTURE.md         # Detailed architecture explanation
│   ├── model_architecture.png        # Architecture diagram
│   └── RETRAINING_GUIDE.md           # How to retrain
├── scripts/                          # Helper scripts
└── prompts/                          # LLM prompts used to generate code
```

## Quick Start

### 1. Data Collection

Collect training data (requires physical machine with RAPL):

```bash
cd src/
sudo python3 power_data_collector.py --output ../data/training_data.csv --duration 3600
```

Generate controlled workloads during collection:

```bash
./load_generator.py --sequence cpu-focused --duration 3600
```

### 2. Training

Train the model:

```bash
python3 train_model.py --data ../data/training_data.csv --output ../models/my_model.pth
```

**Key arguments:**
- `--epochs`: Training epochs (default: 200)
- `--batch-size`: Batch size (default: 32)
- `--split-mode`: Data split strategy (random or temporal, default: random)

### 3. Inference

Live prediction:

```bash
# High-frequency data collection, low-frequency prediction
sudo python3 power_predictor.py \
  --model ../models/best_model.pth \
  --live \
  --scroll \
  --interval 0.016 \
  --frequency 0.1 \
  --save ../results/predictions/live.csv
```

**Key arguments:**
- `--interval`: Data collection rate (0.016 = 62.5Hz, matches training)
- `--frequency`: Prediction rate (0.1 = once per 10 seconds, reduces CPU overhead)

Predict on CSV file:

```bash
python3 power_predictor.py --model ../models/best_model.pth --csv test_data.csv
```

## Model Architecture

MS-TCN combines:
- **Multi-scale convolution** (kernels 3/5/7) - captures patterns at different timescales
- **Dilated temporal blocks** (6 layers) - expands receptive field to 64 timesteps
- **Multi-head attention** (8 heads) - focuses on important timesteps
- **Dual output heads** - separate predictions for package and core power

See [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) for full details.

**Model size:** 740K parameters (~3MB)
**Inference time:** ~5ms per prediction on CPU

## Input Features (19)

All metrics from `/proc`:
- CPU utilization (user, system, idle, iowait, irq, softirq)
- Context switches and interrupts per second
- Memory usage (used, cached, buffers, free, swap)
- Page faults per second
- System load averages (1/5/15 min)
- Process counts (running, blocked)

## Output Targets (2)

- **Package power** (W) - Total CPU package including cores + uncore
- **Core power** (W) - CPU cores only

## Key Findings

### What Works:
✅ MS-TCN architecture effective for power prediction
✅ 0.1Hz prediction frequency reduces inference overhead to ~3% CPU
✅ Random data split ensures balanced validation
✅ Model learns temporal dependencies well

### Current Limitations:
❌ **Training data distribution is heavily skewed to low power:**
   - 98% of samples < 30W
   - Only 0.4% > 40W
   - Model underpredicts on high-power workloads not seen during training

❌ **Workload-specific power signatures:**
   - Different instruction mixes produce different power for same CPU%
   - Model only predicts well on workload types present in training data

❌ **Temporal resolution critical:**
   - Must use 62.5Hz sampling (--interval 0.016) to match training
   - 64-sample window = 1.024 seconds of history
   - Lower sampling rates cause temporal mismatch

### Recommendations for Improvement:

1. **Collect diverse training data:**
   - Use multiple stress-ng cpu-methods (ackermann, matrixprod, fft, etc.)
   - Include real workloads (browsers, compilers, databases)
   - Ensure balanced power distribution (equal samples at all power levels)

2. **Longer sustained high-power phases:**
   - Current load_generator produces brief spikes
   - Need sustained 60-90W loads for minutes, not seconds

3. **Add more features (if available):**
   - Perf counters (instructions, cache misses, branch mispredicts)
   - Thermal sensors (junction temps affect power)
   - CPU frequency (DVFS state)

4. **For VM deployment:**
   - Train on VM-specific metrics only
   - Use host-side power attribution for validation
   - Consider separate model for VMs vs bare metal

## Files

**Models:**
- `best_model.pth` - Best validation R² (epoch 22, R²=0.9096)
- `model_v2.pth` - Final epoch model (more overfitted)

**Data:**
- `training_data_v2.csv` - 60 minutes, 225K samples @ 62.5Hz
- `power_data.csv` - Early 10-minute test data

**Results:**
- `training_summary.json` - Final metrics
- `training_history.png` - Loss/R² curves
- `training_data_analysis.png` - Power distribution analysis

## Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

For workload generation:
```bash
sudo apt install stress-ng  # Ubuntu/Debian
```

## License

See main repository LICENSE file.

## References

- **MS-TCN Paper:** "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation" (CVPR 2019)
- **TCN Survey:** "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (arXiv:1803.01271)
- **Attention:** "Attention Is All You Need" (NIPS 2017)
