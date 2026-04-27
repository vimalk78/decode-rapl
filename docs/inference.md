# DECODE-RAPL v2 Inference Guide

Complete guide for real-time power prediction using trained models.

## Overview

The inference system provides:
- **Dynamic tau support** - Automatically adapts to model's tau value (1, 4, or 8)
- **Real-time prediction** - Buffer-based inference with <1ms latency
- **Live monitoring** - Color-coded predictions vs. actual RAPL power
- **CSV replay** - Batch prediction on collected data
- **Flexible frequency** - Configurable prediction rate

## Quick Start

### Live Monitoring (Recommended)

For real-time power prediction with visual feedback:

```bash
cd /path/to/decode-rapl

# Live monitoring with scrolling output
sudo python src/power_predictor.py \
  --model checkpoints/v2_tau1/best_model.pt \
  --live \
  --scroll

# Predict once per second (data collected continuously)
sudo python src/power_predictor.py \
  --model checkpoints/v2_tau8/best_model.pt \
  --live \
  --scroll \
  --frequency 1.0

# Save predictions to CSV
sudo python src/power_predictor.py \
  --model checkpoints/v2_tau1/best_model.pt \
  --live \
  --scroll \
  --save live_predictions.csv
```

**Why sudo?** - Required for reading RAPL power counters (`/sys/class/powercap/intel-rapl`)

### CSV Replay

For offline prediction on collected data:

```bash
# Predict on CSV file
python src/power_predictor.py \
  --model checkpoints/v2_tau1/best_model.pt \
  --csv data/test_workload.csv \
  --save predictions.csv
```

**CSV format requirements:**
```csv
user_percent,system_percent,iowait_percent,ctx_switches_per_sec,package_power_watts
12.5,8.3,0.5,2500,35.2
15.2,9.1,0.8,2800,38.5
...
```

(`package_power_watts` is optional - only needed for comparison)

## How It Works

### Delay Embedding Buffer

The inference system maintains a **sliding buffer** that adapts to the model's tau value:

| Tau | Buffer Size | Lookback Time | Use Case |
|-----|------------|---------------|----------|
| **tau=1** | 25 samples | 384ms | Short-term patterns, rapid changes |
| **tau=4** | 97 samples | 1.54s | Medium-term context |
| **tau=8** | 193 samples | 3.07s | Long-term sustained workloads |

**Buffer size formula:** `(d - 1) × tau + 1` where `d = 25` delays

### Prediction Pipeline

```
1. Read CPU metrics (/proc/stat)
   ↓
2. Add to buffer [user%, system%, iowait%, ctx_switches]
   ↓
3. When buffer full: Create delay-embedded vector (100-dim)
   ↓
4. Feed to model → Power prediction
   ↓
5. Compare with RAPL (if available)
```

### Feature-Grouped Delay Embedding

The 100-dim vector is constructed as:

```python
# Positions 0-24:   user%(t), user%(t-τ), ..., user%(t-24τ)
# Positions 25-49:  sys%(t), sys%(t-τ), ..., sys%(t-24τ)
# Positions 50-74:  iowait%(t), iowait%(t-τ), ..., iowait%(t-24τ)
# Positions 75-99:  log_ctx(t), log_ctx(t-τ), ..., log_ctx(t-24τ)
```

**Key insight:** Same 100-dim output regardless of tau, but different temporal spacing!

## CLI Reference

### power_predictor.py

**Live Monitoring Mode:**
```bash
python src/power_predictor.py --model <path> --live [OPTIONS]
```

**Options:**
- `--scroll` - Scrolling one-line output (recommended for logging)
- `--interval SECONDS` - Sampling interval (default: 0.1s = 100ms)
- `--frequency HZ` - Prediction frequency in Hz (e.g., 1.0 = once per second)
- `--save FILE` - Save predictions to CSV

**CSV Replay Mode:**
```bash
python src/power_predictor.py --model <path> --csv <file> [OPTIONS]
```

**Options:**
- `--save FILE` - Save predictions to CSV

### Output Modes

**Scrolling Mode** (`--scroll`):
```
[10:30:15] #    1 (  1.0Hz) | CPU:  45.2% (U:30.1% S:15.1% IO: 0.5%) | Pred:  42.3W | Actual:  41.8W | Err:  +1.2% | MAPE:  1.2%
[10:30:16] #    2 (  1.0Hz) | CPU:  48.5% (U:32.3% S:16.2% IO: 0.3%) | Pred:  45.1W | Actual:  44.5W | Err:  +1.3% | MAPE:  1.3%
```

**Full-Screen Mode** (default):
```
Live Power Prediction - DECODE-RAPL v2
Samples: 150 | Predictions: 150 | Elapsed: 15.0s
Sample rate: 10.0 Hz | Prediction rate: 10.0 Hz
Timestamp: 1735567815.23

Metric                    Value
----------------------------------------
CPU Total%                     45.2%
  User%                        30.1%
  System%                      15.1%
  IOWait%                       0.5%
Context Switches/s            2500

Predicted Power              42.3W
Actual Power                 41.8W
Error                        +0.5W
Error %                      +1.2%
Running MAPE                  1.3%
```

## Selecting the Right Tau Model

After training with multiple tau values (1, 4, 8), choose based on:

### 1. Validation Metrics

Pick the model with lowest validation MAE/MAPE:

```bash
# Check training logs
cat results/v2_tau1/training.log | grep "Val MAE"
cat results/v2_tau4/training.log | grep "Val MAE"
cat results/v2_tau8/training.log | grep "Val MAE"
```

### 2. Target Workload Characteristics

**tau=1 (384ms lookback):**
- ✅ Bursty workloads (short CPU spikes)
- ✅ Interactive applications
- ✅ Rapid changes in CPU usage
- ❌ Sustained compute-heavy loads

**tau=4 (1.54s lookback):**
- ✅ Balanced workloads
- ✅ General-purpose server tasks
- ✅ Mix of burst and sustained patterns

**tau=8 (3.07s lookback):**
- ✅ Sustained workloads (HPC, batch processing)
- ✅ Stable long-running jobs
- ✅ Workloads with slow power transitions
- ❌ Rapidly changing bursty loads

### 3. Prediction Latency Requirements

**Startup delay (buffer fill time):**
- tau=1: 25 samples × 16ms = **400ms**
- tau=4: 97 samples × 16ms = **1.55s**
- tau=8: 193 samples × 16ms = **3.1s**

For applications requiring instant predictions, prefer lower tau.

## Integration with an external monitoring system

To integrate with a monitoring system on your bare-metal host:

### 1. Deploy Trained Model

```bash
# Copy best model to the target host
scp checkpoints/v2_tau1/best_model.pt <your-bare-metal-host>:/opt/powermon/models/

# Verify model loads
ssh <your-bare-metal-host> "cd /opt/powermon && python -c 'from decode_rapl_v2.inference import RAPLPredictor; p = RAPLPredictor(\"models/best_model.pt\"); print(p.get_buffer_info())'"
```

### 2. Python Integration Example

```python
from decode_rapl_v2.inference import RAPLPredictor
import time

# Initialize predictor
predictor = RAPLPredictor("/opt/powermon/models/best_model.pt")

# Get buffer requirements
info = predictor.get_buffer_info()
print(f"Need {info['buffer_size']} samples before first prediction")

# Fill buffer (collect from /proc/stat)
for i in range(info['buffer_size']):
    metrics = read_cpu_metrics()  # Your implementation
    predictor.update_metrics(
        metrics['user_percent'],
        metrics['system_percent'],
        metrics['iowait_percent'],
        metrics['ctx_switches_per_sec']
    )
    time.sleep(0.016)  # 16ms sampling

# Now ready for predictions
while True:
    metrics = read_cpu_metrics()
    predictor.update_metrics(
        metrics['user_percent'],
        metrics['system_percent'],
        metrics['iowait_percent'],
        metrics['ctx_switches_per_sec']
    )

    power = predictor.predict()
    print(f"Predicted power: {power:.2f}W")

    time.sleep(0.016)
```

### 3. Real-time Daemon Example

```python
#!/usr/bin/env python3
"""
Real-time power monitoring daemon for powermon
"""
import sys
import time
from collections import deque
from pathlib import Path

sys.path.append('/opt/powermon/decode-rapl-v2/src')
from inference import RAPLPredictor

def read_proc_stat():
    """Read /proc/stat and calculate CPU metrics"""
    # Implementation similar to CPUReader in power_predictor.py
    pass

def main():
    # Initialize predictor
    predictor = RAPLPredictor("/opt/powermon/models/best_model.pt")

    # Open output file for logging
    with open("/opt/powermon/logs/power_predictions.csv", "w") as f:
        f.write("timestamp,user%,system%,iowait%,ctx_switches,predicted_power_w\n")

        # Fill buffer
        info = predictor.get_buffer_info()
        for _ in range(info['buffer_size']):
            metrics = read_proc_stat()
            predictor.update_metrics(**metrics)
            time.sleep(0.016)

        # Continuous prediction
        while True:
            metrics = read_proc_stat()
            predictor.update_metrics(**metrics)
            power = predictor.predict()

            # Log
            f.write(f"{time.time()},{metrics['user_percent']},{metrics['system_percent']},"
                   f"{metrics['iowait_percent']},{metrics['ctx_switches_per_sec']},{power:.2f}\n")
            f.flush()

            time.sleep(0.016)  # 16ms = 62.5 Hz

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Buffer Not Filling

**Symptom:**
```
Buffer not full yet: 10/25
```

**Solution:** Keep collecting metrics. The buffer needs to fill before first prediction.

### RAPL Not Available

**Symptom:**
```
⚠ RAPL not available - showing predictions only
```

**Causes:**
1. Not running with sudo
2. RAPL not supported on CPU
3. Kernel module not loaded

**Solutions:**
```bash
# Check if RAPL exists
ls /sys/class/powercap/intel-rapl/

# Run with sudo
sudo python src/power_predictor.py --model ... --live

# Load MSR module (if needed)
sudo modprobe msr
```

### Model Loading Error

**Symptom:**
```
Error: Checkpoint missing 'config'
```

**Cause:** Model trained before inference support was added.

**Solution:** Retrain model with updated training pipeline:
```bash
python src/train.py --config config/v2_default.yaml
```

### Incorrect Predictions

**Symptom:** Predictions are wildly off (e.g., predicting 5W when actual is 50W)

**Possible causes:**
1. **Wrong tau model** - Using tau=8 model on bursty workload
2. **Feature mismatch** - CPU metrics not matching training data
3. **Normalization issues** - Check that context switches are in correct units

**Debugging:**
```python
# Check buffer info
info = predictor.get_buffer_info()
print(info)

# Inspect first few predictions
predictor.reset()
for i in range(10):
    metrics = read_cpu_metrics()
    predictor.update_metrics(**metrics)
    power = predictor.predict()
    if power:
        print(f"Sample {i}: {metrics} → {power:.2f}W")
```

### High Latency

**Symptom:** Predictions are slow (>10ms)

**Causes:**
1. GPU transfer overhead
2. Large batch processing

**Solutions:**
```python
# Use CPU for inference (models are small)
# Prediction is <1ms on CPU

# If using GPU, ensure model stays on GPU
predictor = RAPLPredictor(model_path)
# Model automatically moved to GPU if available
```

## Performance Expectations

### Latency

| Component | Time |
|-----------|------|
| Buffer update | <0.1ms |
| Delay embedding creation | <0.2ms |
| Model forward pass (CPU) | <0.5ms |
| Model forward pass (GPU) | <0.3ms |
| **Total (CPU)** | **<1ms** |

### Accuracy (Expected on Validation Set)

Based on training targets:

| Metric | v2 Target | v1 Actual |
|--------|-----------|-----------|
| MAE | <3W | ~5W |
| R² | >0.95 | 0.85 |
| MAPE | <10% | ~15% |

**On Prometheus workloads** (where v1 failed):
- v1 achieved R² = 0.12 (failed)
- v2 expected: R² > 0.90 (success criterion)

### Throughput

- **Max prediction rate**: Limited by sampling interval (62.5 Hz at 16ms)
- **Configurable rate**: Use `--frequency` flag (e.g., 1 Hz for logging)
- **Batch processing**: ~1000 predictions/second on CPU

## Advanced Usage

### Custom Sampling Intervals

For non-standard sampling rates:

```python
# 10ms sampling instead of 16ms
predictor = RAPLPredictor(model_path)

# Adjust tau interpretation
# tau=1 at 10ms → 240ms lookback (vs 384ms at 16ms)
```

**Note:** Model was trained on 16ms data. Different sampling may affect accuracy.

### Batch Prediction Optimization

For processing large CSV files:

```python
# Use vectorized batch prediction
predictions = predictor.predict_from_sequence(
    user_pct_array,      # (N,)
    system_pct_array,    # (N,)
    iowait_pct_array,    # (N,)
    ctx_switches_array   # (N,)
)
# Returns: (N - buffer_size + 1,) predictions
```

### Multi-Model Ensemble

Run multiple tau models and combine:

```python
predictor_tau1 = RAPLPredictor("checkpoints/v2_tau1/best_model.pt")
predictor_tau4 = RAPLPredictor("checkpoints/v2_tau4/best_model.pt")
predictor_tau8 = RAPLPredictor("checkpoints/v2_tau8/best_model.pt")

# Fill buffers (different sizes!)
# ...

# Predict with all three
power_tau1 = predictor_tau1.predict()
power_tau4 = predictor_tau4.predict()
power_tau8 = predictor_tau8.predict()

# Weighted average (tune weights based on validation)
power_ensemble = 0.5 * power_tau1 + 0.3 * power_tau4 + 0.2 * power_tau8
```

## Next Steps

After successful inference testing:

1. **Validate on Test Set** - Run evaluation script (TODO)
2. **Test on Prometheus** - Deploy to your bare-metal host and test real workloads
3. **Compare with v1** - Side-by-side accuracy comparison
4. **Tune Prediction Frequency** - Balance accuracy vs. overhead
5. **Integrate with Powermon** - Production deployment

## References

- **Training Guide:** [docs/training.md](training.md)
- **Architecture Details:** [docs/architecture.md](architecture.md)
- **Inference Module:** [src/inference.py](../src/inference.py)
- **CLI Tool:** [src/power_predictor.py](../src/power_predictor.py)
