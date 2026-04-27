# Model Retraining Guide

## Problem Identified

Your current model has **poor accuracy (R² = 0.715)**:
- ✗ Validation R² = 0.715 (need >0.90 for good predictions)
- ✗ Model plateaued after epoch 10 (no improvement for 60+ epochs)
- ✗ Prediction errors of 20-40% on live data
- ✗ Was training on DRAM power = 0 (confusing the model)

## Solution: Collect Better Data & Retrain

### Step 1: Collect High-Quality Training Data (30-60 minutes)

**Option A: Use the helper script (recommended)**
```bash
# Collect 30 minutes of diverse workload data
./collect_training_data.sh 1800 training_data_v2.csv

# Or for 60 minutes (better for more variety)
./collect_training_data.sh 3600 training_data_v2.csv
```

**Option B: Manual collection**
```bash
# Terminal 1: Start data collector
chmod -R a+r /sys/class/powercap/intel-rapl  # Enable RAPL access
python power_data_collector.py --duration 3600 --output training_data_v2.csv

# Terminal 2: Start load generator (wait 5 seconds after starting collector)
python load_generator.py --sequence comprehensive --duration 3600
```

**What this does:**
- Runs comprehensive workload patterns (idle, 25%, 50%, 75%, 100% CPU)
- Tests different instruction mixes (AES, SHA, matrix operations)
- Exercises memory and cache hierarchy
- Creates diverse power signatures for the model to learn

### Step 2: Retrain the Model

```bash
# Basic retraining (uses improved defaults)
python train_model.py --data training_data_v2.csv --output model_v2.pth

# Or with custom parameters for better results
python train_model.py \
    --data training_data_v2.csv \
    --output model_v2.pth \
    --epochs 300 \
    --patience 50 \
    --batch-size 64 \
    --hidden-dim 128
```

**Monitor training output for:**
- ✓ Validation R² should reach **>0.90** (ideally >0.95)
- ✓ Validation MAE should be **<1.0W**
- ✓ Training should run at least 50-100 epochs
- ✗ Stop if R² plateaus below 0.85 (need more/better data)

### Step 3: Validate the New Model

```bash
# Test on the training data (should show R² > 0.9)
python power_predictor.py --model model_v2.pth --csv training_data_v2.csv

# Test on live system
python power_predictor.py --model model_v2.pth --live --scroll
```

**Expected results:**
- Package power error: <5% (green)
- Core power error: <5% (green)
- Stable predictions that track actual power

## Changes Made

1. ✓ **Removed DRAM target** - No longer trains on constant zero values
2. ✓ **Increased patience** - Default 30 epochs (was 20)
3. ✓ **Better early stopping** - Won't waste time training if no improvement

## Training Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Validation R² | 0.715 | >0.90 | ✗ Poor |
| Validation MAE | 0.163W | <1.0W | ✓ OK |
| Training convergence | Plateaued at epoch 10 | Smooth descent | ✗ Poor |
| Prediction accuracy | 20-40% error | <5% error | ✗ Poor |

## Troubleshooting

**If R² is still <0.85 after retraining:**

1. **Collect more data** (60-120 minutes instead of 30)
   ```bash
   ./collect_training_data.sh 7200 training_data_extended.csv
   ```

2. **Try different sequence length**
   ```bash
   python train_model.py --data training_data_v2.csv --sequence-length 128
   ```

3. **Reduce model complexity** (if overfitting)
   ```bash
   python train_model.py --data training_data_v2.csv --hidden-dim 64
   ```

4. **Increase learning rate** (if underfitting)
   ```bash
   python train_model.py --data training_data_v2.csv --learning-rate 5e-4
   ```

**If you get "out of memory" errors:**
- Reduce `--batch-size` to 16 or 8
- Reduce `--hidden-dim` to 64

## Quick Reference

```bash
# Full workflow
./collect_training_data.sh 3600 training_data_v2.csv
python train_model.py --data training_data_v2.csv --output model_v2.pth
python power_predictor.py --model model_v2.pth --live --scroll

# Expected training output
#   Epoch 50/200
#   Train Loss: 0.0042, Val Loss: 0.0089
#   Val MAE: 0.52, Val R2: 0.9534  ← This is what you want!
#   Saved best model!
```

## Notes

- **Minimum data:** 30 minutes with varied workloads
- **Recommended:** 60 minutes for better generalization
- **Ideal:** Multiple 30-60 minute sessions with different workload patterns
- **Sample rate:** ~62 Hz (16ms intervals) = ~225,000 samples per hour
