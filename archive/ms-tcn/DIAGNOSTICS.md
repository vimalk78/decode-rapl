# MS-TCN Model Diagnostics

This document describes diagnostic tools to investigate why the trained model predicts ~40W for idle instead of ~27W, even though training data contained plenty of idle samples at 27W.

## The Mystery

**Observation**: Model predicts ~40W at idle when actual is ~27W (50% error), but predicts accurately at high loads (0.5-8% error).

**Critical Clue**: Training data has NO samples around 40W! Power distribution jumps from ~28W (idle) to ~50W (mid-load) to ~70W (high-load).

**Question**: Why does the model predict a value that doesn't exist in training data?

## Diagnostic Tools

### 1. Feature Distribution Analyzer
**File**: `scripts/analyze_feature_distribution.py`

**Purpose**: Check if live features differ from training features.

**What it tests**:
- Compares raw feature values: training idle vs live idle
- Compares normalized features (what model actually sees)
- Identifies out-of-distribution features

**Usage**:
```bash
# Capture live features and compare
python3 scripts/analyze_feature_distribution.py \
    --training-data data/training_data_beaker_3600s.csv \
    --live

# Or use pre-captured features
python3 scripts/analyze_feature_distribution.py \
    --training-data data/training_data_beaker_3600s.csv \
    --live-csv data/live_idle_features.csv
```

**Expected outcome**:
- ✓ If features match: Distribution mismatch is NOT the issue
- ⚠️  If features differ: Model sees features it never saw during training

---

### 2. Sequence Buffer Inspector
**File**: `scripts/inspect_sequence_buffer.py`

**Purpose**: Check if the 64-sample buffer initialization causes issues.

**What it tests**:
- How predictions evolve as buffer fills from startup
- Whether predictions stabilize after buffer is fully filled
- If buffer contamination affects results

**Usage**:
```bash
python3 scripts/inspect_sequence_buffer.py \
    --model models/model_beaker.pth \
    --test-data data/test_idle.csv \
    --output results/buffer_evolution.csv
```

**Expected outcome**:
- ✓ If predictions change after buffer fills: Buffer initialization was the issue
- ⚠️  If predictions stay constant: Buffer is NOT the issue

---

### 3. Model Activation Analyzer
**File**: `scripts/analyze_model_activations.py`

**Purpose**: Check for dead/saturated neurons in model layers.

**What it tests**:
- Activation statistics for each layer (Stage 1, 2, 3, FC)
- Detects neurons that always output zero (dead)
- Compares idle vs high-load activations

**Usage**:
```bash
python3 scripts/analyze_model_activations.py \
    --model models/model_beaker.pth
```

**Expected outcome**:
- ✓ If activations change with input: Model layers are functional
- ⚠️  If >50% neurons dead: Architecture or training issue
- ⚠️  If activations don't change: Model collapsed to constant output

---

### 4. Prediction Sensitivity Analyzer
**File**: `scripts/analyze_prediction_sensitivity.py`

**Purpose**: Identify which features the model actually uses.

**What it tests**:
- Perturbs each feature individually (±10%, ±50%, ±100%)
- Measures how much predictions change
- Identifies ignored features

**Usage**:
```bash
# Test at idle
python3 scripts/analyze_prediction_sensitivity.py \
    --model models/model_beaker.pth \
    --scenario idle

# Test at high load
python3 scripts/analyze_prediction_sensitivity.py \
    --model models/model_beaker.pth \
    --scenario high_load
```

**Expected outcome**:
- ⚠️  If `cpu_user_percent` is ignored: Model can't distinguish idle from load
- ⚠️  If `cpu_idle_percent` is ignored: Model can't detect idle state
- ✓ If CPU features are sensitive: Model uses correct features

---

### 5. Training Loss Distribution Analyzer
**File**: `scripts/analyze_training_loss_distribution.py`

**Purpose**: Check if predicting 40W minimizes MSE loss.

**What it tests**:
- Calculates MSE for different constant predictions (27W, 30W, 35W, 40W, 50W, etc.)
- Identifies which constant value minimizes loss
- Shows loss landscape

**Usage**:
```bash
python3 scripts/analyze_training_loss_distribution.py \
    --training-data data/training_data_beaker_3600s.csv \
    --output-plot results/plots/loss_distribution.png
```

**Expected outcome**:
- ⚠️  If 40W minimizes MSE: Loss function is biasing predictions
- ⚠️  If optimal constant is 35-45W: MSE loss on imbalanced data is the problem
- ✓ If optimal is ~27W: Loss function is NOT the issue

**Interpretation**:
If predicting 40W minimizes MSE, it means:
- Model learned to avoid large errors on common cases
- Being wrong by 13W on idle (40 vs 27) is acceptable
- Because it avoids being wrong by 30W+ on high loads
- This is a fundamental issue with MSE on imbalanced data

**Solutions**:
1. Use weighted MSE (higher weight for idle samples)
2. Use balanced sampling during training
3. Use different loss function (Huber loss, quantile loss)

---

### 6. Temporal Feature Importance Analyzer
**File**: `scripts/analyze_temporal_importance.py`

**Purpose**: Check if MS-TCN actually uses temporal context.

**What it tests**:
- Whether model uses 64-sample history or just recent samples
- Sensitivity to sample position in sequence
- Whether temporal dynamics (gradual vs sudden) matter

**Usage**:
```bash
python3 scripts/analyze_temporal_importance.py \
    --model models/model_beaker.pth
```

**Expected outcome**:
- ✓ If position matters: Model uses temporal context correctly
- ⚠️  If only recent samples matter: Model ignores history
- ⚠️  If position doesn't matter: Temporal convolutions failed

**Interpretation**:
If model ignores temporal context:
- Dilated convolutions have zero-valued kernels
- Gradient flow issues during training
- Learning rate too high/low
- Architecture needs redesign

---

## Recommended Diagnostic Sequence

Run tools in this order to narrow down the issue:

### Phase 1: Quick Checks
1. **Training Loss Distribution Analyzer** (fastest, no model inference)
   - If this shows 40W minimizes MSE → **Root cause found: Loss function**

2. **Prediction Sensitivity Analyzer** (synthetic tests, fast)
   - If this shows CPU features ignored → **Root cause found: Feature learning failure**

### Phase 2: Deep Analysis
3. **Temporal Feature Importance Analyzer** (synthetic tests)
   - If this shows no temporal context → **Root cause found: Architecture failure**

4. **Model Activation Analyzer** (internal inspection)
   - If this shows dead neurons → **Root cause found: Training failure**

### Phase 3: Data-Specific
5. **Feature Distribution Analyzer** (requires live data or training data)
   - If this shows distribution mismatch → **Root cause found: Data pipeline issue**

6. **Sequence Buffer Inspector** (requires test data)
   - If this shows buffer contamination → **Root cause found: Inference bug**

---

## Interpreting Results

### Scenario A: Loss Function Issue
**Symptoms**:
- ✓ Tool 5 shows 40W minimizes MSE
- ✓ Tool 4 shows model uses CPU features correctly
- ✓ Tool 6 shows temporal context works

**Diagnosis**: MSE loss on imbalanced data biases toward "safe" predictions

**Solution**: Retrain with weighted loss or balanced sampling

### Scenario B: Feature Learning Failure
**Symptoms**:
- ✓ Tool 4 shows `cpu_user_percent` ignored
- ⚠️  Tool 3 shows dead neurons in early layers

**Diagnosis**: Model failed to learn CPU features properly

**Solution**: Check feature scaling, try different architecture

### Scenario C: Architecture Failure
**Symptoms**:
- ✓ Tool 6 shows no temporal context
- ✓ Tool 3 shows activations don't change
- ⚠️  Tool 2 shows constant predictions after buffer fills

**Diagnosis**: Temporal convolutions not working

**Solution**: Check dilated conv parameters, gradient flow, try simpler architecture

### Scenario D: Data Pipeline Issue
**Symptoms**:
- ✓ Tool 1 shows normalized features differ significantly
- ✓ All other tools show model works correctly

**Diagnosis**: Feature collection differs between training and inference

**Solution**: Fix data pipeline, ensure consistency

---

## Next Steps After Diagnosis

Based on the diagnostic results:

### If Loss Function Issue:
1. Implement weighted MSE: `weights = 1 / power_frequency`
2. Or use Huber loss: `torch.nn.SmoothL1Loss()`
3. Or use balanced sampling during training
4. Retrain model

### If Architecture Issue:
1. Simplify architecture (fewer dilations)
2. Add skip connections
3. Check kernel initialization
4. Monitor gradients during training
5. Try alternative architecture (LSTM, Transformer)

### If Feature Learning Issue:
1. Check feature scaling (StandardScaler parameters)
2. Try different normalization (MinMaxScaler, RobustScaler)
3. Add feature engineering (derivatives, rolling means)
4. Increase model capacity (more layers/channels)

### If Data Issue:
1. Collect more diverse idle data
2. Add synthetic noise to idle samples
3. Ensure feature collection is consistent
4. Verify scaler parameters match

---

## Quick Reference

| Tool | Run Time | Requires | Key Question |
|------|----------|----------|--------------|
| Loss Distribution | <1 min | Training data | Does 40W minimize MSE? |
| Sensitivity | 1-2 min | Model | Does model use CPU features? |
| Temporal Importance | 2-3 min | Model | Does model use history? |
| Activations | 1 min | Model | Are neurons dead? |
| Feature Distribution | 1-5 min | Training + live data | Do features match? |
| Buffer Inspector | 2-5 min | Model + test data | Does buffer matter? |

---

## Common Issues and Solutions

### Issue: "Model file missing preprocessor parameters"
**Cause**: Model was saved with old training script
**Solution**: Retrain model with current `train_model.py`

### Issue: "psutil not installed" (Tool 1)
**Solution**: `pip install psutil` or provide `--live-csv` instead of `--live`

### Issue: "Plots not showing"
**Solution**: Install matplotlib: `pip install matplotlib`

### Issue: All tools show "model works correctly" but predictions still wrong
**Cause**: Issue is likely in the training data or process, not the model itself
**Solution**:
1. Check training data quality
2. Review training logs
3. Compare validation predictions to training predictions
4. Check for data leakage between train/val sets
