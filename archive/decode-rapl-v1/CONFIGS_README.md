# DECODE-RAPL Configuration Files

This directory contains multiple configuration files for comparing different τ (tau) values in delay embedding experiments.

## Available Configurations

### config_tau1.yaml - Dense Embedding (Baseline)
- **τ = 1 sample (16ms)**
- **window_size = 120**
- **Total context**: ~1920ms
- **Characteristics**:
  - Maximum temporal detail
  - High redundancy (ACF ≈ 0.95)
  - Captures complete burst dynamics
  - Slower training
- **Use for**: Baseline comparison, when fine-grained dynamics matter

### config_tau4.yaml - Balanced
- **τ = 4 samples (64ms)**
- **window_size = 30**
- **Total context**: ~1920ms
- **Characteristics**:
  - Moderate decorrelation (ACF ≈ 0.5)
  - Still captures burst structure
  - Good efficiency/detail balance
  - Faster than τ=1
- **Use for**: Balance between efficiency and accuracy

### config_tau8.yaml - Theoretical Optimum
- **τ = 8 samples (128ms)**
- **window_size = 15**
- **Total context**: ~1920ms
- **Characteristics**:
  - Strong decorrelation (ACF ≈ 0.35, near 1/e)
  - Efficient representation
  - Captures burst-to-burst transitions
  - May lose fine-grained dynamics
- **Use for**: Maximum efficiency, following traditional delay embedding theory

## Usage

### Training with Different Configs

```bash
# Baseline (τ=1)
python src/train.py --config config_tau1.yaml

# Balanced (τ=4)
python src/train.py --config config_tau4.yaml

# Theoretical optimum (τ=8)
python src/train.py --config config_tau8.yaml
```

### Comparison Experiment

```bash
# Train all three models
for config in config_tau1.yaml config_tau4.yaml config_tau8.yaml; do
    echo "Training with $config"
    python src/train.py --config $config
done

# Compare results
python scripts/compare_tau_results.py \
    --results results/tau1 results/tau4 results/tau8
```

## Expected Trade-offs

| Config | Training Speed | Memory | Expected MAPE | Detail Level |
|--------|---------------|--------|---------------|--------------|
| tau1   | Slowest       | High   | Best (?)      | Maximum      |
| tau4   | Medium        | Medium | Good (?)      | Good         |
| tau8   | Fastest       | Low    | Good (?)      | Moderate     |

**Note**: MAPE expectations are hypothetical and need empirical validation!

## Key Differences Summary

### Total Temporal Context
All configs maintain ~1920ms total temporal context by adjusting window_size:
- tau1: 120 steps × 16ms = 1920ms
- tau4: 30 steps × 64ms = 1920ms
- tau8: 15 steps × 128ms = 1920ms

### Embedding Lookback
Different configs have different delay embedding lookback spans:
- tau1: (25-1) × 16ms = 384ms
- tau4: (25-1) × 64ms = 1536ms
- tau8: (25-1) × 128ms = 3072ms

### LSTM Input Shape
- tau1: (batch, 120, 16) - longer sequence
- tau4: (batch, 30, 16) - medium sequence
- tau8: (batch, 15, 16) - shorter sequence

All have latent_dim=16 after encoder compression.

## Recommendations

1. **Start with tau1** (config_tau1.yaml)
   - Fixes sampling rate mismatch (1ms → 16ms)
   - Establishes baseline performance
   - Minimal change from original approach

2. **Run comparison experiment**
   - Train all three configs on same data split
   - Compare test MAPE, training time, convergence

3. **Choose based on results**
   - If MAPE comparable: Use faster config (tau4 or tau8)
   - If accuracy differs significantly: Use best MAPE config
   - If training time critical: Accept small MAPE increase for speedup

## References

See `docs/tau-selection-guide.md` for detailed explanation of τ selection theory and trade-offs.

## Outputs

Each config writes to separate directories:
- `results/tau1/` - τ=1 results
- `results/tau4/` - τ=4 results
- `results/tau8/` - τ=8 results
- `checkpoints/tau1/` - τ=1 model checkpoints
- `checkpoints/tau4/` - τ=4 model checkpoints
- `checkpoints/tau8/` - τ=8 model checkpoints
