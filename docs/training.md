# DECODE-RAPL v2 Training Guide

Complete guide for training the v2 model with background execution support.

## Overview

The training pipeline (`src/train.py`) provides a complete training workflow with:

- **Background execution** - Run training on remote GPU machines with nohup
- **Checkpoint management** - Automatic saving and resume capability
- **Early stopping** - Prevent overfitting with patience-based stopping
- **Learning rate scheduling** - Adaptive learning rate reduction
- **Progress tracking** - Logs, metrics, and plots
- **Graceful shutdown** - Handle interruptions cleanly

## Quick Start

### Foreground Training (Interactive)

For quick experimentation or when you want to see real-time progress:

```bash
cd /path/to/decode-rapl
python src/train.py --config config/v2_tau1.yaml
```

**Features:**
- Real-time progress bar (tqdm)
- Interactive output
- Ctrl+C for graceful shutdown

### Background Training (Recommended for GPU)

For long training runs on remote machines:

```bash
cd /path/to/decode-rapl

# Create logs directory
mkdir -p logs

# Start training in background
nohup python src/train.py --config config/v2_tau1.yaml > logs/train.out 2>&1 &

# Save the process ID
echo $! > logs/train.pid
```

**Monitor progress:**
```bash
# View training log with timestamps
tail -f results/v2_tau1/training.log

# View stdout/stderr
tail -f logs/train.out

# Check progress file
cat results/v2_tau1/training_progress.txt
```

**Stop training gracefully:**
```bash
# Send SIGTERM for graceful shutdown (saves checkpoint)
kill $(cat logs/train.pid)

# Or use SIGINT (Ctrl+C equivalent)
kill -INT $(cat logs/train.pid)
```

## Configuration

Training is configured via YAML files. The default configuration is `config/v2_tau1.yaml`.

### Key Configuration Sections

#### Data Paths
```yaml
data:
  processed_dir: "data/processed/tau1"  # Change for tau4 or tau8
  train_file: "train.npz"
  val_file: "val.npz"
  test_file: "test.npz"
```

#### Output Directories
```yaml
output:
  checkpoint_dir: "checkpoints/v2_tau1"
  results_dir: "results/v2_tau1"
  plots_dir: "results/v2_tau1/plots"
```

#### Model Architecture
```yaml
model:
  input_dim: 100          # 4 features × 25 delays
  latent_dim: 64          # Wider than v1 (16)
  encoder_layers: [512, 128]
  decoder_layers: [128, 512]
  power_head_layers: [128, 64]
  dropout: 0.2
```

#### Training Parameters
```yaml
training:
  batch_size: 256                    # Fits in GPU memory, good gradient estimates
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001               # L2 regularization
  early_stopping_patience: 15        # Stop after 15 epochs without improvement

  loss_weights:
    power: 1.0                       # Main objective
    reconstruction: 0.1              # Autoencoder quality (reduced from v1's 0.5)

  scheduler:
    type: "ReduceLROnPlateau"
    factor: 0.5                      # Reduce LR by half
    patience: 5                      # After 5 epochs without improvement
    min_lr: 0.00001

  device: "cuda"                     # "cuda" or "cpu"
```

### Creating Custom Configurations

For training with different tau values or hyperparameters:

```bash
# Copy default config
cp config/v2_tau1.yaml config/v2_tau4.yaml

# Edit the config
vim config/v2_tau4.yaml
# Change:
#   data.processed_dir: "data/processed/tau4"
#   output.checkpoint_dir: "checkpoints/v2_tau4"
#   output.results_dir: "results/v2_tau4"

# Train with new config
python src/train.py --config config/v2_tau4.yaml
```

## Output Files

Training produces several output files for monitoring and analysis.

### Directory Structure
```
results/v2_tau1/
├── training.log              # Timestamped log (all epochs)
├── training_progress.txt     # Brief progress summary
├── training_history.json     # Metrics history (for plotting)
└── plots/
    ├── training_curves.png   # Loss curves (updated every 5 epochs)
    └── final_predictions.png # Final evaluation plots

checkpoints/v2_tau1/
├── best_model.pt             # Best model (lowest val loss)
├── checkpoint_epoch_0.pt     # Checkpoint after epoch 0
├── checkpoint_epoch_1.pt     # Checkpoint after epoch 1
├── ...
└── interrupted.pt            # Saved if training interrupted
```

### Log Files

**`training.log`** - Complete training log with timestamps:
```
[2025-10-19 10:30:15] Starting training
[2025-10-19 10:30:15] Model created: 267,941 parameters
[2025-10-19 10:30:20] Train batches: 21875
[2025-10-19 10:30:20] Val batches: 2735
[2025-10-19 10:35:42] Epoch 0/100 Summary:
[2025-10-19 10:35:42]   Train Loss: 45.2341
[2025-10-19 10:35:42]   Val Loss:   42.1234
[2025-10-19 10:35:42]   Val MAE:    5.23W
[2025-10-19 10:35:42]   Val R²:     0.8234
[2025-10-19 10:35:42]   LR:         0.001000
[2025-10-19 10:35:42]   Time:       322.1s
[2025-10-19 10:35:42]   ✓ New best model! Val loss: 42.1234
...
```

**`training_progress.txt`** - Brief progress tracking:
```
Epoch 0/100: train_loss=45.2341, val_loss=42.1234
Epoch 1/100: train_loss=38.5432, val_loss=36.7890
Epoch 2/100: train_loss=34.2156, val_loss=32.4567
...
```

**`training_history.json`** - Metrics for plotting:
```json
{
  "train_loss": [45.2341, 38.5432, 34.2156, ...],
  "val_loss": [42.1234, 36.7890, 32.4567, ...],
  "learning_rates": [0.001, 0.001, 0.001, ...]
}
```

### Checkpoint Files

Each checkpoint (`.pt` file) contains:
```python
{
    'epoch': 5,
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state
    'train_loss': 28.1234,
    'val_loss': 26.7890
}
```

**Best model:** `best_model.pt` is automatically updated when validation loss improves.

**Regular checkpoints:** Saved after every epoch as `checkpoint_epoch_N.pt`.

**Interrupted checkpoint:** Saved as `interrupted.pt` if training is stopped via signal.

## Resume Training

If training is interrupted or stopped early, resume from a checkpoint:

```bash
# Resume from best model
python src/train.py \
  --config config/v2_tau1.yaml \
  --resume checkpoints/v2_tau1/best_model.pt

# Resume from specific epoch
python src/train.py \
  --config config/v2_tau1.yaml \
  --resume checkpoints/v2_tau1/checkpoint_epoch_25.pt

# Resume from interrupted checkpoint
python src/train.py \
  --config config/v2_tau1.yaml \
  --resume checkpoints/v2_tau1/interrupted.pt
```

**What gets restored:**
- Model weights
- Optimizer state (momentum, etc.)
- Current epoch number
- Best validation loss (for early stopping)

**What continues from config:**
- All training hyperparameters
- Learning rate schedule (continues from restored epoch)
- Early stopping patience (resets counter)

## Training Process

### Training Loop

For each epoch:

1. **Training phase:**
   - Iterate through all training batches
   - Forward pass: encoder → latent → decoder + power head
   - Compute combined loss (power MSE + reconstruction MSE)
   - Backward pass and optimizer step
   - Log progress every N batches

2. **Validation phase:**
   - Evaluate on validation set (no gradients)
   - Calculate validation loss and metrics (MAE, R², RMSE, MAPE)
   - Collect predictions for plotting

3. **Scheduler update:**
   - ReduceLROnPlateau checks validation loss
   - Reduces learning rate if no improvement for `patience` epochs

4. **Checkpoint saving:**
   - Save checkpoint for current epoch
   - Save as best model if validation loss improved
   - Update progress files

5. **Early stopping check:**
   - Track epochs without improvement
   - Stop if patience exceeded

6. **Periodic plotting:**
   - Generate training curves every N epochs

### Loss Function

The model uses a combined loss:

```python
total_loss = power_weight * power_mse + reconstruction_weight * reconstruction_mse
```

**Default weights:**
- `power_weight = 1.0` - Main objective (power prediction accuracy)
- `reconstruction_weight = 0.1` - Regularization (ensure latent space quality)

**Components:**

1. **Power MSE Loss:**
   ```python
   power_loss = MSE(predicted_power, actual_power)
   ```
   - Directly measures prediction accuracy
   - Main training signal

2. **Reconstruction MSE Loss:**
   ```python
   reconstruction_loss = MSE(reconstructed_input, original_input)
   ```
   - Ensures encoder preserves input information
   - Prevents latent space collapse
   - Acts as regularization

### Metrics

**Tracked during validation:**

- **MSE** - Mean Squared Error (lower is better)
- **RMSE** - Root Mean Squared Error (in Watts)
- **MAE** - Mean Absolute Error (primary metric, in Watts)
- **R²** - Coefficient of determination (0-1, higher is better)
- **MAPE** - Mean Absolute Percentage Error (%)

**Target performance (v2 vs v1):**
| Metric | v1 (stress-ng) | v1 (Prometheus) | v2 Target |
|--------|----------------|-----------------|-----------|
| R² | 0.85 | 0.12 (failed) | > 0.95 |
| MAE | ~5W | ~15W | < 3W |

## Monitoring Training

### Real-time Monitoring

**Terminal 1 - Training log:**
```bash
tail -f results/v2_tau1/training.log
```

**Terminal 2 - System resources:**
```bash
watch -n 1 nvidia-smi  # GPU utilization
```

**Terminal 3 - Disk space:**
```bash
watch -n 60 du -sh checkpoints/v2_tau1/
```

### Check Training Status

```bash
# Latest epoch
tail -n 20 results/v2_tau1/training_progress.txt

# Best validation loss so far
grep "New best model" results/v2_tau1/training.log | tail -n 1

# Current learning rate
grep "LR:" results/v2_tau1/training.log | tail -n 1

# Training time per epoch
grep "Time:" results/v2_tau1/training.log | tail -n 10
```

### Visualize Progress

Training curves are automatically generated every 5 epochs:

```bash
# View training curves
xdg-open results/v2_tau1/plots/training_curves.png

# Or copy to local machine
scp remote:/path/to/results/v2_tau1/plots/training_curves.png .
```

## Troubleshooting

### Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 128  # Default is 256
   ```

2. Use gradient accumulation (requires code modification)

3. Use CPU instead of GPU:
   ```yaml
   training:
     device: "cpu"
   ```

### Training Diverges (Loss → NaN)

**Symptoms:**
```
Epoch 5: train_loss=nan, val_loss=nan
```

**Solutions:**
1. Reduce learning rate:
   ```yaml
   training:
     learning_rate: 0.0001  # Default is 0.001
   ```

2. Increase weight decay:
   ```yaml
   training:
     weight_decay: 0.001  # Default is 0.0001
   ```

3. Check for data issues:
   ```bash
   # Look for NaN or inf in training data
   python -c "
   import numpy as np
   data = np.load('data/processed/tau1/train.npz')
   print('NaN in X:', np.isnan(data['X']).any())
   print('Inf in X:', np.isinf(data['X']).any())
   print('NaN in y:', np.isnan(data['y']).any())
   print('Inf in y:', np.isinf(data['y']).any())
   "
   ```

### Training Stalls (No Improvement)

**Symptoms:**
- Validation loss stops decreasing
- Early stopping triggered too early

**Solutions:**
1. Increase early stopping patience:
   ```yaml
   training:
     early_stopping_patience: 30  # Default is 15
   ```

2. Adjust learning rate schedule:
   ```yaml
   training:
     scheduler:
       patience: 10  # Default is 5
       factor: 0.3   # More aggressive reduction
   ```

3. Check if overfitting:
   - If train loss << val loss, add regularization
   - Increase dropout or weight decay

### Slow Training

**Expected time per epoch (5.6M samples, batch_size=256):**
- GPU (V100): ~5-10 minutes
- GPU (RTX 3090): ~3-5 minutes
- CPU: ~30-60 minutes

**If slower:**
1. Check GPU utilization:
   ```bash
   nvidia-smi
   # GPU-Util should be > 80%
   ```

2. Increase num_workers in DataLoader (requires code edit):
   ```python
   # In src/train.py, increase from 4 to 8
   num_workers=8
   ```

3. Enable pin_memory (already enabled for CUDA)

## Advanced Usage

### Multi-GPU Training

To use multiple GPUs, modify `src/train.py`:

```python
# After model creation
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    self.model = nn.DataParallel(self.model)
```

### Custom Loss Weights

To adjust the balance between power prediction and reconstruction:

```yaml
training:
  loss_weights:
    power: 1.0            # Prioritize power accuracy
    reconstruction: 0.05  # Reduce reconstruction importance
```

**Recommendations:**
- Higher reconstruction weight (e.g., 0.5) if latent space visualization is poor
- Lower reconstruction weight (e.g., 0.05) if power MAE is the only concern

### Hyperparameter Search

For systematic hyperparameter tuning:

```bash
# Create configs for different hyperparameters
for lr in 0.0001 0.001 0.01; do
  for batch in 128 256 512; do
    cp config/v2_tau1.yaml config/v2_lr${lr}_batch${batch}.yaml
    # Edit learning_rate and batch_size
    # Then train:
    python src/train.py --config config/v2_lr${lr}_batch${batch}.yaml &
  done
done
```

## Expected Results

### Training Progress

**Typical training curve:**
- **Epochs 0-20:** Rapid improvement, loss drops significantly
- **Epochs 20-50:** Steady improvement, learning rate may reduce
- **Epochs 50-80:** Slow convergence, early stopping may trigger
- **Epochs 80-100:** Minimal improvement or stopped

**Early stopping typically triggers around epoch 50-70.**

### Final Performance

**Target metrics on validation set (tau=1):**
- **MAE:** < 3W (v1 achieved ~5W)
- **R²:** > 0.95 (v1 achieved 0.85)
- **RMSE:** < 5W
- **MAPE:** < 10%

**If metrics are worse:**
- Check data quality (preprocessing correctness)
- Try different tau values (tau=4 or tau=8 may work better)
- Adjust loss weights or model architecture

## Next Steps After Training

Once training completes with satisfactory metrics:

1. **Evaluate on test set** (TODO - needs implementation)
2. **Test on Prometheus workloads** (real-world validation)
3. **Visualize latent space** (t-SNE/UMAP analysis)
4. **Export model for inference** (deploy to powermon)

See [Architecture Guide](architecture.md) for details on model inference and deployment.

## References

- **Model Architecture:** [docs/architecture.md](architecture.md)
- **Preprocessing:** [docs/preprocessing.md](preprocessing.md)
- **Training Script:** [src/train.py](../src/train.py)
- **Config Example:** [config/v2_tau1.yaml](../config/v2_tau1.yaml)
