# DECODE-RAPL Training Guide

## GPU Machine Setup

### 1. Transfer Files to GPU Machine

Transfer the entire `decode-rapl/` directory to your GPU machine:

```bash
# From your local machine
scp -r /path/to/decode-rapl user@gpu-machine:/path/to/destination/
```

Or use rsync for better handling of large files:

```bash
rsync -avz --progress /path/to/decode-rapl/ user@gpu-machine:/path/to/destination/decode-rapl/
```

### 2. Install Dependencies on GPU Machine

```bash
# SSH into GPU machine
ssh user@gpu-machine

# Navigate to decode-rapl directory
cd /path/to/destination/decode-rapl

# Install Python dependencies
pip install -r requirements.txt

# Verify GPU is available
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
GPU available: True
GPU: NVIDIA GeForce RTX 3090 (or your GPU model)
```

### 3. Verify Training Data

Make sure the training data file exists:

```bash
ls -lh data/real_data_2hr_clean.csv
```

If missing, transfer from local machine:

```bash
# From local machine
scp /path/to/your/training_data.csv user@gpu-machine:/path/to/decode-rapl/data/
```

## Training Options

### Option A: Train Single Configuration (Background)

Train one tau configuration at a time in background:

```bash
# Train tau=1 (baseline, maximum temporal detail)
./scripts/train_tau_bg.sh config_tau1.yaml

# Monitor progress in real-time
tail -f logs/training_YYYYMMDD_HHMMSS.log

# Check status
cat logs/training_YYYYMMDD_HHMMSS.status
```

**When to use:**
- You want to run one configuration and evaluate before trying others
- You want to run different configs in parallel on multiple GPUs
- You need fine control over each training run

### Option B: Train All Configurations Sequentially

Train all three tau configurations one after another:

```bash
# Train tau1 → tau4 → tau8 sequentially
./scripts/train_all_tau.sh

# This will:
# 1. Train config_tau1.yaml (τ=1 sample = 16ms)
# 2. Train config_tau4.yaml (τ=4 samples = 64ms)
# 3. Train config_tau8.yaml (τ=8 samples = 128ms)
```

**When to use:**
- You want to run the complete comparison experiment
- You can leave the machine running for extended time (~6 hours total)
- You want a comprehensive comparison of all tau values

**Note:** This script waits for each training to complete before starting the next one. You can safely close your terminal (uses nohup internally).

## Monitoring Training

### Real-time Progress

Monitor training in real-time with tail:

```bash
# For single training
tail -f logs/training_YYYYMMDD_HHMMSS.log

# For sequential training (all tau)
tail -f logs/train_all_tau_YYYYMMDD_HHMMSS.log
```

Look for:
- `Epoch X/100` - Training progress
- `Val MAPE: X.XX%` - Validation accuracy
- `Saved best model` - Model checkpoints
- `Final Test MAPE: X.XX%` - Final accuracy

### Check Training Status

```bash
# Check if training process is still running
ps aux | grep train.py

# For background training, check PID file
ps -p $(cat logs/training_YYYYMMDD_HHMMSS.pid)

# Check status file (JSON format)
cat logs/training_YYYYMMDD_HHMMSS.status
```

### Watch for Completion

Use `watch` to periodically check progress:

```bash
# For single training
watch -n 10 'grep -E "Epoch|MAPE|completed" logs/training_YYYYMMDD_HHMMSS.log | tail -20'

# For sequential training
watch -n 30 'tail -50 logs/train_all_tau_YYYYMMDD_HHMMSS.log'
```

## Training Outputs

### Directory Structure

After training completes, you'll have:

```
decode-rapl/
├── checkpoints/
│   ├── tau1/
│   │   └── best_model.pth          # Best tau=1 model
│   ├── tau4/
│   │   └── best_model.pth          # Best tau=4 model
│   └── tau8/
│       └── best_model.pth          # Best tau=8 model
├── results/
│   ├── tau1/plots/
│   │   ├── training_curves.png     # Loss curves
│   │   ├── test_predictions.png    # Predictions scatter plot
│   │   └── test_errors.png         # Error distribution
│   ├── tau4/plots/
│   │   └── ...
│   └── tau8/plots/
│       └── ...
└── logs/
    ├── training_*.log               # Training logs
    ├── training_*.status            # Status files
    └── decode_rapl_tau*.log         # Application logs
```

### Extracting Results

Get final test MAPE for each configuration:

```bash
# tau=1 results
grep "Final Test MAPE:" logs/train_tau1_*.log

# tau=4 results
grep "Final Test MAPE:" logs/train_tau4_*.log

# tau=8 results
grep "Final Test MAPE:" logs/train_tau8_*.log

# Or from sequential training log
grep "Test MAPE:" logs/train_all_tau_*.log
```

### Model Checkpoints

Each checkpoint file contains:
- Model state dict (trained weights)
- Optimizer state
- Configuration used
- Machine ID mapping
- Data scalers (normalization parameters)
- Delay embedder
- Training history

Load checkpoint for inference:

```python
import torch

checkpoint = torch.load('checkpoints/tau1/best_model.pth')
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Config: {checkpoint['config']['embedding']['tau']} samples tau")
print(f"Final MAPE: {checkpoint['history']['val_mape'][-1]:.2f}%")
```

## Retrieving Results from GPU Machine

After training completes, transfer results back to local machine:

```bash
# From local machine

# Transfer all results
rsync -avz --progress user@gpu-machine:/path/to/decode-rapl/results/ ./results/
rsync -avz --progress user@gpu-machine:/path/to/decode-rapl/checkpoints/ ./checkpoints/
rsync -avz --progress user@gpu-machine:/path/to/decode-rapl/logs/ ./logs/

# Or just specific tau results
rsync -avz user@gpu-machine:/path/to/decode-rapl/results/tau1/ ./results/tau1/
rsync -avz user@gpu-machine:/path/to/decode-rapl/checkpoints/tau1/ ./checkpoints/tau1/
```

## Expected Training Time

Rough estimates with GPU (RTX 3090 or similar):

| Config | τ Value | Window Size | Sequences | Time/Epoch | Total Time |
|--------|---------|-------------|-----------|------------|------------|
| tau1   | 1       | 120         | ~450k     | ~2 min     | ~3-4 hours |
| tau4   | 4       | 30          | ~450k     | ~1 min     | ~1-2 hours |
| tau8   | 8       | 15          | ~450k     | ~30 sec    | ~30-60 min |

**Total for sequential training:** ~5-7 hours (assuming early stopping)

Times will vary based on:
- GPU model and memory
- Batch size (default: 32)
- Early stopping (patience: 10 epochs)
- Number of training samples (~450k from 2-hour data)

## Troubleshooting

### Out of Memory (OOM) Errors

If training fails with CUDA OOM:

```bash
# Reduce batch size in config files
# Edit config_tau1.yaml, config_tau4.yaml, config_tau8.yaml
training:
  batch_size: 16  # Reduce from 32 to 16
```

### Training Too Slow

If training on CPU (no GPU detected):

```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# May need to install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Missing Data File

If training fails with file not found:

```bash
# Check data path in config
grep "train_csv:" config_tau1.yaml

# Verify file exists
ls -lh data/real_data_2hr_clean.csv

# If missing, create from MS-TCN data
python scripts/convert_mstcn_data.py \
    ../ms-tcn/data/training_diverse_2hr.csv \
    data/real_data_2hr_clean.csv
```

### Process Killed Unexpectedly

If background process stops:

```bash
# Check system logs
dmesg | tail -50

# Check if OOM killer terminated it
grep -i "killed process" /var/log/messages

# Reduce batch size or free up memory
```

## Next Steps After Training

1. **Compare Results**: Review test MAPE for tau1, tau4, tau8
2. **Analyze Plots**: Check prediction scatter plots and error distributions
3. **Select Best Config**: Choose based on MAPE vs training time trade-off
4. **Run Inference**: Use best model for real-time power prediction

See `CONFIGS_README.md` for detailed comparison guidelines.
