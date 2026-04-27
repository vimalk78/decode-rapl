# DECODE-RAPL v2 Training Guide

Complete guide for training models on GPU machines (local or cloud).

## Overview

You'll train **3 separate models** (tau=1, tau=4, tau=8) on ~4.4-4.7M samples each.

**Model specs:**
- Architecture: Encoder (100→64) + Decoder (64→100) + Power Head (64→1)
- Parameters: 267,941 (~268K)
- Batch size: 256
- Epochs: 100 (with early stopping)

## Prerequisites

### 1. Preprocessed Data

You need the preprocessed data (created by `scripts/prepare_training_data.py`):

```
data/processed/
├── tau1/
│   ├── train.npz  (~1.3GB)
│   ├── val.npz    (~157MB)
│   └── test.npz   (~157MB)
├── tau4/
│   ├── train.npz  (~1.2GB)
│   ├── val.npz    (~154MB)
│   └── test.npz   (~154MB)
└── tau8/
    ├── train.npz  (~1.2GB)
    ├── val.npz    (~149MB)
    └── test.npz   (~149MB)
```

**Total size:** ~4.6GB

### 2. Python Environment

```bash
# Python 3.8+
pip install -r requirements.txt
```

**Required packages:**
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- PyYAML >= 6.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- tqdm

## Quick Start

### Step 1: Verify System

```bash
cd /path/to/decode-rapl
./scripts/check_gpu.sh
```

This checks:
- Python environment
- PyTorch + CUDA
- GPU availability and memory
- Training data presence
- Estimated training time

### Step 2: Transfer Data (if training remotely)

```bash
# From your local machine with preprocessed data
rsync -avz --progress data/processed/ user@gpu-machine:/path/to/decode-rapl/data/processed/

# Or use scp
scp -r data/processed user@gpu-machine:/path/to/decode-rapl/data/
```

### Step 3: Start Training

**Train single model:**
```bash
./scripts/start_training.sh tau1
```

**Train all models sequentially:**
```bash
./scripts/start_training.sh all
```

**Train specific models:**
```bash
./scripts/start_training.sh tau1 tau4
```

### Step 4: Monitor Progress

```bash
# One-time status check
./scripts/monitor_training.sh

# Continuous monitoring (updates every 10s)
./scripts/monitor_training.sh --watch

# Or tail individual logs
tail -f results/v2_tau1/training.log
tail -f logs/train_tau1.out
```

## Training Time Estimates

| GPU | Time per Epoch | Time per Model | Total (3 models) | Cost (cloud) |
|-----|----------------|----------------|------------------|--------------|
| **GTX 1650** | 30-60 min | 1-3 days | 3-9 days | N/A (local) |
| **RTX 3090** | 8-12 min | 8-12 hours | 24-36 hours | ~$10-15 |
| **RTX 4090** | 5-8 min | 5-8 hours | 15-24 hours | ~$15-25 |
| **A100 (40GB)** | 3-5 min | 3-5 hours | 9-15 hours | ~$15-20 |
| **V100** | 5-10 min | 5-10 hours | 15-30 hours | ~$20-30 |

**Note:** Times assume ~50 epochs with early stopping (not full 100 epochs)

## Cloud GPU Recommendations

If your GTX 1650 is too slow, consider renting a cloud GPU:

### 1. Lambda Labs (Recommended)
**Website:** https://lambdalabs.com/service/gpu-cloud

**Pros:**
- Simple interface
- Pre-configured PyTorch environment
- Fast GPUs (RTX 4090, A100)
- Good documentation

**Pricing:**
- RTX 4090 (24GB): **$0.99/hr** → ~$25 total for all 3 models
- A100 (40GB): **$1.10/hr** → ~$15 total
- A100 (80GB): $1.89/hr

**Setup:**
```bash
# SSH to instance
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>

# Clone repo
git clone <your-repo-url>
cd decode-rapl

# Install requirements
pip install -r requirements.txt

# Transfer data (from local machine)
rsync -avz -e "ssh -i ~/.ssh/lambda_key" data/processed/ ubuntu@<instance-ip>:/home/ubuntu/decode-rapl/data/processed/

# Start training
./scripts/start_training.sh all

# Detach (training continues in background)
exit
```

### 2. Vast.ai (Cheapest)
**Website:** https://vast.ai/

**Pros:**
- Very cheap (spot pricing)
- Many GPU options
- Pay-per-second billing

**Cons:**
- Instances can be interrupted (less reliable)
- More setup required

**Pricing:**
- RTX 3090: **$0.30-0.50/hr** → ~$10-15 total
- RTX 4090: **$0.60-0.80/hr** → ~$15-20 total

**Setup:**
```bash
# Search for instance
# Filter: GPU RAM >= 16GB, CUDA >= 11.8, Storage >= 50GB

# SSH to instance
ssh -p <port> root@<instance-ip>

# Install dependencies
apt update && apt install -y git rsync
pip install -r requirements.txt

# Transfer data and start training (same as Lambda)
```

### 3. RunPod
**Website:** https://www.runpod.io/

**Pros:**
- Good balance of price and reliability
- Easy to use
- Persistent storage option

**Pricing:**
- RTX 3090: **$0.44/hr** → ~$15 total
- RTX 4090: **$0.79/hr** → ~$20 total

**Setup:** Similar to Lambda Labs

### 4. Google Colab Pro+ (Not Recommended)
**Website:** https://colab.research.google.com/

**Pros:**
- Familiar Jupyter interface
- No instance management

**Cons:**
- 12-hour runtime limit (training takes longer)
- Needs manual checkpointing and resuming
- More expensive ($50/month)

**Not recommended for this project** due to long training time.

## Configuration Files

Three config files for the three models:

### config/v2_tau1.yaml
- Data: `data/processed/tau1/`
- Output: `checkpoints/v2_tau1/`, `results/v2_tau1/`
- Temporal lookback: 384ms (25 samples)

### config/v2_tau4.yaml
- Data: `data/processed/tau4/`
- Output: `checkpoints/v2_tau4/`, `results/v2_tau4/`
- Temporal lookback: 1536ms (97 samples)

### config/v2_tau8.yaml
- Data: `data/processed/tau8/`
- Output: `checkpoints/v2_tau8/`, `results/v2_tau8/`
- Temporal lookback: 3072ms (193 samples)

**All models share the same architecture** - only training data differs.

## Manual Training (without scripts)

If you prefer to run training manually:

```bash
# Create directories
mkdir -p logs checkpoints results

# Train tau=1
nohup python src/train.py --config config/v2_tau1.yaml > logs/train_tau1.out 2>&1 &

# Monitor
tail -f results/v2_tau1/training.log

# After tau=1 completes, train tau=4
nohup python src/train.py --config config/v2_tau4.yaml > logs/train_tau4.out 2>&1 &

# Then tau=8
nohup python src/train.py --config config/v2_tau8.yaml > logs/train_tau8.out 2>&1 &
```

## Resume Training

If training is interrupted, resume from checkpoint:

```bash
python src/train.py \
  --config config/v2_tau1.yaml \
  --resume checkpoints/v2_tau1/checkpoint_epoch_25.pt
```

## Training Output

Each model creates:

```
results/v2_tau1/
├── training.log              # Detailed training log
├── training_progress.txt     # Epoch summary
├── training_history.json     # Metrics for plotting
└── plots/
    ├── training_curves.png   # Loss curves (updated every 5 epochs)
    └── final_predictions.png # Scatter plot at end

checkpoints/v2_tau1/
├── best_model.pt             # Best model based on val loss
├── checkpoint_epoch_0.pt     # Checkpoints for each epoch
├── checkpoint_epoch_1.pt
└── ...
```

## What to Expect

### Early Epochs (0-10)
- Train loss drops quickly: ~100 → ~20
- Val loss follows train loss
- Learning rate: 0.001

### Middle Epochs (10-40)
- Train/val loss stabilize: ~10-15
- Learning rate may reduce (ReduceLROnPlateau)
- Val MAE should be <5W

### Late Epochs (40+)
- Minimal improvement
- Early stopping may trigger
- Val R² should be >0.90

### Target Metrics (at convergence)
- **Val MAE**: <3W
- **Val R²**: >0.95
- **Val MAPE**: <10%

If metrics are worse:
- Check data preprocessing
- Try increasing latent_dim from 64 to 128
- Increase model size (encoder_layers: [512, 256, 128])

## Troubleshooting

### Out of Memory (OOM)
**Error:** `CUDA out of memory`

**Solution:**
```yaml
# In config file, reduce batch_size
training:
  batch_size: 128  # or 64
```

### Training Hangs
**Symptom:** Progress bar frozen

**Solution:**
```bash
# Check GPU usage
nvidia-smi

# Kill process
pkill -f train.py

# Reduce num_workers in DataLoader
# Edit src/train.py line 142:
num_workers: 2  # or 0
```

### Slow Training
**Symptom:** 100+ min per epoch on GPU

**Possible causes:**
1. CPU bottleneck (DataLoader)
2. GPU not being used
3. Small GPU (GTX 1650)

**Check:**
```bash
# Verify GPU usage
nvidia-smi -l 1

# Should show ~80-100% GPU utilization
```

### Poor Validation Metrics
**Symptom:** Val MAE >5W, R² <0.85

**Possible causes:**
1. Underfitting (model too small)
2. Data quality issues
3. Wrong preprocessing

**Solutions:**
- Check data: `python -c "import numpy as np; d=np.load('data/processed/tau1/train.npz'); print(d['X'].shape, d['y'].shape)"`
- Increase model size (see "What to Expect" section)
- Check preprocessing metadata: `cat data/processed/tau1/metadata.json`

## After Training

### 1. Evaluate on Test Set

```bash
# TODO: Create evaluation script
python src/evaluate.py --model checkpoints/v2_tau1/best_model.pt
```

### 2. Test on Prometheus Workloads

Deploy to your bare-metal host and test on real workloads:

```bash
# Copy model to the host
scp checkpoints/v2_tau1/best_model.pt <your-bare-metal-host>:/opt/powermon/models/

# Run live inference
ssh <your-bare-metal-host>
cd /path/to/decode-rapl
sudo python src/power_predictor.py \
  --model checkpoints/v4_tau1/best_model.pt \
  --live --scroll
```

### 3. Compare Tau Models

After all 3 models are trained:
- Compare validation metrics
- Test on different workload types
- Pick best model (likely tau=1 for general use)

### 4. Next Steps

- [ ] Latent space visualization
- [ ] Ablation studies
- [ ] Production deployment
- [ ] Integration with powermon

## Cost Summary

**Local training (GTX 1650):**
- Cost: $0 (but takes 3-9 days)

**Cloud training (recommended for speed):**
- Lambda Labs RTX 4090: ~$25 total
- Vast.ai RTX 3090: ~$10-15 total
- RunPod RTX 4090: ~$20 total

**Choose based on:**
- Budget: Use GTX 1650 if you can wait
- Speed: Use RTX 4090 on Lambda Labs (~24 hours total)
- Balance: Use RTX 3090 on Vast.ai (~36 hours, cheapest)

## References

- [Data Collection Guide](docs/data_collection.md)
- [Preprocessing Guide](docs/preprocessing.md)
- [Training Pipeline](docs/training.md)
- [Inference Guide](docs/inference.md)
- [Architecture Details](docs/architecture.md)
