# DECODE-RAPL (v1 — archived)

> **This is the first iteration of DECODE-RAPL and has been superseded.**
> The current work lives at the repo root (v2/v3/v4 — same code tree, three architecture variants).
> See the top-level [README](../../README.md) for the project overview and the
> canonical interim report at [`../../DECORE-RAPL-REPORT.md`](../../DECORE-RAPL-REPORT.md)
> (§3 covers what was learned from this v1 attempt).
>
> Kept here for context: the v1 architecture used `Delay Embedding + Autoencoder + LSTM`,
> and the analysis docs in this folder ([`LSTM_ANALYSIS_SUMMARY.md`](LSTM_ANALYSIS_SUMMARY.md),
> [`MULTIFEATURE_README.md`](MULTIFEATURE_README.md)) explain why the LSTM stage
> was eventually dropped and why the latent dimension was a bottleneck.

---

**Delayed Embedding and Coherent Dynamics for Emulating RAPL**

Machine learning framework for predicting CPU power consumption from usage patterns, with the goal of enabling power monitoring in virtual machines where Intel RAPL is unavailable. Note: data collection and experiments here were performed on bare metal (RAPL is not exposed to VM guests); the model design constrains itself to features that *would* be visible from a VM.

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy pandas scikit-learn pyyaml matplotlib

# 2. Train model (uses synthetic data for demo)
./scripts/train_model.sh

# 3. Run inference
./scripts/run_inference.sh
```

## What is DECODE-RAPL?

DECODE-RAPL reverse-engineers Intel RAPL power estimation using:
- **Time-delay embedding**: Reconstructs CPU-power dynamics from usage history
- **Autoencoder**: Learns compact latent representation of system state
- **LSTM**: Predicts power from temporal patterns in latent space
- **Adversarial training**: Generalizes across different machines

**Key Features:**
- Train on bare-metal machines with RAPL access
- Deploy in VMs without RAPL for power prediction
- <5% MAPE target accuracy
- Real-time inference with buffering
- Cross-machine generalization

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Linux (for MSR access on bare-metal)

### Setup

```bash
# Clone repository
cd decode-rapl

# Install dependencies
pip install torch numpy pandas scikit-learn pyyaml matplotlib

# For data collection on bare-metal (optional)
sudo apt-get install msr-tools stress-ng
sudo modprobe msr  # Load MSR kernel module
```

## Usage

### 1. Data Collection (Bare-Metal Only)

Collect CPU usage and RAPL power data from bare-metal machines:

```bash
# Run with root privileges for MSR access
sudo python3 -m src.data_collector \
    --machine-id skylake1 \
    --duration 10 \
    --sampling-rate 1 \
    --output data/rapl_data.csv
```

**Parameters:**
- `--machine-id`: Unique machine identifier
- `--duration`: Collection duration in hours
- `--sampling-rate`: Sampling rate in milliseconds (default: 1ms)
- `--output`: Output CSV path

Repeat on 3-5 machines for best generalization.

### 2. Training

Train the model on collected data:

```bash
./scripts/train_model.sh
```

Or manually:

```bash
python3 -m src.train
```

**Configuration**: Edit `config.yaml` to adjust hyperparameters.

**Outputs:**
- Trained model: `checkpoints/best_model.pth`
- Training curves: `results/plots/training_curves.png`
- Evaluation plots: `results/plots/test_predictions.png`

### 3. Inference

#### Bare-Metal Inference

```python
from src.inference import RAPLPredictor

# Load model
predictor = RAPLPredictor('checkpoints/best_model.pth')

# Real-time prediction
predictor.update_usage(cpu_usage_percent)  # 0-100
power_watts = predictor.predict()
```

#### VM Inference

```python
predictor = RAPLPredictor(
    'checkpoints/best_model.pth',
    vm_mode=True,
    vm_vcpus=4,        # Your VM's vCPUs
    host_cores=16      # Host physical cores
)

predictor.update_usage(vm_cpu_usage)
vm_power = predictor.predict()
```

Run inference test:

```bash
# Bare-metal mode
./scripts/run_inference.sh

# VM mode
./scripts/run_inference.sh --vm --vm-vcpus 4 --host-cores 16
```

### 4. Testing with Synthetic Data

No bare-metal access? Use synthetic data for testing:

```python
from src.utils import generate_synthetic_data

# Generate test data
generate_synthetic_data(
    num_machines=3,
    duration_hours=0.5,
    sampling_rate_ms=1,
    output_path='data/rapl_train.csv'
)

# Train on synthetic data
python3 -m src.train
```

## Architecture

```
CPU Usage (0-100%)
    ↓
[Delay Embedding] → h(t) ∈ ℝ²⁵
    ↓
[Encoder] → z(t) ∈ ℝ⁸ (latent)
    ↓
[LSTM] → Power (Watts)

Training only:
z(t) → [Discriminator] → Machine ID (for generalization)
```

**Model Components:**
1. **Delay Embedding** (d=25, τ=1ms): Captures temporal structure
2. **Autoencoder** (256→64→32→8): Compresses to latent space
3. **LSTM** (hidden=64): Predicts power from latent sequences
4. **Discriminator**: Encourages machine-invariant features

See [docs/decode-rapl.md](docs/decode-rapl.md) for detailed methodology.

## Configuration

Edit `config.yaml` to customize:

```yaml
embedding:
  tau: 1        # Time delay (ms)
  d: 25         # Embedding dimension

model:
  latent_dim: 8
  lstm_hidden_size: 64

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100

inference:
  buffer_size: 60      # Sequence length
  vm_mode: false
  vm_vcpus: 4
  host_cores: 16
```

## Project Structure

```
decode-rapl/
├── config.yaml              # Configuration
├── README.md                # This file
├── docs/
│   └── decode-rapl.md       # Detailed documentation
├── prompt/
│   ├── model.txt            # Model specification
│   └── grok.txt             # Research conversation
├── src/
│   ├── utils.py             # Utilities (synthetic data, metrics)
│   ├── preprocessing.py     # Delay embedding, dataset
│   ├── model.py             # Neural network architecture
│   ├── train.py             # Training pipeline
│   ├── inference.py         # Real-time prediction
│   └── data_collector.py    # MSR reader (bare-metal)
├── scripts/
│   ├── train_model.sh       # Training script
│   └── run_inference.sh     # Inference script
├── data/                    # Training data
├── checkpoints/             # Saved models
├── results/plots/           # Visualizations
└── logs/                    # Training logs
```

## Performance

**Target Metrics:**
- Test MAPE: <5%
- Training time: ~1-2 hours (GPU) for 30min/machine data
- Inference latency: <1ms (CPU), <0.1ms (GPU)
- Model size: ~500K parameters

**Tested On:**
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)
- Intel Skylake, Cascade Lake CPUs

## Troubleshooting

### MSR Access Denied

```bash
sudo modprobe msr
sudo chmod +r /dev/cpu/*/msr
```

### Training Loss Not Decreasing

- Check data quality (sufficient variation in CPU usage)
- Increase embedding dimension (d=30-50)
- Reduce learning rate (0.0001)
- Add more machines to dataset

### VM Inference Inaccurate

- Verify `vm_vcpus` and `host_cores` are correct
- Check for host-level CPU contention
- Increase sampling rate (10ms → 100ms for smoother signal)

## Limitations

1. **Architecture-specific**: Train separate models for different CPU families
2. **Package-level only**: Matches RAPL (no per-core attribution)
3. **MSR requirement**: 1ms sampling needs bare-metal access
4. **VM scaling assumptions**: Assumes linear resource allocation

## Contributing

Contributions welcome! Areas for improvement:
- Multi-architecture training
- Additional input features (IPC, cache, memory BW)
- Online learning / domain adaptation
- Uncertainty quantification

## Citation

If you use DECODE-RAPL in research:

```bibtex
@software{decode_rapl_2024,
  title={DECODE-RAPL: Delayed Embedding and Coherent Dynamics for Emulating RAPL},
  author={Vimal Kumar},
  year={2024},
  url={https://github.com/vimalk78/decode-rapl}
}
```

## License

Apache License 2.0 - see LICENSE file at the repo root.

## References

1. Bakarji et al., "Discovering Governing Equations from Partial Measurements with Deep Delay Autoencoders" (2021)
2. Intel RAPL: David et al., "RAPL: Memory Power Estimation and Capping" (2010)
3. Takens, F., "Detecting strange attractors in turbulence" (1981)

## Contact

Issues: https://github.com/vimalk78/decode-rapl/issues

---

**DECODE-RAPL** - Bringing RAPL to VMs through Machine Learning
