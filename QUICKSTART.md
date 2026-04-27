# Quickstart — running the DECODE-RAPL model code

> **Note.** This is a developer-level quickstart for the v2/v3/v4 model code (originally written when v2 was the active version; v3 and v4 share the same `src/`, just with different model classes and configs). For the project overview, the headline result, and the analysis of why the model fails on real workloads, read [`README.md`](README.md) and [`DECORE-RAPL-REPORT.md`](DECORE-RAPL-REPORT.md) at the repo root first.

The architecture summary below describes v2's simplification over v1 (in `archive/decode-rapl-v1/`); v3 then dropped the decoder, and v4 swapped the MLP encoder for a 1D-CNN.

## Key Changes from v1

### Architecture (Implemented ✓)
- **Remove LSTM** - delay embedding already provides temporal encoding
- **Increase latent space**: 16 → 64 dimensions
- **Remove window dimension** - process single delay-embedded vectors (batch, 100)
- **New MLP power head** - direct feedforward instead of LSTM
- **4 features**: user%, system%, iowait%, log(1+ctx_switches)
- **Total parameters**: ~267K (simpler than v1)

### Data Collection (Implemented ✓)
- **Go-based collector** - minimal overhead, 16ms sampling
- **Combinatorial workload generator** - covers full CPU state space (2025 combinations)
- **Diverse workloads**: CPU compute, syscalls, I/O, pipes, memory, cache
- **Idle baseline** - captures base system power (~25W)
- **Multi-tau preprocessing** - generates datasets for tau=1,4,8

## Quick Start

### 1. Data Collection

```bash
# Generate workload configuration
cd scripts/
./generate_config.sh

# Start collection (runs for ~35 hours on 20-core system)
export DATA_DIR=/opt/rapl-training-data
sudo -E ./start_collection.sh

# Monitor progress
tail -f /opt/rapl-training-data/run.log
```

See [Data Collection Guide](docs/data_collection.md) for full details.

### 2. Data Preprocessing

```bash
# Convert CSVs to delay-embedded training datasets
python scripts/prepare_training_data.py \
  --data-dir data/all-combinations-temp-0 \
  --output-dir data/processed \
  --tau 1 4 8 \
  --skip-startup 100 \
  --seed 42
```

Generates train/val/test splits for 3 tau values (~15 minutes).

See [Preprocessing Guide](docs/preprocessing.md) for full details.

### 3. Model Training

**Quick Start** (recommended):

```bash
# Check GPU and system requirements
./scripts/check_gpu.sh

# Train single model (tau=1)
./scripts/start_training.sh tau1

# Train all models sequentially
./scripts/start_training.sh all

# Monitor progress
./scripts/monitor_training.sh --watch
```

**Manual Training**:

```bash
# Train tau=1 model
nohup python src/train.py --config config/v2_tau1.yaml > logs/train_tau1.out 2>&1 &

# Train tau=4 model
nohup python src/train.py --config config/v2_tau4.yaml > logs/train_tau4.out 2>&1 &

# Train tau=8 model
nohup python src/train.py --config config/v2_tau8.yaml > logs/train_tau8.out 2>&1 &

# Monitor progress
tail -f results/v2_tau1/training.log
```

**Training Details:**
- Model parameters: 267,941 (~268K)
- Training samples: ~4.4-4.7M per tau
- Batch size: 256
- Expected epochs: ~50 (with early stopping)
- GTX 1650: ~30-60 min/epoch (1-3 days per model)
- RTX 4090: ~5-8 min/epoch (5-8 hours per model)

**See [TRAINING.md](TRAINING.md) for:**
- Cloud GPU recommendations (Lambda Labs, Vast.ai, RunPod)
- Detailed setup instructions
- Cost estimates (~$10-25 for cloud GPU)
- Troubleshooting guide

Training features:
- Background execution support (nohup compatible)
- Automatic checkpoint saving and resume capability
- Early stopping based on validation loss
- Learning rate scheduling (ReduceLROnPlateau)
- Progress logging to file and stdout
- Graceful shutdown on SIGTERM/SIGINT

See [Training Guide](docs/training.md) for full details.

### 4. Model Inference

```bash
# Live monitoring (requires sudo for RAPL)
sudo python src/power_predictor.py --model checkpoints/v2_tau1/best_model.pt --live --scroll

# Custom prediction frequency
sudo python src/power_predictor.py --model checkpoints/v2_tau1/best_model.pt --live --scroll --frequency 1.0

# CSV replay
python src/power_predictor.py --model checkpoints/v2_tau1/best_model.pt --csv test_data.csv --save predictions.csv
```

Inference features:
- Dynamic tau support (auto-adapts to model's tau value)
- Real-time buffered prediction (<1ms latency)
- Live RAPL comparison with color-coded accuracy
- CSV replay mode for batch processing
- Flexible prediction frequency

See [Inference Guide](docs/inference.md) for full details.

## Directory Structure

```
decode-rapl/
├── collector/              # Go data collector (16ms sampling)
│   ├── my_data_collector.go
│   └── my_data_collector      # Compiled binary
├── scripts/                # Data collection & preprocessing
│   ├── generate_config.sh     # Generate workload combinations
│   ├── start_collection.sh    # Start background collection
│   ├── stop_collection.sh     # Stop collection gracefully
│   ├── run_workloads.sh       # Main collection loop
│   ├── validate_collection.py # Validate collected data
│   ├── plot_workload.py       # Visualize single CSV
│   └── prepare_training_data.py # Preprocess to training format
├── src/                    # Python model code
│   ├── model.py               # v2 architecture (Encoder + Decoder + Power Head)
│   ├── train.py               # Training pipeline with background execution
│   ├── inference.py           # Real-time prediction with dynamic tau support
│   ├── power_predictor.py     # CLI tool for live monitoring
│   └── utils.py               # Utilities (metrics, plotting, config)
├── config/                 # Training configs
│   ├── v2_tau1.yaml           # tau=1 config (384ms lookback)
│   ├── v2_tau4.yaml           # tau=4 config (1.5s lookback)
│   └── v2_tau8.yaml           # tau=8 config (3s lookback)
├── docs/                   # Detailed documentation
│   ├── data_collection.md     # Collection guide
│   ├── preprocessing.md       # Preprocessing guide
│   ├── training.md            # Training guide
│   ├── inference.md           # Inference and deployment guide
│   ├── workload_types.md      # Workload patterns explained
│   └── architecture.md        # Model architecture details
├── data/                   # Training data (gitignored)
├── requirements.txt        # Python dependencies
└── README.md
```

## Requirements

### System Requirements
- **stress-ng >= 0.19.00** - Required for `--syscall-method` option
  - Fedora 39 ships with 0.17.x which is too old
  - Install from: https://github.com/ColinIanKing/stress-ng
- **Go 1.x** - For building the data collector
- **Python 3.8+** - For preprocessing and training
- **Root access** - Required for RAPL energy measurements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- PyYAML >= 6.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0

## Data Collection Features

### Workload Types

The collector generates 6 workload types + idle baseline:
- **Idle baseline** - base system power (~25W)
- **CPU compute** - floating-point operations (HIGH power: 70-90W)
- **Syscalls** - lightweight system calls (LOW power: 28-35W) ⭐ **Critical pattern missing from v1**
- **I/O** - disk operations with iowait%
- **Pipes** - high context switches
- **Memory (VM)** - memory allocation/access
- **Cache** - L3 cache thrashing

Combinatorial mixing creates **2025 combinations** covering full power spectrum (23W-90W).

See [Workload Types Guide](docs/workload_types.md) for detailed explanation.

### Collection Management

- **Background execution** - runs in background, logs to file
- **Resume capability** - interrupted collections resume automatically
- **Configuration snapshot** - each collection is reproducible
- **State file** - `/tmp/decode-rapl-collection.state` for process management

### Output Format

Each CSV file (one per workload combination):
```csv
timestamp_unix,user_percent,system_percent,iowait_percent,ctx_switches_per_sec,package_power_watts
1760876345.676,0.000,0.000,0.000,2162.5,26.240
1760876345.693,0.000,0.000,0.000,2155.8,26.086
...
```

- Sampling rate: 16ms (62.5 Hz)
- Duration: ~48 seconds per workload (~3000 samples)
- Filename encodes workload: `run_N_of_2025-cpu*-sys*-io*-pipe*-vm*-cache*.csv`

## Preprocessing Features

- **Startup filtering** - removes first 100 samples to eliminate transients
- **Log transform** - `log(1 + ctx_switches)` to handle outliers
- **Delay embedding** - creates 100-dim vectors (4 features × 25 delays)
- **Feature-grouped ordering** - temporal history contiguous per feature
- **Multi-tau support** - generates datasets for tau=1, 4, 8 (384ms, 1.5s, 3s lookback)
- **Global shuffle + split** - 80/10/10 train/val/test

Output: ~5.6M training samples per tau value, saved as compressed NPZ files.

## Documentation

- **[TRAINING.md](TRAINING.md)** - **START HERE** - GPU setup, cloud rental, training workflow
- **[Data Collection Guide](docs/data_collection.md)** - Complete collection workflow
- **[Preprocessing Guide](docs/preprocessing.md)** - Data preparation details
- **[Training Guide](docs/training.md)** - Training pipeline and background execution
- **[Inference Guide](docs/inference.md)** - Real-time prediction and deployment
- **[Workload Types Guide](docs/workload_types.md)** - Understanding workload patterns
- **[Architecture Guide](docs/architecture.md)** - Model architecture and design decisions

## Project Status

1. ✅ Data collection infrastructure
2. ✅ Preprocessing pipeline (completed for tau=1, 4, 8)
3. ✅ Model architecture (Encoder + Decoder + Power Head)
4. ✅ Training pipeline (background execution, checkpoints, early stopping)
5. ✅ Inference system (dynamic tau, live monitoring, CSV replay)
6. ✅ Training automation scripts (start_training.sh, monitor_training.sh, check_gpu.sh)
7. 🔄 **CURRENT: Model training** (ready to train on GPU)
8. ⏳ Evaluation on test set
9. ⏳ Testing on Prometheus workloads
10. ⏳ Latent space visualization

## Training Status

**Preprocessed Data:** ✅ Complete (~4.6GB, 3 tau variants)
- tau=1: 4,692,254 train samples, 586,531 val, 586,533 test
- tau=4: 4,575,614 train samples, 571,951 val, 571,953 test
- tau=8: 4,420,094 train samples, 552,511 val, 552,513 test

**Models to Train:** 3 models (tau=1, tau=4, tau=8)
- Architecture: 267,941 parameters (~268K)
- Config files: config/v2_tau1.yaml, v2_tau4.yaml, v2_tau8.yaml

**Next Action:** Transfer data to GPU machine and start training
- See [TRAINING.md](TRAINING.md) for complete instructions

## References

- v1 archived as `decode-rapl-v1/` with git tag `v1-archive`
- Expert consultation: `decode-rapl-v1/gemini/GEMINI_RESPONSE.md`
- v1 analysis: `decode-rapl-v1/LSTM_ANALYSIS_SUMMARY.md`
