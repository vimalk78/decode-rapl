# LSTM Power Prediction Model for VMs

Complete PyTorch implementation for predicting per-VM power consumption from CPU usage time series and vCPU count.

## Overview

This LSTM model captures non-linear relationships and temporal dependencies in power consumption, accounting for:
- Non-linear CPU-power relationships (frequency scaling)
- Thermal effects and lag
- Variable VM sizes (2, 4, 10+ vCPUs)

The model is trained on multi-VM bare metal data and can be deployed for real-time inference in VMs.

## Features

- **Multivariate LSTM**: Handles CPU usage time series + vCPU count as static feature
- **Data preprocessing**: Normalization, sliding windows, train/val/test split
- **Training**: MSE loss, Adam optimizer, early stopping
- **Evaluation**: MSE, MAE, MAPE metrics, stratified by vCPU count
- **Inference**: Lightweight predictor for VM deployment (CPU-only)
- **ONNX export**: For easier deployment across platforms
- **Synthetic data generation**: For testing without real hardware

## Requirements

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Quick Start

### 1. Generate Synthetic Data (for testing)

```bash
python power_lstm.py --mode generate --num_samples 10000
```

This creates `multi_vm_training_data.csv` with 10,000 samples from 3 VMs (2, 4, 10 vCPUs).

### 2. Train Model

```bash
python power_lstm.py --mode train --data multi_vm_training_data.csv --epochs 100
```

Outputs:
- `power_lstm_model.pth` - Best model checkpoint
- `preprocessor.pkl` - Scaler parameters for inference
- `training_loss.png` - Training/validation loss curves
- `predictions.png` - Test set predictions vs actuals

Training on GPU (if available):
```bash
python power_lstm.py --mode train --epochs 100 --batch_size 64
```

### 3. Inference (in VM)

```bash
python power_lstm.py --mode inference --model power_lstm_model.pth
```

For real VM deployment, integrate with `psutil`:

```python
import psutil
from power_lstm import PowerPredictor
import pickle

# Load model and preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

predictor = PowerPredictor('power_lstm_model.pth', device='cpu')
predictor.set_power_scaler(preprocessor.power_scaler)

# Get VM vCPUs
vcpus = psutil.cpu_count()

# Collect CPU usage over 60 seconds (1Hz sampling)
cpu_history = []
for _ in range(60):
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_history.append(cpu_usage)

# Predict power
predicted_power = predictor.predict(cpu_history, vcpus)
print(f"Predicted power: {predicted_power:.2f} W")
```

## Architecture

```
Input: (batch_size, 60, 2)  # 60 timesteps, 2 features (cpu_usage, vcpus)
  ↓
LSTM(input_size=2, hidden_size=64, num_layers=1)
  ↓
Dropout(0.2)
  ↓
Linear(64, 1)
  ↓
Output: (batch_size, 1)  # Predicted power
```

## Data Format

CSV with columns:
- `timestamp`: datetime or index
- `vm_id`: VM identifier (string/int)
- `vcpus`: Number of vCPUs (int)
- `cpu_usage`: CPU usage percentage (0-100)
- `power`: Power consumption in Watts (float)

Example:
```csv
timestamp,vm_id,vcpus,cpu_usage,power
2025-01-01 00:00:00,vm-2vcpu,2,45.2,28.3
2025-01-01 00:00:01,vm-2vcpu,2,47.8,29.1
2025-01-01 00:00:02,vm-4vcpu,4,62.3,48.7
```

## Collecting Real Data

Use the companion script `../collect_vm_data.py` to collect multi-VM training data on bare metal:

```bash
# On bare metal host with KVM/libvirt
sudo python ../collect_vm_data.py --duration 1800 --vm_configs "2,4,10" --output training_data.csv
```

This:
1. Creates VMs with different vCPU counts
2. Runs stress-ng workloads
3. Monitors per-VM CPU usage and host RAPL power
4. Apportions power: `vm_power = total_power * (vm_cpu_usage / total_host_usage)`
5. Outputs CSV ready for LSTM training

## Advanced Options

### Custom hyperparameters

```bash
python power_lstm.py --mode train \
    --seq_length 120 \
    --hidden_size 128 \
    --lr 0.0005 \
    --batch_size 64 \
    --epochs 200 \
    --patience 15
```

### Export to ONNX

```bash
python power_lstm.py --mode inference --model power_lstm_model.pth
# Generates power_lstm_model.onnx
```

Load ONNX model:
```python
import onnxruntime as ort

session = ort.InferenceSession('power_lstm_model.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_data})
```

## Model Performance

On synthetic data (10K samples):
- **RMSE**: ~2-4 W
- **MAE**: ~1-3 W
- **MAPE**: ~5-10%

Performance varies by:
- Data quality (RAPL accuracy, VM monitoring granularity)
- VM workload diversity (more varied training data → better generalization)
- Sequence length (longer = more context, but slower inference)

## Troubleshooting

### High errors on test set

1. **Poor data quality**:
   - Ensure RAPL readings are accurate (check `/sys/class/powercap/intel-rapl`)
   - Verify per-VM CPU usage is correctly monitored
   - Collect longer duration data (30+ minutes per VM config)

2. **Insufficient training data**:
   - Increase `--num_samples` for synthetic data
   - Collect more diverse workloads on bare metal

3. **Model underfitting**:
   - Increase `--hidden_size` (64 → 128)
   - Increase `--seq_length` (60 → 120)
   - Add more LSTM layers (modify code)

4. **Model overfitting**:
   - Reduce `--hidden_size`
   - Increase dropout (modify code)
   - Collect more training data

### Poor generalization across vCPU counts

- Ensure training data has sufficient samples for each vCPU configuration
- Check normalization (vCPU should be scaled consistently)
- Try including more VM configurations in training (e.g., 2, 4, 6, 8, 10 vCPUs)

### RAPL not available

If running in a VM or without RAPL support:
- Use synthetic data for testing: `--mode generate`
- For production, train on bare metal with RAPL, deploy inference in VMs

## File Structure

```
lstm/
├── power_lstm.py          # Main program (all-in-one)
├── README.md              # This file
├── multi_vm_training_data.csv  # Generated data
├── power_lstm_model.pth   # Trained model
├── preprocessor.pkl       # Scaler parameters
├── training_loss.png      # Training curves
├── predictions.png        # Test predictions
└── power_lstm_model.onnx  # ONNX export
```

## License

MIT

## Contributing

To improve model performance:
1. Collect diverse multi-VM workloads on bare metal
2. Experiment with architecture (bidirectional LSTM, attention, GRU)
3. Add more features (CPU frequency, memory usage, network I/O)
4. Try ensemble methods or hybrid physics-ML models