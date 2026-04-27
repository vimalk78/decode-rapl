#!/bin/bash
#
# DECODE-RAPL Model Evaluation Script
# Evaluate already-trained model without retraining
#

set -e  # Exit on error

echo "======================================================================"
echo "DECODE-RAPL Model Evaluation"
echo "======================================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Check if checkpoint exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "Error: No trained model found at checkpoints/best_model.pth"
    echo "Please train a model first using: ./scripts/train_model.sh"
    exit 1
fi

echo -e "\nFound trained model checkpoint"
echo "Running evaluation only (no training)..."

# Run evaluation
python3 << 'EOF'
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path

from src.model import create_model
from src.preprocessing import load_and_split_data, create_dataloaders
from src.utils import (
    load_config, calculate_metrics,
    plot_predictions, plot_error_distribution
)

print("=" * 70)
print("Loading checkpoint and data...")
print("=" * 70)

# Load config
config = load_config()

# Load checkpoint
checkpoint_path = 'checkpoints/best_model.pth'
checkpoint = torch.load(checkpoint_path, weights_only=False)

print(f"\nCheckpoint Info:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Training history: {len(checkpoint['history']['train_loss'])} epochs")
print(f"  Best validation MAPE: {min(checkpoint['history']['val_mape']):.2f}%")

# Load data
csv_path = config['data']['train_csv']
train_df, val_df, test_df = load_and_split_data(csv_path, config)

# Create dataloaders
dataloaders = create_dataloaders(train_df, val_df, test_df, config)

# Get scalers and machine mapping from checkpoint
power_scaler = checkpoint['scalers']['power']
machine_id_map = checkpoint['machine_id_map']

# Create model and load weights
num_machines = len(machine_id_map)
model = create_model(config, num_machines)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"\nModel loaded on {device}")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Evaluate on test set
print("\n" + "=" * 70)
print("Evaluating on Test Set")
print("=" * 70)

all_predictions = []
all_targets = []
all_machine_ids = []

with torch.no_grad():
    for batch in dataloaders['test']:
        x = batch['embedding'].to(device)
        power_target = batch['power'].to(device)

        outputs = model(x)

        all_predictions.append(outputs['power_pred'].cpu().numpy())
        all_targets.append(power_target.cpu().numpy())
        all_machine_ids.extend(batch['machine_id'])

# Concatenate
predictions = np.concatenate(all_predictions, axis=0).flatten()
targets = np.concatenate(all_targets, axis=0).flatten()

# Denormalize
if power_scaler is not None:
    predictions = power_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets = power_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

# Calculate metrics
metrics = calculate_metrics(targets, predictions)

print(f"\nTest Set Results:")
print(f"  MSE:  {metrics['mse']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  MAPE: {metrics['mape']:.2f}%")

# Generate plots
print("\nGenerating evaluation plots...")
plots_dir = Path(config['data']['plots_dir'])
plots_dir.mkdir(parents=True, exist_ok=True)

# Encode machine IDs for plotting
unique_machines = sorted(set(all_machine_ids))
machine_colors = np.array([unique_machines.index(mid) for mid in all_machine_ids])

plot_predictions(
    targets, predictions,
    machine_ids=machine_colors,
    save_path=plots_dir / 'test_predictions.png'
)

plot_error_distribution(
    targets, predictions,
    save_path=plots_dir / 'test_errors.png'
)

print(f"Saved plots to {plots_dir}/")

# Final summary
print("\n" + "=" * 70)
print("Evaluation Complete!")
print("=" * 70)

print(f"\nFinal Test MAPE: {metrics['mape']:.2f}%")
if metrics['mape'] < 5.0:
    print("✓ Target accuracy (<5% MAPE) achieved!")
else:
    print(f"✗ Target accuracy not achieved. Current: {metrics['mape']:.2f}%, Target: <5%")
    print("\nSuggestions for improvement:")
    print("  - Collect real RAPL data (synthetic data is simplified)")
    print("  - Increase model capacity (encoder layers, latent dim)")
    print("  - Adjust loss weights (reduce adversarial/reconstruction)")
    print("  - Increase sequence length (window_size)")

EOF

echo ""
echo "======================================================================"
echo "Evaluation completed!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - results/plots/test_predictions.png"
echo "  - results/plots/test_errors.png"
