#!/bin/bash
#
# DECODE-RAPL Training Script
# End-to-end model training with synthetic or real data
#

set -e  # Exit on error

echo "======================================================================"
echo "DECODE-RAPL Model Training"
echo "======================================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check dependencies
echo -e "\nChecking dependencies..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" || {
    echo "Error: PyTorch not installed"
    echo "Install with: pip install torch"
    exit 1
}

python3 -c "import numpy, pandas, sklearn, yaml" || {
    echo "Error: Missing dependencies"
    echo "Install with: pip install numpy pandas scikit-learn pyyaml matplotlib"
    exit 1
}

# Check if data exists
DATA_FILE="data/rapl_train.csv"

if [ ! -f "$DATA_FILE" ]; then
    echo -e "\nWarning: Training data not found at $DATA_FILE"
    echo "Generating synthetic data for testing..."

    mkdir -p data

    python3 -c "
from src.utils import generate_synthetic_data, set_seed

set_seed(42)
generate_synthetic_data(
    num_machines=3,
    duration_hours=0.5,  # 30 minutes of data
    sampling_rate_ms=1,
    output_path='$DATA_FILE'
)
"
    echo "Synthetic data generated successfully"
fi

# Create necessary directories
echo -e "\nCreating directories..."
mkdir -p checkpoints
mkdir -p results/plots
mkdir -p logs

# Run training
echo -e "\n======================================================================"
echo "Starting Training"
echo "======================================================================"

python3 -m src.train 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "\n======================================================================"
    echo "Training Completed Successfully!"
    echo "======================================================================"
    echo -e "\nModel saved to: checkpoints/best_model.pth"
    echo "Plots saved to: results/plots/"
    echo "Logs saved to: logs/"

    # Show training artifacts
    echo -e "\nGenerated files:"
    ls -lh checkpoints/best_model.pth
    ls -lh results/plots/

else
    echo -e "\nError: Training failed"
    exit 1
fi
