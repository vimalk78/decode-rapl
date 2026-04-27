#!/bin/bash
# Train all 3 multi-feature models (tau1, tau4, tau8)

set -e

echo "==============================================="
echo "DECODE-RAPL Multi-Feature Training"
echo "Training models with 3 features:"
echo "  - user_percent"
echo "  - system_percent"
echo "  - context_switches"
echo "==============================================="
echo ""

# Set PYTHONPATH to include current directory for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

# Train tau8 (fastest, most efficient)
echo "=========================================="
echo "Training tau8_multifeature..."
echo "=========================================="
python src/train.py --config config_tau8_multifeature.yaml 2>&1 | tee logs/train_tau8_multifeature.log

# Train tau4 (medium granularity)
echo ""
echo "=========================================="
echo "Training tau4_multifeature..."
echo "=========================================="
python src/train.py --config config_tau4_multifeature.yaml 2>&1 | tee logs/train_tau4_multifeature.log

# Train tau1 (finest granularity)
echo ""
echo "=========================================="
echo "Training tau1_multifeature..."
echo "=========================================="
python src/train.py --config config_tau1_multifeature.yaml 2>&1 | tee logs/train_tau1_multifeature.log

echo ""
echo "=========================================="
echo "All multi-feature models trained!"
echo "=========================================="
echo ""
echo "Results:"
echo "  tau8: results/tau8_multifeature/"
echo "  tau4: results/tau4_multifeature/"
echo "  tau1: results/tau1_multifeature/"
