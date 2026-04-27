#!/bin/bash
#
# DECODE-RAPL Inference Script
# Run power prediction inference (bare-metal or VM mode)
#

set -e  # Exit on error

echo "======================================================================"
echo "DECODE-RAPL Inference"
echo "======================================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Default parameters
VM_MODE="false"
VM_VCPUS=4
HOST_CORES=16

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vm)
            VM_MODE="true"
            shift
            ;;
        --vm-vcpus)
            VM_VCPUS="$2"
            shift 2
            ;;
        --host-cores)
            HOST_CORES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --vm              Enable VM inference mode"
            echo "  --vm-vcpus N      Number of VM vCPUs (default: 4)"
            echo "  --host-cores N    Number of host cores (default: 16)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "Error: Trained model not found at checkpoints/best_model.pth"
    echo "Please train a model first using: ./scripts/train_model.sh"
    exit 1
fi

# Print configuration
echo -e "\nInference Configuration:"
echo "  VM Mode: $VM_MODE"
if [ "$VM_MODE" = "true" ]; then
    echo "  VM vCPUs: $VM_VCPUS"
    echo "  Host Cores: $HOST_CORES"
fi

# Run inference test
echo -e "\n======================================================================"
echo "Running Inference Test"
echo "======================================================================"

python3 -c "
import sys
sys.path.append('.')

from src.inference import RAPLPredictor, run_realtime_inference
import numpy as np

print('Loading model...')
predictor = RAPLPredictor(
    checkpoint_path='checkpoints/best_model.pth',
    vm_mode=$VM_MODE,
    vm_vcpus=$VM_VCPUS if $VM_MODE else None,
    host_cores=$HOST_CORES if $VM_MODE else None
)

print('\\nTesting batch prediction...')
test_usage = np.random.uniform(20, 80, 200)
power_pred = predictor.predict_from_sequence(test_usage)
print(f'Predicted power: {power_pred:.2f}W')

print('\\nTesting real-time inference (10 seconds)...')
def get_synthetic_cpu_usage():
    return np.random.uniform(30, 70)

predictions, timestamps = run_realtime_inference(
    predictor,
    get_synthetic_cpu_usage,
    duration_seconds=10,
    sampling_rate_ms=100
)

print(f'\\nInference completed!')
print(f'  Total predictions: {len(predictions)}')
print(f'  Average power: {np.mean(predictions):.2f}W')
print(f'  Power range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]W')
"

echo -e "\n======================================================================"
echo "Inference Test Completed"
echo "======================================================================"
echo ""
echo "To use inference in your own code:"
echo ""
echo "from src.inference import RAPLPredictor"
echo ""
echo "# Initialize predictor"
if [ "$VM_MODE" = "true" ]; then
    echo "predictor = RAPLPredictor("
    echo "    checkpoint_path='checkpoints/best_model.pth',"
    echo "    vm_mode=True,"
    echo "    vm_vcpus=$VM_VCPUS,"
    echo "    host_cores=$HOST_CORES"
    echo ")"
else
    echo "predictor = RAPLPredictor("
    echo "    checkpoint_path='checkpoints/best_model.pth'"
    echo ")"
fi
echo ""
echo "# Real-time prediction"
echo "predictor.update_usage(cpu_usage)  # Update with current CPU usage %"
echo "power = predictor.predict()         # Get power prediction"
echo ""
