#!/bin/bash
#
# DECODE-RAPL v3 Training Launcher
#
# Usage:
#   ./scripts/start_training.sh tau1        # Train single model
#   ./scripts/start_training.sh all         # Train all 3 models sequentially
#   ./scripts/start_training.sh tau1 tau4   # Train specific models
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Create directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to start training for a specific tau
start_tau_training() {
    local tau=$1

    # Use common training script with v3 config prefix
    ./scripts/train_common.sh "$tau" "v3"

    # Add spacing between launches
    echo ""
}

# Function to check if training is already running
check_running() {
    local tau=$1
    local pid_file="logs/train_v3_tau${tau}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}WARNING: Training for v3 tau=${tau} is already running (PID: ${pid})${NC}"
            return 0
        else
            # PID file exists but process is dead
            rm "$pid_file"
        fi
    fi
    return 1
}

# Main logic
if [ $# -eq 0 ]; then
    echo "Usage: $0 <tau1|tau4|tau8|all> [tau1|tau4|tau8...]"
    echo ""
    echo "Examples:"
    echo "  $0 tau1              # Train tau=1 model"
    echo "  $0 all               # Train all 3 models sequentially"
    echo "  $0 tau1 tau4         # Train tau=1 and tau=4"
    echo ""
    exit 1
fi

# Parse arguments
TAUS=()
for arg in "$@"; do
    case $arg in
        tau1)
            TAUS+=(1)
            ;;
        tau4)
            TAUS+=(4)
            ;;
        tau8)
            TAUS+=(8)
            ;;
        all)
            TAUS=(1 4 8)
            ;;
        *)
            echo -e "${RED}ERROR: Unknown argument: $arg${NC}"
            echo "Valid options: tau1, tau4, tau8, all"
            exit 1
            ;;
    esac
done

# Remove duplicates
TAUS=($(echo "${TAUS[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

echo "========================================"
echo "DECODE-RAPL v3 Training Launcher"
echo "========================================"
echo "Models to train: tau=${TAUS[@]}"
echo ""

# Verify Python environment
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}ERROR: PyTorch not found. Please install requirements:${NC}"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check CUDA availability
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}GPU available:${NC}"
    python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
    echo ""
else
    echo -e "${YELLOW}WARNING: CUDA not available. Training will use CPU (very slow!)${NC}"
    echo ""
fi

# Start training for each tau
for tau in "${TAUS[@]}"; do
    if check_running "$tau"; then
        continue
    fi

    start_tau_training "$tau"

    # If training multiple models, wait a bit between launches
    if [ ${#TAUS[@]} -gt 1 ]; then
        sleep 5
    fi
done

echo "========================================"
echo -e "${GREEN}All training jobs launched!${NC}"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  ./scripts/monitor_training.sh"
echo ""
echo "Or tail individual logs:"
for tau in "${TAUS[@]}"; do
    echo "  tail -f logs/train_v3_tau${tau}.out"
done
echo ""
