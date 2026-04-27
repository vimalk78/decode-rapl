#!/bin/bash
#
# Common Training Launcher for DECODE-RAPL
#
# Usage:
#   ./scripts/train_common.sh <tau> <config_prefix>
#
# Examples:
#   ./scripts/train_common.sh 1 v3            # Uses config/v3_tau1.yaml
#   ./scripts/train_common.sh 1 v3_decoder    # Uses config/v3_decoder_tau1.yaml
#

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <tau> <config_prefix>"
    echo ""
    echo "Arguments:"
    echo "  tau:           1, 4, or 8"
    echo "  config_prefix: v3, v3_decoder, etc."
    echo ""
    echo "Examples:"
    echo "  $0 1 v3"
    echo "  $0 1 v3_decoder"
    exit 1
fi

TAU=$1
CONFIG_PREFIX=$2

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

# Validate tau
if [[ ! "$TAU" =~ ^[1,4,8]$ ]]; then
    echo -e "${RED}ERROR: Invalid tau value: $TAU${NC}"
    echo "Valid values: 1, 4, 8"
    exit 1
fi

CONFIG_FILE="config/${CONFIG_PREFIX}_tau${TAU}.yaml"
LOG_FILE="logs/train_${CONFIG_PREFIX}_tau${TAU}.out"
PID_FILE="logs/train_${CONFIG_PREFIX}_tau${TAU}.pid"

echo "========================================"
echo "DECODE-RAPL Training Launcher"
echo "========================================"
echo -e "${GREEN}Starting training: ${CONFIG_PREFIX} tau=${TAU}${NC}"
echo "  Config: ${CONFIG_FILE}"
echo "  Log: ${LOG_FILE}"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Check if data exists
DATA_DIR="data/processed/tau${TAU}"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}ERROR: Data directory not found: ${DATA_DIR}${NC}"
    echo "Please preprocess data first."
    exit 1
fi

if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo -e "${RED}ERROR: Training data not found: ${DATA_DIR}/train.npz${NC}"
    exit 1
fi

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: Training already running (PID: ${PID})${NC}"
        echo "To stop it: kill $PID"
        exit 1
    else
        # PID file exists but process is dead
        rm "$PID_FILE"
    fi
fi

# Verify Python environment
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}ERROR: PyTorch not found${NC}"
    exit 1
fi

# Check CUDA availability
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}GPU available:${NC}"
    python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}')"
    echo ""
else
    echo -e "${YELLOW}WARNING: CUDA not available. Training will use CPU (very slow!)${NC}"
    echo ""
fi

# Start training in background
nohup python src/train.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
PID=$!

echo -e "${GREEN}Training started!${NC}"
echo "  PID: ${PID}"
echo "  Monitor: tail -f ${LOG_FILE}"
echo ""

# Save PID for monitoring
echo "${PID}" > "$PID_FILE"

echo "========================================"
