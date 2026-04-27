#!/bin/bash
#
# DECODE-RAPL Live Power Prediction Launcher
#
# Usage:
#   ./scripts/predict_live.sh tau1                    # Live monitoring with tau=1 model
#   ./scripts/predict_live.sh tau8 --save output.csv  # Save predictions to CSV
#   ./scripts/predict_live.sh tau4 --csv data.csv     # Run on existing CSV
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse first argument as tau value
if [ $# -eq 0 ]; then
    echo "Usage: $0 <tau1|tau4|tau8> [additional args...]"
    echo ""
    echo "Examples:"
    echo "  $0 tau1                          # Live monitoring with tau=1"
    echo "  $0 tau8 --save predictions.csv   # Live with CSV output"
    echo "  $0 tau4 --csv data.csv           # Run on existing CSV"
    echo "  $0 tau8 --frequency 1.0          # Predict once per second"
    echo ""
    exit 1
fi

# Extract tau value
TAU_ARG=$1
shift  # Remove first argument

case $TAU_ARG in
    tau1)
        TAU=1
        ;;
    tau4)
        TAU=4
        ;;
    tau8)
        TAU=8
        ;;
    *)
        echo -e "${RED}ERROR: Invalid tau value: $TAU_ARG${NC}"
        echo "Valid options: tau1, tau4, tau8"
        exit 1
        ;;
esac

# Check for v3 model first, fallback to v2
MODEL_PATH="checkpoints/v3_tau${TAU}/best_model.pt"

if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="checkpoints/v2_tau${TAU}/best_model.pt"
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Model not found${NC}"
        echo "  Tried: checkpoints/v3_tau${TAU}/best_model.pt"
        echo "  Tried: checkpoints/v2_tau${TAU}/best_model.pt"
        echo ""
        echo "Please train a model first:"
        echo "  ./scripts/start_training.sh tau${TAU}"
        exit 1
    fi
fi

echo "========================================"
echo "DECODE-RAPL Live Power Prediction"
echo "========================================"
echo "Model: $MODEL_PATH"
echo ""

# Check if --csv or --live is specified
HAS_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--csv" ] || [ "$arg" = "--live" ]; then
        HAS_MODE=true
        break
    fi
done

# If no mode specified, default to --live --scroll
if [ "$HAS_MODE" = false ]; then
    echo -e "${YELLOW}No mode specified, defaulting to --live --scroll${NC}"
    echo ""
    EXTRA_ARGS="--live --scroll"
else
    EXTRA_ARGS=""
fi

# Check if running as root (needed for RAPL access in live mode)
if [ "$EUID" -ne 0 ] && [[ "$EXTRA_ARGS" == *"--live"* || "$@" == *"--live"* ]]; then
    echo -e "${YELLOW}WARNING: Live monitoring requires sudo for RAPL access${NC}"
    echo "Restarting with sudo..."
    echo ""
    exec sudo $(which python3) src/power_predictor.py --model "$MODEL_PATH" $EXTRA_ARGS "$@"
fi

# Run prediction
python3 src/power_predictor.py --model "$MODEL_PATH" $EXTRA_ARGS "$@"
