#!/bin/bash
#
# DECODE-RAPL v3-decoder Training Launcher
#
# Usage:
#   ./scripts/start_training_decoder.sh tau1
#

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tau1|tau4|tau8>"
    echo ""
    echo "Examples:"
    echo "  $0 tau1    # Train v3-decoder with tau=1"
    echo "  $0 tau4    # Train v3-decoder with tau=4"
    echo "  $0 tau8    # Train v3-decoder with tau=8"
    exit 1
fi

# Parse tau argument
case $1 in
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
        echo "ERROR: Unknown argument: $1"
        echo "Valid options: tau1, tau4, tau8"
        exit 1
        ;;
esac

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use common training script with v3_decoder config prefix
"$SCRIPT_DIR/train_common.sh" "$TAU" "v3_decoder"
