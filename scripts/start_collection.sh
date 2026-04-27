#!/bin/bash
#
# Background wrapper for data collection
# Starts run_workloads.sh in the background with nohup
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/opt/rapl-training-data}"
STATE_FILE="/tmp/decode-rapl-collection.state"

# Check if collection is already running
if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "ERROR: Data collection is already running!"
        echo "PID: $PID"
        echo "Data dir: $DATA_DIR"
        echo "To stop: sudo -E ./stop_collection.sh"
        exit 1
    else
        echo "WARNING: Stale state file found (process not running). Cleaning up..."
        rm -f "$STATE_FILE"
    fi
fi

mkdir -p "$DATA_DIR"

# Copy workload config to data dir on first run (for reproducibility)
if [ ! -f "$DATA_DIR/workload_config.sh" ]; then
    if [ -f "$SCRIPT_DIR/workload_config.sh" ]; then
        cp "$SCRIPT_DIR/workload_config.sh" "$DATA_DIR/workload_config.sh"
        echo "Copied workload config to $DATA_DIR"
    else
        echo "ERROR: workload_config.sh not found in $SCRIPT_DIR"
        echo "Run ./generate_config.sh first"
        exit 1
    fi
else
    echo "Using existing workload config from $DATA_DIR"
fi

cd "$SCRIPT_DIR"
nohup ./run_workloads.sh > "$DATA_DIR/nohup.log" 2>&1 &

PID=$!

# Write state file
cat > "$STATE_FILE" << EOF
PID=$PID
DATA_DIR=$DATA_DIR
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "Data collection started in background."
echo "PID: $PID"
echo "Data dir: $DATA_DIR"
echo "State file: $STATE_FILE"
echo ""
echo "Monitor: tail -f $DATA_DIR/run.log"
echo "Stop: sudo -E ./stop_collection.sh"
