#!/bin/bash
#
# Stop data collection gracefully
# Kills run_workloads.sh and all child processes
#

STATE_FILE="/tmp/decode-rapl-collection.state"
DATA_DIR=""
COLLECTION_PID=""

echo "Stopping DECODE-RAPL data collection..."

# Try to read state file first
if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    echo "Found state file: $STATE_FILE"
    echo "Collection PID: $PID"
    echo "Data directory: $DATA_DIR"
    COLLECTION_PID="$PID"
else
    echo "No state file found at $STATE_FILE"
    echo "Will search for running processes..."
    DATA_DIR="${DATA_DIR:-/opt/rapl-training-data}"
fi

echo ""

# Function to check if any processes are running
check_running() {
    pgrep -f "run_workloads.sh" > /dev/null 2>&1 || \
    pgrep -f "my_data_collector" > /dev/null 2>&1 || \
    pgrep -f "stress-ng" > /dev/null 2>&1
}

# Kill run_workloads.sh
if [ -n "$COLLECTION_PID" ] && ps -p "$COLLECTION_PID" > /dev/null 2>&1; then
    echo "Stopping collection (PID $COLLECTION_PID)..."
    kill -TERM "$COLLECTION_PID" 2>/dev/null || true
    sleep 2
elif pgrep -f "run_workloads.sh" > /dev/null 2>&1; then
    echo "Stopping run_workloads.sh..."
    pkill -TERM -f "run_workloads.sh"
    sleep 2
else
    echo "No run_workloads.sh process found."
fi

# Kill data collectors
if pgrep -f "my_data_collector" > /dev/null 2>&1; then
    echo "Stopping data collectors..."
    pkill -TERM -f "my_data_collector"
    sleep 1
else
    echo "No data collector processes found."
fi

# Kill any lingering stress-ng
if pgrep -f "stress-ng" > /dev/null 2>&1; then
    echo "Stopping stress-ng workloads..."
    pkill -TERM -f "stress-ng"
    sleep 1
else
    echo "No stress-ng processes found."
fi

# Force kill if anything still running
if check_running; then
    echo "Some processes still running, force killing..."
    pkill -KILL -f "run_workloads.sh" 2>/dev/null || true
    pkill -KILL -f "my_data_collector" 2>/dev/null || true
    pkill -KILL -f "stress-ng" 2>/dev/null || true
    sleep 1
fi

# Verify everything stopped
if check_running; then
    echo ""
    echo "WARNING: Some processes may still be running:"
    ps aux | grep -E "(run_workloads|my_data_collector|stress-ng)" | grep -v grep || true
    # Don't clean up state file if processes still running
    exit 1
else
    echo ""
    echo "✓ All collection processes stopped successfully."
fi

# Clean up state file
if [ -f "$STATE_FILE" ]; then
    rm -f "$STATE_FILE"
    echo "✓ Cleaned up state file"
fi

# Show progress info
if [ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/.progress.txt" ]; then
    COMPLETED=$(wc -l < "$DATA_DIR/.progress.txt")
    echo "✓ Progress saved: $COMPLETED runs completed"
    echo "✓ Resume with: sudo -E ./start_collection.sh"
elif [ -n "$DATA_DIR" ]; then
    echo "No progress file found at $DATA_DIR/.progress.txt"
else
    echo "Could not determine data directory location"
fi

echo ""
if [ -n "$DATA_DIR" ]; then
    echo "Data directory: $DATA_DIR"
fi
