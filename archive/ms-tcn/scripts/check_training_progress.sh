#!/bin/bash
# Monitor the progress of a background training pipeline
# Usage: ./check_training_progress.sh [timestamp]
#        ./check_training_progress.sh              # Auto-detect latest

TIMESTAMP=$1

# ============================================================
# Find pipeline files
# ============================================================
if [ -z "$TIMESTAMP" ]; then
    # Find most recent pipeline
    LATEST_PID=$(ls -t logs/pipeline_*.pid 2>/dev/null | head -n 1)

    if [ -z "$LATEST_PID" ]; then
        echo "No active pipelines found."
        echo ""
        echo "Looking for completed runs..."
        LATEST_LOG=$(ls -t logs/pipeline_*.log 2>/dev/null | head -n 1)
        if [ -n "$LATEST_LOG" ]; then
            echo "Most recent log: $LATEST_LOG"
            echo "View with: tail -f $LATEST_LOG"
        fi
        exit 1
    fi

    # Extract timestamp from filename
    TIMESTAMP=$(basename $LATEST_PID .pid | sed 's/pipeline_//')
    echo "Auto-detected most recent pipeline: $TIMESTAMP"
    echo ""
fi

PID_FILE="logs/pipeline_${TIMESTAMP}.pid"
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"
STATUS_FILE="logs/pipeline_${TIMESTAMP}.status"

# ============================================================
# Check if files exist
# ============================================================
if [ ! -f "$PID_FILE" ] && [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Pipeline not found for timestamp: $TIMESTAMP"
    echo ""
    echo "Available pipelines:"
    ls -lh logs/pipeline_*.pid 2>/dev/null || echo "  None"
    exit 1
fi

# ============================================================
# Display status
# ============================================================
echo "============================================================"
echo "Training Pipeline Status"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check if process is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: ✓ RUNNING"
        echo "Process ID: $PID"

        # Show CPU and memory usage
        if command -v ps &> /dev/null; then
            echo ""
            echo "Resource Usage:"
            ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | awk '{printf "  PID: %s\n  CPU: %s%%\n  Memory: %s%%\n  Running time: %s\n", $1, $3, $4, $5}'
        fi
    else
        echo "Status: ✗ STOPPED (PID $PID not running)"
    fi
else
    echo "Status: ✓ COMPLETED or ✗ FAILED"
fi

echo ""

# Show status file if available
if [ -f "$STATUS_FILE" ]; then
    echo "Pipeline Status:"
    if command -v jq &> /dev/null; then
        cat $STATUS_FILE | jq .
    else
        cat $STATUS_FILE
    fi
    echo ""
fi

# ============================================================
# Log file analysis
# ============================================================
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LOG_LINES=$(wc -l < "$LOG_FILE")

    echo "Log File: $LOG_FILE"
    echo "  Size: $LOG_SIZE"
    echo "  Lines: $LOG_LINES"
    echo ""

    # Try to detect current phase
    echo "Current Phase:"
    if grep -q "PHASE 1A: CPU-Focused" "$LOG_FILE" && ! grep -q "Phase 1A Complete" "$LOG_FILE"; then
        echo "  📊 Phase 1A: Collecting CPU-Focused data..."
        SAMPLES=$(grep -o "Progress:.*" "$LOG_FILE" | tail -n 1 || echo "  Starting...")
        echo "  $SAMPLES"
    elif grep -q "Phase 1A Complete" "$LOG_FILE" && ! grep -q "Phase 1B Complete" "$LOG_FILE"; then
        echo "  📊 Phase 1B: Collecting CPU-Random data..."
        SAMPLES=$(grep -o "Progress:.*" "$LOG_FILE" | tail -n 1 || echo "  Starting...")
        echo "  $SAMPLES"
    elif grep -q "Phase 1B Complete" "$LOG_FILE" && ! grep -q "PHASE 2: Model Training" "$LOG_FILE"; then
        echo "  🔗 Combining datasets..."
    elif grep -q "PHASE 2: Model Training" "$LOG_FILE" && ! grep -q "Pipeline Complete" "$LOG_FILE"; then
        echo "  🧠 Training model..."
        # Try to find current epoch
        EPOCH_INFO=$(grep "Epoch" "$LOG_FILE" | tail -n 1 | grep -o "Epoch [0-9]*/[0-9]*" || echo "")
        if [ -n "$EPOCH_INFO" ]; then
            echo "  $EPOCH_INFO"
        fi
        # Show best metrics so far
        BEST_R2=$(grep "R²=" "$LOG_FILE" | tail -n 5 | head -n 3 || echo "")
        if [ -n "$BEST_R2" ]; then
            echo ""
            echo "  Recent metrics:"
            echo "$BEST_R2" | sed 's/^/    /'
        fi
    elif grep -q "Pipeline Complete" "$LOG_FILE"; then
        echo "  ✓ Pipeline completed successfully!"
        echo ""
        echo "Summary:"
        grep -A 20 "Pipeline Complete - Summary Report" "$LOG_FILE" | head -n 25
    else
        echo "  Initializing..."
    fi
    echo ""

    # Show last few lines
    echo "Last 10 lines of log:"
    echo "----------------------------------------"
    tail -n 10 "$LOG_FILE"
    echo "----------------------------------------"
fi

echo ""
echo "============================================================"
echo "Monitoring Commands"
echo "============================================================"
echo "  Live log: tail -f $LOG_FILE"
echo "  Full log: less $LOG_FILE"
if [ -f "$PID_FILE" ]; then
    echo "  Kill process: kill \$(cat $PID_FILE)"
fi
echo "============================================================"
