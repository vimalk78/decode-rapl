#!/bin/bash
# Background wrapper for collect_and_train.sh
# Runs the full pipeline immune to terminal disconnection
# Uses nohup to survive SIGHUP signals

# ============================================================
# Parse arguments
# ============================================================
TOTAL_DURATION=${1:-7200}  # Default 2 hours
EPOCHS=${2:-200}
BATCH_SIZE=${3:-32}
LEARNING_RATE=${4:-0.001}

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"
PID_FILE="logs/pipeline_${TIMESTAMP}.pid"
STATUS_FILE="logs/pipeline_${TIMESTAMP}.status"

# Create logs directory
mkdir -p logs

# ============================================================
# Check if already running
# ============================================================
if ls logs/pipeline_*.pid 2>/dev/null | grep -q .; then
    echo "============================================================"
    echo "WARNING: Another training pipeline may be running"
    echo "============================================================"
    echo "Existing PID files found:"
    ls -lh logs/pipeline_*.pid 2>/dev/null
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# ============================================================
# Display configuration
# ============================================================
echo "============================================================"
echo "MS-TCN: Background Training Pipeline"
echo "============================================================"
echo "Configuration:"
echo "  Total duration: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60)) minutes)"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""
echo "Output files:"
echo "  Log file: ${LOG_FILE}"
echo "  PID file: ${PID_FILE}"
echo "  Status file: ${STATUS_FILE}"
echo "============================================================"
echo ""

# ============================================================
# Create status file
# ============================================================
cat > ${STATUS_FILE} <<EOF
{
  "status": "starting",
  "start_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "total_duration": ${TOTAL_DURATION},
  "epochs": ${EPOCHS},
  "phase": "initialization",
  "progress": 0
}
EOF

# ============================================================
# Launch background process
# ============================================================
echo "Starting background pipeline..."
echo ""

# Use nohup to run in background, immune to SIGHUP
nohup bash -c "
    # Update status
    echo '{\"status\": \"running\", \"phase\": \"data_collection_1a\", \"start_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\"}' > ${STATUS_FILE}

    # Run the actual pipeline
    cd '$(pwd)' && ./scripts/collect_and_train.sh ${TOTAL_DURATION} ${EPOCHS} ${BATCH_SIZE} ${LEARNING_RATE}
    EXIT_CODE=\$?

    # Update final status
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '{\"status\": \"completed\", \"end_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"exit_code\": '\$EXIT_CODE'}' > ${STATUS_FILE}
    else
        echo '{\"status\": \"failed\", \"end_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"exit_code\": '\$EXIT_CODE'}' > ${STATUS_FILE}
    fi

    # Remove PID file
    rm -f ${PID_FILE}
" >> ${LOG_FILE} 2>&1 &

# Save PID
BG_PID=$!
echo $BG_PID > ${PID_FILE}

echo "✓ Pipeline started in background"
echo "  Process ID: $BG_PID"
echo "  Log file: ${LOG_FILE}"
echo ""

# ============================================================
# Monitoring instructions
# ============================================================
echo "============================================================"
echo "Monitoring Instructions"
echo "============================================================"
echo ""
echo "1. Check if process is still running:"
echo "   ps -p $BG_PID"
echo ""
echo "2. Monitor progress in real-time:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "3. Check current status:"
echo "   cat ${STATUS_FILE}"
echo ""
echo "4. Find the process later:"
echo "   cat ${PID_FILE}   # Shows PID"
echo "   ps -p \$(cat ${PID_FILE})   # Check if running"
echo ""
echo "5. Stop the pipeline (if needed):"
echo "   kill \$(cat ${PID_FILE})"
echo ""
echo "6. Use the monitoring script:"
echo "   ./scripts/check_training_progress.sh ${TIMESTAMP}"
echo ""
echo "============================================================"
echo ""
echo "You can now safely close this terminal."
echo "The pipeline will continue running in the background."
echo "============================================================"
