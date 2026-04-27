#!/bin/bash
# Background wrapper for collect_training_data.sh
# Runs data collection immune to terminal disconnection using nohup

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
DURATION=${1:-7200}  # Default 2 hours
OUTPUT_FILE=${2:-data/training_data_bg.csv}
SEQUENCE=${3:-cpu-random}

# Generate timestamp for unique file naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/collection_${TIMESTAMP}.log"
PID_FILE="logs/collection_${TIMESTAMP}.pid"
STATUS_FILE="logs/collection_${TIMESTAMP}.status"

# Create logs directory
mkdir -p logs data

# ============================================================
# Display configuration
# ============================================================
echo "============================================================"
echo "MS-TCN: Background Data Collection"
echo "============================================================"
echo "Configuration:"
echo "  Duration: ${DURATION}s ($((DURATION / 60)) minutes)"
echo "  Output file: ${OUTPUT_FILE}"
echo "  Workload sequence: ${SEQUENCE}"
echo ""
echo "Output files:"
echo "  Log file: ${LOG_FILE}"
echo "  PID file: ${PID_FILE}"
echo "  Status file: ${STATUS_FILE}"
echo "============================================================"
echo ""

# ============================================================
# Validation
# ============================================================

# Check we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/power_data_collector.py" ]; then
    echo "ERROR: Must run from ms-tcn/ directory"
    echo "Usage: cd ms-tcn && ./scripts/collect_training_data_bg.sh [duration] [output] [sequence]"
    echo ""
    echo "Arguments:"
    echo "  duration: Duration in seconds (default: 7200 = 2 hours)"
    echo "  output: Output CSV file (default: data/training_data_bg.csv)"
    echo "  sequence: Workload sequence - cpu-focused, cpu-random, or comprehensive (default: cpu-random)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/collect_training_data_bg.sh 7200 data/my_data.csv cpu-random"
    echo "  ./scripts/collect_training_data_bg.sh 3600  # 1 hour with defaults"
    exit 1
fi

# Check if required tools are available
if ! command -v stress-ng &> /dev/null; then
    echo "ERROR: stress-ng not found. Install with:"
    echo "  sudo apt-get install stress-ng"
    exit 1
fi

# ============================================================
# Create status file
# ============================================================
cat > ${STATUS_FILE} <<EOF
{
  "status": "starting",
  "start_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "duration": ${DURATION},
  "output": "${OUTPUT_FILE}",
  "sequence": "${SEQUENCE}"
}
EOF

# ============================================================
# Launch background process
# ============================================================
echo "Starting background data collection..."
echo ""

# Use nohup to run in background, immune to SIGHUP
# Redirect stdin from /dev/null to prevent suspension
nohup bash -c "
    # Update status
    echo '{\"status\": \"running\", \"start_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\"}' > ${STATUS_FILE}

    # Run the data collection script
    cd '$(pwd)' && ./scripts/collect_training_data.sh ${DURATION} ${OUTPUT_FILE} ${SEQUENCE}
    EXIT_CODE=\$?

    # Update final status
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '{\"status\": \"completed\", \"end_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"exit_code\": '\$EXIT_CODE'}' > ${STATUS_FILE}
    else
        echo '{\"status\": \"failed\", \"end_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"exit_code\": '\$EXIT_CODE'}' > ${STATUS_FILE}
    fi

    # Remove PID file
    rm -f ${PID_FILE}
" < /dev/null >> ${LOG_FILE} 2>&1 &

# Save PID
BG_PID=$!
echo $BG_PID > ${PID_FILE}

echo "✓ Data collection started in background"
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
echo "4. Check data file size (should be growing):"
echo "   ls -lh ${OUTPUT_FILE}"
echo "   watch -n 10 ls -lh ${OUTPUT_FILE}"
echo ""
echo "5. Find the process later:"
echo "   cat ${PID_FILE}   # Shows PID"
echo "   ps -p \$(cat ${PID_FILE})   # Check if running"
echo ""
echo "6. Stop the collection (if needed):"
echo "   kill \$(cat ${PID_FILE})"
echo ""
echo "Expected data file size:"
echo "  ~10 MB per hour at 10 Hz sampling rate"
echo "  For ${DURATION}s ($((DURATION / 60)) min): ~$((DURATION / 360)) MB"
echo ""
echo "============================================================"
echo ""
echo "You can now safely close this terminal."
echo "The data collection will continue running in the background."
echo "============================================================"
