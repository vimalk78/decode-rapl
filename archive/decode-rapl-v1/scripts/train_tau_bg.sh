#!/bin/bash
# Background wrapper for DECODE-RAPL training
# Runs model training immune to terminal disconnection using nohup

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
CONFIG_FILE=${1:-config_tau1.yaml}

# Generate timestamp for unique file naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
PID_FILE="logs/training_${TIMESTAMP}.pid"
STATUS_FILE="logs/training_${TIMESTAMP}.status"

# Create necessary directories
mkdir -p logs

# ============================================================
# Display configuration
# ============================================================
echo "============================================================"
echo "DECODE-RAPL: Background Model Training"
echo "============================================================"
echo "Configuration:"
echo "  Config file: ${CONFIG_FILE}"
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
if [ ! -d "src" ] || [ ! -f "src/train.py" ]; then
    echo "ERROR: Must run from decode-rapl/ directory"
    echo "Usage: cd decode-rapl && ./scripts/train_tau_bg.sh [config_file]"
    echo ""
    echo "Arguments:"
    echo "  config_file: Configuration YAML file (default: config_tau1.yaml)"
    echo ""
    echo "Examples:"
    echo "  # Train with tau=1 (baseline):"
    echo "  ./scripts/train_tau_bg.sh config_tau1.yaml"
    echo ""
    echo "  # Train with tau=4 (balanced):"
    echo "  ./scripts/train_tau_bg.sh config_tau4.yaml"
    echo ""
    echo "  # Train with tau=8 (theoretical optimum):"
    echo "  ./scripts/train_tau_bg.sh config_tau8.yaml"
    exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    echo "Available configs:"
    ls -1 config_tau*.yaml 2>/dev/null || echo "  No config_tau*.yaml files found"
    exit 1
fi

# Parse config to extract key parameters
TAU=$(grep "tau:" ${CONFIG_FILE} | head -1 | awk '{print $2}' | sed 's/#.*//')
EPOCHS=$(grep "epochs:" ${CONFIG_FILE} | awk '{print $2}')
TRAIN_CSV=$(grep "train_csv:" ${CONFIG_FILE} | awk '{print $2}' | tr -d '"')

echo "Parsed configuration:"
echo "  τ (tau): ${TAU} samples"
echo "  Epochs: ${EPOCHS}"
echo "  Training data: ${TRAIN_CSV}"
echo ""

# Check if data file exists
if [ ! -f "${TRAIN_CSV}" ]; then
    echo "ERROR: Training data file not found: ${TRAIN_CSV}"
    exit 1
fi

# Check Python dependencies
if ! python3 -c "import torch, pandas, numpy, sklearn, yaml" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check for GPU
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "GPU detected: ${GPU_INFO}"
else
    echo "WARNING: No GPU detected. Training will run on CPU (slow!)"
fi
echo ""

# ============================================================
# Estimate training time
# ============================================================
SAMPLES=$(wc -l < ${TRAIN_CSV})
SAMPLES=$((SAMPLES - 1))  # Subtract header
# Rough estimate: ~1-2 min per epoch depending on tau and GPU
EST_TIME_MIN=$((EPOCHS * 2))

echo "Training data: ${SAMPLES} samples"
echo "Estimated training time: ~${EST_TIME_MIN} minutes (rough estimate, GPU-dependent)"
echo ""

# ============================================================
# Create status file
# ============================================================
cat > ${STATUS_FILE} <<EOF
{
  "status": "starting",
  "start_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "config_file": "${CONFIG_FILE}",
  "tau": ${TAU},
  "epochs": ${EPOCHS},
  "train_csv": "${TRAIN_CSV}",
  "samples": ${SAMPLES}
}
EOF

# ============================================================
# Launch background process
# ============================================================
echo "Starting background training..."
echo ""

# Use nohup to run in background, immune to SIGHUP
# Redirect stdin from /dev/null to prevent suspension
nohup bash -c "
    # Update status
    echo '{\"status\": \"running\", \"start_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\"}' > ${STATUS_FILE}

    # Run the training
    cd '$(pwd)' && PYTHONPATH='$(pwd)' python3 src/train.py --config ${CONFIG_FILE}

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

echo "✓ Training started in background"
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
echo "2. Monitor training progress in real-time:"
echo "   tail -f ${LOG_FILE}"
echo "   # Look for 'Epoch X/Y' lines to track progress"
echo ""
echo "3. Check current status:"
echo "   cat ${STATUS_FILE}"
echo ""
echo "4. Check if model checkpoint was created:"
echo "   # Extract output_dir from config"
echo "   OUTPUT_DIR=\$(grep 'checkpoint_dir:' ${CONFIG_FILE} | awk '{print \$2}' | tr -d '\"')"
echo "   ls -lh \${OUTPUT_DIR}/best_model.pth"
echo ""
echo "5. Find the process later:"
echo "   cat ${PID_FILE}   # Shows PID"
echo "   ps -p \$(cat ${PID_FILE})   # Check if running"
echo ""
echo "6. Stop training (if needed):"
echo "   kill \$(cat ${PID_FILE})"
echo ""
echo "7. Watch for completion:"
echo "   watch -n 10 'grep -E \"Epoch|MAPE|completed\" ${LOG_FILE} | tail -20'"
echo ""
echo "Expected output:"
echo "  - Training will run for ${EPOCHS} epochs"
echo "  - Config: ${CONFIG_FILE} (τ=${TAU})"
echo "  - Best model saved to checkpoints/tau${TAU}/best_model.pth"
echo "  - Training plots saved to results/tau${TAU}/plots/"
echo "  - Estimated completion: ~${EST_TIME_MIN} minutes"
echo ""
echo "============================================================"
echo ""
echo "You can now safely close this terminal."
echo "The training will continue running in the background."
echo "============================================================"
