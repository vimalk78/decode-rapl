#!/bin/bash
# Background wrapper for train_model.py
# Runs model training immune to terminal disconnection using nohup

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
DATA_FILE=${1:-data/training_data.csv}
OUTPUT_MODEL=${2:-models/model_bg.pth}
EPOCHS=${3:-200}
BATCH_SIZE=${4:-32}
LEARNING_RATE=${5:-0.0001}
HIDDEN_DIM=${6:-128}
PATIENCE=${7:-30}
OUTPUT_DIR=${8:-results}
SPLIT_MODE=${9:-random}

# Generate timestamp for unique file naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
PID_FILE="logs/training_${TIMESTAMP}.pid"
STATUS_FILE="logs/training_${TIMESTAMP}.status"

# Create necessary directories
mkdir -p logs models ${OUTPUT_DIR}

# ============================================================
# Display configuration
# ============================================================
echo "============================================================"
echo "MS-TCN: Background Model Training (with AttentionPooling)"
echo "============================================================"
echo "Configuration:"
echo "  Training data: ${DATA_FILE}"
echo "  Output model: ${OUTPUT_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Hidden dimension: ${HIDDEN_DIM}"
echo "  Patience: ${PATIENCE}"
echo "  Split mode: ${SPLIT_MODE}"
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
if [ ! -d "src" ] || [ ! -f "src/train_model.py" ]; then
    echo "ERROR: Must run from ms-tcn/ directory"
    echo "Usage: cd ms-tcn && ./scripts/train_model_bg.sh [data] [output] [epochs] [batch_size] [lr] [hidden_dim] [patience] [output_dir] [split_mode]"
    echo ""
    echo "Arguments (all optional, use defaults if not provided):"
    echo "  data:        Training data CSV file (default: data/training_data.csv)"
    echo "  output:      Output model file (default: models/model_bg.pth)"
    echo "  epochs:      Number of training epochs (default: 200)"
    echo "  batch_size:  Training batch size (default: 32)"
    echo "  lr:          Learning rate (default: 0.0001)"
    echo "  hidden_dim:  Hidden dimension size (default: 128)"
    echo "  patience:    Early stopping patience (default: 30)"
    echo "  output_dir:  Output directory for plots (default: results)"
    echo "  split_mode:  Data split mode: random or temporal (default: random)"
    echo ""
    echo "Examples:"
    echo "  # Use all defaults:"
    echo "  ./scripts/train_model_bg.sh"
    echo ""
    echo "  # Custom data and output:"
    echo "  ./scripts/train_model_bg.sh data/my_data.csv models/my_model.pth"
    echo ""
    echo "  # Custom training params:"
    echo "  ./scripts/train_model_bg.sh data/my_data.csv models/my_model.pth 200 64 0.0001"
    echo ""
    echo "  # Full customization:"
    echo "  ./scripts/train_model_bg.sh data/my_data.csv models/my_model.pth 200 32 0.0001 128 30 results random"
    exit 1
fi

# Check if data file exists
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data file not found: ${DATA_FILE}"
    exit 1
fi

# Check Python dependencies
if ! python3 -c "import torch, pandas, numpy, sklearn" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# ============================================================
# Estimate training time
# ============================================================
SAMPLES=$(wc -l < ${DATA_FILE})
SAMPLES=$((SAMPLES - 1))  # Subtract header
# Rough estimate: ~0.5s per epoch for 20K samples
EST_TIME_MIN=$((EPOCHS / 2))

echo "Training data: ${SAMPLES} samples"
echo "Estimated training time: ~${EST_TIME_MIN} minutes (rough estimate)"
echo ""

# ============================================================
# Create status file
# ============================================================
cat > ${STATUS_FILE} <<EOF
{
  "status": "starting",
  "start_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "data_file": "${DATA_FILE}",
  "output_model": "${OUTPUT_MODEL}",
  "output_dir": "${OUTPUT_DIR}",
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "learning_rate": ${LEARNING_RATE},
  "hidden_dim": ${HIDDEN_DIM},
  "patience": ${PATIENCE},
  "split_mode": "${SPLIT_MODE}",
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
    cd '$(pwd)' && python3 src/train_model.py \
        --data ${DATA_FILE} \
        --output ${OUTPUT_MODEL} \
        --output-dir ${OUTPUT_DIR} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LEARNING_RATE} \
        --hidden-dim ${HIDDEN_DIM} \
        --patience ${PATIENCE} \
        --split-mode ${SPLIT_MODE}

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
echo "4. Check if model file was created:"
echo "   ls -lh ${OUTPUT_MODEL}"
echo "   ls -lh models/best_model.pth  # Best model checkpoint"
echo ""
echo "5. Find the process later:"
echo "   cat ${PID_FILE}   # Shows PID"
echo "   ps -p \$(cat ${PID_FILE})   # Check if running"
echo ""
echo "6. Stop training (if needed):"
echo "   kill \$(cat ${PID_FILE})"
echo ""
echo "7. Watch for completion:"
echo "   watch -n 10 'grep -E \"Epoch|Best|Complete\" ${LOG_FILE} | tail -20'"
echo ""
echo "Expected output:"
echo "  - Training will run for ${EPOCHS} epochs (patience: ${PATIENCE})"
echo "  - Model uses AttentionPooling (learnable temporal importance)"
echo "  - Hidden dimension: ${HIDDEN_DIM}"
echo "  - Best model saved to: models/best_model.pth"
echo "  - Final model saved to: ${OUTPUT_MODEL}"
echo "  - Training plots saved to: ${OUTPUT_DIR}/"
echo "  - Estimated completion: ~${EST_TIME_MIN} minutes"
echo ""
echo "============================================================"
echo ""
echo "You can now safely close this terminal."
echo "The training will continue running in the background."
echo "============================================================"
