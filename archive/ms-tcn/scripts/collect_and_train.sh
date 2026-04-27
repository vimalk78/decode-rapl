#!/bin/bash
# Combined script to collect training data and immediately train the model
# Collects BOTH cpu-focused and cpu-random workloads for hybrid training
# Automates the full training pipeline on remote servers
# Must be run from ms-tcn/ directory

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
TOTAL_DURATION=${1:-7200}  # Default 2 hours (7200 seconds)
EPOCHS=${2:-200}  # Default: 200 epochs
BATCH_SIZE=${3:-32}  # Default: 32
LEARNING_RATE=${4:-0.001}  # Default: 0.001

# Split duration 20/80 between cpu-focused and cpu-random
DURATION_FOCUSED=$((TOTAL_DURATION / 5))
DURATION_RANDOM=$((TOTAL_DURATION * 4 / 5))

# Generate timestamp for unique file naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_FILE_FOCUSED="data/training_cpu-focused_${TIMESTAMP}.csv"
DATA_FILE_RANDOM="data/training_cpu-random_${TIMESTAMP}.csv"
DATA_FILE_COMBINED="data/training_hybrid_${TIMESTAMP}.csv"
MODEL_FILE="models/model_hybrid_${TIMESTAMP}.pth"
LOG_FILE="logs/training_hybrid_${TIMESTAMP}.log"

# ============================================================
# Display configuration
# ============================================================
echo "============================================================"
echo "MS-TCN: Hybrid Collect and Train Pipeline"
echo "============================================================"
echo "Configuration:"
echo "  Total collection time: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60)) minutes)"
echo "  Phase 1 - CPU-Focused: ${DURATION_FOCUSED}s ($((DURATION_FOCUSED / 60)) minutes) [20%]"
echo "  Phase 2 - CPU-Random: ${DURATION_RANDOM}s ($((DURATION_RANDOM / 60)) minutes) [80%]"
echo "  Training epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""
echo "Output files:"
echo "  Phase 1 data: ${DATA_FILE_FOCUSED}"
echo "  Phase 2 data: ${DATA_FILE_RANDOM}"
echo "  Combined data: ${DATA_FILE_COMBINED}"
echo "  Model: ${MODEL_FILE}"
echo "  Log: ${LOG_FILE}"
echo "============================================================"
echo ""

# ============================================================
# Validation
# ============================================================

# Check we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/power_data_collector.py" ]; then
    echo "ERROR: Must run from ms-tcn/ directory"
    echo "Usage: cd ms-tcn && sudo ./scripts/collect_and_train.sh [total_duration] [epochs] [batch_size] [lr]"
    echo ""
    echo "Arguments:"
    echo "  total_duration: Total collection duration in seconds (default: 7200 = 2 hours)"
    echo "                  Will be split 20/80 between cpu-focused and cpu-random"
    echo "  epochs: Number of training epochs (default: 200)"
    echo "  batch_size: Training batch size (default: 32)"
    echo "  lr: Learning rate (default: 0.001)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/collect_and_train.sh              # 2 hours (24min focused + 96min random)"
    echo "  ./scripts/collect_and_train.sh 3600         # 1 hour (12min + 48min)"
    echo "  ./scripts/collect_and_train.sh 7200 100     # 2 hours, 100 epochs"
    exit 1
fi

# Check if required tools are available
if ! command -v stress-ng &> /dev/null; then
    echo "ERROR: stress-ng not found. Install with:"
    echo "  sudo apt-get install stress-ng"
    exit 1
fi

# Check Python dependencies
if ! python3 -c "import torch, pandas, numpy, sklearn" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Create output directories
mkdir -p data models logs results/plots

# Check permissions for RAPL
if [ ! -r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj ]; then
    echo "WARNING: Cannot read RAPL. Setting permissions..."
    sudo chmod -R a+r /sys/class/powercap/intel-rapl
fi

# ============================================================
# Phase 1A: CPU-Focused Data Collection
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 1A: CPU-Focused Data Collection"
echo "============================================================"
echo "Duration: ${DURATION_FOCUSED}s ($((DURATION_FOCUSED / 60)) minutes)"
echo ""

echo "Starting power data collector..."
PYTHON_PATH=$(which python3)
echo "  Python path: ${PYTHON_PATH}"
echo "  Output file: ${DATA_FILE_FOCUSED}"

# Start collector with stderr logged
sudo ${PYTHON_PATH} src/power_data_collector.py --duration ${DURATION_FOCUSED} --output ${DATA_FILE_FOCUSED} 2>> ${LOG_FILE}.collector_1a.err &
SUDO_PID=$!
echo "  Sudo PID: ${SUDO_PID}"

# Wait a moment and find the actual Python process
sleep 2
COLLECTOR_PID=$(pgrep -P ${SUDO_PID} || echo ${SUDO_PID})
echo "  Collector PID: ${COLLECTOR_PID}"

echo "Waiting 5 seconds for collector to initialize..."
sleep 5

echo "Starting load generator with cpu-focused workload sequence..."
python3 src/load_generator.py --sequence cpu-focused --duration ${DURATION_FOCUSED}
GENERATOR_EXIT=$?

echo ""
echo "Waiting for data collector to finish..."
# Poll for process instead of wait (which doesn't work reliably with sudo)
while sudo ps -p ${COLLECTOR_PID} > /dev/null 2>&1; do
    sleep 5
    echo "  Collector still running (PID ${COLLECTOR_PID})..."
done
echo "  Collector process finished"

# Check collector exit status via file existence (since we can't get real exit code)
COLLECTOR_EXIT=0
if [ ! -f "${DATA_FILE_FOCUSED}" ]; then
    COLLECTOR_EXIT=1
fi

# Check if data collection succeeded
if [ ${COLLECTOR_EXIT} -ne 0 ] || [ ${GENERATOR_EXIT} -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "ERROR: Phase 1A data collection failed"
    echo "============================================================"
    echo "Collector exit code: ${COLLECTOR_EXIT}"
    echo "Generator exit code: ${GENERATOR_EXIT}"
    exit 1
fi

# Verify data file exists and has content
if [ ! -f "${DATA_FILE_FOCUSED}" ]; then
    echo "ERROR: Data file not created: ${DATA_FILE_FOCUSED}"
    exit 1
fi

SAMPLES_FOCUSED=$(wc -l < ${DATA_FILE_FOCUSED})
SAMPLES_FOCUSED=$((SAMPLES_FOCUSED - 1))  # Subtract header

echo ""
echo "============================================================"
echo "Phase 1A Complete"
echo "============================================================"
echo "✓ CPU-Focused data collected"
echo "  Output file: ${DATA_FILE_FOCUSED}"
echo "  Samples: ${SAMPLES_FOCUSED}"
echo "  Sample rate: $((SAMPLES_FOCUSED / DURATION_FOCUSED)) Hz"
echo "============================================================"

# ============================================================
# Phase 1B: CPU-Random Data Collection
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 1B: CPU-Random Data Collection"
echo "============================================================"
echo "Duration: ${DURATION_RANDOM}s ($((DURATION_RANDOM / 60)) minutes)"
echo "Starting in 10 seconds..."
sleep 10
echo ""

echo "Starting power data collector..."
PYTHON_PATH=$(which python3)
echo "  Python path: ${PYTHON_PATH}"
echo "  Output file: ${DATA_FILE_RANDOM}"

# Start collector with stderr logged
sudo ${PYTHON_PATH} src/power_data_collector.py --duration ${DURATION_RANDOM} --output ${DATA_FILE_RANDOM} 2>> ${LOG_FILE}.collector_1b.err &
SUDO_PID=$!
echo "  Sudo PID: ${SUDO_PID}"

# Wait a moment and find the actual Python process
sleep 2
COLLECTOR_PID=$(pgrep -P ${SUDO_PID} || echo ${SUDO_PID})
echo "  Collector PID: ${COLLECTOR_PID}"

echo "Waiting 5 seconds for collector to initialize..."
sleep 5

echo "Starting load generator with cpu-random workload sequence..."
python3 src/load_generator.py --sequence cpu-random --duration ${DURATION_RANDOM}
GENERATOR_EXIT=$?

echo ""
echo "Waiting for data collector to finish..."
# Poll for process instead of wait (which doesn't work reliably with sudo)
while sudo ps -p ${COLLECTOR_PID} > /dev/null 2>&1; do
    sleep 5
    echo "  Collector still running (PID ${COLLECTOR_PID})..."
done
echo "  Collector process finished"

# Check collector exit status via file existence (since we can't get real exit code)
COLLECTOR_EXIT=0
if [ ! -f "${DATA_FILE_RANDOM}" ]; then
    COLLECTOR_EXIT=1
fi

# Check if data collection succeeded
if [ ${COLLECTOR_EXIT} -ne 0 ] || [ ${GENERATOR_EXIT} -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "ERROR: Phase 1B data collection failed"
    echo "============================================================"
    echo "Collector exit code: ${COLLECTOR_EXIT}"
    echo "Generator exit code: ${GENERATOR_EXIT}"
    echo ""
    echo "Phase 1A data is still available: ${DATA_FILE_FOCUSED}"
    exit 1
fi

# Verify data file exists and has content
if [ ! -f "${DATA_FILE_RANDOM}" ]; then
    echo "ERROR: Data file not created: ${DATA_FILE_RANDOM}"
    exit 1
fi

SAMPLES_RANDOM=$(wc -l < ${DATA_FILE_RANDOM})
SAMPLES_RANDOM=$((SAMPLES_RANDOM - 1))  # Subtract header

echo ""
echo "============================================================"
echo "Phase 1B Complete"
echo "============================================================"
echo "✓ CPU-Random data collected"
echo "  Output file: ${DATA_FILE_RANDOM}"
echo "  Samples: ${SAMPLES_RANDOM}"
echo "  Sample rate: $((SAMPLES_RANDOM / DURATION_RANDOM)) Hz"
echo "============================================================"

# ============================================================
# Combine Datasets
# ============================================================
echo ""
echo "============================================================"
echo "Combining Datasets"
echo "============================================================"

# Copy header from first file
head -n 1 ${DATA_FILE_FOCUSED} > ${DATA_FILE_COMBINED}

# Append data from both files (skip headers)
tail -n +2 ${DATA_FILE_FOCUSED} >> ${DATA_FILE_COMBINED}
tail -n +2 ${DATA_FILE_RANDOM} >> ${DATA_FILE_COMBINED}

SAMPLES_COMBINED=$(wc -l < ${DATA_FILE_COMBINED})
SAMPLES_COMBINED=$((SAMPLES_COMBINED - 1))  # Subtract header

if [ ${SAMPLES_COMBINED} -lt 100 ]; then
    echo "ERROR: Insufficient combined samples: ${SAMPLES_COMBINED}"
    echo "Need at least 100 samples for training"
    exit 1
fi

echo "✓ Datasets combined successfully"
echo "  Combined file: ${DATA_FILE_COMBINED}"
echo "  Total samples: ${SAMPLES_COMBINED}"
echo "  CPU-Focused: ${SAMPLES_FOCUSED} samples"
echo "  CPU-Random: ${SAMPLES_RANDOM} samples"
echo "============================================================"

# ============================================================
# Phase 2: Model Training
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 2: Model Training"
echo "============================================================"
echo ""
echo "Starting training in 5 seconds..."
echo "(Press Ctrl+C within 5 seconds to skip training)"
sleep 5

echo "Training model on combined dataset..."
echo "Output will be logged to: ${LOG_FILE}"
echo ""

# Run training and tee output to both console and log file
python3 src/train_model.py \
    --data ${DATA_FILE_COMBINED} \
    --output ${MODEL_FILE} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    2>&1 | tee ${LOG_FILE}

TRAINING_EXIT=${PIPESTATUS[0]}

# Check if training succeeded
if [ ${TRAINING_EXIT} -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "ERROR: Training failed"
    echo "============================================================"
    echo "Check log file: ${LOG_FILE}"
    exit 1
fi

# ============================================================
# Generate Summary Report
# ============================================================
echo ""
echo "============================================================"
echo "Pipeline Complete - Summary Report"
echo "============================================================"
echo ""
echo "Data Collection:"
echo "  Total duration: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60)) minutes)"
echo "  Phase 1A (CPU-Focused): ${SAMPLES_FOCUSED} samples"
echo "  Phase 1B (CPU-Random): ${SAMPLES_RANDOM} samples"
echo "  Combined samples: ${SAMPLES_COMBINED}"
echo "  Combined file: ${DATA_FILE_COMBINED}"
echo ""
echo "Model Training:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Model: ${MODEL_FILE}"
echo ""

# Extract best metrics from log file
if [ -f "${LOG_FILE}" ]; then
    echo "Best Training Results:"
    # Look for "Saved best model!" lines and extract the metrics before it
    grep -B 2 "Saved best model!" ${LOG_FILE} | tail -n 20 | head -n 15
fi

echo ""
echo "Files created:"
echo "  📊 Phase 1A data: ${DATA_FILE_FOCUSED}"
echo "  📊 Phase 1B data: ${DATA_FILE_RANDOM}"
echo "  📊 Combined data: ${DATA_FILE_COMBINED}"
echo "  🧠 Model file: ${MODEL_FILE}"
echo "  📝 Training log: ${LOG_FILE}"
echo ""
echo "Next steps:"
echo "  1. Test predictions:"
echo "     python3 src/predict_power.py --model ${MODEL_FILE}"
echo ""
echo "  2. Visualize training data:"
echo "     python3 scripts/plot_power_data.py --with-power --output results/plots/data_${TIMESTAMP}.png ${DATA_FILE_COMBINED}"
echo ""
echo "  3. Compare workload patterns:"
echo "     python3 scripts/plot_power_data.py --with-power --output results/plots/focused_${TIMESTAMP}.png ${DATA_FILE_FOCUSED}"
echo "     python3 scripts/plot_power_data.py --with-power --output results/plots/random_${TIMESTAMP}.png ${DATA_FILE_RANDOM}"
echo ""
echo "============================================================"
