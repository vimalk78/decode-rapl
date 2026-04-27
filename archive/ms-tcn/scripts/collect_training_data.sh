#!/bin/bash
# Helper script to collect high-quality training data
# Runs load generator and power collector simultaneously
# Must be run from ms-tcn/ directory: ./scripts/collect_training_data.sh

DURATION=${1:-1800}  # Default 30 minutes (1800 seconds)
OUTPUT_FILE=${2:-data/training_data_v2.csv}
SEQUENCE=${3:-cpu-focused}  # Default: cpu-focused. Options: cpu-focused, cpu-random, comprehensive

echo "============================================================"
echo "Training Data Collection Script"
echo "============================================================"
echo "Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"
echo "Output: ${OUTPUT_FILE}"
echo "Workload sequence: ${SEQUENCE}"
echo "============================================================"
echo ""

# Check we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/power_data_collector.py" ]; then
    echo "ERROR: Must run from ms-tcn/ directory"
    echo "Usage: cd ms-tcn && sudo ./scripts/collect_training_data.sh [duration] [output] [sequence]"
    echo "  duration: Duration in seconds (default: 1800)"
    echo "  output: Output CSV file (default: data/training_data_v2.csv)"
    echo "  sequence: Workload sequence - cpu-focused, cpu-random, or comprehensive (default: cpu-focused)"
    exit 1
fi

# Check if required tools are available
if ! command -v stress-ng &> /dev/null; then
    echo "ERROR: stress-ng not found. Install with:"
    echo "  sudo apt-get install stress-ng"
    exit 1
fi

echo "Starting power data collector..."

# Determine python path - prefer venv if available
if [ -f "venv/bin/python3" ]; then
    PYTHON_PATH="$(pwd)/venv/bin/python3"
elif [ -f "../venv/bin/python3" ]; then
    PYTHON_PATH="$(cd .. && pwd)/venv/bin/python3"
else
    PYTHON_PATH="/usr/bin/python3"
fi

echo "  Using Python: ${PYTHON_PATH}"

sudo ${PYTHON_PATH} $(pwd)/src/power_data_collector.py --duration ${DURATION} --output ${OUTPUT_FILE} &
COLLECTOR_PID=$!

echo "Waiting 5 seconds for collector to initialize..."
sleep 5

echo "Starting load generator with ${SEQUENCE} workload sequence..."
python3 src/load_generator.py --sequence ${SEQUENCE} --duration ${DURATION}
GENERATOR_EXIT=$?

echo ""
echo "Waiting for data collector to finish..."
wait ${COLLECTOR_PID}
COLLECTOR_EXIT=$?

echo ""
echo "============================================================"
echo "Data Collection Complete"
echo "============================================================"

if [ ${COLLECTOR_EXIT} -eq 0 ] && [ ${GENERATOR_EXIT} -eq 0 ]; then
    # Count samples
    SAMPLES=$(wc -l < ${OUTPUT_FILE})
    SAMPLES=$((SAMPLES - 1))  # Subtract header

    echo "✓ Success!"
    echo "  Output file: ${OUTPUT_FILE}"
    echo "  Total samples: ${SAMPLES}"
    echo "  Sample rate: $((SAMPLES / DURATION)) Hz"
    echo ""
    echo "Next step: Train the model with:"
    echo "  python3 src/train_model.py --data ${OUTPUT_FILE} --output models/new_model.pth"
else
    echo "✗ Error occurred during collection"
    exit 1
fi
