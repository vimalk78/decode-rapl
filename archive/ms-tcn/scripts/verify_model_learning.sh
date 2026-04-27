#!/bin/bash
# Systematic test to verify if the model actually learned CPU-power relationship
# This script runs controlled CPU loads and checks if model predictions respond

echo "============================================================"
echo "Model Learning Verification Test"
echo "============================================================"
echo ""
echo "This test will run controlled CPU loads and verify if the"
echo "model's predictions respond appropriately to load changes."
echo ""
echo "IMPORTANT: Run this while monitoring predictions in another terminal:"
echo "  sudo \$(which python3) src/power_predictor.py --model models/model_beaker.pth --live --scroll --frequency 1.0"
echo ""
echo "============================================================"
echo ""

# Check if stress-ng is available
if ! command -v stress-ng &> /dev/null; then
    echo "ERROR: stress-ng not found. Install with:"
    echo "  sudo dnf install stress-ng  # Fedora/RHEL"
    echo "  sudo apt install stress-ng  # Debian/Ubuntu"
    exit 1
fi

NUM_CPUS=$(nproc)
DURATION=60  # seconds per test

echo "Configuration:"
echo "  CPUs available: $NUM_CPUS"
echo "  Duration per test: ${DURATION}s"
echo "  Total test time: ~$((DURATION * 6))s (~$((DURATION * 6 / 60)) minutes)"
echo ""

# Function to run a test
run_test() {
    local test_num=$1
    local cpu_load=$2
    local description=$3

    echo "============================================================"
    echo "Test $test_num: $description"
    echo "============================================================"
    echo "CPU Load: ${cpu_load}%"
    echo "Duration: ${DURATION}s"
    echo ""
    echo "Expected behavior:"
    if [ "$cpu_load" -eq 0 ]; then
        echo "  - Actual power: ~25-30W (idle)"
        echo "  - Model should predict: ~25-30W"
        echo "  - If model predicts ~40W or higher → FAILED to learn idle"
    elif [ "$cpu_load" -eq 25 ]; then
        echo "  - Actual power: ~35-45W"
        echo "  - Model should predict: ~35-45W"
    elif [ "$cpu_load" -eq 50 ]; then
        echo "  - Actual power: ~50-60W"
        echo "  - Model should predict: ~50-60W"
    elif [ "$cpu_load" -eq 75 ]; then
        echo "  - Actual power: ~65-75W"
        echo "  - Model should predict: ~65-75W"
    else
        echo "  - Actual power: ~70-80W (max)"
        echo "  - Model should predict: ~70-80W"
    fi
    echo ""

    if [ "$cpu_load" -eq 0 ]; then
        echo "Running idle for ${DURATION}s..."
        echo "(No stress-ng, just waiting)"
        sleep $DURATION
    else
        echo "Starting stress-ng with ${cpu_load}% load..."
        stress-ng --cpu $NUM_CPUS --cpu-load $cpu_load --timeout ${DURATION}s --quiet
    fi

    echo ""
    echo "Test $test_num complete. Check predictions in other terminal."
    echo "Press Enter to continue to next test..."
    read
}

echo "============================================================"
echo "Starting Tests"
echo "============================================================"
echo ""
echo "Make sure you have the prediction monitor running in another"
echo "terminal BEFORE starting these tests!"
echo ""
read -p "Press Enter when prediction monitor is running..."

# Test 1: Idle (0% load)
run_test 1 0 "Idle - Baseline Low Power"

# Test 2: 25% load
run_test 2 25 "Light Load"

# Test 3: 50% load
run_test 3 50 "Medium Load"

# Test 4: 75% load
run_test 4 75 "High Load"

# Test 5: 100% load
run_test 5 100 "Maximum Load"

# Test 6: Back to idle
run_test 6 0 "Idle - Return to Baseline"

echo "============================================================"
echo "All Tests Complete"
echo "============================================================"
echo ""
echo "Analysis:"
echo ""
echo "If the model LEARNED the CPU-power relationship:"
echo "  ✓ Predictions should have increased from Test 1 → Test 5"
echo "  ✓ Predictions should match actual power (±10% error)"
echo "  ✓ Predictions should return to idle in Test 6"
echo ""
echo "If the model FAILED to learn (predicts constants/random):"
echo "  ✗ Predictions stayed constant across all tests"
echo "  ✗ Predictions don't match actual power (>30% error)"
echo "  ✗ Predictions don't respond to load changes"
echo ""
echo "Based on previous observations:"
echo "  - If predictions at idle (~Test 1, 6) are ~40W while actual is ~27W"
echo "    → Model failed to learn idle/low-power states"
echo "  - If predictions at high load (Test 4, 5) match actual ~70W"
echo "    → Model successfully learned high-power states"
echo ""
echo "Conclusion: Model may have PARTIALLY learned (high loads only)"
echo "============================================================"
