#!/bin/bash
# Diagnostic script to check CPU frequency behavior
# Tests whether CPU frequency is fixed or varies with load

echo "============================================================"
echo "CPU Frequency Diagnostic Tool"
echo "============================================================"
echo ""

# 1. CPU Model Info
echo "[1] CPU Model Information:"
lscpu | grep "Model name"
echo ""

# 2. Check frequency scaling driver
echo "[2] CPU Frequency Driver Status:"
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    echo "✓ cpufreq interface available"

    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver ]; then
        DRIVER=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver)
        echo "  Driver: $DRIVER"
    fi

    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        echo "  Governor: $GOVERNOR"
    fi

    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq ]; then
        MIN_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
        MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
        echo "  Min frequency: $((MIN_FREQ / 1000)) MHz"
        echo "  Max frequency: $((MAX_FREQ / 1000)) MHz"
    fi
else
    echo "✗ cpufreq interface NOT available"
    echo "  Possible reasons:"
    echo "  - Hardware P-states (HWP) enabled"
    echo "  - Frequency scaling disabled in BIOS"
    echo "  - Kernel driver not loaded"
fi
echo ""

# 3. Current CPU frequencies
echo "[3] Current CPU Frequencies (all cores):"
if command -v lscpu &> /dev/null; then
    FREQ=$(lscpu | grep "CPU MHz" | awk '{print $3}')
    if [ -n "$FREQ" ]; then
        echo "  Average: ${FREQ} MHz"
    fi
fi

# Read from /proc/cpuinfo
echo "  Per-core frequencies:"
grep "cpu MHz" /proc/cpuinfo | head -n 10
TOTAL_CORES=$(grep "cpu MHz" /proc/cpuinfo | wc -l)
if [ $TOTAL_CORES -gt 10 ]; then
    echo "  ... (showing 10 of $TOTAL_CORES cores)"
fi
echo ""

# 4. Check dmesg for frequency scaling info
echo "[4] Kernel Messages (frequency scaling):"
if dmesg | grep -i "intel_pstate\|acpi-cpufreq\|HWP" | tail -5 > /dev/null 2>&1; then
    dmesg | grep -i "intel_pstate\|acpi-cpufreq\|HWP" | tail -5
else
    echo "  No frequency scaling messages in dmesg"
fi
echo ""

# 5. Dynamic frequency test
echo "[5] Dynamic Frequency Test:"
echo "  Measuring idle frequency..."
sleep 2
IDLE_FREQ=$(grep "cpu MHz" /proc/cpuinfo | head -n 1 | awk '{print $4}')
echo "  Idle: ${IDLE_FREQ} MHz (CPU 0)"

echo "  Generating CPU load for 5 seconds..."
# Generate CPU load using dd and sha256sum
(for i in {1..4}; do dd if=/dev/zero bs=1M count=500 2>/dev/null | sha256sum > /dev/null & done)
sleep 2

LOAD_FREQ=$(grep "cpu MHz" /proc/cpuinfo | head -n 1 | awk '{print $4}')
echo "  Under load: ${LOAD_FREQ} MHz (CPU 0)"

# Kill background jobs
pkill -P $$ dd 2>/dev/null
wait 2>/dev/null

# Compare
FREQ_DIFF=$(echo "$LOAD_FREQ - $IDLE_FREQ" | bc 2>/dev/null || echo "0")
echo ""
if command -v bc &> /dev/null; then
    FREQ_INCREASE=$(echo "$LOAD_FREQ - $IDLE_FREQ" | bc)
    if (( $(echo "$FREQ_INCREASE > 100" | bc -l) )); then
        echo "  ✓ CPU frequency VARIES with load (+${FREQ_INCREASE} MHz)"
        echo "  → Frequency scaling is WORKING"
    elif (( $(echo "$FREQ_INCREASE > 10" | bc -l) )); then
        echo "  ~ CPU frequency changes slightly (+${FREQ_INCREASE} MHz)"
        echo "  → Limited frequency scaling"
    else
        echo "  ✗ CPU frequency is FIXED (±${FREQ_INCREASE} MHz)"
        echo "  → No frequency scaling (locked at base frequency)"
    fi
else
    echo "  (Install 'bc' for frequency comparison)"
fi
echo ""

# 6. Turbo Boost status
echo "[6] Turbo Boost Status:"
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
    if [ "$NO_TURBO" = "0" ]; then
        echo "  ✓ Turbo Boost ENABLED"
    else
        echo "  ✗ Turbo Boost DISABLED"
    fi
elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    BOOST=$(cat /sys/devices/system/cpu/cpufreq/boost)
    if [ "$BOOST" = "1" ]; then
        echo "  ✓ CPU Boost ENABLED"
    else
        echo "  ✗ CPU Boost DISABLED"
    fi
else
    echo "  ? Cannot determine turbo boost status"
fi
echo ""

echo "============================================================"
echo "Diagnostic Complete"
echo "============================================================"
