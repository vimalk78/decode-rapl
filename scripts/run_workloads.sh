#!/bin/bash

echo "Starting Combinatorial Workload Generator..."

# --- 1. SOURCE THE AUTO-GENERATED CONFIG ---
# This line imports the arrays (CPU_WORKERS, etc.)
# Source from DATA_DIR for reproducibility (fallback to current dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/opt/rapl-training-data}"

if [ -f "$DATA_DIR/workload_config.sh" ]; then
    echo "Loading workload config from: $DATA_DIR/workload_config.sh"
    source "$DATA_DIR/workload_config.sh"
elif [ -f "$SCRIPT_DIR/workload_config.sh" ]; then
    echo "WARNING: Using workload config from scripts dir (should be in DATA_DIR)"
    source "$SCRIPT_DIR/workload_config.sh"
else
    echo "ERROR: workload_config.sh not found in $DATA_DIR or $SCRIPT_DIR"
    exit 1
fi

# --- 1.5 CHECK STRESS-NG VERSION ---
# This script requires stress-ng >= 0.19.00 for --syscall-method option
# Older versions (e.g. 0.17.x in Fedora 39) don't support it
REQUIRED_VERSION="0.19.00"
CURRENT_VERSION=$(stress-ng --version 2>&1 | grep -oP 'version \K[0-9.]+' | head -1)

if [ -z "$CURRENT_VERSION" ]; then
    echo "ERROR: Could not detect stress-ng version"
    echo "Please install stress-ng: https://github.com/ColinIanKing/stress-ng"
    exit 1
fi

# Simple version comparison (assumes format X.Y.Z)
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$CURRENT_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: stress-ng version $CURRENT_VERSION is too old"
    echo "Required: >= $REQUIRED_VERSION"
    echo "Current:  $CURRENT_VERSION"
    echo ""
    echo "Update stress-ng on this machine:"
    echo "  https://github.com/ColinIanKing/stress-ng/releases/latest"
    exit 1
fi

echo "✓ Using stress-ng version $CURRENT_VERSION"

# --- 2. DEFINE RUN PARAMETERS ---
DURATION="45s"
COOLDOWN="15s"

mkdir -p "$DATA_DIR"

# Progress tracking for resume capability
PROGRESS_FILE="$DATA_DIR/.progress.txt"
LOG_FILE="$DATA_DIR/run.log"
touch $PROGRESS_FILE

# --- 3. THE COMBINATORIAL LOOP ---
# We use 'eval' to count the total runs from the sourced arrays
# This is a bit of bash magic to get the product of all array lengths.
eval "TOTAL_RUNS=\$(( \${#CPU_WORKERS[@]} * \${#SYS_WORKERS[@]} * \${#IO_WORKERS[@]} * \${#PIPE_WORKERS[@]} * \${#VM_WORKERS[@]} * \${#CACHE_WORKERS[@]} ))"
CURRENT_RUN=0

echo "Starting a total of $TOTAL_RUNS workload combinations..."

for cpu in "${CPU_WORKERS[@]}"; do
  for sys in "${SYS_WORKERS[@]}"; do
    for io in "${IO_WORKERS[@]}"; do
      for pipe in "${PIPE_WORKERS[@]}"; do
        for vm in "${VM_WORKERS[@]}"; do
          for cache in "${CACHE_WORKERS[@]}"; do

            CURRENT_RUN=$((CURRENT_RUN + 1))

            RUN_NAME="run_${CURRENT_RUN}_of_${TOTAL_RUNS}-cpu${cpu}-sys${sys}-io${io}-pipe${pipe}-vm${vm}-cache${cache}"

            # --- Check if already completed (resume capability) ---
            if grep -q "^$RUN_NAME$" $PROGRESS_FILE 2>/dev/null; then
              echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Already completed: $RUN_NAME" | tee -a $LOG_FILE
              continue
            fi

            # --- Handle IDLE baseline (all zeros) ---
            if [ $cpu -eq 0 ] && [ $sys -eq 0 ] && [ $io -eq 0 ] && [ $pipe -eq 0 ] && [ $vm -eq 0 ] && [ $cache -eq 0 ]; then
              echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTING: $RUN_NAME (IDLE BASELINE) ($CURRENT_RUN / $TOTAL_RUNS)" | tee -a $LOG_FILE

              # Start collector without any workload
              ../collector/my_data_collector --outfile "$DATA_DIR/$RUN_NAME.csv" &
              COLLECTOR_PID=$!
              sleep 1

              # No stress-ng, just wait for collection duration
              echo "Collecting idle baseline (no workload)"
              sleep 45

              # Stop collector
              sleep 2
              kill $COLLECTOR_PID
              wait $COLLECTOR_PID 2>/dev/null

              echo "$RUN_NAME" >> $PROGRESS_FILE
              echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: $RUN_NAME (IDLE BASELINE). Cooling down for $COOLDOWN..." | tee -a $LOG_FILE
              sleep $COOLDOWN
              continue
            fi

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTING: $RUN_NAME ($CURRENT_RUN / $TOTAL_RUNS)" | tee -a $LOG_FILE

            # --- Start your data collector in the background ---
            ../collector/my_data_collector --outfile "$DATA_DIR/$RUN_NAME.csv" &
            COLLECTOR_PID=$!
            sleep 1 # Give collector a second to start

            # --- Dynamically Build the stress-ng command ---
            CMD="stress-ng --metrics-brief --timeout $DURATION"

            if [ $cpu -gt 0 ]; then
              CMD="$CMD --cpu $cpu --cpu-method float"
            fi
            if [ $sys -gt 0 ]; then
              # Use fast10 for sustained high system% load
              CMD="$CMD --syscall $sys --syscall-method fast10"
            fi
            if [ $io -gt 0 ]; then
              CMD="$CMD --io $io"
            fi
            if [ $pipe -gt 0 ]; then
              CMD="$CMD --pipe $pipe"
            fi
            if [ $vm -gt 0 ]; then
              CMD="$CMD --vm $vm --vm-bytes 1G --vm-populate" # Added --vm-populate
            fi
            if [ $cache -gt 0 ]; then
              CMD="$CMD --cache $cache --cache-level 3"
            fi

            # --- Execute the dynamically built command ---
            echo "Executing: $CMD"
            eval $CMD

            # --- Stop data collection ---
            sleep 2 # Capture ramp-down
            # Send TERM signal to gracefully stop the collector
            kill $COLLECTOR_PID
            wait $COLLECTOR_PID 2>/dev/null

            # Mark as completed for resume tracking
            echo "$RUN_NAME" >> $PROGRESS_FILE
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: $RUN_NAME. Cooling down for $COOLDOWN..." | tee -a $LOG_FILE
            sleep $COOLDOWN

          done
        done
      done
    done
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Workload generation complete. Data saved to $DATA_DIR" | tee -a $LOG_FILE

# Clean up state file on successful completion
STATE_FILE="/tmp/decode-rapl-collection.state"
if [ -f "$STATE_FILE" ]; then
    rm -f "$STATE_FILE"
    echo "Cleaned up state file: $STATE_FILE"
fi
