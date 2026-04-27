#!/bin/bash
#
# DECODE-RAPL Training Monitor
#
# Usage:
#   ./scripts/monitor_training.sh               # Monitor v3 models
#   ./scripts/monitor_training.sh v3_decoder    # Monitor v3_decoder models
#   ./scripts/monitor_training.sh --watch       # Watch mode for v3
#   ./scripts/monitor_training.sh v3_decoder -w # Watch mode for v3_decoder
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to get latest metrics from training log
get_training_status() {
    local tau=$1
    local config_prefix=$2
    local log_file="results/${config_prefix}_tau${tau}/training.log"
    local pid_file="logs/train_${config_prefix}_tau${tau}.pid"

    echo -e "${BLUE}=== ${config_prefix} TAU=${tau} ===${NC}"

    # Check if training is running
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "Status: ${GREEN}RUNNING${NC} (PID: ${pid})"
        else
            echo -e "Status: ${RED}STOPPED${NC} (PID file exists but process is dead)"
            rm "$pid_file"
            return
        fi
    else
        if [ -f "$log_file" ]; then
            echo -e "Status: ${YELLOW}COMPLETED or STOPPED${NC}"
        else
            echo -e "Status: ${RED}NOT STARTED${NC}"
            return
        fi
    fi

    # Extract latest metrics from log
    if [ -f "$log_file" ]; then
        # Get latest epoch summary
        local latest_epoch=$(grep -E "Epoch [0-9]+/[0-9]+ Summary:" "$log_file" | tail -1 | sed 's/.*Epoch \([0-9]*\)\/\([0-9]*\) Summary:.*/\1 \2/')

        if [ -n "$latest_epoch" ]; then
            read current_epoch total_epochs <<< "$latest_epoch"
            echo "Progress: Epoch ${current_epoch}/${total_epochs} ($(( current_epoch * 100 / total_epochs ))%)"

            # Get metrics from latest epoch (v3 format with timestamps)
            # Format: [2025-10-22 15:30:48]   Train Power Loss: 18.5858
            local train_loss=$(grep -A 10 "Epoch ${current_epoch}/${total_epochs} Summary:" "$log_file" | grep "Train Power Loss:" | tail -1 | awk '{print $6}')
            local val_loss=$(grep -A 10 "Epoch ${current_epoch}/${total_epochs} Summary:" "$log_file" | grep "Val Power Loss:" | tail -1 | awk '{print $6}')
            local val_mae=$(grep -A 10 "Epoch ${current_epoch}/${total_epochs} Summary:" "$log_file" | grep "Val MAE:" | tail -1 | awk '{print $5}')
            local val_r2=$(grep -A 10 "Epoch ${current_epoch}/${total_epochs} Summary:" "$log_file" | grep "Val R²:" | tail -1 | awk '{print $5}')
            local lr=$(grep -A 10 "Epoch ${current_epoch}/${total_epochs} Summary:" "$log_file" | grep "LR:" | tail -1 | awk '{print $4}')

            echo "Train Loss: ${train_loss}"
            echo "Val Loss: ${val_loss}"
            echo "Val MAE: ${val_mae}W"
            echo "Val R²: ${val_r2}"
            echo "Learning Rate: ${lr}"

            # Check if best model
            if grep -q "New best model" "$log_file"; then
                local best_line=$(grep "New best model" "$log_file" | tail -1)
                local best_val_loss=$(echo "$best_line" | sed 's/.*Val power loss: \([0-9.]*\)/\1/')
                # Extract epoch by looking for the Summary line before the "New best model" message
                local best_epoch=$(grep -B 5 "New best model" "$log_file" | tail -6 | grep "Epoch.*Summary:" | tail -1 | sed 's/.*Epoch \([0-9]*\)\/[0-9]* Summary:.*/\1/')
                if [ -n "$best_epoch" ]; then
                    echo -e "Best Model: ${GREEN}Epoch ${best_epoch}${NC} (Val Loss: ${best_val_loss})"
                else
                    echo -e "Best Val Loss: ${GREEN}${best_val_loss}${NC}"
                fi
            fi
        else
            echo "No epoch data yet (training just started)"
        fi

        # Show last few lines from log
        echo ""
        echo "Recent log entries:"
        tail -5 "$log_file" | sed 's/^/  /'
    fi

    echo ""
}

# Function to show quick summary
show_summary() {
    local config_prefix=$1

    echo "========================================"
    echo "DECODE-RAPL ${config_prefix} Training Status"
    echo "========================================"
    echo ""

    for tau in 1 4 8; do
        get_training_status "$tau" "$config_prefix"
    done

    echo "========================================"
    echo "Commands:"
    echo "  Watch this script: watch -n 10 ./scripts/monitor_training.sh ${config_prefix}"
    echo "  Tail log: tail -f results/${config_prefix}_tau1/training.log"
    echo "  View plots: ls results/${config_prefix}_tau1/plots/"
    echo "========================================"
}

# Function to continuously monitor (watch mode)
watch_mode() {
    local config_prefix=$1
    while true; do
        clear
        show_summary "$config_prefix"
        echo ""
        echo "Refreshing in 10 seconds... (Ctrl+C to exit)"
        sleep 10
    done
}

# Main
CONFIG_PREFIX="v3"  # Default
WATCH_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --watch|-w)
            WATCH_MODE=true
            ;;
        v3|v3_decoder|v4)
            CONFIG_PREFIX=$arg
            ;;
        *)
            echo "Unknown argument: $arg"
            echo ""
            echo "Usage: $0 [v3|v3_decoder|v4] [--watch|-w]"
            echo ""
            echo "Examples:"
            echo "  $0                    # Monitor v3 models"
            echo "  $0 v3_decoder         # Monitor v3_decoder models"
            echo "  $0 v4                 # Monitor v4 models"
            echo "  $0 --watch            # Watch mode for v3"
            echo "  $0 v4 -w              # Watch mode for v4"
            exit 1
            ;;
    esac
done

if [ "$WATCH_MODE" = true ]; then
    watch_mode "$CONFIG_PREFIX"
else
    show_summary "$CONFIG_PREFIX"
fi
