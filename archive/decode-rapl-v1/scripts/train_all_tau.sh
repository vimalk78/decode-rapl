#!/bin/bash
# Sequential training script for all tau configurations
# Runs tau1 → tau4 → tau8 one after another

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================
CONFIGS=("config_tau1.yaml" "config_tau4.yaml" "config_tau8.yaml")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/train_all_tau_${TIMESTAMP}.log"

# Create logs directory
mkdir -p logs

# ============================================================
# Display banner
# ============================================================
echo "============================================================" | tee -a ${MASTER_LOG}
echo "DECODE-RAPL: Sequential Training of All Tau Configurations" | tee -a ${MASTER_LOG}
echo "============================================================" | tee -a ${MASTER_LOG}
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${MASTER_LOG}
echo "Master log: ${MASTER_LOG}" | tee -a ${MASTER_LOG}
echo "" | tee -a ${MASTER_LOG}
echo "Training sequence:" | tee -a ${MASTER_LOG}
for config in "${CONFIGS[@]}"; do
    TAU=$(grep "tau:" ${config} | head -1 | awk '{print $2}' | sed 's/#.*//')
    EPOCHS=$(grep "epochs:" ${config} | awk '{print $2}')
    echo "  - ${config} (τ=${TAU}, ${EPOCHS} epochs)" | tee -a ${MASTER_LOG}
done
echo "" | tee -a ${MASTER_LOG}
echo "============================================================" | tee -a ${MASTER_LOG}
echo "" | tee -a ${MASTER_LOG}

# ============================================================
# Validation
# ============================================================

# Check we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/train.py" ]; then
    echo "ERROR: Must run from decode-rapl/ directory" | tee -a ${MASTER_LOG}
    echo "Usage: cd decode-rapl && ./scripts/train_all_tau.sh" | tee -a ${MASTER_LOG}
    exit 1
fi

# Check if all config files exist
for config in "${CONFIGS[@]}"; do
    if [ ! -f "${config}" ]; then
        echo "ERROR: Config file not found: ${config}" | tee -a ${MASTER_LOG}
        exit 1
    fi
done

# Check Python dependencies
if ! python3 -c "import torch, pandas, numpy, sklearn, yaml" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies. Install with:" | tee -a ${MASTER_LOG}
    echo "  pip install -r requirements.txt" | tee -a ${MASTER_LOG}
    exit 1
fi

# Check for GPU
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "GPU detected: ${GPU_INFO}" | tee -a ${MASTER_LOG}
else
    echo "WARNING: No GPU detected. Training will run on CPU (very slow!)" | tee -a ${MASTER_LOG}
    read -p "Continue without GPU? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "" | tee -a ${MASTER_LOG}

# ============================================================
# Sequential training loop
# ============================================================

TOTAL_CONFIGS=${#CONFIGS[@]}
COMPLETED=0
FAILED=0

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    config_num=$((i + 1))

    TAU=$(grep "tau:" ${config} | head -1 | awk '{print $2}' | sed 's/#.*//')
    EPOCHS=$(grep "epochs:" ${config} | awk '{print $2}')

    echo "============================================================" | tee -a ${MASTER_LOG}
    echo "Training ${config_num}/${TOTAL_CONFIGS}: ${config}" | tee -a ${MASTER_LOG}
    echo "τ=${TAU}, ${EPOCHS} epochs" | tee -a ${MASTER_LOG}
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${MASTER_LOG}
    echo "============================================================" | tee -a ${MASTER_LOG}
    echo "" | tee -a ${MASTER_LOG}

    # Individual log file for this config
    CONFIG_LOG="logs/train_tau${TAU}_${TIMESTAMP}.log"

    # Run training (not in background, wait for completion)
    PYTHONPATH=$(pwd) python3 src/train.py --config ${config} 2>&1 | tee ${CONFIG_LOG}
    EXIT_CODE=${PIPESTATUS[0]}

    echo "" | tee -a ${MASTER_LOG}

    if [ $EXIT_CODE -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "✓ ${config} completed successfully" | tee -a ${MASTER_LOG}
        echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${MASTER_LOG}
        echo "  Log saved to: ${CONFIG_LOG}" | tee -a ${MASTER_LOG}

        # Extract final test MAPE from log
        FINAL_MAPE=$(grep "Final Test MAPE:" ${CONFIG_LOG} | tail -1 | awk '{print $4}')
        if [ ! -z "${FINAL_MAPE}" ]; then
            echo "  Final Test MAPE: ${FINAL_MAPE}" | tee -a ${MASTER_LOG}
        fi
    else
        FAILED=$((FAILED + 1))
        echo "✗ ${config} FAILED with exit code ${EXIT_CODE}" | tee -a ${MASTER_LOG}
        echo "  Check log for details: ${CONFIG_LOG}" | tee -a ${MASTER_LOG}

        # Ask whether to continue
        echo "" | tee -a ${MASTER_LOG}
        read -p "Continue with remaining configs? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping training sequence." | tee -a ${MASTER_LOG}
            break
        fi
    fi

    echo "" | tee -a ${MASTER_LOG}

    # Brief pause between trainings
    if [ $config_num -lt $TOTAL_CONFIGS ]; then
        echo "Waiting 5 seconds before next training..." | tee -a ${MASTER_LOG}
        sleep 5
        echo "" | tee -a ${MASTER_LOG}
    fi
done

# ============================================================
# Final summary
# ============================================================
echo "============================================================" | tee -a ${MASTER_LOG}
echo "Training Sequence Complete" | tee -a ${MASTER_LOG}
echo "============================================================" | tee -a ${MASTER_LOG}
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${MASTER_LOG}
echo "Summary:" | tee -a ${MASTER_LOG}
echo "  Total configs: ${TOTAL_CONFIGS}" | tee -a ${MASTER_LOG}
echo "  Completed: ${COMPLETED}" | tee -a ${MASTER_LOG}
echo "  Failed: ${FAILED}" | tee -a ${MASTER_LOG}
echo "" | tee -a ${MASTER_LOG}

if [ $COMPLETED -gt 0 ]; then
    echo "Results:" | tee -a ${MASTER_LOG}
    for config in "${CONFIGS[@]}"; do
        TAU=$(grep "tau:" ${config} | head -1 | awk '{print $2}' | sed 's/#.*//')
        CONFIG_LOG="logs/train_tau${TAU}_${TIMESTAMP}.log"

        if [ -f "${CONFIG_LOG}" ]; then
            FINAL_MAPE=$(grep "Final Test MAPE:" ${CONFIG_LOG} | tail -1 | awk '{print $4}')
            CHECKPOINT_DIR=$(grep "checkpoint_dir:" ${config} | awk '{print $2}' | tr -d '"')

            echo "" | tee -a ${MASTER_LOG}
            echo "  Config: ${config} (τ=${TAU})" | tee -a ${MASTER_LOG}
            if [ ! -z "${FINAL_MAPE}" ]; then
                echo "    Test MAPE: ${FINAL_MAPE}" | tee -a ${MASTER_LOG}
            fi
            if [ -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
                MODEL_SIZE=$(ls -lh ${CHECKPOINT_DIR}/best_model.pth | awk '{print $5}')
                echo "    Model: ${CHECKPOINT_DIR}/best_model.pth (${MODEL_SIZE})" | tee -a ${MASTER_LOG}
            fi
        fi
    done
fi

echo "" | tee -a ${MASTER_LOG}
echo "Master log saved to: ${MASTER_LOG}" | tee -a ${MASTER_LOG}
echo "============================================================" | tee -a ${MASTER_LOG}

# Exit with error if any training failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
