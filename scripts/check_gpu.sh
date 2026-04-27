#!/bin/bash
#
# DECODE-RAPL v2 GPU Verification Script
#
# Checks system requirements and estimates training time
#

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "DECODE-RAPL v2 System Check"
echo "========================================"
echo ""

# Check Python
echo -e "${BLUE}1. Python Environment${NC}"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "  Python: ${GREEN}${PYTHON_VERSION}${NC}"
else
    echo -e "  Python: ${RED}NOT FOUND${NC}"
    exit 1
fi

# Check PyTorch
echo ""
echo -e "${BLUE}2. PyTorch Installation${NC}"
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "  PyTorch: ${GREEN}${TORCH_VERSION}${NC}"
else
    echo -e "  PyTorch: ${RED}NOT INSTALLED${NC}"
    echo "  Install: pip install torch torchvision torchaudio"
    exit 1
fi

# Check CUDA
echo ""
echo -e "${BLUE}3. CUDA/GPU Status${NC}"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "  CUDA Available: ${GREEN}YES${NC}"

    # GPU details
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    GPU_MEMORY=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}')")

    echo "  GPU: ${GPU_NAME}"
    echo "  CUDA Version: ${CUDA_VERSION}"
    echo "  GPU Memory: ${GPU_MEMORY} GB"

    # Check if it's GTX 1650
    if echo "$GPU_NAME" | grep -qi "1650"; then
        echo -e "  ${YELLOW}Note: GTX 1650 detected - training will be slow (30-60 min/epoch)${NC}"
        ESTIMATED_EPOCH_TIME=45  # minutes
    elif echo "$GPU_NAME" | grep -qi "V100\|A100\|RTX 3090\|RTX 4090"; then
        echo -e "  ${GREEN}Fast GPU detected - training will be quick (5-10 min/epoch)${NC}"
        ESTIMATED_EPOCH_TIME=7
    else
        echo "  Mid-range GPU - estimated 15-30 min/epoch"
        ESTIMATED_EPOCH_TIME=22
    fi
else
    echo -e "  CUDA Available: ${RED}NO${NC}"
    echo -e "  ${YELLOW}WARNING: Training will use CPU (VERY SLOW - hours per epoch!)${NC}"
    ESTIMATED_EPOCH_TIME=180
fi

# Check required packages
echo ""
echo -e "${BLUE}4. Required Packages${NC}"
# Format: "display_name:import_name"
PACKAGES=("numpy:numpy" "pandas:pandas" "matplotlib:matplotlib" "scikit-learn:sklearn" "pyyaml:yaml" "tqdm:tqdm")
ALL_INSTALLED=true

for pkg_pair in "${PACKAGES[@]}"; do
    IFS=':' read -r display_name import_name <<< "$pkg_pair"
    if python -c "import $import_name" 2>/dev/null; then
        VERSION=$(python -c "import $import_name; print($import_name.__version__)")
        echo -e "  ${display_name}: ${GREEN}${VERSION}${NC}"
    else
        echo -e "  ${display_name}: ${RED}NOT INSTALLED${NC}"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = false ]; then
    echo ""
    echo -e "${RED}Missing packages! Install with:${NC}"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check data
echo ""
echo -e "${BLUE}5. Training Data${NC}"
DATA_FOUND=true
for tau in 1 4 8; do
    DATA_DIR="data/processed/tau${tau}"
    if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/train.npz" ]; then
        SIZE=$(du -sh "$DATA_DIR" | awk '{print $1}')
        SAMPLES=$(python -c "import numpy as np; d=np.load('$DATA_DIR/train.npz'); print(f'{len(d[\"X\"]):,}')" 2>/dev/null || echo "unknown")
        echo -e "  tau=${tau}: ${GREEN}FOUND${NC} (${SIZE}, ${SAMPLES} samples)"
    else
        echo -e "  tau=${tau}: ${RED}NOT FOUND${NC}"
        DATA_FOUND=false
    fi
done

if [ "$DATA_FOUND" = false ]; then
    echo ""
    echo -e "${YELLOW}Missing data! Transfer preprocessed data to this machine:${NC}"
    echo "  rsync -avz --progress data/processed/ user@gpu-machine:/path/to/decode-rapl/data/processed/"
fi

# Estimate training time
echo ""
echo -e "${BLUE}6. Training Time Estimate${NC}"
echo "  Model parameters: 267,941 (~268K)"
echo "  Training samples per tau: ~4.4-4.7M"
echo "  Batch size: 256"
echo "  Batches per epoch: ~18,000"
echo ""

if [ -n "$ESTIMATED_EPOCH_TIME" ]; then
    TOTAL_EPOCHS=100
    EARLY_STOP_EPOCHS=50  # Realistic with early stopping

    TOTAL_TIME_HOURS=$(echo "scale=1; $ESTIMATED_EPOCH_TIME * $EARLY_STOP_EPOCHS / 60" | bc)
    TOTAL_TIME_DAYS=$(echo "scale=1; $TOTAL_TIME_HOURS / 24" | bc)

    echo "  Estimated time per epoch: ~${ESTIMATED_EPOCH_TIME} minutes"
    echo "  Expected epochs (with early stopping): ~${EARLY_STOP_EPOCHS}"
    echo "  Time per model: ~${TOTAL_TIME_HOURS} hours (~${TOTAL_TIME_DAYS} days)"
    echo "  Time for all 3 models: ~$(echo "scale=1; $TOTAL_TIME_HOURS * 3" | bc) hours (~$(echo "scale=1; $TOTAL_TIME_DAYS * 3" | bc) days)"
fi

# Memory check
echo ""
echo -e "${BLUE}7. Memory Requirements${NC}"
echo "  Model size: ~1 MB"
echo "  Batch activations: ~15-20 MB"
echo "  Optimizer state: ~2 MB"
echo "  Total per batch: ~20-25 MB"
echo ""
if [ -n "$GPU_MEMORY" ]; then
    if (( $(echo "$GPU_MEMORY < 4" | bc -l) )); then
        echo -e "  ${RED}WARNING: GPU has <4GB memory - may need to reduce batch size${NC}"
    else
        echo -e "  ${GREEN}GPU memory sufficient for training${NC}"
    fi
fi

# Recommendations
echo ""
echo "========================================"
echo -e "${BLUE}Recommendations${NC}"
echo "========================================"

if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    if echo "$GPU_NAME" | grep -qi "1650"; then
        echo -e "${YELLOW}Your GTX 1650 will work but training will be slow.${NC}"
        echo "Consider using a cloud GPU service for faster training:"
        echo ""
        echo "  Recommended services:"
        echo "    - Lambda Labs: https://lambdalabs.com/service/gpu-cloud"
        echo "      - RTX 4090: \$0.99/hr (~\$25 total for 3 models)"
        echo "      - A100 (40GB): \$1.10/hr (~\$15 total)"
        echo ""
        echo "    - Vast.ai: https://vast.ai/"
        echo "      - RTX 3090: \$0.30-0.50/hr (~\$10-15 total)"
        echo "      - RTX 4090: \$0.60-0.80/hr (~\$15-20 total)"
        echo ""
        echo "    - RunPod: https://www.runpod.io/"
        echo "      - RTX 3090: \$0.44/hr (~\$15 total)"
        echo "      - RTX 4090: \$0.79/hr (~\$20 total)"
        echo ""
    else
        echo -e "${GREEN}Your GPU is suitable for training!${NC}"
        echo "Expected training time is reasonable."
    fi
else
    echo -e "${RED}No GPU detected!${NC}"
    echo "CPU training is not recommended (will take weeks)."
    echo "Use a cloud GPU service (see recommendations above)."
fi

echo ""
echo "========================================"
echo "Ready to start training!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Ensure data is in data/processed/tau{1,4,8}/"
echo "  2. Run: ./scripts/start_training.sh tau1"
echo "  3. Monitor: ./scripts/monitor_training.sh --watch"
echo ""
