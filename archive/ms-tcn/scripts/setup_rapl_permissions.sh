#!/bin/bash
# Setup RAPL permissions for passwordless background execution
# This script configures sudoers to allow specific Python scripts to access RAPL without password

set -e

echo "============================================================"
echo "RAPL Passwordless Sudo Setup"
echo "============================================================"
echo ""
echo "This script will configure passwordless sudo access for:"
echo "  - power_data_collector.py (needs RAPL for data collection)"
echo "  - power_predictor.py (needs RAPL for live monitoring)"
echo ""
echo "This allows background execution without password prompts."
echo ""

# Get current user
CURRENT_USER=$(whoami)

# Get Python path from current environment
PYTHON_PATH=$(which python3)

# Get absolute paths to scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COLLECTOR_PATH="${PROJECT_DIR}/src/power_data_collector.py"
PREDICTOR_PATH="${PROJECT_DIR}/src/power_predictor.py"

echo "Configuration:"
echo "  User: ${CURRENT_USER}"
echo "  Python: ${PYTHON_PATH}"
echo "  Data collector: ${COLLECTOR_PATH}"
echo "  Power predictor: ${PREDICTOR_PATH}"
echo ""

# Check if files exist
if [ ! -f "${COLLECTOR_PATH}" ]; then
    echo "ERROR: power_data_collector.py not found at ${COLLECTOR_PATH}"
    exit 1
fi

if [ ! -f "${PREDICTOR_PATH}" ]; then
    echo "ERROR: power_predictor.py not found at ${PREDICTOR_PATH}"
    exit 1
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run this script as root/sudo"
    echo "Run as: ./scripts/setup_rapl_permissions.sh"
    echo "(The script will ask for sudo password when needed)"
    exit 1
fi

SUDOERS_FILE="/etc/sudoers.d/powermon-rapl"

echo "This will create a sudoers configuration at:"
echo "  ${SUDOERS_FILE}"
echo ""
echo "Granting passwordless sudo for:"
echo "  sudo ${PYTHON_PATH} ${COLLECTOR_PATH} ..."
echo "  sudo ${PYTHON_PATH} ${PREDICTOR_PATH} ..."
echo ""
read -p "Continue? (Y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Creating sudoers configuration..."

# Create temporary file with sudoers rules
TEMP_SUDOERS=$(mktemp)
cat > ${TEMP_SUDOERS} <<EOF
# Allow passwordless sudo for RAPL-accessing Python scripts
# Created by setup_rapl_permissions.sh on $(date)
# User: ${CURRENT_USER}

${CURRENT_USER} ALL=(root) NOPASSWD: ${PYTHON_PATH} ${COLLECTOR_PATH}*
${CURRENT_USER} ALL=(root) NOPASSWD: ${PYTHON_PATH} ${PREDICTOR_PATH}*
EOF

echo "Sudoers rules to be created:"
echo "---"
cat ${TEMP_SUDOERS}
echo "---"
echo ""

# Validate sudoers syntax
echo "Validating sudoers syntax..."
if ! sudo visudo -c -f ${TEMP_SUDOERS}; then
    echo "ERROR: Invalid sudoers syntax"
    rm ${TEMP_SUDOERS}
    exit 1
fi

# Install sudoers file
echo "Installing sudoers file (requires sudo password)..."
sudo cp ${TEMP_SUDOERS} ${SUDOERS_FILE}
sudo chmod 440 ${SUDOERS_FILE}
sudo chown root:root ${SUDOERS_FILE}
rm ${TEMP_SUDOERS}

echo "✓ Sudoers configuration installed: ${SUDOERS_FILE}"
echo ""

# Test it
echo "Testing passwordless sudo access..."
if sudo -n ${PYTHON_PATH} --version >/dev/null 2>&1; then
    echo "✓ Passwordless sudo is working!"
else
    echo "⚠️  Warning: Passwordless sudo test failed"
    echo "   You may need to log out and back in for changes to take effect"
fi

echo ""
echo "============================================================"
echo "Setup Complete"
echo "============================================================"
echo ""
echo "You can now run background data collection without passwords:"
echo "  ./scripts/collect_and_train_bg.sh 3600 100"
echo ""
echo "The process will run in background even if you disconnect."
echo ""
echo "To monitor progress:"
echo "  ./scripts/check_training_progress.sh"
echo ""
echo "To remove this configuration later:"
echo "  sudo rm ${SUDOERS_FILE}"
echo ""
