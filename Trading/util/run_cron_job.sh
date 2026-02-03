#!/bin/bash

# Wrapper script for running Python scripts via Cron
# Usage: ./util/run_cron_job.sh path/to/script.py [args]

# 1. Determine Project Root (assuming script is in Trading/util/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

# 2. Source the Python virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
else
    echo "Error: .venv not found in $PROJECT_DIR"
    exit 1
fi

# 3. Execute the passed script with arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script> [args...]"
    exit 1
fi

# Setup logging
LOG_DIR="$PROJECT_DIR/trading_logs/cron"
mkdir -p "$LOG_DIR"

SCRIPT_NAME=$(basename "$1")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log"

echo "[$(date)] Starting job: python $*" | tee -a "$LOG_FILE"
python "$@" 2>&1 | tee -a "$LOG_FILE"
exit ${PIPESTATUS[0]}