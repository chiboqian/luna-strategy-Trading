#!/bin/bash
# Wrapper script to run Python scripts with the configured environment.
# Usage: ./run.sh <script_path> [arguments...]

# Get the directory of this script (Project Root)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Virtual Environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Add Project Root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if a script was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <script_path> [arguments...]"
    exit 1
fi

# Get the script to run
SCRIPT=$1
shift

# Run the script with the remaining arguments
python3 -u "$SCRIPT" "$@"