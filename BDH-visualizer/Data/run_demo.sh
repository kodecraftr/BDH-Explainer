#!/bin/bash
# Launcher script for demo.py
# Works with conda environment 'ml' or any activated Python environment

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Try to activate conda environment 'ml' if conda is available
if command -v conda &> /dev/null; then
    # Try different conda initialization locations
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    
    # Activate ml environment if it exists
    if conda env list | grep -q "^ml "; then
        conda activate ml
        echo "Using conda environment: ml"
    else
        echo "Warning: conda environment 'ml' not found. Using current Python environment."
    fi
else
    echo "Conda not found. Using current Python environment."
fi

# Run demo.py
echo "Running demo.py..."
python demo.py "$@"
