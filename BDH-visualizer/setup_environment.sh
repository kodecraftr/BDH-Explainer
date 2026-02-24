#!/bin/bash
# Setup script for BDH Visuals project
# Creates conda environment named 'ml' and installs all dependencies

set -e  # Exit on error

echo "================================================"
echo "BDH Visuals - Environment Setup"
echo "================================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^ml "; then
    echo "Environment 'ml' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ml -y
    else
        echo "Using existing environment. Updating packages..."
        conda env update -n ml -f environment.yml
        echo ""
        echo "================================================"
        echo "Environment updated successfully!"
        echo "================================================"
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate ml"
        exit 0
    fi
fi

echo "Creating conda environment 'ml'..."
conda env create -f environment.yml

echo ""
echo "================================================"
echo "Environment setup complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ml"
echo ""
echo "To verify installation:"
echo "  conda activate ml"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import tiktoken; print(\"tiktoken: OK\")'"
echo ""
