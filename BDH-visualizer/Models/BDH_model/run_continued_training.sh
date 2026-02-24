#!/bin/bash

echo "============================================================"
echo "BDH Continued Training Setup & Execution"
echo "============================================================"
echo "This script will:"
echo "1. Prepare OpenWebText dataset for training"
echo "2. Continue training your existing BDH model"
echo "============================================================"

# Change to BDH model directory
cd "$(dirname "$0")"

# Check if existing model exists
if [ ! -f "checkpoints/bdh_final.pt" ]; then
    echo "❌ Error: bdh_final.pt not found in checkpoints/"
    echo "Please make sure you have a trained BDH model first."
    exit 1
fi

echo "✓ Found existing BDH model: checkpoints/bdh_final.pt"

# Step 1: Prepare OpenWebText data (if not already done)
if [ ! -d "openwebtext_bin" ] || [ ! -f "openwebtext_bin/train.bin" ]; then
    echo ""
    echo "📊 Step 1: Preparing OpenWebText dataset..."
    python prepare_openwebtext_data.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Data preparation failed!"
        exit 1
    fi
else
    echo "✓ OpenWebText dataset already prepared"
fi

# Step 2: Continue training
echo ""
echo "🚀 Step 2: Starting continued training..."
echo "Loading existing BDH weights and training on OpenWebText..."
echo ""

python continue_train.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "🎉 CONTINUED TRAINING COMPLETE!"
    echo "============================================================"
    echo "Your enhanced BDH model has been saved as:"
    echo "  • checkpoints/bdh_continued_final.pt"
    echo ""
    echo "You can now use this enhanced model for inference!"
    echo "============================================================"
else
    echo "❌ Training failed! Check the error messages above."
    exit 1
fi