#!/bin/bash
# BDH Activation Analysis Quick Start Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/Models/BDH_model/checkpoints"
OUTPUT_DIR="$SCRIPT_DIR/Activation_Analysis/explorer_output"

echo -e "${GREEN}=== BDH Activation Analysis ===${NC}"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${YELLOW}Warning: Checkpoint directory not found: $CHECKPOINT_DIR${NC}"
    echo "Please ensure checkpoints are in the correct location."
    exit 1
fi

# Count checkpoints
CKPT_COUNT=$(ls -1 "$CHECKPOINT_DIR"/ckpt_*.pt 2>/dev/null | wc -l)
echo "Found $CKPT_COUNT checkpoints in $CHECKPOINT_DIR"
echo ""

# Function to show menu
show_menu() {
    echo "Select an option:"
    echo ""
    echo "  1) Run full analysis (feature mapping + clustering + dynamics)"
    echo "  2) Interactive exploration mode"
    echo "  3) Analyze single text"
    echo "  4) Install dependencies"
    echo "  5) Exit"
    echo ""
    read -p "Enter choice [1-5]: " choice
}

# Function to run full analysis
run_full_analysis() {
    echo -e "\n${GREEN}Running full analysis...${NC}"
    echo "This may take several minutes depending on the number of checkpoints."
    echo ""
    
    cd "$SCRIPT_DIR"
    python -m Activation_Analysis.run_analysis explore \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output "$OUTPUT_DIR" \
        --n_clusters 50 \
        --skip 2 \
        --max_checkpoints 20
    
    echo -e "\n${GREEN}Analysis complete!${NC}"
    echo "Output saved to: $OUTPUT_DIR"
}

# Function to run interactive mode
run_interactive() {
    echo -e "\n${GREEN}Starting interactive mode...${NC}"
    echo ""
    
    cd "$SCRIPT_DIR"
    python -m Activation_Analysis.run_analysis interactive \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output "$OUTPUT_DIR"
}

# Function to analyze single text
analyze_text() {
    read -p "Enter text to analyze: " text
    
    if [ -z "$text" ]; then
        text="Once upon a time"
    fi
    
    echo -e "\n${GREEN}Analyzing: '$text'${NC}"
    echo ""
    
    cd "$SCRIPT_DIR"
    
    # Use the latest checkpoint
    LATEST_CKPT=$(ls -1 "$CHECKPOINT_DIR"/ckpt_*.pt | tail -1)
    
    python -m Activation_Analysis.run_analysis extract \
        --checkpoint "$LATEST_CKPT" \
        --text "$text" \
        --output "$OUTPUT_DIR/quick_analysis.json"
    
    echo -e "\n${GREEN}Analysis saved to: $OUTPUT_DIR/quick_analysis.json${NC}"
}

# Function to install dependencies
install_deps() {
    echo -e "\n${GREEN}Installing dependencies...${NC}"
    pip install -r "$SCRIPT_DIR/Activation_Analysis/requirements.txt"
    echo -e "\n${GREEN}Dependencies installed!${NC}"
}

# Main loop
while true; do
    show_menu
    
    case $choice in
        1) run_full_analysis ;;
        2) run_interactive ;;
        3) analyze_text ;;
        4) install_deps ;;
        5) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    echo ""
done
