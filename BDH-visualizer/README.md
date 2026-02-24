# BDH Visuals — Neural Language Model Research & Visualization

A comprehensive research project featuring custom neural language models with interactive visualization tools for understanding model behavior and neuron analysis.

## 🧠 Models

Two PyTorch language models with distinct architectures:

- **BDH Model**: Custom architecture with rotary positional embeddings and Hebbian-style gating
- **Transformer Model**: GPT-style causal Transformer for baseline comparison

## 🎨 Interactive Frontend

Next.js-based web interface for:
- **Text Generation**: Interactive text generation with temperature and sampling controls
- **Neuron Visualization**: Real-time visualization of neuron activations and weights
- **Token Encoding**: Visual representation of token encoding processes
- **Model Comparison**: Side-by-side analysis of different model behaviors

## 📁 Repository Structure

```
Backend/              # FastAPI server and utilities
├── server.py         # FastAPI server with model endpoints
├── utils.py          # BDHModelHandler and helper functions
└── __init__.py       # Python package initialization
Data/                 # Data processing and token prediction utilities
├── next_token.py     # Script for next token prediction
Frontend/             # Next.js visualization interface
├── app/              # Next.js app router pages
├── components/       # React components for visualizations
└── public/           # Static assets
Models/               # Model implementations and analysis
├── BDH_model/        # Custom BDH architecture
├── Transformer_model/# Standard transformer baseline
├── comparison/       # Model comparison scripts
└── scripts/          # Analysis and visualization scripts
.gitignore           # Git ignore rules
README.md            # This file
requirements.txt     # Python dependencies (pip)
environment.yml      # Conda environment specification
setup_environment.sh # Automated conda setup script
start_server.sh      # Script to start the backend server
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (3.10 recommended)
- Conda (Miniconda or Anaconda) - **recommended**
- Node.js 16+ (for Frontend)
- Git

### Option 1: Conda Environment Setup (Recommended)

The easiest way to set up the project:

```bash
# Clone the repository
git clone <your-repo-url>
cd BDH_visuals

# Run the automated setup script
chmod +x setup_environment.sh
./setup_environment.sh

# Activate the environment
conda activate ml
```

The script creates a conda environment named **"ml"** and installs all dependencies automatically.

### Option 2: Manual Conda Setup

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ml
```

### Option 3: pip/venv Setup

If you prefer pip and virtual environments:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` and `environment.yml` files are cross-platform compatible. All hardcoded paths have been removed.

### Running the Backend Server

The FastAPI backend server provides REST API endpoints for model inference:

```bash
# Start the backend server
python -m uvicorn Backend.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at [http://localhost:8000](http://localhost:8000)

API Endpoints:
- `GET /` - Server status
- `POST /predict` - Generate next token predictions
- `POST /embeddings` - Get token embeddings
- `POST /neurons` - Get neuron activations for a specific layer

### Frontend Setup (Next.js Interface)

```bash
# Navigate to frontend directory
cd Frontend

# Install Node.js dependencies
npm install

# Start development server
npm run dev
```

Access the visualization interface at [http://localhost:3000](http://localhost:3000)

## 🔬 Training Models

The training scripts work both on Google Colab (with GPU) and local systems:

### On Google Colab (Recommended for GPU training):
1. The scripts automatically detect Colab environment and mount Google Drive
2. Upload `train.bin` and `val.bin` to your Drive
3. Run the training script

### On Local Systems:
```bash
# Set up data directory (optional - defaults to project/data)
export BDH_DATA_DIR=/path/to/your/data
export BDH_OUT_DIR=/path/to/save/checkpoints

# Run training
python Models/BDH_model/train.py
# or
python Models/Transformer_model/train.py
```

Prepare `train.bin` and `val.bin` token binaries (TinyStories dataset with GPT-2 tokenization).

For more details on training configuration, see [Models/README.md](Models/README.md).

Checkpoints are written every 100 iterations (`ckpt_<iter>.pt`) and final weights saved as `bdh_final.pt` or `transformer_final.pt`.

## 🔬 Model Inference

### Command Line Interface

```bash
# BDH Model inference
cd Models/BDH_model && python run.py

# Transformer Model inference  
cd Models/Transformer_model && python run.py

# Next token prediction utility
python Data/next_token.py "Your input text here"
```

### Web Interface

Start the frontend development server for interactive model exploration:

```bash
cd Frontend && npm run dev
```

## 📊 Analysis & Visualization

- **Neuron Analysis**: Automated scripts analyze neuron activations across model checkpoints
- **Performance Comparison**: Compare BDH vs Transformer architectures
- **Memory Usage**: Track computational efficiency
- **Training Metrics**: Visualize loss curves and model convergence

## ✨ Portability & Cross-Platform Support

This project has been designed to work seamlessly across different systems:

### Key Features:
- **No Hardcoded Paths**: All file paths are relative to the project root
- **Environment Variable Support**: Override default paths using `BDH_DATA_DIR` and `BDH_OUT_DIR`
- **Automatic Environment Detection**: Training scripts detect Colab vs local environments
- **Cross-Platform Requirements**: Clean `requirements.txt` without system-specific file paths
- **Flexible Model Loading**: All model handlers use relative paths by default

### Path Configuration:
All scripts automatically determine paths relative to the current file location. You can override these defaults using environment variables:

```bash
# Example: Custom data directory
export BDH_DATA_DIR=/custom/path/to/data
export BDH_OUT_DIR=/custom/path/to/outputs
```

## 🛠️ Development

Both training scripts support Google Colab (GPU) and local systems. Local CPUs may be slower for the configured batch sizes.

### Training Requirements
- Google Colab Pro (recommended) or GPU with 8GB+ VRAM
- TinyStories dataset with GPT-2 tokenization
- `tiktoken` library for tokenization

## 📚 Documentation

See [Models/README.md](Models/README.md) for detailed guidance on:
- Data preparation and preprocessing
- Training configuration and hyperparameters
- Model architecture details
- Advanced analysis techniques

## 🌐 Frontend Documentation

See [Frontend/README.md](Frontend/README.md) for:
- Component architecture
- Visualization features
- Development setup
- Deployment instructions
