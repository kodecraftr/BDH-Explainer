# BDH Visualizer

An interactive research toolkit for exploring model behavior and internal representations of the Baby Dragon Hypernetwork (BDH) and related transformer baselines. The project pairs a Python/FastAPI backend with a Next.js frontend, enabling interactive inspection of token encodings, neuron activations, and generation behavior.

---

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Primary Components](#primary-components)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Developer Notes](#developer-notes)
- [Contributing](#contributing)

---

## Overview

The BDH Visualizer provides a browser-based interface for researchers and developers to:

- Inspect token encodings and embedding representations
- Visualize neuron activations across model layers
- Run text generation with configurable sampling and temperature settings
- Compare BDH and standard Transformer baselines side by side

The **BDH model** is a custom language model architecture featuring rotary positional embeddings and Hebbian-style gating, trained on the TinyStories dataset. A standard GPT-style Transformer is included for comparison.

---

## Repository Layout

```
BDH-visualizer/
├── Backend/                    # FastAPI server and model utilities
│   ├── server.py               # API endpoints (inference, embeddings, neurons)
│   └── utils.py                # Model handler and helper utilities
├── Data/                       # Data preparation and prediction utilities
├── Frontend/                   # Next.js visualization interface
│   ├── app/                    # Next.js app router pages
│   ├── components/             # React components for visualizations
│   └── lib/                    # Shared frontend utilities
├── Models/                     # Model implementations and training scripts
│   ├── BDH_model/              # BDH architecture, training, and inference
│   ├── Transformer_model/      # GPT-style baseline
│   ├── comparison/             # Cross-model analysis utilities
│   └── scripts/                # Standalone analysis scripts
├── environment.yml             # Conda environment definition
├── requirements.txt            # Pip dependency manifest
├── setup_environment.sh        # One-step environment setup
├── start_server.sh             # Backend server launch script
└── run_activation_analysis.sh  # Activation analysis convenience script
```

---

## Quick Start

### Prerequisites

- Python 3.8+ (3.10 recommended)
- Conda (recommended) or a Python virtual environment
- Node.js 16+

### Option 1 — Conda (Recommended)

**1. Set up the environment:**

```bash
chmod +x setup_environment.sh
./setup_environment.sh
conda activate ml
```

**2. Start the backend server:**

```bash
python -m uvicorn Backend.server:app --host 0.0.0.0 --port 8000 --reload
```

**3. Start the frontend development server:**

```bash
cd Frontend
npm install
npm run dev
```

The backend will be available at `http://localhost:8000` and the frontend at `http://localhost:3000`.

### Option 2 — pip/venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then follow steps 2 and 3 above to launch the servers.

---

## Primary Components

| Path | Description |
|---|---|
| `Backend/server.py` | FastAPI server exposing `/predict`, `/embeddings`, and `/neurons` endpoints |
| `Backend/utils.py` | `BDHModelHandler` class — loads checkpoints and runs inference |
| `Frontend/` | Next.js app with React components for visualization and UI |
| `Data/` | Preprocessing scripts and next-token prediction utilities |
| `Models/BDH_model/` | BDH architecture, training script, and local inference runner |
| `Models/Transformer_model/` | GPT-style baseline with equivalent training and inference scripts |

---

## Configuration

Scripts resolve data and checkpoint paths relative to the project root by default. Override these with environment variables:

| Variable | Purpose |
|---|---|
| `BDH_DATA_DIR` | Directory containing `train.bin` and `val.bin` token binaries |
| `BDH_OUT_DIR` | Output directory for checkpoints and training artifacts |

Example:

```bash
export BDH_DATA_DIR=/path/to/data
export BDH_OUT_DIR=/path/to/checkpoints
```

---

## Usage Examples

**Next-token prediction from the CLI:**

```bash
python Data/next_token.py "Once upon a time"
```

**Run a BDH model inference session:**

```bash
cd Models/BDH_model
python run.py
```

**Run activation analysis:**

```bash
chmod +x run_activation_analysis.sh
./run_activation_analysis.sh
```

---

## Developer Notes

- Training scripts support both Google Colab (GPU) and local GPU environments. A GPU with at least 8 GB VRAM is recommended for practical batch sizes.
- Tokenization follows GPT-2 conventions via `tiktoken`. Install it with `pip install tiktoken` if not present.
- Model checkpoints are saved every 100 iterations as `ckpt_<iter>.pt`. Final weights are saved as `bdh_final.pt` and `transformer_final.pt` respectively.
- Neuron analysis CSVs and training curve plots are written to `Models/<model>/neuron_analysis/` and `Models/<model>/visuals/` after training.

