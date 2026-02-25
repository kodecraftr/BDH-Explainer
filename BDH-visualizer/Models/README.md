# BDH Models Architecture & Training

This directory contains the core PyTorch implementations for the **Baby Dragon Hatchling (BDH)** model and a comparative **Transformer** baseline. It includes all necessary scripts for model definition, training, checkpointing, and local inference testing, alongside utilities for generating performance visuals and neuron statistics.

---

## 🏗️ Architecture Overview

The repository explores two distinct language modeling paradigms:

1. **BDH Model (`BDH_model/`)**: 
   A highly customized architecture leveraging Rotary Positional Embeddings (RoPE) paired with Hebbian-style gating mechanisms. This model is designed to study alternative sparse activation mapping and token encoding strategies.
   - **Core Implementation:** `train.py` (Training logic), `run.py` (Inference execution).

2. **Transformer Baseline (`Transformer_model/`)**: 
   A standard GPT-style causal language model utilizing scaled dot-product attention and Multi-Layer Perceptrons (MLP). This serves as the primary baseline against which the BDH architecture is measured.
   - **Core Implementation:** `train.py` (Training logic), `run.py` (Inference execution).

---

## 📂 Directory Structure

Both architectural branches follow an identical structural layout for consistency in analysis:

```text
Models/
├── BDH_model/
│   ├── train.py                # Primary training script (Supports local & Colab)
│   ├── run.py                  # Local CPU/GPU inference script
│   ├── checkpoints/            # Serialized model weights (`ckpt_*.pt`)
│   ├── visuals/                # Generated loss curves and metrics (`.png`, `.csv`)
│   └── neuron_analysis/        # Granular per-checkpoint neuron statistics (`.csv`)
│
├── Transformer_model/
│   ├── train.py                
│   ├── run.py                  
│   ├── checkpoints/            
│   ├── visuals/                
│   └── neuron_analysis/        
└── scripts/                    # Utility scripts for batch analysis
```

---

## 🚀 Setup & Environment

Ensure you have a modern Python environment (**Python 3.10+** is highly recommended). To avoid system-wide dependency conflicts, utilize the project-wide Conda environment or initialize a dedicated virtual environment here.

```bash
# Using standard virtual environment:
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Requirements are defined in the repository root
pip install -r ../requirements.txt 
```

---

## 💾 Data Preparation

The training scripts ingest raw tokenized binaries natively format-compatible with `tiktoken` (GPT-2 BPE standard). 

**Required Files:** 
Ensure your dataset is pre-processed into `uint16` binaries named `train.bin` and `val.bin`. Place these within your designated data directory before initiating training.

---

## 🏋️‍♂️ Training Workflows

The provided training scripts are hybrid—they are configured to detect and run identically on both **Google Colab (GPU)** and **Local Systems**.

### Local Execution Strategy

By default, scripts resolve relative paths targeting `<project_root>/data` for inputs and `<model_folder>/checkpoints` for outputs.

You can explicitly override these paths using environmental variables:

*   `BDH_DATA_DIR`: Absolute path to the directory housing `train.bin` and `val.bin`.
*   `BDH_OUT_DIR`: Absolute path for saving the compiled `.pt` checkpoint dictionaries.

**Example Local Execution:**
```bash
export BDH_DATA_DIR=/dataset/tinystories/
export BDH_OUT_DIR=/projects/BDH/output_checkpoints/

python Models/BDH_model/train.py
```

### Google Colab Execution Strategy

1. Open a GPU-accelerated notebook instance.
2. Clone this repository into the workspace.
3. The scripts will automatically detect the Colab environment and attempt to mount Google Drive. 
4. Ensure your tokenized binaries are located at the default Drive mounting point: `/content/drive/My Drive/BDH_Data`.

**Checkpoint Intervals:**
*   Intermediate weights serialize every 100 iterations as `ckpt_<iter>.pt`.
*   Final states compile as either `bdh_final.pt` or `transformer_final.pt`.

---

## 🧠 Local Inference

To validate output quality interactively, execute the `run.py` script within either model's respective directory. 

*Note: The inference script requires a valid checkpoint (either `ckpt_*.pt` or `*_final.pt`) to be present in the working directory.*

**Transformer Baseline:**
```bash
cd Models/Transformer_model
python run.py
```

**BDH Architecture:**
```bash
cd Models/BDH_model
python run.py
```

The script will stream auto-regressive token generation directly to the console. *(Requires `tiktoken` for decoding).*

---

## 📊 Analytics & Telemetry

Automated telemetry is generated routinely during the training lifecycle:

*   **Training Curves (`visuals/`)**: Plots visualizing `loss_vs_iter.png` and `param_norm_vs_iter.png`.
*   **Performance Metrics (`visuals/metrics.csv`)**: A historical log mapping iterations to loss bounds.
*   **Activation Statistics (`neuron_analysis/*.csv`)**: Deep-dive statistical outputs mapping the activation density of individual neurons per checkpoint. This data heavily feeds the BDH Visualizer UI.

---

## 🛠 Troubleshooting Common Issues

*   **`google.colab` Import Errors (Local Run):** If your local environment strictly enforces import checks, you may need to manually comment out the Colab-specific Drive mount block headers at the top of the `train.py` scripts.
*   **Missing Dependencies (`tiktoken`):** Ensure the tokenizer is installed via `pip install tiktoken`.
*   **Checkpoint Resolution Failures:** Verify that `run.py` is executed from *within* its specific model sub-directory, and that a `.pt` file exists.
