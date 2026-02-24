# Models

This directory contains two PyTorch language models plus their checkpoints, visuals, and neuron analyses. It also explains how to set up a Python environment, prepare data, train in Google Colab, and run local inference.

## Overview
- **BDH Model**: Custom architecture with rotary positional embeddings and Hebbian-style gating. See [Models/BDH_model/train.py](BDH_model/train.py) and [Models/BDH_model/run.py](BDH_model/run.py).
- **Transformer Model**: GPT-style causal Transformer (scaled dot-product attention + MLP). See [Models/Transformer_model/train.py](Transformer_model/train.py) and [Models/Transformer_model/run.py](Transformer_model/run.py).
- **Checkpoints & Visuals**: Each model subfolder includes `checkpoints/`, `visuals/` (loss/metrics plots), and `neuron_analysis/` CSVs where available.

## Folder Layout
```
Models/
   BDH_model/
      train.py           # Colab-oriented training script
      run.py             # Local CPU inference
      checkpoints/       # Saved checkpoints (ckpt_*.pt)
      visuals/           # loss_vs_iter.png, param_norm_vs_iter.png, metrics.csv
      neuron_analysis/   # per-ckpt neuron statistics (CSV)
      bdh_final.pt       # Final trained weights (example)
   Transformer_model/
      train.py           # Colab-oriented training script
      run.py             # Local CPU inference
      checkpoints/       # Saved checkpoints (ckpt_*.pt)
      visuals/           # loss_vs_iter.png, param_norm_vs_iter.png, metrics.csv
      transformer_final.pt  # Final trained weights (example)
```

## Setup
- **Python**: 3.10+ is recommended.
- **Environment**: Use a virtualenv to avoid system-wide changes.

Create and activate a venv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Data Preparation
Training expects `uint16` token binaries (`train.bin`, `val.bin`) produced from TinyStories (GPT‑2 tokenization via `tiktoken`). If your data-prep script is not included in this repo, prepare the `.bin` files and place them in the data directory.

## Training (Colab or Local)
The scripts in [Models/BDH_model/train.py](BDH_model/train.py) and [Models/Transformer_model/train.py](Transformer_model/train.py) work both on Google Colab with GPU and on local systems.

**On Google Colab:**
- The scripts automatically detect Colab environment and mount Google Drive
- Default data directory: `/content/drive/My Drive/BDH_Data`
- Steps:
   1. Open a GPU-backed Colab notebook.
   2. Upload/clone this repository.
   3. Ensure `train.bin` and `val.bin` exist under a Drive folder referenced by `DATA_DIR`.
   4. Run the chosen training script.

**On Local Systems:**
- The scripts use relative paths from the project directory
- Default data directory: `<project_root>/data`
- Default output directory: `<model_folder>/checkpoints`

**Custom Paths:**
You can override the default paths using environment variables:
- `BDH_DATA_DIR`: Path to directory containing `train.bin` and `val.bin`
- `BDH_OUT_DIR`: Path to directory for saving checkpoints

Example:
```bash
export BDH_DATA_DIR=/path/to/your/data
export BDH_OUT_DIR=/path/to/your/checkpoints
python Models/BDH_model/train.py
```

Checkpoints:
- Saved every 100 iterations as `ckpt_<iter>.pt`.
- Final models saved as `bdh_final.pt` (BDH) and `transformer_final.pt` (Transformer).

## Inference (Local CPU)
Copy a checkpoint into the respective model folder and run `run.py`.

Transformer:
```bash
cd Models/Transformer_model
python run.py
```

BDH:
```bash
cd Models/BDH_model
python run.py
```

Both scripts stream generated tokens. `tiktoken` is required; install via `pip install tiktoken` or use the provided `requirements.txt`.

## Visuals & Neuron Analysis
- `visuals/metrics.csv`: Summary metrics per checkpoint.
- `visuals/loss_vs_iter.png`, `visuals/param_norm_vs_iter.png`: Training curves.
- `neuron_analysis/*.csv`: per-checkpoint neuron stats (BDH and Transformer), useful for interpretability.

## Training Locally (Optional)
To train outside Colab, remove Drive‑mount code and set `DATA_DIR`/`OUT_DIR` to local paths. Reduce `batch_size`/`max_iters` and use a CUDA-enabled environment.

## Troubleshooting
- `google.colab` import errors locally: Training scripts are Colab‑specific; use Colab or adapt paths and remove Drive mount.
- Missing `tiktoken`: `pip install tiktoken`.
- Checkpoint not found: ensure the expected `.pt` file is in the working directory used by `run.py`.

