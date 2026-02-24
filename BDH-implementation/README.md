## Project Overview

BDH-Diffusion is a generative framework for predicting longitudinal brain tumor progression. It pairs diffusion-based generation with BDH-style linear attention to reduce memory overhead versus standard quadratic attention.

In this repository’s main training/inference path, the model operates on 2D slices (not full 3D volumes in a single forward pass), while still modeling longitudinal progression across sessions.

## Repository Structure

```text
BDH-implementation/
├── config/                 # Model and training configurations
│   ├── arg_parse.py        # Command-line argument parsing
│   └── cfg_tadiff_net.py   # Hyperparameters, paths, and DiT & BDH config
├── data/                   # Dataset preprocessing scripts and metadata
│   ├── preproc_prepare_data.py  # Script to preprocess SAILOR NIfTI files
│   └── README.md           # Data format and preparation notes
│   └── sailor_info.csv     # Metadata and patient info for the SAILOR dataset
├── src/                    # Source code modules
│   ├── data/               # Data loaders (MONAI-based) and PyTorch Dataset
│   ├── evaluation/         # Metrics calculation (Dice, SSIM, MAE, PSNR)
│   ├── net/                # Network architecture (Diffusion, BDH blocks, SEAL adapter)
│   ├── visualization/      # Plotting, overlays, and uncertainty map generation
│   └── tadiff_model.py     # PyTorch Lightning module wrapper
├── inference.py            # Script for running model inference
├── train.py                # Main training script for the Bdh-DiT model
├── utils_net.py            # Core neural network utilities
└── requirements.txt        # Python dependencies
```



https://github.com/user-attachments/assets/a9fdb518-9116-4246-a198-65f841f9149d



## Model Architecture & Diffusion Pipeline

> **Bdh-DiT**: A treatment-aware diffusion transformer that uses BDH-style linear attention (ELU+1 feature map, linear-form normalization) to reduce memory growth compared with quadratic softmax attention.

---

### End-to-End Execution Flowchart

Architecture diagram placeholder (add your local figure path if available, e.g. `docs/architecture.png`).

---

## Key Features & Key Functionality of the Model

| Feature                                 | Implementation (code)                                                                                                                                                                                                        | Notes                                                                                                                                                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BDH Linear Attention**                | Active architecture block is `BDH_Attention` in `src/net/tadiff_dit_arch.py` (ELU+1 feature map, K^T·V-first formulation, normalizer clamp). Utility linear attention also exists in `src/net/utils.py`.                     | `bdh_expansion` is configurable (commonly 2 in corrected configs). Avoid hard-coding claims like “128× expansion” unless your run config explicitly sets it.                                       |
| **Hybrid CNN Stem**                     | `HybridPatchEmbed` (in `src/net/tadiff_dit_arch.py`) builds a convolutional stem before patching.                                                                                                                            | Preserves local spatial features prior to tokenization; useful for medical images where locality matters.                                                                                          |
| **Conditioning Inputs**                 | The model accepts timestep, session-day, treatment, and target-session index (see `train.py`, `inference.py`, and `src/net/tadiff_dit_arch.py`). Embedding/modulation utilities include `timestep_embedding` and `modulate`. | Conditioning is deeply integrated through AdaLN-style modulation in transformer blocks.                                                                                                            |
| **Joint Image + Segmentation Support**  | Training code (`train.py`, `tadiff_model.py`) uses multiple output channels and mixes diffusion/image losses with auxiliary segmentation losses (Dice/BCE/SSIM/L1/Charbonnier/TV).                                           | The number of image/mask channels is configurable via `out_channels` and loss weights. Avoid hard-coded assertions like "3 image channels + 4 masks" unless you lock `out_channels` to that value. |
| **SEAL Adapter (Test-Time Adaptation)** | `src/net/seal_adapter.py` provides a SEALAdapter for test-time training with self-consistency checks (SSIM + variance) and limited steps of adaptation.                                                                      | This is a practical mechanism for per-scan adaptation; it is included and usable as-is.                                                                                                            |

---

### Evaluation Metrics

The model evaluates its longitudinal predictions across two critical clinical dimensions: **Image Reconstruction Quality** and **Tumor Segmentation Accuracy**.

#### Image Quality Metrics

| Metric   | Target   | Purpose                                                                                             |
| -------- | -------- | --------------------------------------------------------------------------------------------------- |
| **SSIM** | ↑ Higher | Measures structural perceptual similarity (luminance, contrast, structure).                         |
| **PSNR** | ↑ Higher | Evaluates signal fidelity (higher dB means reconstruction error is minimal relative to the signal). |
| **MAE**  | ↓ Lower  | Calculates the average pixel-level intensity deviation.                                             |

#### Tumor Segmentation Metrics

_These are evaluated across three confidence thresholds: `0.25` (lenient), `0.50` (standard), and `0.75` (strict)._

| Metric               | Target   | Purpose                                                                                                           |
| -------------------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| **Dice Coefficient** | ↑ Higher | Checks the exact voxel-level overlap between the predicted and ground-truth tumor masks.                          |
| **RAVD**             | → 0      | Relative Absolute Volume Difference. Positive indicates over-segmentation; negative indicates under-segmentation. |

---

## What Insights Does Our Project Reveal About BDH?

Integrating Baby Dragon Hatchling into a medical diffusion pipeline surfaces a clear gap between architectural promise and practical inference behavior — and shows exactly what it takes to close it.

### Sparse Linear Attention Is the Real Strength

The core contribution is sparse/linear attention behavior in the BDH blocks that reduces memory growth relative to standard attention and improves practical scalability.

### SEAL Extends BDH Into Adaptive Territory

SEAL (`src/net/seal_adapter.py`) adds test-time adaptation through self-consistency checks and lightweight update steps, enabling patient-specific adjustment during inference.

### From Sparsity to Full Anatomical Consistency

BDH attention improves efficiency while SEAL adds adaptive robustness. Together, they improve practical inference behavior on longitudinal multi-modal MRI slices.

---

## Future Scope

The following table outlines the prioritized focus areas for BDH's continued development:

| Priority | Focus Area                       | Key Action Items                                                                                                                                                           | Clinical & Technical Impact                                                                                                                                                                                                    |
| -------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1️⃣       | **Clinical Integration (BraTS)** | Build a reproducible preprocessing pipeline (skull-stripping, N4 bias correction, resampling) and patient-wise splits to track core metrics (Dice, HD95, SSIM).            | Standardized data would prevent the model from learning scanner artifacts.                                                                                                                                                     |
| 2️⃣       | **Inference Acceleration**       | Maximize the O(N) linear attention efficiency on consumer GPUs using FP16 mixed precision (AMP), ONNX quantization, and targeted Triton/CUDA QKV kernel fusion.            | Optimizations would allow the model to run on standard hospital GPUs.                                                                                                                                                          |
| 3️⃣       | **Uncertainty & Robustness**     | Implement Monte Carlo sampling on the reverse diffusion chain for per-pixel variance heatmaps, paired with ComBat intensity harmonization for cross-scanner reliability.   | Uncertainty heatmaps would flag ambiguous regions for doctors, while harmonization would ensure consistent performance across different hospital MRI machines.                                                                 |
| 4️⃣       | **Targeted Adaptation**          | Encode treatment schedules/doses as learnable embeddings, and fine-tune the SEAL adapter specifically on post-operative resection cases using synthetic cavity boundaries. | Treating time/dosage as continuous variables would allow the model to predict treatment trajectories, while training on synthetic surgical cavities would prevent the model from mistaking post-op fluid for recurring tumors. |

---

## How to Run Locally

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended; CPU works but will be slow)
- [PyTorch](https://pytorch.org/get-started/locally/) installed for your CUDA version

### Minimum Requirements

#### Inference

- Python 3.9+
- RAM: 16 GB minimum (32 GB recommended)
- GPU: 8 GB VRAM minimum for quick inference (12–16 GB recommended)

#### Training

- Python 3.9+
- RAM: 32 GB minimum (64 GB recommended)
- GPU: 24 GB VRAM practical minimum for default full architecture

### 1. Clone the Repository

```bash
git clone https://github.com/spandan11106/BDH-Explainer.git
cd BDH-Explainer/BDH-implementation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

Install PyTorch first (select the command matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)), then install the remaining requirements:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place your raw SAILOR NIfTI data and run the preprocessing script to convert it into `.npy` format:

```bash
python data/preproc_prepare_data.py \
    --root /path/to/sailor-raw \
    --output ./data/sailor_npy
```

### 5. Train the Model

```bash
python train.py --data_dir ./data/sailor_npy
```

Common training flags:

| Flag                        | Default      | Description                                    |
| --------------------------- | ------------ | ---------------------------------------------- |
| `--data_dir`                | _(required)_ | Path to preprocessed `.npy` data               |
| `--batch_size`              | `1`          | Batch size per GPU                             |
| `--max_epochs`              | `3000`       | Maximum training epochs                        |
| `--lr`                      | `1e-4`       | Learning rate                                  |
| `--hidden_size`             | `1024`       | Transformer hidden dimension                   |
| `--depth`                   | `18`         | Number of DiT blocks                           |
| `--patch_size`              | `8`          | Patch size (must divide `image_size`)          |
| `--bdh_expansion`           | `2`          | BDH sparse expansion factor                    |
| `--devices`                 | `1`          | Number of GPUs                                 |
| `--precision`               | `32`         | Training precision (`32` or `16-mixed`)        |
| `--accumulate_grad_batches` | `1`          | Gradient accumulation steps                    |
| `--lr_scheduler`            | `onecycle`   | LR scheduler (`onecycle`, `cosine`, `plateau`) |

Checkpoints and TensorBoard logs are saved to `./logs/<experiment_name>/` (default experiment name: `bdh-dit`).

### 6. Run Inference

```bash
python inference.py \
    --checkpoint ./logs/tadiff-dit/checkpoints/last.ckpt \
    --data_dir ./data/inference_npy \
    --patient_ids sub-03 sub-04 \
    --output_dir ./inference_results
```

Or run on a single patient file:

```bash
python inference.py \
    --checkpoint ./logs/tadiff-dit/checkpoints/last.ckpt \
    --patient_file ./data/inference_npy/sub-03_image.npy \
    --output_dir ./inference_results
```

```bash
python inference.py \
    --checkpoint ./logs/tadiff-dit/checkpoints/last.ckpt \
    --patient_file ./data/inference_npy/sub-04_image.npy \
    --output_dir ./inference_results
```

### 7. Monitor Training

```bash
tensorboard --logdir ./logs
```

### Command Verification (CLI)

The following command surfaces were checked successfully in this workspace:

```bash
python train.py --help
python inference.py --help
python data/preproc_prepare_data.py --help
python -m src.evaluation.test_pipeline --help
```

---
