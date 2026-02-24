# TaDiff-DiT: Treatment-Aware Diffusion Transformer

A medical image synthesis framework combining **Diffusion Transformers (DiT)** with **Sparse Linear Attention (BDH)** for treatment-aware longitudinal medical image generation, featuring **SEAL-based Test-Time Training** for continual adaptation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TaDiff-DiT Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, 12, 192, 192]  (4 sessions × 3 modalities: T1, T2, FLAIR)      │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────┐                                        │
│  │         PatchEmbed              │  Conv2d-based patchifier              │
│  │    [B,12,192,192] → [B,144,768] │  patch_size=16, 144 patches           │
│  └─────────────────────────────────┘                                        │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────┐                                        │
│  │   Sinusoidal Position Embed     │  2D position encoding                 │
│  └─────────────────────────────────┘                                        │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DiT Blocks (×12)                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  AdaLN-Zero + BDH Attention + MLP                             │  │   │
│  │  │                                                               │  │   │
│  │  │  • Adaptive Layer Norm with treatment conditioning            │  │   │
│  │  │  • BDH Sparse Linear Attention: O(N) complexity               │  │   │
│  │  │  • Feed-forward MLP with GELU                                 │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼              Treatment Conditioning                    │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │        FinalLayer               │◄─│  Timestep + Days + Treatment    │  │
│  │  Depatchify + AdaLN             │  │  Fourier Features → MLP → 768d  │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  Output: [B, 7, 192, 192]  (4 mask channels + 3 image channels)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🔷 BDH Sparse Linear Attention
- **O(N) complexity** instead of O(N²) standard attention
- Expansion → Sparsity (ReLU) → Associative Memory → Compression
- Enables efficient processing of high-resolution medical images

### 🔷 Treatment-Aware Conditioning
- Multi-session input (4 timepoints × 3 MRI modalities)
- Treatment code embedding (radiation, chemotherapy, etc.)
- Day interval encoding via Fourier features
- Target session masking for flexible prediction

### 🔷 SEAL Adapter (Test-Time Training)
- Self-Adapting framework for continual learning
- Per-sample gradient updates with consistency checking
- Amnesia detection and rollback mechanism
- Warmup + cosine LR scheduling

### 🔷 Gaussian Diffusion
- Linear and cosine noise schedules
- DDPM, DDIM, and DPM-Solver++ samplers
- Configurable timesteps (default T=1000)

## 📁 Project Structure

```
abcd/
├── src/
│   ├── net/
│   │   ├── tadiff_dit_arch.py    # Main DiT architecture (218M params)
│   │   ├── diffusion.py          # Gaussian diffusion process
│   │   ├── seal_adapter.py       # SEAL TTT/Continual Learning adapter
│   │   ├── ssim.py               # SSIM loss module
│   │   └── utils.py              # Utility functions
│   ├── data/
│   │   ├── data_loader.py        # MONAI-based data loading pipeline
│   │   └── sailor_dataset.py     # SAILOR dataset loader
│   ├── evaluation/
│   │   ├── metrics.py            # Evaluation metrics (SSIM, Dice, RAVD)
│   │   └── test_pipeline.py      # Connected test/inference pipeline
│   ├── utils/
│   │   └── image_processing.py   # Image processing utilities
│   ├── visualization/
│   │   └── visualizer.py         # Visualization tools
│   └── tadiff_model.py           # Model wrapper
├── config/
│   ├── arg_parse.py              # Argument parsing
│   ├── cfg_tadiff_net.py         # Model configuration
│   └── test_config.py            # Test/Train/Inference configuration
├── data/
│   ├── preproc_prepare_data.py   # SAILOR preprocessing from NIfTI
│   └── README.md                 # Data preparation instructions
├── tests/
│   ├── test_architecture_integration.py  # Comprehensive arch tests (33)
│   └── test_seal_adapter.py      # SEAL adapter tests (7)
├── train.py                      # Training script
├── inference.py                  # Basic inference script
├── inference_seal.py             # Inference with SEAL TTT
├── test.py                       # Unified test runner
└── README.md                     # This file
```

## 🔄 Data Preprocessing Pipeline

The project includes a comprehensive preprocessing pipeline compatible with TaDiff-Net for converting raw NIfTI MRI data into the SAILOR format.

### Preprocessing from Raw NIfTI Data

```bash
# Preprocess SAILOR dataset from raw NIfTI files
python data/preproc_prepare_data.py \
    --root /path/to/sailor-raw \
    --output ./data/sailor_npy \
    --modalities T1 T1c FLAIR T2 \
    --reorient  # Optional: reorient to RAS+

# Create synthetic test data for pipeline validation
python data/preproc_prepare_data.py \
    --synthetic \
    --output ./data/test_synthetic

# Validate preprocessed data
python data/preproc_prepare_data.py \
    --validate \
    --root ./data/sailor_npy
```

### MONAI-Based Data Loading

The data loader uses MONAI transforms for medical image handling:

```python
from src.data.data_loader import create_dataloaders, prepare_batch_for_model

# Create data loaders
train_loader, val_loader = create_dataloaders(
    data_dir="./data/sailor_npy",
    batch_size=4,
    num_workers=4,
    max_T=5,  # Maximum sessions per patient
    slice_size=(192, 192),
)

# Prepare batch for model input
for batch in train_loader:
    x, masks, days, treatment = prepare_batch_for_model(
        batch, 
        num_sessions=4, 
        num_modalities=4
    )
    # x: [B, 12, 192, 192] - input images
    # masks: [B, 4, 192, 192] - segmentation masks
    # days: [B, 4] - day intervals
    # treatment: [B, 4] - treatment codes
```

### Connected Test Pipeline

Run the connected pipeline test to verify preprocessing → data loading → model inference:

```bash
# Run full pipeline test
python test.py --pipeline

# Run from the test_pipeline module directly
python -m src.evaluation.test_pipeline --quick_test
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nishantNRC/abcd.git
cd abcd

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📋 Complete End-to-End Guide

This guide walks you through preparing your dataset, training the model, and running inference with continual learning.

### Step 1: Prepare Your Dataset

Your data must follow the **SAILOR format**. You can either:

**Option A: Use the preprocessing script (recommended for raw NIfTI data):**

```bash
# Preprocess SAILOR dataset from raw NIfTI files
python data/preproc_prepare_data.py \
    --root /path/to/sailor-raw \
    --output ./data/sailor_npy

# Expected raw data structure:
# sailor-raw/
# ├── sub-001/
# │   ├── ses-01/
# │   │   ├── sub-001_ses-01_T1.nii.gz
# │   │   ├── sub-001_ses-01_T1c.nii.gz
# │   │   ├── sub-001_ses-01_FLAIR.nii.gz
# │   │   ├── sub-001_ses-01_T2.nii.gz
# │   │   └── sub-001_ses-01_seg.nii.gz
# │   ├── ses-02/
# │   │   └── ...
# │   └── metadata.csv  # Contains days and treatment info
# └── sub-002/
#     └── ...
```

**Option B: Prepare .npy files directly:**

For each patient, create 4 `.npy` files:

```
your_data/
├── sub-001_image.npy      # Shape: (M×T, H, W, D) - MRI volumes
├── sub-001_label.npy      # Shape: (T, H, W, D) - Segmentation masks
├── sub-001_days.npy       # Shape: (T,) - Days since baseline
├── sub-001_treatment.npy  # Shape: (T,) - Treatment codes
├── sub-002_image.npy
├── sub-002_label.npy
...
```

#### Data Format Specification

| File | Shape | Description |
|------|-------|-------------|
| `*_image.npy` | `(M×T, H, W, D)` | M modalities × T sessions, stacked. E.g., 4 modalities × 5 sessions = shape `(20, 240, 240, 155)` |
| `*_label.npy` | `(T, H, W, D)` | Segmentation mask per session. Values: 0=background, 1=edema, 2=non-enhancing, 3=enhancing |
| `*_days.npy` | `(T,)` | Days since first session. E.g., `[0, 30, 60, 120, 180]` |
| `*_treatment.npy` | `(T,)` | Treatment code per session. 0=CRT (radiation), 1=TMZ (chemo), 2=combined |

**Example creation script:**
```python
import numpy as np

patient_id = "sub-001"
num_sessions = 5      # T = number of timepoints
num_modalities = 4    # M = T1, T1c, T2, FLAIR
H, W, D = 240, 240, 155  # Volume dimensions

# Create image: (M*T, H, W, D) = (20, 240, 240, 155)
image = np.zeros((num_modalities * num_sessions, H, W, D), dtype=np.float32)
for t in range(num_sessions):
    for m in range(num_modalities):
        image[t * num_modalities + m] = your_mri_data[t, m]  # Normalize to [0,1]

# Create label: (T, H, W, D)
label = np.zeros((num_sessions, H, W, D), dtype=np.float32)
for t in range(num_sessions):
    label[t] = your_segmentation_data[t]  # Values 0-3

# Create days: (T,)
days = np.array([0, 30, 60, 120, 180], dtype=np.float32)

# Create treatment: (T,)
treatment = np.array([0, 0, 1, 1, 1], dtype=np.float32)  # CRT then TMZ

# Save
np.save(f"your_data/{patient_id}_image.npy", image)
np.save(f"your_data/{patient_id}_label.npy", label)
np.save(f"your_data/{patient_id}_days.npy", days)
np.save(f"your_data/{patient_id}_treatment.npy", treatment)
```

### Step 2: Verify Data Loading

```bash
# Run data loading test with your data
python test.py --data

# Or test with synthetic data first
python -c "
import tempfile
import numpy as np
from pathlib import Path

# Generate synthetic test data
tmpdir = './data/test_synthetic'
Path(tmpdir).mkdir(parents=True, exist_ok=True)

patient_id = 'sub-01'
T, M, H, W, D = 4, 4, 64, 64, 32

np.save(f'{tmpdir}/{patient_id}_image.npy', np.random.rand(M*T, H, W, D).astype(np.float32))
np.save(f'{tmpdir}/{patient_id}_label.npy', np.random.randint(0, 4, (T, H, W, D)).astype(np.float32))
np.save(f'{tmpdir}/{patient_id}_days.npy', np.array([0, 30, 60, 90], dtype=np.float32))
np.save(f'{tmpdir}/{patient_id}_treatment.npy', np.array([0, 0, 1, 1], dtype=np.float32))
print(f'Created synthetic data in {tmpdir}')
"
```

### Step 3: Configure Training

Edit training parameters or pass via command line:

```python
# Key hyperparameters in train.py
TrainingConfig(
    # Data
    data_dir="/path/to/your_data",
    image_size=192,              # Resize to this resolution
    num_sessions=4,              # Number of timepoints to use
    
    # Model
    hidden_size=768,             # Transformer hidden dim
    depth=12,                    # Number of DiT blocks
    num_heads=12,                # Attention heads
    patch_size=16,               # Patch size for tokenization
    
    # Training
    batch_size=4,                # Adjust based on GPU memory
    learning_rate=1e-4,
    max_epochs=100,
    
    # Diffusion
    diffusion_steps=1000,
    schedule='cosine',           # 'linear' or 'cosine'
)
```

### Step 4: Run Training

```bash
# Basic training
python train.py \
    --data_dir /path/to/your_data \
    --output_dir ./logs \
    --batch_size 4 \
    --max_epochs 100 \
    --gpus 1

# Multi-GPU training
python train.py \
    --data_dir /path/to/your_data \
    --output_dir ./logs \
    --batch_size 8 \
    --max_epochs 100 \
    --gpus 4 \
    --strategy ddp

# Resume from checkpoint
python train.py \
    --data_dir /path/to/your_data \
    --resume_from ./logs/last.ckpt
```

**Expected output:**
```
[INFO] Loading SAILOR dataset from /path/to/your_data
[INFO] Found 50 patients, 2500 samples
[INFO] Train: 2000, Val: 500
[INFO] Model: TaDiff-DiT (218M params)
[INFO] Starting training...
Epoch 1/100: 100%|███████████████| 500/500 [05:23<00:00]
  train_loss: 0.0823, val_loss: 0.0912, val_ssim: 0.847
...
```

### Step 5: Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir ./logs/tensorboard

# Or use Weights & Biases (if configured)
wandb login
python train.py --use_wandb --wandb_project tadiff
```

### Step 6: Run Inference (Basic)

```bash
# Standard inference without adaptation
python inference.py \
    --checkpoint ./logs/best.ckpt \
    --data_dir /path/to/test_data \
    --output_dir ./results \
    --diffusion_steps 50
```

### Step 7: Run Inference with SEAL Continual Learning

```bash
# Inference with Test-Time Training (adapts to each patient)
python inference_seal.py \
    --checkpoint ./logs/best.ckpt \
    --data_dir /path/to/test_data \
    --output_dir ./results \
    --enable_ttt \
    --ttt_steps 3 \
    --ttt_lr 1e-4

# Process single patient with SEAL
python inference_seal.py \
    --checkpoint ./logs/best.ckpt \
    --patient_file /path/to/sub-001_image.npy \
    --output_dir ./results \
    --enable_ttt
```

**SEAL adaptation process:**
```
Processing sub-001...
  [TTT] Adapting to patient (3 steps)...
  [TTT] Step 1/3: loss=0.0821, confidence=0.912
  [TTT] Step 2/3: loss=0.0654, confidence=0.934
  [TTT] Step 3/3: loss=0.0598, confidence=0.948
  [TTT] Adaptation complete. Generating predictions...
  Saved: ./results/sub-001_pred.npy
```

### Step 8: Evaluate Results

```python
from src.evaluation.metrics import MetricsCalculator
import numpy as np
import torch

# Load predictions and ground truth
pred = torch.from_numpy(np.load("./results/sub-001_pred.npy"))
gt = torch.from_numpy(np.load("/path/to/sub-001_label.npy"))

# Calculate metrics
calc = MetricsCalculator(device='cuda')
metrics = calc.calculate_metrics(
    pred_img=pred[:, 4:7],   # Image channels
    gt_img=gt_images,
    pred_mask=pred[:, :4],   # Mask channels
    gt_mask=gt,
)
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"Dice: {metrics['dice_50']:.4f}")
```

---

## 🔄 Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TaDiff-DiT Training Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. PREPARE DATA                                                    │
│     └─ Convert MRI volumes to SAILOR format (.npy files)           │
│        • image: (M×T, H, W, D) - stacked modalities × sessions     │
│        • label: (T, H, W, D) - segmentation per session            │
│        • days: (T,) - days since baseline                          │
│        • treatment: (T,) - treatment codes                         │
│                                                                     │
│  2. VERIFY DATA                                                     │
│     └─ python test.py --data                                        │
│                                                                     │
│  3. TRAIN MODEL                                                     │
│     └─ python train.py --data_dir ./data --max_epochs 100          │
│                                                                     │
│  4. MONITOR                                                         │
│     └─ tensorboard --logdir ./logs/tensorboard                      │
│                                                                     │
│  5. INFERENCE (choose one)                                          │
│     ├─ Basic: python inference.py --checkpoint ./logs/best.ckpt    │
│     └─ SEAL:  python inference_seal.py --enable_ttt --ttt_steps 3  │
│                                                                     │
│  6. EVALUATE                                                        │
│     └─ Calculate SSIM, Dice, RAVD metrics                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Basic Usage

```python
import torch
from src.net.tadiff_dit_arch import TaDiff_DiT
from src.net.diffusion import GaussianDiffusion

# Initialize model
model = TaDiff_DiT(
    image_size=192,
    in_channels=12,      # 4 sessions × 3 modalities
    out_channels=7,      # 4 masks + 3 images
    hidden_size=768,
    depth=12,
    num_heads=12,
    patch_size=16,
)

# Initialize diffusion
diffusion = GaussianDiffusion(T=1000, schedule='cosine')

# Forward pass
x = torch.randn(2, 12, 192, 192)  # Input images
t = torch.randint(1, 1000, (2,)).float()  # Timesteps
intv_t = [torch.rand(2) * 300 for _ in range(4)]  # Day intervals
treat_code = [torch.randint(0, 3, (2,)).float() for _ in range(4)]  # Treatment codes
i_tg = -torch.ones(2, dtype=torch.int8)  # Target session

output = model(x, t, intv_t, treat_code, i_tg)
print(output.shape)  # [2, 7, 192, 192]
```

### Using SEAL Adapter for Test-Time Training

```python
from src.net.seal_adapter import SEALAdapter, SEALConfig

# Configure SEAL
config = SEALConfig(
    ttt_lr=1e-4,
    ttt_steps=3,
    consistency_threshold=0.1,
    enable_rollback=True,
    use_cosine_schedule=True,
)

# Wrap model with SEAL adapter
adapter = SEALAdapter(model, config=config, device='cuda')

# Adapt to new sample
test_input = torch.randn(1, 12, 192, 192)
adapted_output = adapter.adapt_and_predict(
    x=test_input,
    t=torch.tensor([500.0]),
    intv_t=[torch.tensor([100.0]) for _ in range(4)],
    treat_code=[torch.tensor([1.0]) for _ in range(4)],
    i_tg=torch.tensor([-1], dtype=torch.int8),
)
```

## 🧪 Running Tests

```bash
# Run all tests (quick mode)
python test.py --quick

# Run all tests including pipeline
python test.py --all --pipeline

# Run specific test suites
python test.py --arch        # Architecture tests only
python test.py --seal        # SEAL adapter tests only
python test.py --diffusion   # Diffusion tests only
python test.py --data        # Data loading tests only
python test.py --pipeline    # Connected pipeline tests

# Run pytest-based tests
pytest tests/ -v

# Run architecture integration tests
pytest tests/test_architecture_integration.py -v

# Run SEAL adapter tests
pytest tests/test_seal_adapter.py -v
```

### Test Coverage

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| Architecture Integration | 33 | Dimensions, gradients, edge cases, numerical stability |
| SEAL Adapter | 7 | State management, rollback, LR scheduling |
| Architecture Validation | 8 | Component tests, memory, batch sizes |
| Diffusion | 4 | Schedule, sampling, training loop |
| Connected Pipeline | 5 | Preprocessing, data loading, batch prep, forward pass |

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | 218M |
| **Hidden Size** | 768 |
| **Transformer Depth** | 12 blocks |
| **Attention Heads** | 12 |
| **Patch Size** | 16×16 |
| **Input Resolution** | 192×192 |
| **Input Channels** | 12 (4 sessions × 3 modalities) |
| **Output Channels** | 7 (4 masks + 3 images) |
| **Attention Type** | BDH Sparse Linear (O(N)) |

## 🔧 Configuration

### Model Configuration (`config/cfg_tadiff_net.py`)

```python
config = {
    'image_size': 192,
    'in_channels': 12,
    'out_channels': 7,
    'model_channels': 128,
    'hidden_size': 768,
    'depth': 12,
    'num_heads': 12,
    'patch_size': 16,
    'mlp_ratio': 4.0,
    'dropout': 0.0,
    'use_checkpoint': False,
}
```

### SEAL Configuration

```python
seal_config = SEALConfig(
    ttt_lr=1e-4,              # Test-time training learning rate
    ttt_steps=3,              # Gradient steps per sample
    consistency_threshold=0.1, # Threshold for consistency check
    enable_rollback=True,     # Enable amnesia rollback
    warmup_steps=5,           # LR warmup steps
    use_cosine_schedule=True, # Cosine annealing
)
```

## 📈 Training

Using PyTorch Lightning:

```python
from src.tadiff_model import TaDiffLitModel
import pytorch_lightning as pl

# Initialize Lightning model
lit_model = TaDiffLitModel(config)

# Train
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
)
trainer.fit(lit_model, train_dataloader, val_dataloader)
```

## 🔬 Research Background

This project implements:

1. **Diffusion Transformers (DiT)** - [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
2. **BDH Attention** - Baby Dragon Hatchling Sparse Linear Attention for O(N) complexity
3. **SEAL Framework** - [Self-Adapting Language Models](https://github.com/Continual-Intelligence/SEAL) adapted for diffusion models
4. **Treatment-Aware Synthesis** - Longitudinal medical image prediction with treatment conditioning

## 📝 Citation

```bibtex
@software{tadiff_dit_2026,
  title={TaDiff-DiT: Treatment-Aware Diffusion Transformer},
  author={TaDiff-Net Project},
  year={2026},
  url={https://github.com/nishantNRC/abcd}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Note**: This is the `tadiff-continual` branch featuring SEAL-based continual learning capabilities.
