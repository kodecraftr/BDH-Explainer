# Data Preprocessing 

This directory contains the preprocessing pipeline for converting raw SAILOR MRI data into the `.npy` format required by this model.

## Overview

The preprocessing script `preproc_prepare_data.py` handles:

- Loading NIfTI (.nii.gz) MRI volumes
- Normalizing image intensities
- Stacking multiple modalities (T1, T1c, FLAIR, T2)
- Merging segmentation labels
- Saving in the SAILOR format

## Minimum Requirements

- Python 3.9+
- RAM: 16 GB minimum (32 GB recommended for larger cohorts)
- Disk: at least 10–20 GB free for intermediate `.npy` outputs
- Required Python packages: `numpy`, `pandas`, `nibabel`
- Optional package: `reorient-nii` (for NIfTI reorientation support)

## Usage

### 1. Preprocess Raw NIfTI Data

```bash
python data/preproc_prepare_data.py \
    --root /path/to/sailor-raw \
    --output ./data/sailor_npy
```

### 2. Create Synthetic Test Data

```bash
python data/preproc_prepare_data.py \
    --synthetic \
    --output ./data/test_synthetic
```

### 3. Validate Processed Data

```bash
python data/preproc_prepare_data.py \
    --validate \
    --data_dir ./data/sailor_npy \
    --patient sub-17
```

## Expected Raw Data Structure

```
sailor-raw/
├── sub-01/
│   ├── ses-01/
│   │   ├── T1.nii.gz
│   │   ├── T1c.nii.gz
│   │   ├── Flair.nii.gz
│   │   ├── T2.nii.gz
│   │   ├── EdemaMask-CL.nii.gz
│   │   └── ContrastEnhancedMask-CL.nii.gz
│   ├── ses-02/
│   │   └── ...
│   ├── ses-03/
│   │   └── ...
├── sub-02/
│   └── ...
└── sailor_info.csv
```

### sailor_info.csv Format

```csv
patients,interval_days
sub-01,"[30,30,60]"
sub-02,"[45,30,30]"
```

### If `sailor_info.csv` is in another location

Current CLI behavior in `data/preproc_prepare_data.py` loads metadata from:

`<root>/sailor_info.csv`

So if your CSV is stored elsewhere, use one of these options before preprocessing:

- Copy it into your selected `--root` folder as `sailor_info.csv`.
- Create a symbolic link named `sailor_info.csv` inside `--root`.
- Or edit `save_session_data(..., info_csv=...)` in the script to pass your external path.

### Treatment Codes

- `0`: baseline/CRT phase (early sessions)
- `1`: TMZ phase (later sessions)

Note: treatment array is generated from session index in preprocessing unless your metadata workflow is customized.

## Output Format

For each patient, 4 `.npy` files are created:

| File                    | Shape            | Description                       |
| ----------------------- | ---------------- | --------------------------------- |
| `sub-XXX_image.npy`     | `(M×T, H, W, D)` | M modalities × T sessions stacked |
| `sub-XXX_label.npy`     | `(T, H, W, D)`   | Segmentation mask per session     |
| `sub-XXX_days.npy`      | `(T,)`           | Days since first session          |
| `sub-XXX_treatment.npy` | `(T,)`           | Treatment code per session        |

### Image Dimensions

- **M**: Number of modalities (default: 4 for T1, T1c, FLAIR, T2)
- **T**: Number of sessions/timepoints
- **H, W, D**: Spatial dimensions (typically 240×240×155 for brain MRI)

### Segmentation Labels

- `0`: Background
- `1`: Peritumoral edema
- `3`: Enhancing tumor

Note: label `2` is reserved in constants, but the current merge path writes edema (`1`) and enhancing (`3`).

## Data Loading

Once preprocessed, load data using the MONAI pipeline:

```python
from src.data.data_loader import create_dataloaders, prepare_batch_for_model
import torch

# Create data loaders
train_loader, val_loader = create_dataloaders(
    data_root="./data/sailor_npy",
    batch_size=4,
    num_workers=4,
    image_size=192,
)

# Prepare batch for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch in train_loader:
    prepared = prepare_batch_for_model(
        batch,
        device=device,
        num_modalities=4
    )
    # prepared['image'], prepared['label'], prepared['days'], prepared['treatment']
```

## Testing the Pipeline

Run the connected pipeline test:

```bash
# From project root
python -m src.evaluation.test_pipeline --quick_test

# Data-loader focused check
python -m src.evaluation.test_pipeline --test_data
```

## Notes

1. **Memory Usage**: Large datasets may require significant RAM. Consider processing in batches.

2. **Normalization**: Preprocessing uses non-zero normalization (z-score then [0, 1] scaling).

3. **Orientation**: Reorientation is handled only if `reorient-nii` is installed; there is no `--reorient` CLI flag in this script.

4. **Dependencies**: The preprocessing requires `nibabel`, `numpy`, `pandas`, and optionally `reorient-nii` for reorientation.

5. **Metadata Path**: During CLI preprocessing, `sailor_info.csv` is expected under `--root`.
