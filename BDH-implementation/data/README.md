# Data Preprocessing for TaDiff-DiT

This directory contains the preprocessing pipeline for converting raw SAILOR MRI data into the `.npy` format required by TaDiff-DiT.

## Overview

The preprocessing script `preproc_prepare_data.py` handles:
- Loading NIfTI (.nii.gz) MRI volumes
- Normalizing image intensities
- Stacking multiple modalities (T1, T1c, FLAIR, T2)
- Merging segmentation labels
- Saving in the SAILOR format

## Usage

### 1. Preprocess Raw NIfTI Data

```bash
python preproc_prepare_data.py \
    --root /path/to/sailor-raw \
    --output ./sailor_npy \
    --modalities T1 T1c FLAIR T2
```

### 2. Create Synthetic Test Data

```bash
python preproc_prepare_data.py \
    --synthetic \
    --output ./test_synthetic
```

### 3. Validate Processed Data

```bash
python preproc_prepare_data.py \
    --validate \
    --root ./sailor_npy
```

## Expected Raw Data Structure

```
sailor-raw/
├── sub-001/
│   ├── ses-01/
│   │   ├── sub-001_ses-01_T1.nii.gz       # T1-weighted MRI
│   │   ├── sub-001_ses-01_T1c.nii.gz      # T1 contrast-enhanced
│   │   ├── sub-001_ses-01_FLAIR.nii.gz    # FLAIR sequence
│   │   ├── sub-001_ses-01_T2.nii.gz       # T2-weighted MRI
│   │   └── sub-001_ses-01_seg.nii.gz      # Segmentation mask
│   ├── ses-02/
│   │   └── ...
│   ├── ses-03/
│   │   └── ...
│   └── metadata.csv
├── sub-002/
│   └── ...
└── participants.csv
```

### metadata.csv Format

```csv
session,days_from_baseline,treatment_code
ses-01,0,0
ses-02,30,0
ses-03,60,1
ses-04,120,1
```

### Treatment Codes
- `0`: CRT (Concurrent Radiotherapy)
- `1`: TMZ (Temozolomide chemotherapy)
- `2`: Combined treatment

## Output Format

For each patient, 4 `.npy` files are created:

| File | Shape | Description |
|------|-------|-------------|
| `sub-XXX_image.npy` | `(M×T, H, W, D)` | M modalities × T sessions stacked |
| `sub-XXX_label.npy` | `(T, H, W, D)` | Segmentation mask per session |
| `sub-XXX_days.npy` | `(T,)` | Days since first session |
| `sub-XXX_treatment.npy` | `(T,)` | Treatment code per session |

### Image Dimensions
- **M**: Number of modalities (default: 4 for T1, T1c, FLAIR, T2)
- **T**: Number of sessions/timepoints
- **H, W, D**: Spatial dimensions (typically 240×240×155 for brain MRI)

### Segmentation Labels
- `0`: Background
- `1`: Peritumoral edema
- `2`: Non-enhancing tumor core
- `3`: Enhancing tumor

## Data Loading

Once preprocessed, load data using the MONAI pipeline:

```python
from src.data.data_loader import create_dataloaders, prepare_batch_for_model

# Create data loaders
train_loader, val_loader = create_dataloaders(
    data_dir="./sailor_npy",
    batch_size=4,
    num_workers=4,
    max_T=5,
    slice_size=(192, 192),
)

# Prepare batch for model
for batch in train_loader:
    x, masks, days, treatment = prepare_batch_for_model(
        batch,
        num_sessions=4,
        num_modalities=4
    )
    # Use in model training
```

## Testing the Pipeline

Run the connected pipeline test:

```bash
# From project root
python test.py --pipeline

# Or directly
python -m src.evaluation.test_pipeline --quick_test
```

## Notes

1. **Memory Usage**: Large datasets may require significant RAM. Consider processing in batches.

2. **Normalization**: Images are normalized to [0, 1] range using non-zero voxel statistics.

3. **Orientation**: Use `--reorient` flag to reorient volumes to RAS+ (requires ANTsPy).

4. **Dependencies**: The preprocessing requires `nibabel`, `numpy`, `pandas`, and optionally `ants` for reorientation.
