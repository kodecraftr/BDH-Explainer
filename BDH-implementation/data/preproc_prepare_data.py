"""
SAILOR Dataset Preprocessing Script

This script preprocesses SAILOR (brain tumor MRI) data by:
- Reading and reorienting NIfTI files
- Normalizing image intensities
- Merging segmentation masks
- Saving processed data as numpy arrays

Compatible with TaDiff-Net preprocessing pipeline.
For reference: https://github.com/samleoqh/TaDiff-Net/blob/main/data/preproc_prepare_data.py

Dataset: SAILOR MNI v2
Download: https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8
"""

import os
import argparse
import numpy as np
import pandas as pd

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Install with: pip install nibabel")

try:
    from reorient_nii import reorient
    HAS_REORIENT = True
except ImportError:
    HAS_REORIENT = False
    print("Warning: reorient_nii not installed. Install with: pip install reorient-nii")

# ============================================================================
# Configuration
# ============================================================================

# Segmentation label indices
BG = 0          # background
EDEMA = 1       # edema
NECROTIC = 2    # necrosis (reserved, not used in SAILOR)
ENHANCING = 3   # enhancing tumor

# File naming conventions for SAILOR dataset
KEY_FILENAMES = {
    "edema_mask": "EdemaMask-CL.nii.gz",
    "et_mask": "ContrastEnhancedMask-CL.nii.gz",
    "t1": "T1.nii.gz",
    "t1c": "T1c.nii.gz",
    "flair": "Flair.nii.gz",
    "t2": "T2.nii.gz",
}

# Modality order: T1, T1c, FLAIR, T2
MODALITIES = ['t1', 't1c', 'flair', 't2']

# Patient ID lists (total 27 patients in raw SAILOR dataset)
ALL_PATIENT_IDS = [f'sub-{i:02d}' for i in range(1, 28)]

# Valid patients (excluding those with missing/incorrect data)
VALID_PATIENT_IDS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-09',
    'sub-10', 'sub-11', 'sub-13', 'sub-14', 'sub-16', 'sub-17', 'sub-27',
    'sub-18', 'sub-23', 'sub-26', 'sub-05', 'sub-08', 'sub-12', 'sub-15',
    'sub-20', 'sub-22', 'sub-21', 'sub-25', 'sub-19', 'sub-24',
]


# ============================================================================
# NIfTI I/O Functions
# ============================================================================

def read_nii(path: str, orientation_axcode: str = 'PLI', 
             non_zero_norm: bool = False, clip_percent: float = 0.1) -> np.ndarray:
    """
    Read NIfTI file and return the data array.
    
    Args:
        path: Path to the NIfTI file
        orientation_axcode: Target orientation (e.g., 'PLI', 'RAS', 'LPI')
        non_zero_norm: Whether to normalize non-zero values
        clip_percent: Percentile for intensity clipping (0-0.5)
    
    Returns:
        np.ndarray: Image data array
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    img = nib.load(path)
    
    if HAS_REORIENT:
        img = reorient(img, orientation_axcode)
    
    img_data = img.get_fdata()
    
    if non_zero_norm:
        img_data = nonzero_norm_image(img_data, clip_percent=clip_percent)
    
    return img_data


def nonzero_norm_image(image: np.ndarray, clip_percent: float = 0.1) -> np.ndarray:
    """
    Normalize image based on non-zero values with optional intensity clipping.
    
    Args:
        image: Input image
        clip_percent: Percentile for clipping outliers (0-0.5)
    
    Returns:
        np.ndarray: Normalized image in range [0, 1]
    """
    assert 0 <= clip_percent <= 0.5, f"clip_percent must be in [0, 0.5], got {clip_percent}"
    
    nz_mask = image > 0
    
    if image[nz_mask].size == 0:
        print(f"Warning: Image has no non-zero values. Shape: {image.shape}, "
              f"Min: {image.min():.4f}, Max: {image.max():.4f}")
        return image
    
    # Clip outliers if specified
    if clip_percent > 0:
        minval = np.percentile(image[nz_mask], clip_percent)
        maxval = np.percentile(image[nz_mask], 100 - clip_percent)
        image[nz_mask & (image < minval)] = minval
        image[nz_mask & (image > maxval)] = maxval
    
    # Z-score normalization
    y = image[nz_mask]
    image_mean = np.mean(y)
    image_std = np.std(y)
    
    if image_std == 0.0:
        print(f"Warning: Image std is zero")
        return image
    
    image = (image - image_mean) / image_std
    
    # Scale to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image


# ============================================================================
# Session and File Management
# ============================================================================

def get_session_list(file_csv: str) -> tuple:
    """
    Load time sessions and intervals information from CSV file.
    
    Args:
        file_csv: Path to the CSV file containing patient information
    
    Returns:
        tuple: (times_list, treatment_list) dictionaries keyed by patient ID
            - times_list: Cumulative days from baseline for each session
            - treatment_list: Treatment type (0: CRT, 1: TMZ)
    """
    times_list = {}
    treatment_list = {}
    
    df = pd.read_csv(file_csv)
    df.set_index("patients", inplace=True)
    
    for patient_id in df.index:
        # Parse interval days string to array
        inv_times = df['interval_days'].loc[patient_id]
        time_inv = np.array([int(s) for s in inv_times[1:-1].split(",")])
        
        # Insert baseline (day 0)
        times_list[patient_id] = np.insert(time_inv, 0, 0)
        
        # Treatment encoding: 0=CRT (sessions 0-3), 1=TMZ (sessions 4+)
        treatment_list[patient_id] = np.array(
            [1 if i > 3 else 0 for i in range(len(time_inv) + 1)]
        )
    
    return times_list, treatment_list


def get_file_dict(patient_ids: list, root: str) -> dict:
    """
    Create a nested dictionary of file paths for all patients and sessions.
    
    Args:
        patient_ids: List of patient IDs to process
        root: Root path to SAILOR raw data
    
    Returns:
        dict: Nested dict structure:
            {patient_id: {session_id: {modality: filepath}}}
    """
    file_dict = {}
    
    for patient_id in patient_ids:
        sessions = {}
        patient_path = os.path.join(root, patient_id)
        
        if not os.path.exists(patient_path):
            print(f"Warning: Patient path not found: {patient_path}")
            continue
        
        # Get all session directories
        session_ids = sorted([
            os.path.basename(f.path)
            for f in os.scandir(patient_path)
            if f.is_dir()
        ])
        
        # Build file paths for each session
        for session_id in session_ids:
            files = {
                key: os.path.join(root, patient_id, session_id, filename)
                for key, filename in KEY_FILENAMES.items()
            }
            sessions[session_id] = files
        
        if sessions:
            file_dict[patient_id] = sessions
    
    return file_dict


# ============================================================================
# Main Preprocessing Function
# ============================================================================

def save_session_data(file_dict: dict = None, save_path: str = './sailor_npy', 
                      root: str = None, info_csv: str = None):
    """
    Process and save all patient data as numpy arrays.
    
    For each patient, saves:
        - {patient_id}_image.npy: (M*T, H, W, D) array of MRI modalities
        - {patient_id}_label.npy: (T, H, W, D) array of segmentation masks
        - {patient_id}_days.npy: (T,) array of cumulative days
        - {patient_id}_treatment.npy: (T,) array of treatment types
    
    Where M=4 modalities, T=number of timepoints, H,W,D=spatial dimensions
    
    Args:
        file_dict: File path dictionary from get_file_dict()
        save_path: Output directory for numpy files
        root: Root path to SAILOR raw data
        info_csv: Path to sailor_info.csv
    """
    if file_dict is None:
        print("Error: file_dict is required")
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    # Load session info
    if info_csv is None:
        info_csv = os.path.join(root, "sailor_info.csv")
    
    session_list, treatment_list = get_session_list(info_csv)
    
    patient_ids = list(file_dict.keys())
    
    for patient_id in patient_ids:
        print(f'Processing {patient_id}...')
        
        session_ids = sorted(list(file_dict[patient_id].keys()))
        
        images = []
        labels = []
        
        # Process each modality and session
        for i, modality in enumerate(MODALITIES):
            for session_id in session_ids:
                # Load and normalize image
                img_path = file_dict[patient_id][session_id][modality]
                
                if not os.path.exists(img_path):
                    print(f"  Warning: File not found: {img_path}")
                    continue
                    
                img = read_nii(img_path, non_zero_norm=True, clip_percent=0.2)
                images.append(img)
                
                # Load and merge segmentation masks (only once per session)
                if i == 0:
                    merged_labels = np.zeros_like(img)
                    
                    # Edema mask
                    edema_path = file_dict[patient_id][session_id]['edema_mask']
                    if os.path.exists(edema_path):
                        edema = read_nii(edema_path)
                        merged_labels[edema > 0] = EDEMA
                    
                    # Enhancing tumor mask (overwrites edema)
                    et_path = file_dict[patient_id][session_id]['et_mask']
                    if os.path.exists(et_path):
                        et = read_nii(et_path)
                        merged_labels[et > 0] = ENHANCING
                    
                    labels.append(merged_labels.astype(np.int8))
        
        if not images:
            print(f"  No images found for {patient_id}")
            continue
        
        # Stack and save
        image = np.stack(images, axis=0).astype(np.float32)
        label = np.stack(labels, axis=0).astype(np.int8)
        
        # Calculate cumulative days
        if patient_id in session_list:
            day_interval = session_list[patient_id]
            days = np.array([sum(day_interval[:i+1]) for i in range(len(day_interval))])
            days = days[:len(labels)]  # Match number of sessions
        else:
            # Default: 30-day intervals
            days = np.arange(len(labels)) * 30
        
        if patient_id in treatment_list:
            treatment = treatment_list[patient_id][:len(labels)]
        else:
            # Default: CRT for first 4, TMZ after
            treatment = np.array([1 if i > 3 else 0 for i in range(len(labels))])
        
        # Save to disk
        np.save(os.path.join(save_path, f"{patient_id}_image.npy"), image)
        np.save(os.path.join(save_path, f"{patient_id}_label.npy"), label)
        np.save(os.path.join(save_path, f"{patient_id}_days.npy"), days)
        np.save(os.path.join(save_path, f"{patient_id}_treatment.npy"), treatment)
        
        print(f"  Saved: Image {image.shape}, Label {label.shape}, "
              f"Days {days.shape}, Treatment {treatment.shape}")
    
    print(f"\nPreprocessing complete! Data saved to: {save_path}")


# ============================================================================
# Synthetic Data Generation (for testing)
# ============================================================================

def create_synthetic_patient(
    save_path: str,
    patient_id: str = "sub-99",
    num_sessions: int = 4,
    shape: tuple = (192, 192, 155),
    num_modalities: int = 4,
) -> dict:
    """
    Create synthetic patient data for testing purposes.
    
    Args:
        save_path: Directory to save the data
        patient_id: Patient identifier
        num_sessions: Number of sessions/timepoints
        shape: Spatial dimensions (H, W, D)
        num_modalities: Number of MRI modalities
    
    Returns:
        dict: Dictionary with file paths
    """
    os.makedirs(save_path, exist_ok=True)
    
    H, W, D = shape
    
    # Create synthetic MRI scans
    # Shape: (M * T, H, W, D)
    image = np.random.rand(num_modalities * num_sessions, H, W, D).astype(np.float32)
    
    # Create synthetic segmentation masks
    # Shape: (T, H, W, D)
    label = np.zeros((num_sessions, H, W, D), dtype=np.int8)
    
    # Add synthetic tumor in center
    center_h, center_w, center_d = H // 2, W // 2, D // 2
    tumor_size = 20
    for t in range(num_sessions):
        # Tumor shrinks over time (treatment response)
        size = max(5, tumor_size - t * 3)
        h_slice = slice(center_h - size, center_h + size)
        w_slice = slice(center_w - size, center_w + size)
        d_slice = slice(center_d - size, center_d + size)
        
        # Edema
        label[t, h_slice, w_slice, d_slice] = EDEMA
        
        # Enhancing core (smaller)
        core_size = size // 2
        h_core = slice(center_h - core_size, center_h + core_size)
        w_core = slice(center_w - core_size, center_w + core_size)
        d_core = slice(center_d - core_size, center_d + core_size)
        label[t, h_core, w_core, d_core] = ENHANCING
    
    # Days (cumulative)
    days = np.cumsum([0] + [30] * (num_sessions - 1)).astype(np.float32)
    
    # Treatment (0=CRT for first sessions, 1=TMZ for later)
    treatment = np.array([0 if i < 4 else 1 for i in range(num_sessions)], dtype=np.float32)
    
    # Save files
    files = {
        'image': os.path.join(save_path, f"{patient_id}_image.npy"),
        'label': os.path.join(save_path, f"{patient_id}_label.npy"),
        'days': os.path.join(save_path, f"{patient_id}_days.npy"),
        'treatment': os.path.join(save_path, f"{patient_id}_treatment.npy"),
    }
    
    np.save(files['image'], image)
    np.save(files['label'], label)
    np.save(files['days'], days)
    np.save(files['treatment'], treatment)
    
    print(f"Created synthetic patient {patient_id}:")
    print(f"  Image: {image.shape}")
    print(f"  Label: {label.shape}")
    print(f"  Days: {days}")
    print(f"  Treatment: {treatment}")
    
    return files


def create_synthetic_dataset(
    save_path: str,
    num_patients: int = 3,
    num_sessions: int = 4,
    shape: tuple = (64, 64, 32),
) -> list:
    """
    Create a synthetic dataset with multiple patients.
    
    Args:
        save_path: Directory to save the data
        num_patients: Number of patients to create
        num_sessions: Number of sessions per patient
        shape: Spatial dimensions
    
    Returns:
        list: List of patient IDs created
    """
    os.makedirs(save_path, exist_ok=True)
    
    patient_ids = []
    for i in range(num_patients):
        patient_id = f"sub-{i+1:02d}"
        create_synthetic_patient(
            save_path=save_path,
            patient_id=patient_id,
            num_sessions=num_sessions,
            shape=shape,
        )
        patient_ids.append(patient_id)
    
    print(f"\nCreated {num_patients} synthetic patients in {save_path}")
    return patient_ids


# ============================================================================
# Validation Functions
# ============================================================================

def validate_processed_data(data_dir: str, patient_id: str) -> bool:
    """
    Validate preprocessed data for a patient.
    
    Args:
        data_dir: Directory containing processed .npy files
        patient_id: Patient identifier
    
    Returns:
        bool: True if data is valid
    """
    files = {
        'image': os.path.join(data_dir, f"{patient_id}_image.npy"),
        'label': os.path.join(data_dir, f"{patient_id}_label.npy"),
        'days': os.path.join(data_dir, f"{patient_id}_days.npy"),
        'treatment': os.path.join(data_dir, f"{patient_id}_treatment.npy"),
    }
    
    # Check all files exist
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"✗ Missing {key} file: {path}")
            return False
    
    # Load and validate
    try:
        img = np.load(files['image'])
        lbl = np.load(files['label'])
        days = np.load(files['days'])
        treatment = np.load(files['treatment'])
        
        # Check shapes
        print(f"  Image: {img.shape}")  # e.g., (24, 240, 240, 155) = 4 modalities × 6 sessions
        print(f"  Label: {lbl.shape}")  # e.g., (6, 240, 240, 155) = 6 sessions
        print(f"  Days: {days.shape}")
        print(f"  Treatment: {treatment.shape}")
        
        # Check value ranges
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")  # Should be [0, 1]
        print(f"  Label values: {np.unique(lbl)}")  # Should be [0, 1, 3]
        
        # Check for NaN/Inf
        assert not np.isnan(img).any(), "NaN detected in images"
        assert not np.isinf(img).any(), "Inf detected in images"
        
        # Verify consistency
        num_sessions = lbl.shape[0]
        assert img.shape[0] == 4 * num_sessions, "Image shape mismatch"
        assert days.shape[0] == num_sessions, "Days shape mismatch"
        assert treatment.shape[0] == num_sessions, "Treatment shape mismatch"
        
        print(f"✓ {patient_id} validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAILOR Dataset Preprocessing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preprocess SAILOR raw data
    python preproc_prepare_data.py --root /path/to/sailor-raw --output ./data/sailor_npy
    
    # Create synthetic test data
    python preproc_prepare_data.py --synthetic --output ./data/synthetic
    
    # Validate processed data
    python preproc_prepare_data.py --validate --data_dir ./data/sailor_npy --patient sub-17
        """
    )
    
    parser.add_argument('--root', type=str, default=None,
                        help='Path to SAILOR raw data directory')
    parser.add_argument('--output', type=str, default='./data/sailor_npy',
                        help='Output directory for processed data')
    parser.add_argument('--patients', type=str, nargs='+', default=None,
                        help='Specific patient IDs to process')
    parser.add_argument('--synthetic', action='store_true',
                        help='Create synthetic test data instead')
    parser.add_argument('--num_patients', type=int, default=3,
                        help='Number of synthetic patients to create')
    parser.add_argument('--validate', action='store_true',
                        help='Validate processed data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for validation')
    parser.add_argument('--patient', type=str, default=None,
                        help='Patient ID for validation')
    
    args = parser.parse_args()
    
    if args.synthetic:
        print("Creating synthetic dataset...")
        create_synthetic_dataset(
            save_path=args.output,
            num_patients=args.num_patients,
            shape=(64, 64, 32),
        )
    elif args.validate:
        if args.data_dir is None or args.patient is None:
            print("Error: --data_dir and --patient required for validation")
            return
        print(f"Validating {args.patient} in {args.data_dir}...")
        validate_processed_data(args.data_dir, args.patient)
    else:
        if args.root is None:
            print("Error: --root is required for preprocessing")
            print("Use --synthetic to create test data instead")
            return
        
        patient_ids = args.patients if args.patients else VALID_PATIENT_IDS
        
        print(f"Preprocessing SAILOR data from: {args.root}")
        print(f"Patients: {patient_ids}")
        
        file_dict = get_file_dict(patient_ids, args.root)
        print(f"Found {len(file_dict)} patients")
        
        save_session_data(
            file_dict=file_dict,
            save_path=args.output,
            root=args.root,
        )


if __name__ == "__main__":
    main()
