"""
Data loading utilities for TaDiff model.

Provides MONAI-based data loading pipeline compatible with TaDiff-Net.
Handles dataset creation, transformations, and data loading operations.

Reference: https://github.com/samleoqh/TaDiff-Net/blob/main/src/data/data_loader.py
"""
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, MapTransform, 
    CenterSpatialCropd, SpatialPadd, CropForegroundd,
    RandFlipd, RandRotate90d, ToTensord
)


# ============================================================================
# Custom Transforms
# ============================================================================

class MergeMultiLabels(MapTransform):
    """Convert multi-class labels to binary labels.
    
    Merges all non-zero labels (1, 2, 3, ...) into a single label (1).
    This is useful for binary segmentation tasks.
    
    Args:
        keys: Keys of the corresponding items to be transformed
    """
    def __call__(self, data: Dict) -> Dict:
        """
        Args:
            data: Dictionary containing the data to transform
            
        Returns:
            Transformed data dictionary with binary labels
        """
        for key in self.key_iterator(data):
            data[key] = (data[key] > 0).astype(np.float32)
        return data


class PrepareSessionData(MapTransform):
    """Prepare session data for model input.
    
    Reshapes image data from (M*T, H, W, D) to (T, M, H, W, D) format
    where M=modalities and T=sessions/timepoints.
    
    Args:
        keys: Keys of the corresponding items to be transformed
        num_modalities: Number of MRI modalities (default: 4 for T1, T1c, FLAIR, T2)
    """
    def __init__(self, keys, num_modalities: int = 4):
        super().__init__(keys)
        self.num_modalities = num_modalities
    
    def __call__(self, data: Dict) -> Dict:
        for key in self.key_iterator(data):
            arr = data[key]
            # Input shape: (M*T, H, W, D)
            # Output shape: (T, M, H, W, D)
            total_channels = arr.shape[0]
            num_sessions = total_channels // self.num_modalities
            
            H, W, D = arr.shape[1:]
            reshaped = arr.reshape(self.num_modalities, num_sessions, H, W, D)
            data[key] = reshaped.transpose(1, 0, 2, 3, 4)  # (T, M, H, W, D)
        return data


class ExtractSlices(MapTransform):
    """Extract 2D slices from 3D volumes.
    
    Args:
        keys: Keys of the corresponding items to be transformed
        slice_indices: List of slice indices to extract, or None for all
        axis: Axis to slice along (default: -1 for depth axis)
    """
    def __init__(self, keys, slice_indices: Optional[List[int]] = None, axis: int = -1):
        super().__init__(keys)
        self.slice_indices = slice_indices
        self.axis = axis
    
    def __call__(self, data: Dict) -> Dict:
        for key in self.key_iterator(data):
            arr = data[key]
            if self.slice_indices is not None:
                data[key] = np.take(arr, self.slice_indices, axis=self.axis)
            # If None, keep all slices
        return data


class NormalizeTemporalData(MapTransform):
    """Normalize temporal conditioning data.
    
    Converts days to relative intervals and normalizes.
    
    Args:
        keys: Keys of the corresponding items to be transformed
        max_days: Maximum day value for normalization (default: 365)
    """
    def __init__(self, keys, max_days: float = 365.0):
        super().__init__(keys)
        self.max_days = max_days
    
    def __call__(self, data: Dict) -> Dict:
        for key in self.key_iterator(data):
            data[key] = data[key].astype(np.float32) / self.max_days
        return data


# ============================================================================
# Transform Pipelines
# ============================================================================

# Standard keys for SAILOR dataset
NPZ_KEYS = ['image', 'label', 'days', 'treatment']


def get_val_transforms(image_size: int = 192) -> Compose:
    """
    Create validation/test transformation pipeline.
    
    Args:
        image_size: Target spatial size for images
    
    Returns:
        Composed transformation pipeline
    """
    return Compose([
        LoadImaged(keys=NPZ_KEYS, image_only=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[image_size, image_size, image_size]),
        SpatialPadd(keys=["image", "label"], spatial_size=(image_size, image_size, image_size)),
        MergeMultiLabels(keys=["label"]),
    ])


def get_train_transforms(image_size: int = 192, augment: bool = True) -> Compose:
    """
    Create training transformation pipeline with optional augmentation.
    
    Args:
        image_size: Target spatial size for images
        augment: Whether to apply data augmentation
    
    Returns:
        Composed transformation pipeline
    """
    transforms = [
        LoadImaged(keys=NPZ_KEYS, image_only=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[image_size, image_size, image_size]),
        SpatialPadd(keys=["image", "label"], spatial_size=(image_size, image_size, image_size)),
        MergeMultiLabels(keys=["label"]),
    ]
    
    if augment:
        transforms.extend([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        ])
    
    return Compose(transforms)


# Keep backward compatibility
val_transforms = get_val_transforms()
npz_keys = NPZ_KEYS


def get_transform_pipeline() -> Compose:
    """Backward compatible alias for get_val_transforms."""
    return get_val_transforms()


# ============================================================================
# File Management
# ============================================================================

def get_test_files(
    data_root: str,
    patient_ids: List[str],
    prefix: str = '',
    keys: List[str] = None,
) -> List[Dict[str, str]]:
    """
    Get list of test files for specified patients.
    
    Args:
        data_root: Root directory containing .npy files
        patient_ids: List of patient IDs to process
        prefix: Optional filename prefix (e.g., 'sub-')
        keys: Data keys to load (default: ['image', 'label', 'days', 'treatment'])
    
    Returns:
        List[Dict[str, str]]: List of dictionaries with file paths
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    if keys is None:
        keys = NPZ_KEYS
    
    test_files = []
    
    for patient_id in patient_ids:
        file_dict = {}
        missing_files = []
        
        for key in keys:
            file_path = os.path.join(data_root, f'{prefix}{patient_id}_{key}.npy')
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            file_dict[key] = file_path
        
        if missing_files:
            print(f"Warning: Missing files for {patient_id}: {missing_files}")
            continue
            
        test_files.append(file_dict)
    
    return test_files


def scan_data_directory(data_root: str) -> List[str]:
    """
    Scan directory for available patient IDs.
    
    Args:
        data_root: Directory containing .npy files
    
    Returns:
        List of patient IDs found
    """
    if not os.path.exists(data_root):
        return []
    
    patient_ids = set()
    for filename in os.listdir(data_root):
        if filename.endswith('_image.npy'):
            patient_id = filename.replace('_image.npy', '')
            patient_ids.add(patient_id)
    
    return sorted(list(patient_ids))


# ============================================================================
# DataLoader Creation
# ============================================================================

def load_data(
    test_file_list: Optional[List[Dict[str, str]]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    cache_rate: float = 1.0,
    transforms: Optional[Compose] = None,
) -> DataLoader:
    """
    Create data loader for test data.
    
    Args:
        test_file_list: List of dictionaries containing file paths
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        cache_rate: Fraction of data to cache (0.0-1.0)
        transforms: Transform pipeline to use
                       
    Returns:
        DataLoader instance for the test dataset
    """
    if test_file_list is None:
        print('No test file list provided.')
        return None
    
    if transforms is None:
        transforms = get_val_transforms()
    
    test_dataset = CacheDataset(
        data=test_file_list, 
        transform=transforms,
        cache_rate=cache_rate,
    )
    
    return DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )


def create_dataloaders(
    data_root: str,
    patient_ids: Optional[List[str]] = None,
    train_split: float = 0.8,
    batch_size: int = 1,
    image_size: int = 192,
    augment: bool = True,
    cache_rate: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Directory containing .npy files
        patient_ids: List of patient IDs (auto-scanned if None)
        train_split: Fraction of data for training
        batch_size: Batch size
        image_size: Target spatial size
        augment: Whether to use augmentation for training
        cache_rate: Fraction of data to cache
        num_workers: Number of worker processes
        seed: Random seed for splitting
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if patient_ids is None:
        patient_ids = scan_data_directory(data_root)
    
    if not patient_ids:
        raise ValueError(f"No patients found in {data_root}")
    
    # Split patients
    np.random.seed(seed)
    shuffled_ids = patient_ids.copy()
    np.random.shuffle(shuffled_ids)
    
    split_idx = int(len(shuffled_ids) * train_split)
    train_ids = shuffled_ids[:split_idx]
    val_ids = shuffled_ids[split_idx:]
    
    print(f"Train patients: {len(train_ids)}, Val patients: {len(val_ids)}")
    
    # Get file lists
    train_files = get_test_files(data_root, train_ids)
    val_files = get_test_files(data_root, val_ids)
    
    # Create transforms
    train_transforms = get_train_transforms(image_size, augment)
    val_transforms_local = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
    )
    
    val_dataset = CacheDataset(
        data=val_files,
        transform=val_transforms_local,
        cache_rate=cache_rate,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


# ============================================================================
# Utility Functions
# ============================================================================

def prepare_batch_for_model(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    num_modalities: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch for model input.
    
    Reshapes images from (B, M*T, H, W, D) to model-ready format.
    
    Args:
        batch: Dictionary with 'image', 'label', 'days', 'treatment'
        device: Target device
        num_modalities: Number of modalities (default: 4)
    
    Returns:
        Prepared batch dictionary
    """
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    days = batch['days'].to(device)
    treatments = batch['treatment'].to(device)
    
    # Reshape images: (B, M*T, H, W, D) -> (B, T, M, H, W, D)
    B, total_channels, H, W, D = images.shape
    num_sessions = total_channels // num_modalities
    
    images = images.view(B, num_modalities, num_sessions, H, W, D)
    images = images.permute(0, 2, 1, 3, 4, 5)  # (B, T, M, H, W, D)
    
    return {
        'image': images,
        'label': labels,
        'days': days,
        'treatment': treatments,
        'num_sessions': num_sessions,
    }


def prepare_image_batch(
    images: torch.Tensor,
    num_sessions: int,
    num_modalities: int = 4,
    remove_t2: bool = True,
) -> torch.Tensor:
    """
    Prepare image batch for model input (TaDiff-Net compatible).
    
    Args:
        images: Input tensor of shape (B, M*T, H, W, D)
        num_sessions: Number of sessions
        num_modalities: Number of modalities
        remove_t2: Whether to remove T2 modality (keep T1, T1c, FLAIR)
    
    Returns:
        Prepared image tensor of shape (B, T, M-1, H, W, D) or (B, T, M, H, W, D)
    """
    B = images.shape[0]
    H, W, D = images.shape[2:]
    
    # Reshape to separate modalities and sessions
    images = images.view(B, num_modalities, num_sessions, H, W, D)
    
    # Permute to get sessions first: (B, T, M, H, W, D)
    images = images.permute(0, 2, 1, 3, 4, 5)
    
    # Remove T2 modality if requested (keep T1, T1c, FLAIR)
    if remove_t2:
        images = images[:, :, :-1, :, :, :]
    
    return images


def get_slice_at_index(
    batch: Dict[str, torch.Tensor],
    slice_idx: int,
    session_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract a 2D slice from batch at given indices.
    
    Args:
        batch: Prepared batch dictionary
        slice_idx: Z-axis slice index
        session_idx: Session/timepoint index
    
    Returns:
        Dictionary with 2D tensors
    """
    return {
        'image': batch['image'][:, session_idx, :, :, :, slice_idx],
        'label': batch['label'][:, session_idx, :, :, slice_idx],
        'days': batch['days'][:, session_idx],
        'treatment': batch['treatment'][:, session_idx],
    }
