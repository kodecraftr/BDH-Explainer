"""
SAILOR Dataset for DIT-BDH Training 

Dataset format (per patient):
- {patient_id}_image.npy: Shape (M×T, H, W, D) - 4 modalities × T sessions
- {patient_id}_label.npy: Shape (T, H, W, D) - Segmentation masks  
- {patient_id}_days.npy: Shape (T,) - Cumulative days from baseline
- {patient_id}_treatment.npy: Shape (T,) - Treatment type per session
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SAILORDataset(Dataset):
    """
    PyTorch Dataset for SAILOR brain tumor MRI data 
    
    """
    
    # SAILOR modality indices: 0=T1, 1=T1c, 2=FLAIR, 3=T2
    MODALITY_NAMES = ['T1', 'T1c', 'FLAIR', 'T2']
    DEFAULT_MODALITIES = [1, 2, 3]  # T1c, FLAIR, T2 (skip T1)
    
    def __init__(
        self,
        data_dir: str,
        num_sessions: int = 4,
        num_modalities: int = 3,
        image_size: int = 192,
        num_slices_per_volume: int = 10,
        modality_indices: Optional[List[int]] = None,
        mode: str = 'train',
        augment: bool = True,
        slice_range: Optional[Tuple[int, int]] = None,
        min_tumor_slices: int = 5,
        cache_data: bool = False,
        normalize: bool = True,  # NEW: Control normalization
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.num_sessions = num_sessions
        self.num_modalities = num_modalities
        self.image_size = image_size
        self.num_slices_per_volume = num_slices_per_volume
        self.modality_indices = modality_indices or self.DEFAULT_MODALITIES
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.slice_range = slice_range
        self.min_tumor_slices = min_tumor_slices
        self.cache_data = cache_data
        self.normalize = normalize
        
        # Find all patient files
        self.patient_ids = self._find_patients()
        
        # Cache for loaded data
        self._cache = {} if cache_data else None
        self._valid_slices = {}
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        print(f"[SAILORDataset] Mode: {mode}, Patients: {len(self.patient_ids)}, Samples: {len(self.samples)}")
        
    def _find_patients(self) -> List[str]:
        """Find all patient IDs with complete data."""
        image_files = list(self.data_dir.glob("*_image.npy"))
        patient_ids = []
        
        for f in image_files:
            patient_id = f.stem.replace("_image", "")
            if self._check_patient_files(patient_id):
                patient_ids.append(patient_id)
        
        return sorted(patient_ids)
    
    def _check_patient_files(self, patient_id: str) -> bool:
        """Check if all required files exist."""
        required = ['_image.npy', '_label.npy', '_days.npy', '_treatment.npy']
        return all((self.data_dir / f"{patient_id}{suffix}").exists() for suffix in required)
    
    def _load_patient_data(self, patient_id: str) -> Dict[str, np.ndarray]:
        """Load all data for a patient."""
        if self._cache is not None and patient_id in self._cache:
            return self._cache[patient_id]
        
        data = {
            'image': np.load(self.data_dir / f"{patient_id}_image.npy"),
            'label': np.load(self.data_dir / f"{patient_id}_label.npy"),
            'days': np.load(self.data_dir / f"{patient_id}_days.npy"),
            'treatment': np.load(self.data_dir / f"{patient_id}_treatment.npy"),
        }
        
        if self._cache is not None:
            self._cache[patient_id] = data
        
        return data
    
    def _get_valid_slices(self, patient_id: str) -> List[int]:
        """Get slices containing tumor."""
        if patient_id in self._valid_slices:
            return self._valid_slices[patient_id]
        
        data = self._load_patient_data(patient_id)
        label = data['label']
        
        # Find slices with tumor across all sessions
        tumor_mask = (label > 0).any(axis=0)  # (H, W, D)
        
        valid_slices = []
        for d in range(tumor_mask.shape[-1]):
            if tumor_mask[:, :, d].sum() > 100:
                valid_slices.append(d)
        
        # Apply slice range filter
        if self.slice_range is not None:
            start, end = self.slice_range
            valid_slices = [s for s in valid_slices if start <= s < end]
        
        # Fallback to center slices if no tumor found
        if len(valid_slices) < self.min_tumor_slices:
            D = label.shape[-1]
            center = D // 2
            valid_slices = list(range(max(0, center - 20), min(D, center + 20)))
        
        self._valid_slices[patient_id] = valid_slices
        return valid_slices
    
    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build (patient_idx, slice_idx) samples."""
        samples = []
        
        for patient_idx, patient_id in enumerate(self.patient_ids):
            valid_slices = self._get_valid_slices(patient_id)
            
            if self.mode == 'train':
                # Random slices during training
                for _ in range(self.num_slices_per_volume):
                    samples.append((patient_idx, -1))  # -1 = random
            else:
                # All valid slices during val/test
                for slice_idx in valid_slices:
                    samples.append((patient_idx, slice_idx))
        
        return samples
    
    def _select_sessions(self, num_available: int) -> Tuple[List[int], int]:
        """Select sessions to use."""
        if num_available >= self.num_sessions:
            if self.mode == 'train':
                start = random.randint(0, num_available - self.num_sessions)
                session_indices = list(range(start, start + self.num_sessions))
            else:
                session_indices = list(range(num_available - self.num_sessions, num_available))
        else:
            # Pad with last session
            session_indices = list(range(num_available))
            while len(session_indices) < self.num_sessions:
                session_indices.append(session_indices[-1])
        
        # Target is last session by default
        target_idx = self.num_sessions - 1
        
        # Random target during training
        if self.mode == 'train' and random.random() < 0.3:
            target_idx = random.randint(0, self.num_sessions - 1)
        
        return session_indices, target_idx
    
    def _extract_slice(
        self,
        image: np.ndarray,
        label: np.ndarray,
        slice_idx: int,
        session_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 2D slice from 3D volumes."""
        num_mods_total = 4  # Always 4 modalities in SAILOR
        
        # Extract images for selected sessions and modalities
        selected_images = []
        for s in session_indices:
            for m in self.modality_indices:
                idx = s * num_mods_total + m
                if idx < image.shape[0]:
                    selected_images.append(image[idx, :, :, slice_idx])
                else:
                    selected_images.append(np.zeros((image.shape[1], image.shape[2])))
        
        image_2d = np.stack(selected_images, axis=0)
        
        # Extract labels
        selected_labels = []
        for s in session_indices:
            if s < label.shape[0]:
                selected_labels.append(label[s, :, :, slice_idx])
            else:
                selected_labels.append(np.zeros((label.shape[1], label.shape[2])))
        
        label_2d = np.stack(selected_labels, axis=0)
        
        return image_2d, label_2d
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        FIXED: Normalize image to [-1, 1] range.
        
        This is CRITICAL for diffusion models which assume data in [-1, 1].
        """
        if not self.normalize:
            return image
        
        # Per-channel normalization for each modality
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for c in range(image.shape[0]):
            channel = image[c]
            
            # Robust normalization using percentiles to handle outliers
            p_low, p_high = np.percentile(channel, [1, 99])
            
            if p_high - p_low > 1e-6:
                # Normalize to [0, 1]
                channel = (channel - p_low) / (p_high - p_low)
                channel = np.clip(channel, 0, 1)
            else:
                # FIX: Problem 1.4 — Use 0.5 instead of 0 for constant channels
                # Zero maps to -1.0 after rescaling, which is an extreme bias
                # 0.5 maps to 0.0 in [-1, 1] range, a neutral center value
                channel = np.full_like(channel, 0.5)
            
            # Convert to [-1, 1]
            channel = 2.0 * channel - 1.0
            normalized[c] = channel
        
        return normalized
    
    def _resize(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize to target size."""
        if image.shape[-1] == self.image_size and image.shape[-2] == self.image_size:
            return image, label
        
        img_t = torch.from_numpy(image).float().unsqueeze(0)
        lbl_t = torch.from_numpy(label).float().unsqueeze(0)
        
        img_t = F.interpolate(img_t, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        lbl_t = F.interpolate(lbl_t, size=(self.image_size, self.image_size), mode='nearest')
        
        return img_t.squeeze(0).numpy(), lbl_t.squeeze(0).numpy()
    
    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        if not self.augment:
            return image, label
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=-1).copy()
            label = np.flip(label, axis=-1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=-2).copy()
            label = np.flip(label, axis=-2).copy()
        
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            label = np.rot90(label, k, axes=(-2, -1)).copy()
        
        # FIXED: Small intensity perturbation (keeps data in [-1, 1])
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            shift = random.uniform(-0.05, 0.05)
            image = np.clip(image * scale + shift, -1.0, 1.0)
        
        return image, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_idx, slice_idx = self.samples[idx]
        patient_id = self.patient_ids[patient_idx]
        
        # Load data
        data = self._load_patient_data(patient_id)
        image = data['image']
        label = data['label']
        days = data['days']
        treatment = data['treatment']
        
        num_sessions_total = len(days)
        
        # Select sessions
        session_indices, target_idx = self._select_sessions(num_sessions_total)
        
        # Select slice
        if slice_idx == -1:
            valid_slices = self._get_valid_slices(patient_id)
            slice_idx = random.choice(valid_slices) if valid_slices else 0
        
        # Extract 2D slice
        image_2d, label_2d = self._extract_slice(image, label, slice_idx, session_indices)
        
        # FIXED: Normalize BEFORE resize and augment
        image_2d = self._normalize_image(image_2d)
        
        # Resize
        image_2d, label_2d = self._resize(image_2d, label_2d)
        
        # Augment
        image_2d, label_2d = self._augment(image_2d, label_2d)
        
        # Get days and treatment for selected sessions
        selected_days = np.array([days[min(s, len(days)-1)] for s in session_indices], dtype=np.float32)
        selected_treatment = np.array([treatment[min(s, len(treatment)-1)] for s in session_indices], dtype=np.float32)
        
        # Convert to tensors
        return {
            'image': torch.from_numpy(image_2d.astype(np.float32)),
            'label': torch.from_numpy(label_2d.astype(np.float32)),
            'days': torch.from_numpy(selected_days),
            'treatment': torch.from_numpy(selected_treatment),
            'target_idx': torch.tensor(target_idx, dtype=torch.long),
            'patient_id': patient_id,
            'slice_idx': slice_idx,
        }


class SAILORDatasetFiltered(SAILORDataset):
    """SAILORDataset with explicit patient list."""
    
    def __init__(self, patient_ids: List[str], **kwargs):
        self._explicit_patients = patient_ids
        super().__init__(**kwargs)
    
    def _find_patients(self) -> List[str]:
        valid_patients = []
        for pid in self._explicit_patients:
            if self._check_patient_files(pid):
                valid_patients.append(pid)
        return sorted(valid_patients)


def create_sailor_dataloaders(
    data_dir: str,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 192,
    num_slices_per_volume: int = 10,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    data_path = Path(data_dir)
    image_files = list(data_path.glob("*_image.npy"))
    patient_ids = sorted([f.stem.replace("_image", "") for f in image_files])
    
    # Split patients
    random.seed(seed)
    random.shuffle(patient_ids)
    
    n_train = int(len(patient_ids) * train_ratio)
    n_val = (len(patient_ids) - n_train) // 2
    
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]
    
    print(f"[DataLoaders] Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    train_ds = SAILORDatasetFiltered(
        data_dir=data_dir,
        patient_ids=train_patients,
        mode='train',
        image_size=image_size,
        num_slices_per_volume=num_slices_per_volume,
        normalize=True,
        **kwargs,
    )
    
    val_ds = SAILORDatasetFiltered(
        data_dir=data_dir,
        patient_ids=val_patients,
        mode='val',
        image_size=image_size,
        num_slices_per_volume=num_slices_per_volume,
        normalize=True,
        **kwargs,
    )
    
    test_ds = SAILORDatasetFiltered(
        data_dir=data_dir,
        patient_ids=test_patients,
        mode='test',
        image_size=image_size,
        num_slices_per_volume=num_slices_per_volume,
        normalize=True,
        **kwargs,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
