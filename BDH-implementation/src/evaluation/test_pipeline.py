"""
Connected Test Pipeline for TaDiff Model.

This module provides the complete testing pipeline connecting:
- Data preprocessing (preproc_prepare_data.py)
- Data loading (data_loader.py)
- Model configuration (test_config.py)
- Evaluation and visualization

Compatible with TaDiff-Net pipeline structure.
Reference: https://github.com/samleoqh/TaDiff-Net/blob/main/test.py
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.test_config import TestConfig, InferenceConfig
from src.data.data_loader import (
    get_test_files, 
    load_data, 
    val_transforms,
    prepare_image_batch,
    prepare_batch_for_model,
)


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_tumor_volumes(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate tumor volume per slice.
    
    Args:
        labels: Segmentation tensor of shape (T, H, W, D)
    
    Returns:
        Tensor of shape (D,) with tumor volume per slice
    """
    # Sum across sessions, height, width
    return labels.sum(dim=(0, 1, 2))


def get_slice_indices(z_mask_size: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """
    Get indices of top-k slices with largest tumor volume.
    
    Args:
        z_mask_size: Tensor of tumor volumes per slice
        top_k: Number of top slices to return
    
    Returns:
        Indices of top-k slices
    """
    # Get indices of non-zero slices
    nonzero_mask = z_mask_size > 0
    if nonzero_mask.sum() == 0:
        # No tumor, return middle slices
        mid = len(z_mask_size) // 2
        return torch.tensor([mid - 1, mid, mid + 1])
    
    # Get top-k slices
    _, indices = torch.topk(z_mask_size, min(top_k, nonzero_mask.sum().item()))
    return indices


def extract_session_slices(
    images: torch.Tensor,
    labels: torch.Tensor,
    slice_idx: int,
    session_indices: List[int],
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract 2D slices from 3D volumes for specified sessions.
    
    Args:
        images: Image tensor (B, T, M, H, W, D)
        labels: Label tensor (B, T, H, W, D)
        slice_idx: Z-axis slice index
        session_indices: List of session indices
        num_samples: Number of samples to replicate
        device: Target device
    
    Returns:
        Tuple of (seq_imgs, masks) tensors
    """
    h, w = images.shape[3:5]
    
    # Extract masks for sessions: (T, H, W) for slice
    masks = labels[0, session_indices, :, :, slice_idx]
    masks = masks.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # (samples, T, H, W)
    
    # Extract images for sessions: (T, M, H, W) for slice
    seq_imgs = images[0, session_indices, :, :, :, slice_idx]
    seq_imgs = seq_imgs.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)  # (samples, T, M, H, W)
    
    return seq_imgs.to(device), masks.to(device)


# ============================================================================
# Processing Functions
# ============================================================================

def process_patient(
    patient_id: str,
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    config: TestConfig,
    save_path: str,
) -> Dict[str, Dict]:
    """
    Process a single patient through the test pipeline.
    
    Args:
        patient_id: Patient identifier
        batch: Data batch from dataloader
        model: TaDiff model
        device: Target device
        config: Test configuration
        save_path: Directory to save results
    
    Returns:
        Dictionary of scores for all sessions and slices
    """
    patient_scores = {}
    
    # Extract tensors
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    days = batch['days'].to(device)
    treatments = batch['treatment'].to(device)
    
    # Get number of sessions
    num_sessions = labels.shape[1]
    
    # Prepare images: reshape from (B, M*T, H, W, D) to (B, T, M, H, W, D)
    images = prepare_image_batch(images, num_sessions, remove_t2=True)
    
    # Calculate tumor volumes
    z_mask_size = calculate_tumor_volumes(labels[0])
    mean_vol = z_mask_size[z_mask_size > 0].mean() if z_mask_size.sum() > 0 else 0
    
    # Adjust threshold
    if mean_vol < num_sessions * 100 and z_mask_size.max() > num_sessions * 200:
        mean_vol = num_sessions * 100
    z_mask_size[z_mask_size < mean_vol] = 0
    
    # Get top-k slices
    top_k_indices = get_slice_indices(z_mask_size, config.top_k_slices)
    
    if config.verbose:
        print(f"  Sessions: {num_sessions}, Top slices: {top_k_indices.tolist()}")
    
    # Process each session
    for session_idx in range(min(num_sessions, config.target_session_idx + 1)):
        session_key = f"session_{session_idx}"
        patient_scores[session_key] = {}
        
        # Create session directory
        session_path = os.path.join(save_path, f'p-{patient_id}', f'ses-{session_idx:02d}')
        os.makedirs(session_path, exist_ok=True)
        
        # Process each slice
        for slice_idx in top_k_indices:
            slice_key = f"slice_{slice_idx.item()}"
            
            # Skip if tumor too small
            if z_mask_size[slice_idx] < config.min_tumor_size:
                continue
            
            # Process slice
            try:
                slice_scores = process_slice(
                    slice_idx=slice_idx.item(),
                    session_idx=session_idx,
                    images=images,
                    labels=labels,
                    days=days,
                    treatments=treatments,
                    model=model,
                    device=device,
                    config=config,
                    session_path=session_path,
                )
                patient_scores[session_key][slice_key] = slice_scores
            except Exception as e:
                if config.verbose:
                    print(f"    Error processing slice {slice_idx}: {e}")
    
    return patient_scores


def process_slice(
    slice_idx: int,
    session_idx: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    config: TestConfig,
    session_path: str,
) -> Dict[str, float]:
    """
    Process a single 2D slice through the model.
    
    Args:
        slice_idx: Z-axis slice index
        session_idx: Session index
        images: Prepared image tensor (B, T, M, H, W, D)
        labels: Label tensor (B, T, H, W, D)
        days: Day values tensor
        treatments: Treatment codes tensor
        model: TaDiff model
        device: Target device
        config: Test configuration
        session_path: Directory to save results
    
    Returns:
        Dictionary of metric scores
    """
    num_samples = config.num_samples
    target_idx = config.target_session_idx
    
    # Get session indices for conditioning (last 4 sessions)
    session_indices = list(range(max(0, target_idx - 3), target_idx + 1))
    while len(session_indices) < 4:
        session_indices.insert(0, session_indices[0])
    
    h, w = images.shape[3:5]
    
    # Extract slices
    seq_imgs, masks = extract_session_slices(
        images, labels, slice_idx, session_indices, num_samples, device
    )
    
    # Create noise and prepare inputs
    noise = torch.randn((num_samples, 3, h, w), device=device)
    x_t = seq_imgs.clone()
    x_0 = []
    
    # Target index for each sample
    i_tg = target_idx * torch.ones((num_samples,), dtype=torch.int8, device=device)
    
    for i, j in zip(range(num_samples), i_tg):
        x_0.append(seq_imgs[i, j, :, :, :].unsqueeze(0))
        x_t[i, j, :, :, :] = noise[i, :, :, :]
    
    x_0 = torch.cat(x_0, dim=0)
    x_t = x_t.reshape(num_samples, len(session_indices) * 3, h, w)
    
    # Prepare conditioning
    days_q = days[0, session_indices].repeat(num_samples, 1).to(device)
    treatments_q = treatments[0, session_indices].repeat(num_samples, 1).to(device)
    
    # Create timesteps
    timesteps = torch.tensor([config.diffusion_steps], device=device).float()
    timesteps = timesteps.repeat(num_samples)
    
    # Prepare interval conditioning (4 values for 4 sessions)
    intv_t = [days_q[:, i].to(torch.float32) for i in range(4)]
    treat_code = [treatments_q[:, i].to(torch.float32) for i in range(4)]
    
    # Forward pass
    with torch.no_grad():
        output = model(x_t, timesteps, intv_t, treat_code, i_tg)
    
    # Calculate metrics (placeholder - implement based on model output)
    scores = {
        'mse': float(((output[:, :3] - x_0) ** 2).mean()),
        'slice_idx': slice_idx,
        'session_idx': session_idx,
    }
    
    return scores


# ============================================================================
# Main Test Pipeline
# ============================================================================

def run_test_pipeline(
    config: TestConfig,
    model: Optional[torch.nn.Module] = None,
) -> pd.DataFrame:
    """
    Run the complete test pipeline.
    
    Args:
        config: Test configuration
        model: Pre-loaded model (optional, loaded from checkpoint if None)
    
    Returns:
        DataFrame with test results
    """
    device = torch.device(config.get_device())
    print(f"Device: {device}")
    
    # Load model if not provided
    if model is None:
        print(f"Loading model from {config.model_checkpoint}...")
        # Import here to avoid circular imports
        try:
            from src.net.tadiff_dit_arch import TaDiff_DiT
            model = TaDiff_DiT(
                image_size=config.image_size,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                model_channels=config.model_channels,
                hidden_size=config.hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                patch_size=config.patch_size,
            )
            
            if os.path.exists(config.model_checkpoint):
                checkpoint = torch.load(config.model_checkpoint, map_location=device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print("Model loaded successfully")
            else:
                print(f"Checkpoint not found: {config.model_checkpoint}")
                print("Using untrained model for testing")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    model.to(device)
    model.eval()
    
    # Get test files
    print(f"\nLoading data from {config.data_root}...")
    test_files = get_test_files(
        data_root=config.data_root,
        patient_ids=config.patient_ids,
        prefix=config.prefix,
        keys=config.npz_keys,
    )
    
    if not test_files:
        print("No test files found!")
        return pd.DataFrame()
    
    print(f"Found {len(test_files)} patients")
    
    # Create dataloader
    dataloader = load_data(test_files)
    
    # Create output directory
    os.makedirs(config.save_path, exist_ok=True)
    
    # Process each patient
    all_scores = {}
    for i, batch in enumerate(dataloader):
        patient_id = config.patient_ids[i]
        print(f"\nProcessing patient {patient_id}...")
        
        patient_scores = process_patient(
            patient_id=patient_id,
            batch=batch,
            model=model,
            device=device,
            config=config,
            save_path=config.save_path,
        )
        all_scores[patient_id] = patient_scores
    
    # Convert to DataFrame
    rows = []
    for patient_id, patient_data in all_scores.items():
        for session_key, session_data in patient_data.items():
            for slice_key, scores in session_data.items():
                row = {'patient_id': patient_id, 'session': session_key, 'slice': slice_key}
                row.update(scores)
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    csv_path = os.path.join(config.save_path, 'test_scores.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    return df


# ============================================================================
# Quick Test Functions
# ============================================================================

def test_data_pipeline(data_root: str, patient_ids: Optional[List[str]] = None) -> bool:
    """
    Test the data loading pipeline with actual or synthetic data.
    
    Args:
        data_root: Directory containing data files
        patient_ids: List of patient IDs to test
    
    Returns:
        True if test passed
    """
    print("\n" + "=" * 60)
    print("Data Pipeline Test")
    print("=" * 60)
    
    from src.data.data_loader import scan_data_directory
    
    # Scan for patients if not provided
    if patient_ids is None:
        patient_ids = scan_data_directory(data_root)
    
    if not patient_ids:
        print(f"No patients found in {data_root}")
        return False
    
    print(f"Found patients: {patient_ids}")
    
    # Get test files
    test_files = get_test_files(data_root, patient_ids)
    
    if not test_files:
        print("No valid test files found")
        return False
    
    print(f"Valid test files: {len(test_files)}")
    
    # Create dataloader
    dataloader = load_data(test_files)
    
    if dataloader is None:
        print("Failed to create dataloader")
        return False
    
    # Load one batch
    try:
        batch = next(iter(dataloader))
        print(f"\nBatch loaded successfully:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Label shape: {batch['label'].shape}")
        print(f"  Days: {batch['days']}")
        print(f"  Treatment: {batch['treatment']}")
        
        # Test prepare_batch_for_model
        device = torch.device('cpu')
        prepared = prepare_batch_for_model(batch, device)
        print(f"\nPrepared batch:")
        print(f"  Image shape: {prepared['image'].shape}")
        print(f"  Num sessions: {prepared['num_sessions']}")
        
        print("\n✓ Data pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing_pipeline(output_dir: str = "./data/test_synthetic") -> bool:
    """
    Test the preprocessing pipeline with synthetic data.
    
    Args:
        output_dir: Directory to create synthetic data
    
    Returns:
        True if test passed
    """
    print("\n" + "=" * 60)
    print("Preprocessing Pipeline Test")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    
    from data.preproc_prepare_data import (
        create_synthetic_patient,
        validate_processed_data,
    )
    
    # Create synthetic patient
    print("\n[Step 1] Creating synthetic patient...")
    try:
        files = create_synthetic_patient(
            save_path=output_dir,
            patient_id="test-01",
            num_sessions=4,
            shape=(64, 64, 32),
        )
        print("  ✓ Synthetic patient created")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Validate data
    print("\n[Step 2] Validating processed data...")
    try:
        valid = validate_processed_data(output_dir, "test-01")
        if valid:
            print("  ✓ Data validation passed")
        else:
            print("  ✗ Data validation failed")
            return False
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False
    
    # Test data loading
    print("\n[Step 3] Testing data loading pipeline...")
    result = test_data_pipeline(output_dir, ["test-01"])
    
    if result:
        print("\n✓ Preprocessing pipeline test passed!")
    
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TaDiff Model Test Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_root', type=str, default='./data/sailor_npy',
                        help='Root directory for data files')
    parser.add_argument('--save_path', type=str, default='./test_results',
                        help='Directory to save results')
    parser.add_argument('--checkpoint', type=str, default='./ckpt/model.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--patients', type=str, nargs='+', default=None,
                        help='Patient IDs to test')
    parser.add_argument('--diffusion_steps', type=int, default=600,
                        help='Number of diffusion steps')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of prediction samples')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick pipeline test with synthetic data')
    parser.add_argument('--test_data', action='store_true',
                        help='Test data loading only')
    
    args = parser.parse_args()
    
    if args.quick_test:
        # Run quick test with synthetic data
        success = test_preprocessing_pipeline()
        sys.exit(0 if success else 1)
    
    if args.test_data:
        # Test data loading only
        success = test_data_pipeline(args.data_root, args.patients)
        sys.exit(0 if success else 1)
    
    # Create config
    config = TestConfig(
        data_root=args.data_root,
        save_path=args.save_path,
        model_checkpoint=args.checkpoint,
        diffusion_steps=args.diffusion_steps,
        num_samples=args.num_samples,
    )
    
    if args.patients:
        config.patient_ids = args.patients
    
    # Run test pipeline
    results = run_test_pipeline(config)
    
    if len(results) > 0:
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(results.describe())
    else:
        print("No results generated")


if __name__ == "__main__":
    main()
