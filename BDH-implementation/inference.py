"""
DIT-BDH Inference Script 

Consistent with train.py training pipeline:
- Same model architecture loading (TaDiffDiTLitModule or TaDiff_DiT)
- Same diffusion schedule (T and schedule type from training)
- Same data normalization (per-channel percentile to [-1, 1])
- Same modality selection (T1c, FLAIR, T2 = indices 1, 2, 3)
- DDIM sampling subsamples from full training schedule

Usage:
    python inference.py --checkpoint ./logs/tadiff-dit/checkpoints/last.ckpt \
                        --patient_file ./data/my_custom_data/patient-001_image.npy \
                        --output_dir ./inference_results

    python inference.py --checkpoint ./ckpt/model.ckpt \
                        --data_dir ./data/sailor --patient_ids sub-17 \
                        --slice_idx 102 --input_day 100 --input_treatment 1
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.net.tadiff_dit_arch import TaDiff_DiT
from src.net.diffusion import GaussianDiffusion
from src.visualization.visualizer import Visualizer


# Compatibility shim for checkpoints that pickle TrainingConfig from train.py
class TrainingConfig:
    pass


def _parse_scalar_value(raw: str):
    """Parse simple YAML scalar values without requiring PyYAML."""
    value = raw.strip()
    if value.startswith(('"', "'")) and value.endswith(('"', "'")) and len(value) >= 2:
        return value[1:-1]

    lower = value.lower()
    if lower in {'true', 'false'}:
        return lower == 'true'
    if lower in {'null', 'none'}:
        return None

    if re.fullmatch(r'-?\d+', value):
        try:
            return int(value)
        except ValueError:
            pass

    if re.fullmatch(r'-?\d+(\.\d+)?([eE][-+]?\d+)?', value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def _parse_hparams_yaml(hparams_path: Path) -> Dict[str, object]:
    """Parse training config from Lightning hparams.yaml into a plain dict."""
    if not hparams_path.exists():
        return {}

    parsed: Dict[str, object] = {}
    in_config_block = False

    with hparams_path.open('r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            if stripped.startswith('config:'):
                in_config_block = True
                continue

            if not in_config_block:
                continue

            # Expect lines like: "  key: value"
            if not line.startswith('  '):
                break

            kv = stripped.split(':', 1)
            if len(kv) != 2:
                continue

            key = kv[0].strip()
            raw_value = kv[1].strip()
            parsed[key] = _parse_scalar_value(raw_value)

    return parsed


def _find_nearby_hparams(checkpoint_path: str) -> Optional[Path]:
    """Find nearest hparams.yaml for a checkpoint path."""
    ckpt = Path(checkpoint_path).resolve()
    search_roots = [ckpt.parent, ckpt.parent.parent, ckpt.parent.parent.parent]

    for root in search_roots:
        if not root.exists():
            continue

        direct = root / 'hparams.yaml'
        if direct.exists():
            return direct

        version_dirs = sorted(root.glob('version_*'))
        for vd in version_dirs:
            hp = vd / 'hparams.yaml'
            if hp.exists():
                return hp

    return None


def apply_checkpoint_hparams(args: argparse.Namespace) -> argparse.Namespace:
    """Auto-apply checkpoint training hparams so inference args always match architecture."""
    if not getattr(args, 'auto_hparams', True):
        return args

    hparams_path = _find_nearby_hparams(args.checkpoint)
    if hparams_path is None:
        print("[Inference] hparams.yaml not found near checkpoint, using CLI/default args")
        return args

    hparams = _parse_hparams_yaml(hparams_path)
    if not hparams:
        print(f"[Inference] Could not parse config from {hparams_path}, using CLI/default args")
        return args

    map_keys = [
        'image_size',
        'model_channels',
        'hidden_size',
        'depth',
        'num_heads',
        'patch_size',
        'bdh_expansion',
        'max_T',
        'ddpm_schedule',
    ]

    overrides = []
    for key in map_keys:
        if key in hparams and hparams[key] is not None:
            old = getattr(args, key)
            new = hparams[key]
            if old != new:
                setattr(args, key, new)
                overrides.append((key, old, new))

    print(f"[Inference] Loaded hparams from: {hparams_path}")
    if overrides:
        for key, old, new in overrides:
            print(f"  - Override {key}: {old} -> {new}")
    else:
        print("  - CLI/default args already match checkpoint hparams")

    return args


# =============================================================================
# Model Loading — handles both TaDiffDiTLitModule and Tadiff_model checkpoints
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
    # Architecture params — MUST match what was used during training
    image_size: int = 192,
    in_channels: int = 12,
    out_channels: int = 7,
    model_channels: int = 256,
    hidden_size: int = 1024,
    depth: int = 18,
    num_heads: int = 16,
    patch_size: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    use_hybrid_embed: bool = True,
    bdh_expansion: int = 2,
    bdh_layers: int = 6,
) -> TaDiff_DiT:
    """
    Load TaDiff_DiT model from a Lightning checkpoint.

    Handles key prefix differences between:
    - TaDiffDiTLitModule (train.py): keys are 'model.xxx'
    - Tadiff_model (tadiff_model.py): keys are '_model.xxx'
    """
    model = TaDiff_DiT(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        use_hybrid_embed=use_hybrid_embed,
        bdh_expansion=bdh_expansion,
        bdh_layers=bdh_layers,
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Prefer weights_only to avoid pickle issues with custom classes stored in ckpt
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Try multiple key prefix patterns to find the right one
    prefixes_to_try = ['model.', '_model.', '']
    best_match_count = 0
    best_cleaned = state_dict

    model_keys = set(model.state_dict().keys())

    for prefix in prefixes_to_try:
        if not prefix:
            cleaned = state_dict
        else:
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    cleaned[k[len(prefix):]] = v

        match_count = len(set(cleaned.keys()) & model_keys)
        if match_count > best_match_count:
            best_match_count = match_count
            best_cleaned = cleaned

    total_keys = len(model_keys)
    if best_match_count == 0:
        raise RuntimeError(
            f"Checkpoint has no matching keys. "
            f"Checkpoint keys sample: {list(state_dict.keys())[:5]}. "
            f"Model keys sample: {list(model_keys)[:5]}. "
            f"Ensure architecture params match training."
        )

    missing, unexpected = model.load_state_dict(best_cleaned, strict=False)
    print(f"[Model] Loaded {best_match_count}/{total_keys} keys. "
          f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys (first 5): {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


# =============================================================================
# Data Loading — matches SAILORDataset normalization
# =============================================================================

MODALITY_INDICES = [1, 2, 3]  # T1c, FLAIR, T2 — MUST match training


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    """
    Per-channel percentile normalization to [-1, 1].
    Matches SAILORDataset._normalize_image() exactly.
    """
    p_low, p_high = np.percentile(channel, [1, 99])
    if p_high - p_low > 1e-6:
        channel = (channel - p_low) / (p_high - p_low)
        channel = np.clip(channel, 0, 1)
    else:
        channel = np.full_like(channel, 0.5)
    return 2.0 * channel - 1.0


def load_patient_data(patient_file: str) -> Dict[str, np.ndarray]:
    """Load patient data from _image.npy and associated files."""
    base_name = patient_file.replace('_image.npy', '')
    data = {'image': np.load(patient_file)}

    for key in ['label', 'days', 'treatment']:
        path = f"{base_name}_{key}.npy"
        if os.path.exists(path):
            data[key] = np.load(path)
        else:
            print(f"  Warning: {key} file not found, using zeros")
            if key == 'label':
                data[key] = np.zeros_like(data['image'][0:1])
            else:
                num_sessions = data['image'].shape[0] // 4
                data[key] = np.zeros(num_sessions)

    return data


def prepare_2d_input(
    image: np.ndarray,
    days: np.ndarray,
    treatment: np.ndarray,
    session_indices: List[int],
    slice_idx: int,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Prepare a 2D slice for model input, matching training pipeline.

    Args:
        image: Full patient image array (M*T, H, W, D)
        days: Day values per session (T,)
        treatment: Treatment codes per session (T,)
        session_indices: 4 session indices to use
        slice_idx: Z-axis slice index
        image_size: Target spatial size

    Returns:
        x_input: [1, 12, H, W] normalized input tensor
        intv_t: List of 4 day tensors
        treat_code: List of 4 treatment tensors
    """
    num_mods_total = 4  # SAILOR has 4 modalities per session

    # Extract channels: sessions-first, selected modalities
    channels = []
    for s in session_indices:
        for m in MODALITY_INDICES:
            idx = s * num_mods_total + m
            if idx < image.shape[0]:
                ch = image[idx, :, :, slice_idx].astype(np.float32)
            else:
                ch = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
            channels.append(ch)

    image_2d = np.stack(channels, axis=0)  # (12, H, W)

    # Resize if needed
    if image_2d.shape[-1] != image_size or image_2d.shape[-2] != image_size:
        img_t = torch.from_numpy(image_2d).float().unsqueeze(0)
        img_t = F.interpolate(img_t, size=(image_size, image_size),
                              mode='bilinear', align_corners=False)
        image_2d = img_t.squeeze(0).numpy()

    # Normalize per-channel (matches SAILORDataset._normalize_image)
    for c in range(image_2d.shape[0]):
        image_2d[c] = normalize_channel(image_2d[c])

    x_input = torch.from_numpy(image_2d).float().unsqueeze(0).to(device)  # [1, 12, H, W]

    # Conditioning tensors
    intv_t = [
        torch.tensor([days[min(s, len(days) - 1)]], device=device, dtype=torch.float32)
        for s in session_indices
    ]
    treat_code = [
        torch.tensor([treatment[min(s, len(treatment) - 1)]], device=device, dtype=torch.float32)
        for s in session_indices
    ]

    return x_input, intv_t, treat_code


def extract_gt_target_slice(
    image: np.ndarray,
    session_idx: int,
    slice_idx: int,
    image_size: int,
) -> Optional[np.ndarray]:
    """
    Extract normalized GT image modalities [3, H, W] for one target session/slice.

    Returns None if session index is out of range (e.g., future prediction mode).
    """
    num_sessions = image.shape[0] // 4
    if session_idx < 0 or session_idx >= num_sessions:
        return None

    channels = []
    for m in MODALITY_INDICES:
        idx = session_idx * 4 + m
        ch = image[idx, :, :, slice_idx].astype(np.float32)
        channels.append(ch)

    gt_2d = np.stack(channels, axis=0)  # (3, H, W)

    if gt_2d.shape[-1] != image_size or gt_2d.shape[-2] != image_size:
        gt_t = torch.from_numpy(gt_2d).float().unsqueeze(0)
        gt_t = F.interpolate(gt_t, size=(image_size, image_size),
                             mode='bilinear', align_corners=False)
        gt_2d = gt_t.squeeze(0).numpy()

    for c in range(gt_2d.shape[0]):
        gt_2d[c] = normalize_channel(gt_2d[c])

    return gt_2d


# =============================================================================
# DDIM Sampling — uses training's full schedule with subsampled steps
# =============================================================================

@torch.no_grad()
def ddim_sample(
    model: TaDiff_DiT,
    diffusion: GaussianDiffusion,
    x_input: torch.Tensor,
    intv_t: List[torch.Tensor],
    treat_code: List[torch.Tensor],
    target_idx: torch.Tensor,
    sampling_steps: int = 50,
    eta: float = 0.0,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM sampling consistent with diffusion.py's ddim_inverse.

    Uses the FULL training schedule (diffusion.T timesteps) and subsamples
    `sampling_steps` from it. This ensures alphabar values match training.
    """
    B = x_input.shape[0]
    H, W = x_input.shape[2], x_input.shape[3]

    # Subsample timesteps from full schedule (1-indexed to match training)
    times = torch.linspace(diffusion.T, 1, sampling_steps + 1, device=device).long()
    time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

    # Initialize noise in target session
    x = x_input.clone()
    i_tg_positive = torch.where(target_idx < 0, 4 + target_idx, target_idx).long()

    # Start with pure noise in target session
    noise_init = torch.randn(B, 3, H, W, device=device)
    for b in range(B):
        s = i_tg_positive[b].item()
        x[b, s * 3:(s + 1) * 3] = noise_init[b]

    mask_accum = torch.zeros(B, 4, H, W, device=device)
    step_mask = min(10, sampling_steps)
    alphabar_weights = diffusion.alphabar[:step_mask].to(device)
    w_p = alphabar_weights / alphabar_weights.sum()

    for t_curr, t_next in tqdm(time_pairs, desc="DDIM Sampling", leave=False):
        t_curr_idx = t_curr - 1  # 0-indexed for array access

        alphabar_t = diffusion.alphabar[t_curr_idx].to(device)
        sqrt_ab = torch.sqrt(alphabar_t)
        sqrt_1_ab = torch.sqrt(1.0 - alphabar_t)

        t_batch = torch.full((B,), t_curr, device=device, dtype=torch.float32)

        pred = model(x, t_batch, intv_t=intv_t, treat_code=treat_code, i_tg=target_idx)
        eps_pred = pred[:, 4:7]
        mask_pred = pred[:, 0:4]

        # Extract current noisy target
        xt = torch.stack([
            x[b, i_tg_positive[b].item() * 3:(i_tg_positive[b].item() + 1) * 3]
            for b in range(B)
        ], dim=0)

        # Predict x0
        x0_pred = (xt - sqrt_1_ab * eps_pred) / (sqrt_ab + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if t_next >= 1:
            t_next_idx = t_next - 1
            alphabar_next = diffusion.alphabar[t_next_idx].to(device)
            sqrt_ab_next = torch.sqrt(alphabar_next)

            if eta > 0:
                sigma = eta * torch.sqrt(
                    (1 - alphabar_next) / (1 - alphabar_t + 1e-8)
                ) * torch.sqrt(1 - alphabar_t / (alphabar_next + 1e-8))
                sigma = torch.clamp(sigma, min=0.0)
                noise = torch.randn_like(xt)
            else:
                sigma = 0.0
                noise = 0.0

            dir_coef = torch.sqrt(torch.clamp(1.0 - alphabar_next - sigma ** 2, min=0.0))
            x_next = sqrt_ab_next * x0_pred + dir_coef * eps_pred
            if eta > 0:
                x_next = x_next + sigma * noise
        else:
            x_next = x0_pred

        # Update target session in x
        for b in range(B):
            s = i_tg_positive[b].item()
            x[b, s * 3:(s + 1) * 3] = x_next[b]

        # Accumulate masks
        if t_curr_idx < step_mask:
            mask_accum += mask_pred * w_p[t_curr_idx]

    return x_next, mask_accum


# =============================================================================
# Inference Pipeline
# =============================================================================

def run_inference(
    model: TaDiff_DiT,
    diffusion: GaussianDiffusion,
    image: np.ndarray,
    label: np.ndarray,
    days: np.ndarray,
    treatment: np.ndarray,
    device: torch.device,
    image_size: int = 192,
    sampling_steps: int = 50,
    num_samples: int = 4,
    target_session_mode: str = 'last',
    input_day: Optional[float] = None,
    input_treatment: Optional[float] = None,
    slice_indices: Optional[List[int]] = None,
    quick_inference: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a patient.

    Args:
        model: Loaded TaDiff_DiT model
        diffusion: GaussianDiffusion with SAME schedule as training
        image: Patient images (M*T, H, W, D)
        label: Patient labels (T, H, W, D)
        days: Day values (T,)
        treatment: Treatment codes (T,)
        device: Target device
        image_size: Spatial size
        sampling_steps: DDIM steps (subsampled from full schedule)
        num_samples: Number of samples for uncertainty estimation
        target_session_mode: 'last' to predict last session, 'future' for future prediction
        input_day: Day for future prediction (added to last day)
        input_treatment: Treatment for future prediction
        slice_indices: Specific slices to process, or None for auto-select tumor slices

    Returns:
        Dictionary with 'pred_images', 'pred_masks', 'gt_images', 'slice_indices'
    """
    num_sessions = len(days)
    num_slices = image.shape[-1]

    # Auto-select slices with tumor
    if slice_indices is None:
        tumor_mask = (label > 0).any(axis=0)  # (H, W, D)
        if quick_inference:
            # Fast path: choose ONE best slice only
            areas = [int(tumor_mask[:, :, d].sum()) for d in range(num_slices)]
            best = int(np.argmax(areas)) if max(areas, default=0) > 0 else num_slices // 2
            slice_indices = [best]
        else:
            slice_indices = [d for d in range(num_slices) if tumor_mask[:, :, d].sum() > 100]
            if not slice_indices:
                center = num_slices // 2
                slice_indices = list(range(max(0, center - 10), min(num_slices, center + 10)))
    elif quick_inference and len(slice_indices) > 1:
        # Fast path: if user passed multiple slices, keep only first one in quick mode
        slice_indices = [slice_indices[0]]

    if quick_inference:
        # Force fastest practical settings for quick preview
        num_samples = 1
        sampling_steps = min(sampling_steps, 8)
        print(f"  [QuickInference] Using fast settings: slices={len(slice_indices)}, "
              f"samples={num_samples}, steps={sampling_steps}")

    # Select 4 sessions (last 4 if enough, else pad)
    if target_session_mode == 'future':
        # Use last 3 sessions + 1 future session
        avail = list(range(num_sessions))
        if len(avail) >= 3:
            session_indices = avail[-3:] + [avail[-1]]  # placeholder for future
        else:
            session_indices = (avail * 4)[:4]
        target_idx_val = 3  # last position

        # Extend days/treatment for future session
        last_day = days[-1]
        future_day = last_day + (input_day if input_day is not None else 100)
        future_treatment = input_treatment if input_treatment is not None else treatment[-1]

        ext_days = np.append(days, future_day)
        ext_treatment = np.append(treatment, future_treatment)
        # Update session_indices: last one points to extended session
        session_indices[-1] = len(ext_days) - 1
        days = ext_days
        treatment = ext_treatment
    else:
        # Predict last available session
        if num_sessions >= 4:
            session_indices = list(range(num_sessions - 4, num_sessions))
        else:
            session_indices = list(range(num_sessions))
            while len(session_indices) < 4:
                session_indices.insert(0, session_indices[0])
        target_idx_val = 3  # last position

    target_idx = torch.tensor([target_idx_val], dtype=torch.int8, device=device)
    target_session_idx = session_indices[target_idx_val]

    pred_images_all = []
    pred_masks_all = []
    gt_images_all = []

    print(f"  Processing {len(slice_indices)} slices x {num_samples} samples each...")

    for slice_idx in tqdm(slice_indices, desc="Slices"):
        slice_preds = []
        slice_masks = []

        gt_img_slice = extract_gt_target_slice(
            image=image,
            session_idx=target_session_idx,
            slice_idx=slice_idx,
            image_size=image_size,
        )
        gt_images_all.append(gt_img_slice)

        for sample_i in range(num_samples):
            x_input, intv_t, treat_code = prepare_2d_input(
                image, days, treatment,
                session_indices, slice_idx,
                image_size, device,
            )

            pred_img, pred_mask = ddim_sample(
                model=model,
                diffusion=diffusion,
                x_input=x_input,
                intv_t=intv_t,
                treat_code=treat_code,
                target_idx=target_idx,
                sampling_steps=sampling_steps,
                eta=0.0,
                device=device,
            )

            slice_preds.append(pred_img.cpu())
            slice_masks.append(torch.sigmoid(pred_mask).cpu())

        # Stack samples: [num_samples, 3, H, W] / [num_samples, 4, H, W]
        preds = torch.cat(slice_preds, dim=0)
        masks = torch.cat(slice_masks, dim=0)

        pred_images_all.append(preds.numpy())
        pred_masks_all.append(masks.numpy())

    return {
        'pred_images': pred_images_all,     # List of [num_samples, 3, H, W]
        'pred_masks': pred_masks_all,        # List of [num_samples, 4, H, W]
        'gt_images': gt_images_all,          # List of [3, H, W] or None
        'slice_indices': slice_indices,
    }


# =============================================================================
# Visualization helpers
# =============================================================================

def save_results(
    results: Dict,
    output_dir: Path,
    patient_id: str,
    show_results: bool = True,
    quick_inference: bool = False,
):
    """Save minimal inference outputs with integrated brain enhancement."""
    save_path = output_dir / patient_id
    save_path.mkdir(parents=True, exist_ok=True)

    slice_indices = results['slice_indices']
    gt_images = results.get('gt_images', None)

    def to_uint8_from_unit(arr: np.ndarray) -> np.ndarray:
        """Convert [0,1] float image to uint8 [0,255]."""
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    def pairwise_stretch_to_uint8(pred_unit: np.ndarray, gt_unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply shared percentile normalization to pred/gt for more comparable appearance."""
        both = np.concatenate([pred_unit.ravel(), gt_unit.ravel()])
        p1, p99 = np.percentile(both, [1, 99])
        if p99 - p1 > 1e-6:
            pred_n = np.clip((pred_unit - p1) / (p99 - p1), 0.0, 1.0)
            gt_n = np.clip((gt_unit - p1) / (p99 - p1), 0.0, 1.0)
        else:
            pred_n = np.clip(pred_unit, 0.0, 1.0)
            gt_n = np.clip(gt_unit, 0.0, 1.0)
        return to_uint8_from_unit(pred_n), to_uint8_from_unit(gt_n)

    def build_brain_mask(img_u8: np.ndarray) -> np.ndarray:
        """Build a conservative full-brain mask for display compositing."""
        try:
            import cv2
            den = cv2.GaussianBlur(img_u8, (3, 3), 0)

            # Conservative mask from low-intensity threshold to preserve full brain extent
            mask = (den > 4).astype(np.uint8) * 255

            # Fallback for very dark slices
            if int(mask.sum()) < int(0.01 * mask.size * 255):
                blur_for_mask = cv2.GaussianBlur(den, (15, 15), 0)
                _, mask = cv2.threshold(
                    blur_for_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            # Close gaps and dilate so no anatomy is trimmed at boundaries
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            mask = cv2.dilate(mask, k_dilate, iterations=1)

            # Keep all meaningful connected components (not only largest) to avoid accidental cuts
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                keep = np.zeros_like(mask)
                min_area = max(64, int(0.002 * mask.size))
                for lab in range(1, num_labels):
                    if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                        keep[labels == lab] = 255
                if keep.sum() > 0:
                    mask = keep

            return mask.astype(np.uint8)
        except Exception:
            return ((img_u8 > 0).astype(np.uint8) * 255)

    def enhance_for_display(
        img_u8: np.ndarray,
        background: int = 255,
        brain_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enhance contrast and apply explicit background fill using brain mask."""
        try:
            import cv2
            den = cv2.GaussianBlur(img_u8, (3, 3), 0)

            if brain_mask is None:
                brain_mask = build_brain_mask(den)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(den)

            bg = np.full_like(enhanced, np.uint8(background))
            final = np.where(brain_mask == 255, enhanced, bg).astype(np.uint8)
            return final
        except Exception:
            return img_u8

    # Keep only fresh outputs in patient folder
    for stale in list(save_path.glob("*.png")) + list(save_path.glob("*.npy")):
        try:
            stale.unlink()
        except Exception:
            pass

    if len(slice_indices) == 0:
        print(f"  No slices to save for {patient_id}")
        return

    # Best slice by strongest non-background mask response
    best_i = 0
    best_score = -1.0
    for i in range(len(slice_indices)):
        masks_i = results['pred_masks'][i]
        avg_mask_i = masks_i.mean(axis=0)
        score = float(np.sum(avg_mask_i[1:].max(axis=0)))
        if score > best_score:
            best_score = score
            best_i = i

    slice_idx = slice_indices[best_i]
    preds = results['pred_images'][best_i]
    gt_img = None if gt_images is None else gt_images[best_i]
    avg_img = preds.mean(axis=0)

    pred_t1c_unit = np.clip((avg_img[0] + 1.0) / 2.0, 0.0, 1.0)

    if gt_img is not None:
        gt_t1c_unit = np.clip((gt_img[0] + 1.0) / 2.0, 0.0, 1.0)
        pred_t1c_u8, gt_t1c_u8 = pairwise_stretch_to_uint8(pred_t1c_unit, gt_t1c_unit)
    else:
        print("  GT not available; using prediction for GT panel")
        pred_t1c_u8 = to_uint8_from_unit(pred_t1c_unit)
        gt_t1c_u8 = pred_t1c_u8.copy()

    # Apply enhancement with one shared mask so both sides keep black-only background
    shared_mask_source = gt_t1c_u8 if gt_img is not None else pred_t1c_u8
    shared_mask = build_brain_mask(shared_mask_source)
    pred_t1c_u8 = enhance_for_display(pred_t1c_u8, background=0, brain_mask=shared_mask)
    gt_t1c_u8 = enhance_for_display(gt_t1c_u8, background=0, brain_mask=shared_mask)

    try:
        from PIL import Image

        if quick_inference:
            separator = np.full((pred_t1c_u8.shape[0], 6), 255, dtype=np.uint8)
            quick_panel = np.concatenate([pred_t1c_u8, separator, gt_t1c_u8], axis=1)
            Image.fromarray(quick_panel).save(save_path / "pred_vs_gt.png")
            print(f"  [QuickInference] Best slice {slice_idx}: saved pred_vs_gt.png")
        else:
            separator = np.full((pred_t1c_u8.shape[0], 6), 255, dtype=np.uint8)
            full_panel = np.concatenate([pred_t1c_u8, separator, gt_t1c_u8], axis=1)
            Image.fromarray(full_panel).save(save_path / "pred_vs_gt.png")

            # Save predicted tensors in consolidated .npy files
            pred_images_all = np.stack(results['pred_images'], axis=0)
            pred_masks_all = np.stack(results['pred_masks'], axis=0)
            np.save(save_path / "pred_images.npy", pred_images_all)
            np.save(save_path / "pred_masks.npy", pred_masks_all)

            print(
                f"  [FullInference] Best slice {slice_idx}: saved pred_vs_gt.png, "
                f"pred_images.npy, pred_masks.npy"
            )
    except ImportError:
        pass

    print(f"  Saved results to {save_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='TaDiff Inference (consistent with training)')

    # Input sources (two modes)
    parser.add_argument('--patient_file', type=str, default=None,
                        help='Path to patient *_image.npy file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with patient .npy files')
    parser.add_argument('--patient_ids', nargs='+', default=None,
                        help='Patient IDs to process (used with --data_dir)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Filename prefix (e.g. "sub-")')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Architecture params - MUST match training!
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--bdh_expansion', type=int, default=2)

    # Diffusion params - MUST match training!
    parser.add_argument('--max_T', type=int, default=2000,
                        help='Diffusion timesteps (must match training)')
    parser.add_argument('--ddpm_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Noise schedule (must match training)')

    # Sampling params
    parser.add_argument('--sampling_steps', type=int, default=50,
                        help='DDIM sampling steps (subsampled from max_T)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples for uncertainty')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM stochasticity (0=deterministic)')

    # Target session
    parser.add_argument('--mode', type=str, default='last',
                        choices=['last', 'future'],
                        help="'last': predict last session, 'future': predict future")
    parser.add_argument('--input_day', type=float, default=100,
                        help='Days after last session for future prediction')
    parser.add_argument('--input_treatment', type=float, default=1,
                        help='Treatment code for future prediction')

    # Slice selection
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='Specific slice (None = auto-select tumor slices)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--show_results', dest='show_results', action='store_true',
                        help='Show generated inference images automatically using matplotlib')
    parser.add_argument('--no_show_results', dest='show_results', action='store_false',
                        help='Disable automatic result display window after inference')
    parser.set_defaults(show_results=True)

    parser.add_argument('--auto_hparams', dest='auto_hparams', action='store_true',
                        help='Auto-load architecture/diffusion args from nearby hparams.yaml')
    parser.add_argument('--no_auto_hparams', dest='auto_hparams', action='store_false',
                        help='Disable auto-loading settings from hparams.yaml')
    parser.set_defaults(auto_hparams=True)

    parser.add_argument('--quick_inference', action='store_true',
                        help='Quick mode: save one best-slice image with two parts [Pred T1c | GT T1c]')

    return parser.parse_args()


def main():
    args = parse_args()
    args = apply_checkpoint_hparams(args)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Inference] Device: {device}")

    # Load model with CORRECT architecture params
    print("[Inference] Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        image_size=args.image_size,
        model_channels=args.model_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        bdh_expansion=args.bdh_expansion,
    )

    # Create diffusion with SAME schedule as training
    diffusion = GaussianDiffusion(
        T=args.max_T,
        schedule=args.ddpm_schedule,
    )
    print(f"[Inference] Diffusion: T={args.max_T}, schedule={args.ddpm_schedule}, "
          f"sampling_steps={args.sampling_steps}")

    # Gather patient files
    patient_files = []
    if args.patient_file:
        patient_files.append(args.patient_file)
    elif args.data_dir:
        if args.patient_ids:
            for pid in args.patient_ids:
                patient_files.append(
                    os.path.join(args.data_dir, f"{args.prefix}{pid}_image.npy")
                )
        else:
            # Process all patients in directory
            data_path = Path(args.data_dir)
            for f in sorted(data_path.glob("*_image.npy")):
                patient_files.append(str(f))
    else:
        raise ValueError("Provide either --patient_file or --data_dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each patient
    for pf in patient_files:
        patient_id = Path(pf).stem.replace('_image', '')
        print(f"\n[Inference] Processing patient: {patient_id}")

        data = load_patient_data(pf)
        slice_indices = [args.slice_idx] if args.slice_idx is not None else None

        results = run_inference(
            model=model,
            diffusion=diffusion,
            image=data['image'],
            label=data.get('label', np.zeros_like(data['image'][0:1])),
            days=data['days'],
            treatment=data['treatment'],
            device=device,
            image_size=args.image_size,
            sampling_steps=args.sampling_steps,
            num_samples=args.num_samples,
            target_session_mode=args.mode,
            input_day=args.input_day,
            input_treatment=args.input_treatment,
            slice_indices=slice_indices,
            quick_inference=args.quick_inference,
        )

        save_results(
            results,
            output_dir,
            patient_id,
            show_results=args.show_results,
            quick_inference=args.quick_inference,
        )

    print("\n[Inference] Complete!")


if __name__ == '__main__':
    main()
