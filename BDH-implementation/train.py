

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger

from torch.optim import AdamW

# Add repo root to path
REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import modules
from src.net.tadiff_dit_arch import TaDiff_DiT
from src.net.diffusion import GaussianDiffusion
from src.net.ssim import SSIM
from src.data.sailor_dataset import SAILORDatasetFiltered
from src.visualization.visualizer import Visualizer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with all necessary parameters."""
    
    # Data
    data_dir: str = "./data/sailor"
    train_ratio: float = 0.8
    num_slices_per_volume: int = 10
    
    # Model Architecture
    image_size: int = 192
    in_channels: int = 12  # 4 sessions × 3 modalities (T1c, FLAIR, T2)
    out_channels: int = 7  # 4 masks + 3 image channels
    model_channels: int = 256
    hidden_size: int = 1024
    depth: int = 18
    num_heads: int = 16
    patch_size: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_checkpoint: bool = False
    use_hybrid_embed: bool = True  # Use CNN stem
    bdh_expansion: int = 2
    bdh_layers: int = 6
    
    # Diffusion
    max_T: int = 2000
    ddpm_schedule: str = "cosine"
    
    # Training
    batch_size: int = 1
    max_epochs: int = 3000
    max_steps: int = 80000
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.5
    lr_scheduler: str = "onecycle"  # onecycle | cosine | plateau
    onecycle_pct_start: float = 0.3

    # Convergence helpers
    curriculum_epochs: int = 120
    timestep_sampling_power: float = 1.3
    uniform_timestep_prob: float = 0.35
    snr_loss_weighting: bool = True
    min_diffusion_weight: float = 0.05
    
    # Loss weights
    diffusion_loss_weight: float = 1.0
    seg_loss_weight: float = 0.1
    ssim_loss_weight: float = 0.1
    x0_l1_loss_weight: float = 0.35
    x0_charbonnier_loss_weight: float = 0.2
    x0_tv_loss_weight: float = 0.02
    
    # Hardware
    devices: int = 1
    precision: str = "32"
    num_workers: int = 8
    
    # Logging
    experiment_name: str = "tadiff-dit"
    log_dir: str = "./logs"
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    save_top_k: int = 3
    checkpoint_every_n_epochs: int = 10
    visualize_every_n_checkpoints: int = 5
    show_visualizer_window: bool = False
    
    # Misc
    seed: int = 42
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        assert self.image_size % self.patch_size == 0
        assert self.hidden_size % self.num_heads == 0




class SAILORDataModule(LightningDataModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        data_path = Path(self.config.data_dir)
        
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {self.config.data_dir}")
        
        # Find all patients
        image_files = list(data_path.glob("*_image.npy"))
        patient_ids = sorted([f.stem.replace("_image", "") for f in image_files])
        
        if len(patient_ids) == 0:
            raise ValueError(f"No patient data found in {self.config.data_dir}")
        
        print(f"[DataModule] Found {len(patient_ids)} patients")
        
        # Split patients
        import random
        random.seed(self.config.seed)
        shuffled = patient_ids.copy()
        random.shuffle(shuffled)
        
        num_patients = len(shuffled)
        if num_patients == 1:
            train_patients = shuffled
            val_patients = shuffled
            test_patients = []
        elif num_patients == 2:
            train_patients = shuffled[:1]
            val_patients = shuffled[1:2]
            test_patients = []
        else:
            n_train = int(num_patients * self.config.train_ratio)
            n_train = max(1, min(num_patients - 2, n_train))
            remaining = num_patients - n_train
            n_val = max(1, remaining // 2)

            train_patients = shuffled[:n_train]
            val_patients = shuffled[n_train:n_train + n_val]
            test_patients = shuffled[n_train + n_val:]
        
        print(f"[DataModule] Split: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")
        
        if SAILORDatasetFiltered is None:
            raise ImportError("SAILORDatasetFiltered not available. Check sailor_dataset.py")
        
        if stage == "fit" or stage is None:
            self.train_dataset = SAILORDatasetFiltered(
                data_dir=str(data_path),
                patient_ids=train_patients,
                mode='train',
                image_size=self.config.image_size,
                num_slices_per_volume=self.config.num_slices_per_volume,
                augment=True,
            )
            
            self.val_dataset = SAILORDatasetFiltered(
                data_dir=str(data_path),
                patient_ids=val_patients,
                mode='val',
                image_size=self.config.image_size,
                num_slices_per_volume=5,
                augment=False,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = SAILORDatasetFiltered(
                data_dir=str(data_path),
                patient_ids=test_patients,
                mode='test',
                image_size=self.config.image_size,
                num_slices_per_volume=5,
                augment=False,
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = torch.stack([b['image'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        days = torch.stack([b['days'] for b in batch])
        treatment = torch.stack([b['treatment'] for b in batch])
        target_idx = torch.stack([b['target_idx'] for b in batch])
        
        # Labels should be binary [0, 1]
        labels = (labels > 0.5).float()
        
        return {
            'image': images,
            'label': labels,
            'days': days,
            'treatment': treatment,
            'target_idx': target_idx,
        }




class TaDiffDiTLitModule(LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        
        self.model = TaDiff_DiT(
            image_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            model_channels=config.model_channels,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            patch_size=config.patch_size,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            use_hybrid_embed=config.use_hybrid_embed,
            bdh_expansion=config.bdh_expansion,
            bdh_layers=config.bdh_layers,
        )
        
        
        self.diffusion = GaussianDiffusion(
            T=config.max_T,
            schedule=config.ddpm_schedule,
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, win_size=11, channel=3)
        
        # Metrics
        self.val_losses = []
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] Total parameters: {total_params:,}")
        print(f"[Model] Trainable parameters: {trainable_params:,}")
    
    def forward(self, x, timesteps, intv_t, treat_code, i_tg):
        return self.model(x, timesteps, intv_t, treat_code, i_tg)

    @staticmethod
    def _total_variation_loss(x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w
    
    def _prepare_batch(self, batch: Dict, is_train: bool = True) -> Tuple:
        """Prepare batch data for model input."""
        images = batch['image']      # [B, 12, H, W]
        labels = batch['label']      # [B, 4, H, W]
        days = batch['days']         # [B, 4]
        treatment = batch['treatment']  # [B, 4]
        target_idx = batch['target_idx']  # [B]
        
        B = images.shape[0]
        device = images.device
        
        # Convert to list format expected by model
        intv_t = [days[:, i] for i in range(4)]
        treat_code = [treatment[:, i] for i in range(4)]
        
        # Sample random timesteps (1-indexed) with optional curriculum
        if is_train:
            curriculum_epochs = max(1, getattr(self.config, 'curriculum_epochs', 30))
            progress = min(1.0, float(self.current_epoch + 1) / float(curriculum_epochs))
            t_max = max(10, int(self.config.max_T * progress))

            sampling_power = max(1e-6, float(getattr(self.config, 'timestep_sampling_power', 1.0)))
            u = torch.rand((B,), device=device).pow(sampling_power)
            t_biased = (1 + u * (t_max - 1)).long()

            # Mix in uniform timesteps to keep high-noise denoising well-trained
            # while still benefiting from curriculum/biased sampling.
            uniform_prob = float(getattr(self.config, 'uniform_timestep_prob', 0.0))
            uniform_prob = min(1.0, max(0.0, uniform_prob))
            if uniform_prob > 0.0:
                t_uniform = torch.randint(1, t_max + 1, (B,), device=device)
                choose_uniform = torch.rand((B,), device=device) < uniform_prob
                t = torch.where(choose_uniform, t_uniform, t_biased)
            else:
                t = t_biased

            t = t.float()
        else:
            t = torch.randint(1, self.config.max_T + 1, (B,), device=device).float()
        
        return images, labels, intv_t, treat_code, target_idx, t
    
    def _compute_diffusion_loss(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        intv_t: List[torch.Tensor],
        treat_code: List[torch.Tensor],
        target_idx: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = images.shape[0]
        device = images.device
        
        # Handle target indices (convert negative to positive)
        target_session_idx = target_idx.long()
        target_session_idx = torch.where(target_session_idx < 0, 4 + target_session_idx, target_session_idx)
        
        # Extract ground truth target images
        gt_images = []
        for b in range(B):
            t_idx = target_session_idx[b].item()
            start_ch = t_idx * 3
            gt_images.append(images[b, start_ch:start_ch+3])
        gt_images = torch.stack(gt_images, dim=0)  # [B, 3, H, W]
        
        # Sample noise
        noise = torch.randn_like(gt_images)
        
        
        t_idx = (t.long() - 1).clamp(0, self.config.max_T - 1)
        
        # Explicitly move the CPU-based schedule to the current device
        sqrt_alphabar = self.diffusion.sqrt_alphabar.to(device)[t_idx]
        sqrt_one_minus_alphabar = self.diffusion.sqrt_one_minus_alphabar.to(device)[t_idx]
        # -------------------------------------------------
        
        # Reshape for broadcasting
        sqrt_alphabar = sqrt_alphabar.view(B, 1, 1, 1)
        sqrt_one_minus_alphabar = sqrt_one_minus_alphabar.view(B, 1, 1, 1)
        
        # Forward diffusion equation
        xt = sqrt_alphabar * gt_images + sqrt_one_minus_alphabar * noise
        
        # Create noisy input by replacing target session
        noisy_input = images.clone()
        for b in range(B):
            t_idx_b = target_session_idx[b].item()
            start_ch = t_idx_b * 3
            noisy_input[b, start_ch:start_ch+3] = xt[b]
        
        # Forward pass through model
        output = self.model(noisy_input, t, intv_t, treat_code, target_idx)
        
        # Split output
        pred_masks = output[:, 0:4, :, :]
        pred_noise = output[:, 4:7, :, :]
        
        # Diffusion loss (per-sample)
        mse_per_sample = ((pred_noise - noise) ** 2).mean(dim=(1, 2, 3))  # [B]
        diffusion_loss_raw = mse_per_sample.mean()

        # SNR-weighted diffusion objective to avoid high-noise timestep domination
        if getattr(self.config, 'snr_loss_weighting', True):
            alphabar_t = (sqrt_alphabar[:, 0, 0, 0] ** 2).clamp(1e-8, 1 - 1e-8)
            snr = alphabar_t / (1.0 - alphabar_t)
            weight = (snr / (snr + 1.0)).clamp(min=getattr(self.config, 'min_diffusion_weight', 0.05))
            diffusion_loss = (weight * mse_per_sample).mean()
        else:
            diffusion_loss = diffusion_loss_raw
        
        # Segmentation loss
        gt_masks = labels
        seg_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        
        # SSIM Loss
        x0_pred = (xt - sqrt_one_minus_alphabar * pred_noise) / (sqrt_alphabar + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        x0_pred_01 = (x0_pred + 1.0) / 2.0
        gt_images_01 = (gt_images.detach() + 1.0) / 2.0
        
        ssim_val = self.ssim_loss(x0_pred_01, gt_images_01)
        ssim_loss = 1.0 - ssim_val

        # Direct reconstruction signal improves intensity fidelity and reduces grainy outputs.
        x0_l1_loss = F.l1_loss(x0_pred, gt_images)
        x0_charbonnier_loss = torch.sqrt((x0_pred - gt_images) ** 2 + 1e-6).mean()
        x0_tv_loss = self._total_variation_loss(x0_pred)
        
        # Total weighted loss
        total_loss = (
            self.config.diffusion_loss_weight * diffusion_loss +
            self.config.seg_loss_weight * seg_loss +
            self.config.ssim_loss_weight * ssim_loss +
            self.config.x0_l1_loss_weight * x0_l1_loss +
            self.config.x0_charbonnier_loss_weight * x0_charbonnier_loss +
            self.config.x0_tv_loss_weight * x0_tv_loss
        )
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'diffusion_loss_raw': diffusion_loss_raw,
            'seg_loss': seg_loss,
            'ssim_loss': ssim_loss,
            'x0_l1_loss': x0_l1_loss,
            'x0_charbonnier_loss': x0_charbonnier_loss,
            'x0_tv_loss': x0_tv_loss,
        }
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images, labels, intv_t, treat_code, target_idx, t = self._prepare_batch(batch, is_train=True)
        
        losses = self._compute_diffusion_loss(
            images, labels, intv_t, treat_code, target_idx, t
        )
        
        self.log('train/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('train/diffusion_loss', losses['diffusion_loss'], sync_dist=True)
        self.log('train/diffusion_loss_raw', losses['diffusion_loss_raw'], sync_dist=True)
        self.log('train/seg_loss', losses['seg_loss'], sync_dist=True)
        self.log('train/ssim_loss', losses['ssim_loss'], sync_dist=True)
        self.log('train/x0_l1_loss', losses['x0_l1_loss'], sync_dist=True)
        self.log('train/x0_charbonnier_loss', losses['x0_charbonnier_loss'], sync_dist=True)
        self.log('train/x0_tv_loss', losses['x0_tv_loss'], sync_dist=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images, labels, intv_t, treat_code, target_idx, t = self._prepare_batch(batch, is_train=False)
        
        losses = self._compute_diffusion_loss(
            images, labels, intv_t, treat_code, target_idx, t
        )
        
        self.val_losses.append(losses['loss'].item())
        
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val_loss', losses['loss'], prog_bar=False, sync_dist=True)
        self.log('val/diffusion_loss', losses['diffusion_loss'], sync_dist=True)
        self.log('val/diffusion_loss_raw', losses['diffusion_loss_raw'], sync_dist=True)
        self.log('val/seg_loss', losses['seg_loss'], sync_dist=True)
        self.log('val/ssim_loss', losses['ssim_loss'], sync_dist=True)
        self.log('val/x0_l1_loss', losses['x0_l1_loss'], sync_dist=True)
        self.log('val/x0_charbonnier_loss', losses['x0_charbonnier_loss'], sync_dist=True)
        self.log('val/x0_tv_loss', losses['x0_tv_loss'], sync_dist=True)
        
        return losses['loss']
    
    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = np.mean(self.val_losses)
            self.log('val/avg_loss', avg_loss, sync_dist=True)
        self.val_losses.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        try:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        except:
            steps_per_epoch = 1000 
        
        warmup_steps = self.config.warmup_epochs * steps_per_epoch
        total_steps = self.config.max_epochs * steps_per_epoch
        scheduler_mode = str(getattr(self.config, 'lr_scheduler', 'onecycle')).lower()

        if scheduler_mode == 'onecycle':
            min_pct_start = min(0.9, max(0.05, 2.0 / max(2, total_steps)))
            pct_start = float(getattr(self.config, 'onecycle_pct_start', 0.3))
            pct_start = min(0.9, max(min_pct_start, pct_start))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.lr,
                total_steps=max(1, total_steps),
                pct_start=pct_start,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=max(10.0, self.config.lr / max(self.config.min_lr, 1e-8)),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }

        if scheduler_mode == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=6,
                min_lr=self.config.min_lr,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return self.config.min_lr / self.config.lr + (
                1.0 - self.config.min_lr / self.config.lr
            ) * 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


# =============================================================================
# Inference-During-Training Callback
# =============================================================================

class InferenceDuringTrainingCallback(pl.Callback):
    """
    Run DDIM inference periodically during training so you can visually
    inspect output quality and stop early if the model isn't converging.

    Saves sample images to disk every `every_n_epochs` epochs.
    Also logs to TensorBoard if available.
    """

    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_every_n_epochs: int = 10,
        every_n_checkpoints: int = 5,
        sampling_steps: int = 50,
        output_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.checkpoint_every_n_epochs = max(1, checkpoint_every_n_epochs)
        self.every_n_checkpoints = max(1, every_n_checkpoints)
        self.sampling_steps = sampling_steps
        self.output_dir = Path(output_dir or os.path.join(
            config.log_dir, config.experiment_name, "inference_samples"
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_batch = None
        self.visualizer = Visualizer({
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
        })

    def _get_sample_batch(self, trainer):
        """Cache one validation batch for consistent comparison across epochs."""
        if self._sample_batch is not None:
            return self._sample_batch

        try:
            val_loader = trainer.datamodule.val_dataloader()
            self._sample_batch = next(iter(val_loader))
        except Exception:
            try:
                train_loader = trainer.datamodule.train_dataloader()
                self._sample_batch = next(iter(train_loader))
            except Exception:
                return None

        return self._sample_batch

    @torch.no_grad()
    def _run_ddim(self, model, diffusion, x_input, intv_t, treat_code, target_idx, device):
        """Minimal DDIM sampling matching diffusion.py logic."""
        B = x_input.shape[0]
        H, W = x_input.shape[2], x_input.shape[3]

        times = torch.linspace(diffusion.T, 1, self.sampling_steps + 1, device=device).long()
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        i_tg_positive = torch.where(target_idx < 0, 4 + target_idx, target_idx).long()

        x = x_input.clone()
        noise_init = torch.randn(B, 3, H, W, device=device)
        for b in range(B):
            s = i_tg_positive[b].item()
            x[b, s * 3:(s + 1) * 3] = noise_init[b]

        mask_accum = torch.zeros(B, 4, H, W, device=device)

        for t_curr, t_next in time_pairs:
            t_curr_idx = t_curr - 1
            alphabar_t = diffusion.alphabar[t_curr_idx].to(device)
            sqrt_ab = torch.sqrt(alphabar_t)
            sqrt_1_ab = torch.sqrt(1.0 - alphabar_t)

            t_batch = torch.full((B,), t_curr, device=device, dtype=torch.float32)
            pred = model(x, t_batch, intv_t=intv_t, treat_code=treat_code, i_tg=target_idx)
            eps_pred = pred[:, 4:7]
            mask_pred = pred[:, 0:4]

            xt = torch.stack([
                x[b, i_tg_positive[b].item() * 3:(i_tg_positive[b].item() + 1) * 3]
                for b in range(B)
            ], dim=0)

            x0_pred = (xt - sqrt_1_ab * eps_pred) / (sqrt_ab + 1e-8)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if t_next >= 1:
                t_next_idx = t_next - 1
                alphabar_next = diffusion.alphabar[t_next_idx].to(device)
                sqrt_ab_next = torch.sqrt(alphabar_next)
                dir_coef = torch.sqrt(torch.clamp(1.0 - alphabar_next, min=0.0))
                x_next = sqrt_ab_next * x0_pred + dir_coef * eps_pred
            else:
                x_next = x0_pred

            for b in range(B):
                s = i_tg_positive[b].item()
                x[b, s * 3:(s + 1) * 3] = x_next[b]

            mask_accum += mask_pred

        mask_avg = torch.sigmoid(mask_accum / len(time_pairs))
        return x_next, mask_avg

    def on_train_epoch_end(self, trainer, pl_module):
        """Run inference at the end of selected epochs."""
        epoch = trainer.current_epoch
        epoch_1 = epoch + 1

        # Trigger visualization every N checkpoints, where checkpoints are saved
        # every `checkpoint_every_n_epochs` epochs.
        visualize_every_n_epochs = self.checkpoint_every_n_epochs * self.every_n_checkpoints
        if epoch_1 % visualize_every_n_epochs != 0:
            return
        if pl_module.global_rank != 0:
            return

        batch = self._get_sample_batch(trainer)
        if batch is None:
            return

        device = pl_module.device
        images = batch['image'].to(device)
        days = batch['days'].to(device)
        treatment = batch['treatment'].to(device)
        target_idx = batch['target_idx'].to(device)

        B = images.shape[0]
        intv_t = [days[:, i] for i in range(4)]
        treat_code = [treatment[:, i] for i in range(4)]

        # Extract ground truth target for comparison
        target_session_idx = torch.where(target_idx < 0, 4 + target_idx, target_idx).long()
        gt_images = torch.stack([
            images[b, target_session_idx[b].item() * 3:(target_session_idx[b].item() + 1) * 3]
            for b in range(B)
        ], dim=0)

        pl_module.model.eval()
        pred_img, pred_mask = self._run_ddim(
            model=pl_module.model,
            diffusion=pl_module.diffusion,
            x_input=images,
            intv_t=intv_t,
            treat_code=treat_code,
            target_idx=target_idx,
            device=device,
        )
        pl_module.model.train()

        # Save sample images to disk
        epoch_dir = self.output_dir / f"epoch_{epoch_1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Save and optionally show visual previews using Visualizer
        for b in range(min(B, 4)):
            self.visualizer.save_validation_preview(
                pred_img=pred_img[b].detach().cpu().numpy(),
                gt_img=gt_images[b].detach().cpu().numpy(),
                pred_mask=pred_mask[b].detach().cpu().numpy(),
                save_path=str(epoch_dir / f"b{b}_validation_preview.png"),
                title=f"Epoch {epoch_1} | Batch {b}",
                show=self.config.show_visualizer_window,
            )

        np.save(epoch_dir / "pred_images.npy", pred_img.cpu().numpy())
        np.save(epoch_dir / "gt_images.npy", gt_images.cpu().numpy())
        np.save(epoch_dir / "pred_masks.npy", pred_mask.cpu().numpy())
        print(f"\n[InferenceCallback] Epoch {epoch_1}: Saved validation previews to {epoch_dir}")

        if self.config.show_visualizer_window:
            preview_file = epoch_dir / "b0_validation_preview.png"
            try:
                if os.name == 'nt' and preview_file.exists():
                    os.startfile(str(preview_file))
            except Exception as e:
                print(f"  [InferenceCallback] Could not open preview window: {e}")

        # Log to TensorBoard if available
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_image'):
            try:
                for b in range(min(B, 2)):
                    for ch, name in enumerate(['t1c', 'flair', 't2']):
                        pred_01 = (pred_img[b, ch:ch+1].cpu() + 1) / 2
                        gt_01 = (gt_images[b, ch:ch+1].cpu() + 1) / 2
                        trainer.logger.experiment.add_image(
                            f"inference/b{b}_{name}_pred", pred_01, global_step=epoch
                        )
                        trainer.logger.experiment.add_image(
                            f"inference/b{b}_{name}_gt", gt_01, global_step=epoch
                        )
                    mask_vis = pred_mask[b].max(dim=0, keepdim=True)[0].cpu()
                    trainer.logger.experiment.add_image(
                        f"inference/b{b}_mask", mask_vis, global_step=epoch
                    )
            except Exception as e:
                print(f"  [InferenceCallback] TensorBoard logging failed: {e}")

        # Print quality metrics for quick assessment
        with torch.no_grad():
            mse_val = F.mse_loss(pred_img, gt_images).item()
            pred_01 = (pred_img + 1) / 2
            gt_01 = (gt_images + 1) / 2
            diff = torch.abs(pred_01 - gt_01).mean().item()
            print(f"  [InferenceCallback] MSE: {mse_val:.4f}, MAE: {diff:.4f}")
            print(f"  [InferenceCallback] Pred range: [{pred_img.min():.2f}, {pred_img.max():.2f}]")
            print(f"  [InferenceCallback] If output looks like noise, consider stopping training.\n")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train TaDiff-DiT')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--num_slices_per_volume', type=int, default=10)
    
    # Model — defaults MATCH TrainingConfig to avoid silent mismatches
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bdh_expansion', type=int, default=2)
    parser.add_argument('--bdh_layers', type=int, default=6)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--no_hybrid_embed', action='store_true')
    
    # Diffusion — default matches TrainingConfig
    parser.add_argument('--max_T', type=int, default=2000)
    parser.add_argument('--ddpm_schedule', type=str, default='cosine')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=80000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.5)
    parser.add_argument('--lr_scheduler', type=str, default='onecycle', choices=['onecycle', 'cosine', 'plateau'],
                        help='Learning-rate scheduler mode')
    parser.add_argument('--onecycle_pct_start', type=float, default=0.3,
                        help='Warmup fraction for onecycle scheduler')
    parser.add_argument('--curriculum_epochs', type=int, default=30,
                        help='Epochs to ramp timestep range from easy to full diffusion horizon')
    parser.add_argument('--timestep_sampling_power', type=float, default=1.3,
                        help='>1 biases timestep sampling to lower-noise steps early')
    parser.add_argument('--uniform_timestep_prob', type=float, default=0.35,
                        help='Fraction of batches using uniform timestep sampling within current curriculum range')
    parser.add_argument('--snr_loss_weighting', action='store_true',
                        help='Enable SNR-weighted diffusion loss (recommended for convergence)')
    parser.add_argument('--no_snr_loss_weighting', dest='snr_loss_weighting', action='store_false',
                        help='Disable SNR-weighted diffusion loss')
    parser.set_defaults(snr_loss_weighting=True)
    parser.add_argument('--min_diffusion_weight', type=float, default=0.05,
                        help='Minimum per-sample weight for diffusion loss under SNR weighting')
    
    # Loss
    parser.add_argument('--diffusion_loss_weight', type=float, default=1.0)
    parser.add_argument('--seg_loss_weight', type=float, default=0.1)
    parser.add_argument('--ssim_loss_weight', type=float, default=0.1)
    parser.add_argument('--x0_l1_loss_weight', type=float, default=0.35,
                        help='Weight for direct x0 reconstruction L1 loss')
    parser.add_argument('--x0_charbonnier_loss_weight', type=float, default=0.2,
                        help='Robust x0 reconstruction loss to reduce grainy artifacts')
    parser.add_argument('--x0_tv_loss_weight', type=float, default=0.02,
                        help='Total variation regularizer on x0 prediction')
    
    # Hardware
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Logging
    parser.add_argument('--experiment_name', type=str, default='tadiff-dit')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=10,
                        help='Save periodic checkpoint every N epochs')
    parser.add_argument('--visualize_every_n_checkpoints', type=int, default=5,
                        help='Run validation visualization every N saved checkpoints')
    parser.add_argument('--show_visualizer_window', action='store_true',
                        help='Display matplotlib validation/inference previews (if GUI available)')
    
    # Inference during training
    parser.add_argument('--infer_sampling_steps', type=int, default=80,
                        help='DDIM steps for mid-training inference')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from', type=str, default=None)
    
    return parser.parse_args()


def resolve_resume_checkpoint(config: TrainingConfig, requested_path: Optional[str]) -> Optional[str]:
    
    
    if requested_path:
        requested = Path(requested_path)
        if not requested.exists():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {requested}")
        print(f"[Training] Resuming from explicit checkpoint: {requested}")
        return str(requested)

    auto_last = Path(config.log_dir) / config.experiment_name / "checkpoints" / "last.ckpt"
    if auto_last.exists():
        print(f"[Training] Auto-resume enabled. Found last checkpoint: {auto_last}")
        return str(auto_last)

    print("[Training] No resume checkpoint found. Starting fresh training.")
    return None


def main():
    args = parse_args()
    
    # Create config — all args explicitly passed for full transparency
    config = TrainingConfig(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        num_slices_per_volume=args.num_slices_per_volume,
        image_size=args.image_size,
        model_channels=args.model_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        bdh_expansion=args.bdh_expansion,
        bdh_layers=args.bdh_layers,
        use_checkpoint=args.use_checkpoint,
        use_hybrid_embed=not args.no_hybrid_embed,
        max_T=args.max_T,
        ddpm_schedule=args.ddpm_schedule,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        lr_scheduler=args.lr_scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        curriculum_epochs=args.curriculum_epochs,
        timestep_sampling_power=args.timestep_sampling_power,
        uniform_timestep_prob=args.uniform_timestep_prob,
        snr_loss_weighting=args.snr_loss_weighting,
        min_diffusion_weight=args.min_diffusion_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
        seg_loss_weight=args.seg_loss_weight,
        ssim_loss_weight=args.ssim_loss_weight,
        x0_l1_loss_weight=args.x0_l1_loss_weight,
        x0_charbonnier_loss_weight=args.x0_charbonnier_loss_weight,
        x0_tv_loss_weight=args.x0_tv_loss_weight,
        devices=args.devices,
        precision=args.precision,
        num_workers=args.num_workers,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        visualize_every_n_checkpoints=args.visualize_every_n_checkpoints,
        show_visualizer_window=args.show_visualizer_window,
        seed=args.seed,
        resume_from=args.resume_from,
    )
    
    # Set seed
    pl.seed_everything(config.seed)
    
    # Create data module
    data_module = SAILORDataModule(config)
    
    # Create model
    model = TaDiffDiTLitModule(config)
    
    # Resolve checkpoint resume path (supports continuing after interrupted training)
    resume_ckpt_path = resolve_resume_checkpoint(config, config.resume_from)

    # Callbacks
    periodic_ckpt_dir = Path(config.log_dir) / config.experiment_name / "checkpoints" / "periodic"
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.log_dir) / config.experiment_name / "checkpoints",
            filename="epoch={epoch:03d}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=periodic_ckpt_dir,
            filename="epoch-{epoch:04d}",
            every_n_epochs=config.checkpoint_every_n_epochs,
            save_top_k=-1,
            save_last=False,
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
        ),
    ]
    
    # Add inference-during-training callback on checkpoint-based interval
    callbacks.append(
        InferenceDuringTrainingCallback(
            config=config,
            checkpoint_every_n_epochs=config.checkpoint_every_n_epochs,
            every_n_checkpoints=config.visualize_every_n_checkpoints,
            sampling_steps=args.infer_sampling_steps,
        )
    )
    print(
        f"[Training] Periodic checkpoint every {config.checkpoint_every_n_epochs} epochs; "
        f"validation preview every {config.visualize_every_n_checkpoints} checkpoints "
        f"(every {config.checkpoint_every_n_epochs * config.visualize_every_n_checkpoints} epochs)."
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.experiment_name,
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        devices=config.devices,
        accelerator="auto",
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print("=" * 60)
    print("TaDiff-DiT Training (CORRECTED)")
    print("=" * 60)
    print(f"Data: {config.data_dir}")
    print(f"Architecture: hidden={config.hidden_size}, depth={config.depth}, "
          f"heads={config.num_heads}, patch={config.patch_size}, "
          f"model_ch={config.model_channels}")
    print(f"Diffusion: T={config.max_T}, schedule={config.ddpm_schedule}")
    print(f"Training: epochs={config.max_epochs}, batch={config.batch_size}, "
          f"lr={config.lr}, accum={config.accumulate_grad_batches}")
    print(f"Scheduler: {config.lr_scheduler} (onecycle_pct_start={config.onecycle_pct_start})")
    print(f"Checkpointing: best+last + periodic every {config.checkpoint_every_n_epochs} epochs")
    print(f"Validation previews: every {config.visualize_every_n_checkpoints} checkpoints")
    print(f"Resume ckpt: {resume_ckpt_path if resume_ckpt_path else 'None (fresh start)'}")
    print(f"Precision: {config.precision}")
    
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)
    
    print("\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
