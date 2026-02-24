"""
Treatment-Aware Diffusion Model with BDH Integration 

This module provides the PyTorch Lightning wrapper for the TaDiff-Net
with integrated BDH (Baby Dragon Hatchling) block for efficient
longitudinal medical image modeling.

Key Features:
- Gaussian diffusion process for image generation
- Treatment and temporal conditioning
- Auxiliary segmentation loss
- BDH block for memory-efficient long-range modeling
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from pytorch_lightning import LightningModule, Callback
from torch.optim import AdamW, SGD

# Handle imports
try:
    from src.net.tadiff_dit_arch import TaDiff_DiT as TaDiff_Net
    from src.net.ssim import SSIM
    from src.net.diffusion import GaussianDiffusion
except ImportError:
    from tadiff_dit_arch import TaDiff_DiT as TaDiff_Net
    from ssim import SSIM
    from diffusion import GaussianDiffusion

try:
    from monai.optimizers.lr_scheduler import WarmupCosineSchedule
    from monai.losses.dice import DiceLoss
    from monai.metrics import DiceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    # Fallback implementations
    class DiceLoss:
        def __init__(self, **kwargs):
            pass
        def __call__(self, pred, target):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-5) / (union + 1e-5)
            return 1 - dice
    
    class DiceMetric:
        def __init__(self, **kwargs):
            self.scores = []
        def __call__(self, pred, target):
            pred_bin = (pred > 0.5).float()
            intersection = (pred_bin * target).sum()
            union = pred_bin.sum() + target.sum()
            self.scores.append((2 * intersection / (union + 1e-6)).item())
        def aggregate(self):
            return np.mean(self.scores) if self.scores else 0.0
        def reset(self):
            self.scores = []


class Tadiff_model(LightningModule):
    """
    PyTorch Lightning module for Treatment-Aware Diffusion with BDH Block.
    
    
    Args:
        config: Configuration object containing model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = config
        
        # Extract BDH-specific config with defaults
        bdh_layers = getattr(self.cfg, 'bdh_layers', 4)
        bdh_expansion = getattr(self.cfg, 'bdh_expansion', 4)
        
        # Initialize the TaDiff-Net with BDH block
        self._model = TaDiff_Net(
            image_size=self.cfg.image_size, 
            in_channels=getattr(self.cfg, 'in_channels', 12),
            out_channels=getattr(self.cfg, 'out_channels', 7),
            model_channels=self.cfg.model_channels, 
            num_res_blocks=getattr(self.cfg, 'num_res_blocks', 2),
            channel_mult=getattr(self.cfg, 'channel_mult', (1, 2, 4, 8)),
            attention_resolutions=getattr(self.cfg, 'attention_resolutions', (8, 16)),
            num_heads=getattr(self.cfg, 'num_heads', 8),
            hidden_size=getattr(self.cfg, 'hidden_size', 768),
            depth=getattr(self.cfg, 'depth', 12),
            patch_size=getattr(self.cfg, 'patch_size', 16),
            bdh_layers=bdh_layers,
            bdh_expansion=bdh_expansion,
            use_hybrid_embed=getattr(self.cfg, 'use_hybrid_embed', True),
        )
        
        # Diffusion process 
        self.diffusion = GaussianDiffusion(
            T=self.cfg.max_T, 
            schedule=self.cfg.ddpm_schedule,
            device='cpu'  # Will be moved to correct device during training
        )
        
        #  Precompute alphabar on CPU, move to device during forward
        self.register_buffer(
            'alphabar_np', 
            torch.from_numpy(np.cumprod(1 - np.linspace(1e-4, 2e-2, self.cfg.max_T))).float()
        )
        
        # Training state
        self.best_val_loss = None
        self.best_val_epoch = 0
        self.val_step_outputs = []
        
        # Loss functions
        self.dilation_filters = torch.ones(1, 1, 11, 11) / 121.0  # Normalized
        
        if HAS_MONAI:
            self.dice = DiceLoss(
                smooth_nr=0, 
                smooth_dr=1e-5, 
                squared_pred=True, 
                to_onehot_y=False, 
                sigmoid=True, 
                reduction="none"
            )
            self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        else:
            self.dice = DiceLoss()
            self.dice_metric = DiceMetric()
        
        # SSIM for optional perceptual loss
        #  data_range=2.0 because images are in [-1, 1] range (span = 2.0)
        self.ssim_loss = SSIM(data_range=2.0, size_average=True, win_size=11, channel=3)

    def forward(self, x, timesteps, intv_t, treat_code, i_tg=None):
        """Forward pass through the diffusion model."""
        return self._model(x, timesteps, intv_t, treat_code, i_tg)
    
    def load_model(self, path=None, device='cuda:0'):
        """Load pretrained model weights."""
        if path is not None:
            state_dict = torch.load(path, map_location=device)
            # Handle potential key mismatches
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self._model.load_state_dict(state_dict, strict=False)
        self._model.eval().to(device)
        print('Model loaded successfully!')

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if getattr(self.cfg, 'opt', 'adamw') == 'adamw':
            optimizer = AdamW(
                self.trainer.model.parameters(), 
                lr=float(self.cfg.lr), 
                weight_decay=getattr(self.cfg, 'weight_decay', 0.01),
                betas=(0.9, 0.999),
            )
        else:
            optimizer = SGD(
                self.trainer.model.parameters(), 
                lr=float(self.cfg.lr), 
                momentum=0.9, 
                nesterov=True,
                weight_decay=getattr(self.cfg, 'weight_decay', 0.01)
            )
            
        self.loss_function = F.mse_loss
        
        # Calculate total training steps
        if hasattr(self.trainer, 'num_devices'):
            num_devices = (
                torch.cuda.device_count()
                if self.trainer.num_devices == -1
                else int(self.trainer.num_devices)
            )
        else:
            num_devices = 1
        
        accumulate = getattr(self.cfg, 'accumulate_grad_batches', 1)
        
        if getattr(self.cfg, 'max_epochs', 0) > 0:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // accumulate // num_devices
            total_steps = steps_per_epoch * self.cfg.max_epochs
        else:
            total_steps = getattr(self.cfg, 'max_steps', 100000)
        
        warmup_steps = getattr(self.cfg, 'warmup_steps', total_steps // 10)
        
        if HAS_MONAI:
            scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer, 
                    warmup_steps=warmup_steps, 
                    t_total=total_steps
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
        else:
            # Fallback: cosine annealing with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }

        return [optimizer], [scheduler]

    def get_loss(self, batch, mode='train'):
        """
        Compute training or validation loss - CORRECTED VERSION
        
        FIXES:
        1. Correct noise sampling at timestep t
        2. Proper x0 -> xt forward diffusion
        3. Correct loss weighting by noise level
        4. Fixed target index handling
        """
        imgs = batch["image"]  # [B, S, C, H, W] or [B, S*C, H, W]
        label = batch["label"]  # [B, S, H, W]
        days = batch["days"]    # [B, 4]
        treatments = batch["treatments"] if "treatments" in batch else batch.get("treatment")
        
        # Handle different input formats
        if imgs.dim() == 5:
            b, s, c, h, w = imgs.shape
            imgs = imgs.view(b, s * c, h, w)
        else:
            b, sc, h, w = imgs.shape
            s = 4
            c = sc // s
        
        # Extract day intervals
        s1_days = days[:, 0]
        s2_days = days[:, 1]
        s3_days = days[:, 2]
        t_days = days[:, 3]
        
        # Proper target index selection
        if mode == 'train' and np.random.random() > 0.5:
            # Random target during training for robustness
            i_tg = torch.randint(0, s, (b,), device=self.device)
            # Handle edge cases based on session availability
            mask_same_s2_s3 = (s2_days != s1_days) & (s3_days == s2_days)
            mask_same_s1_s2 = (s2_days == s1_days) & (s3_days != s2_days)
            
            if np.random.random() > 0.5:
                i_tg[mask_same_s2_s3] = -1
            else:
                i_tg[mask_same_s2_s3] = 0
                
            if np.random.random() > 0.5:
                i_tg[mask_same_s1_s2] = -1
            else:
                i_tg[mask_same_s1_s2] = -2
        else:
            # Always predict last session during validation
            i_tg = -torch.ones((b,), dtype=torch.long, device=self.device)
        
        # Convert negative indices to positive
        i_tg_positive = torch.where(i_tg < 0, s + i_tg, i_tg).long()
        
        # Treatment embeddings
        treat_cond = [treatments[:, i].float() for i in range(4)]
        intvs = [days[:, i].float() for i in range(4)]
        
        # Get target images and labels
        imgs_reshaped = imgs.view(b, s, c, h, w)
        gt_img = torch.stack([imgs_reshaped[i, i_tg_positive[i]] for i in range(b)], dim=0)
        gt_label = torch.stack([label[i, i_tg_positive[i]] for i in range(b)], dim=0)
        
        # Sample diffusion timesteps (1-indexed for diffusion.sample)
        t = torch.randint(1, self.diffusion.T + 1, (b,), device=self.device)
        
        # Get proper noise level weights
        t_idx = (t - 1).clamp(0, self.diffusion.T - 1)
        alphabar_t = self.diffusion.alphabar[t_idx].to(self.device)
        
        #Forward diffusion with correct equation
        # x_t = sqrt(αbar_t) * x_0 + sqrt(1 - αbar_t) * ε
        xt, epsilon = self.diffusion.sample(gt_img, t)
        
        # Handle masked sessions (when target equals last available)
        maskout_batch = (s3_days == t_days)
        
        # Prepare input: replace target session with noisy version
        imgs_input = imgs_reshaped.clone()
        label_input = label.clone()
        
        for i in range(b):
            idx = i_tg_positive[i].item()
            if maskout_batch[i]:
                # Mask out all sessions for this batch item
                imgs_input[i] = 0.0
                label_input[i] = 0.0
            # Replace target with noisy version
            label_input[i, idx] = gt_label[i]
            imgs_input[i, idx] = xt[i]
        
        # Reshape for model input
        x_input = imgs_input.view(b, s * c, h, w)
        
        # Forward pass
        out = self.forward(
            x_input,
            t.float(),
            intv_t=intvs,
            treat_code=treat_cond,
            i_tg=i_tg
        )
        
        # Split predictions: [B, 7, H, W] -> masks [B, 4, H, W] + image [B, 3, H, W]
        mask_pred = out[:, 0:4, :, :]
        img_pred = out[:, 4:7, :, :]
        
        #  Compute weighted loss based on tumor region
        # Weight more heavily in tumor regions
        tumor_weight = label_input.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        tumor_weight = tumor_weight * torch.exp(-tumor_weight * 0.01)  # Soft weighting
        tumor_weight = F.conv2d(
            tumor_weight,
            self.dilation_filters.to(tumor_weight.device),
            padding='same'
        ) + 1.0
        
        #  Image reconstruction loss (predict noise, not image)
        # MSE between predicted noise and actual noise
        loss_img = torch.mean(tumor_weight * (img_pred - epsilon) ** 2)
        mse = F.mse_loss(img_pred, epsilon)
        
        # Segmentation loss (Dice)
        dice_loss_raw = self.dice(mask_pred, label_input)
        if isinstance(dice_loss_raw, torch.Tensor) and dice_loss_raw.dim() > 1:
            dice_loss_raw = dice_loss_raw.squeeze()
        
        # Weight segmentation loss WITHOUT in-place modification
        # At high noise, segmentation is harder, so down-weight
        sqrt_alphabar = torch.sqrt(alphabar_t).view(b, 1) if alphabar_t.dim() == 1 else torch.sqrt(alphabar_t)
        
        # Create weight mask (no in-place modification of dice_loss_raw)
        if dice_loss_raw.dim() > 1:
            # dice_loss_raw shape: [B, 4] (4 sessions)
            weight_mask = torch.ones_like(dice_loss_raw)
            
            for i in range(b):
                idx = i_tg_positive[i].item()
                weight = sqrt_alphabar[i] if sqrt_alphabar.dim() > 0 else sqrt_alphabar
                
                if maskout_batch[i]:
                    # Only count target session loss
                    weight_mask[i, :] = 0.0
                    weight_mask[i, idx] = weight
                else:
                    # Weight only the target session
                    weight_mask[i, idx] = weight
            
            # Apply weights via multiplication (preserves computational graph)
            dice_loss_weighted = dice_loss_raw * weight_mask
            dice_loss_final = torch.mean(dice_loss_weighted)
        else:
            # dice_loss_raw is scalar or 1D
            dice_loss_final = torch.mean(dice_loss_raw)
        
        # Combined loss
        aux_weight = getattr(self.cfg, 'aux_loss_w', 0.1)
        loss = loss_img + dice_loss_final * aux_weight
        
        # Compute metrics
        with torch.no_grad():
            mask_pred_bin = (torch.sigmoid(mask_pred) > 0.5).float()
            self.dice_metric(mask_pred_bin, label_input)
            dice_score = self.dice_metric.aggregate()
            self.dice_metric.reset()
        
        return loss, mse, dice_score
        
    def training_step(self, batch, batch_idx):
        """Execute a training step."""
        loss, mse, dice_seg = self.get_loss(batch, mode='train')
        
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse, sync_dist=True, on_epoch=False, prog_bar=False)
        self.log("train_dice", dice_seg, sync_dist=True, on_epoch=False, prog_bar=False)
        
        return {"loss": loss, "mse": mse, "dice_seg": dice_seg}

    def validation_step(self, batch, batch_idx):
        """Execute a validation step."""
        loss, mse, dice = self.get_loss(batch, mode='val')
        
        self.val_step_outputs.append({"val_loss": loss.detach()})
        
        self.log("val_loss", loss.item(), sync_dist=True, prog_bar=True)
        self.log("val_mse", mse.item(), sync_dist=True, prog_bar=False)
        self.log("val_dice", dice if isinstance(dice, float) else dice.item(), sync_dist=True, prog_bar=False)
        
        return {"val_loss": loss, "val_mse": mse, "val_dice": dice}
    
    def on_validation_epoch_end(self):
        """Handle end of validation epoch."""
        if self.val_step_outputs:
            avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
            
            if self.best_val_loss is None or avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.best_val_epoch = self.current_epoch
            
            if self.global_rank == 0:
                print(f"\nValidation Epoch {self.current_epoch}: "
                      f"Loss={avg_loss:.4f}, Best={self.best_val_loss:.4f} (epoch {self.best_val_epoch})")
        
        self.val_step_outputs.clear()


class TaDiffCallback(Callback):
    """
    Validation callback for logging predictions and metrics - CORRECTED VERSION
    """
    
    def __init__(self, val_batch, config):
        super().__init__()
        self.config = config
        self.val_batch = val_batch
        
        # Process validation batch
        self._prepare_val_data()
    
    def _prepare_val_data(self):
        """Prepare validation data for visualization."""
        imgs = self.val_batch["image"]
        labels = self.val_batch["label"]
        days = self.val_batch["days"]
        treatments = self.val_batch.get("treatments", self.val_batch.get("treatment"))
        
        # Handle different input formats
        if imgs.dim() == 5:
            b, s, c, h, w = imgs.shape
            self.imgs = imgs.view(b, s * c, h, w)
        else:
            self.imgs = imgs
            b, sc, h, w = imgs.shape
            s = 4
            c = sc // s
        
        self.labels = labels
        
        # Prepare conditioning
        self.intvs = [days[:, i].float() for i in range(4)]
        self.treat_cond = [treatments[:, i].float() for i in range(4)]
        
        # Prepare noisy input for visualization
        noise = torch.randn_like(self.imgs[:, -3:, :, :])
        self.val_input = self.imgs.clone()
        self.val_input[:, -3:, :, :] = noise
        
        # Create diffusion for sampling
        self.diffusion = GaussianDiffusion(
            T=self.config.max_T,
            schedule=self.config.ddpm_schedule,
            device='cpu'
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate and log sample predictions."""
        if pl_module.global_rank != 0:
            return
        
        device = pl_module.device
        
        # Move data to device
        val_input = self.val_input.to(device)
        intvs = [x.to(device) for x in self.intvs]
        treat_cond = [x.to(device) for x in self.treat_cond]
        
        # Generate predictions using corrected inverse diffusion
        with torch.no_grad():
            # Move diffusion to correct device
            self.diffusion.alphabar = self.diffusion.alphabar.to(device)
            self.diffusion.alpha = self.diffusion.alpha.to(device)
            self.diffusion.beta = self.diffusion.beta.to(device)
            self.diffusion.sqrt_alphabar = self.diffusion.sqrt_alphabar.to(device)
            self.diffusion.sqrt_one_minus_alphabar = self.diffusion.sqrt_one_minus_alphabar.to(device)
            
            preds, masks = self.diffusion.ddim_inverse(
                net=pl_module._model,
                x=val_input,
                intv=intvs,
                treat_cond=treat_cond,
                steps=min(50, self.config.max_T),
                eta=0.0,
                device=device
            )
        
        masks = torch.sigmoid(masks)
        
        # Log images if logger supports it
        if hasattr(trainer.logger, 'log_image'):
            # Log sample predictions
            for i in range(min(2, preds.shape[0])):
                trainer.logger.log_image(
                    key=f"pred_sample_{i}",
                    images=[
                        preds[i, 0].cpu().numpy(),
                        preds[i, 1].cpu().numpy(),
                        preds[i, 2].cpu().numpy(),
                        masks[i, -1].cpu().numpy(),
                    ],
                    caption=["T1c", "FLAIR", "T2", "Mask"]
                )


# Backward compatibility alias
MyCallback = TaDiffCallback
