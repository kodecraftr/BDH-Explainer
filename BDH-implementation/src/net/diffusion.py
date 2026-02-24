"""
Gaussian Diffusion Process for DIT-BDH
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List, Tuple


class GaussianDiffusion:
    """
    Gaussian Diffusion process with linear or cosine beta scheduling.
    
    CRITICAL EQUATIONS (from DDPM/DDIM papers):
    
    Forward: x_t = sqrt(αbar_t) * x_0 + sqrt(1 - αbar_t) * ε
    
    DDPM Reverse: x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-αbar_t)) * ε_θ) + sqrt(β̃_t) * z
    
    DDIM Reverse: x_{t-1} = sqrt(αbar_{t-1}) * x_0_pred + sqrt(1 - αbar_{t-1}) * ε_θ
        where x_0_pred = (x_t - sqrt(1 - αbar_t) * ε_θ) / sqrt(αbar_t)
    """
    
    def __init__(self, T: int, schedule: str, device: str = 'cpu'):
        """
        Initialize diffusion process.
        
        Args:
            T: Number of diffusion timesteps
            schedule: 'linear' or 'cosine'
            device: Target device
        """
        self.T = T
        self.device = device
    
        # Initialize noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = torch.linspace(b0, bT, T, device=self.device)
        elif schedule == 'cosine':
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps, device=self.device)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule}. Choose 'linear' or 'cosine'.")
        
        # Precompute useful quantities
        self.alpha = 1.0 - self.beta
        self.alphabar = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alphabar = torch.sqrt(self.alphabar)
        self.sqrt_one_minus_alphabar = torch.sqrt(1.0 - self.alphabar)
        self.sqrt_recip_alphabar = 1.0 / self.sqrt_alphabar
        
        # For DDPM posterior
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_beta = torch.sqrt(self.beta)
        
        # Posterior variance (for DDPM sampling)
        # β̃_t = β_t * (1 - αbar_{t-1}) / (1 - αbar_t)
        alphabar_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphabar[:-1]])
        self.posterior_variance = self.beta * (1.0 - alphabar_prev) / (1.0 - self.alphabar)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)
        
        # Coefficients for posterior mean
        # μ̃_t = (sqrt(αbar_{t-1}) * β_t / (1 - αbar_t)) * x_0 + (sqrt(α_t) * (1 - αbar_{t-1}) / (1 - αbar_t)) * x_t
        self.posterior_mean_coef1 = self.beta * torch.sqrt(alphabar_prev) / (1.0 - self.alphabar)
        self.posterior_mean_coef2 = (1.0 - alphabar_prev) * self.sqrt_alpha / (1.0 - self.alphabar)

    def sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t given x_0.
        
        x_t = sqrt(αbar_t) * x_0 + sqrt(1 - αbar_t) * ε
        
        Args:
            x0: Clean image [B, C, H, W]
            t: Timesteps [B] (1-indexed, range [1, T])
            
        Returns:
            xt: Noisy image at timestep t
            epsilon: The noise that was added
        """
        
        t = t.to(x0.device)
        
        # Convert to 0-indexed for array access
        t_idx = (t - 1).long().clamp(0, self.T - 1)
        
        # Get coefficients with proper shape for broadcasting
        sqrt_alphabar_t = self.sqrt_alphabar[t_idx].to(x0.device)
        sqrt_one_minus_alphabar_t = self.sqrt_one_minus_alphabar[t_idx].to(x0.device)
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        while sqrt_alphabar_t.dim() < x0.dim():
            sqrt_alphabar_t = sqrt_alphabar_t.unsqueeze(-1)
            sqrt_one_minus_alphabar_t = sqrt_one_minus_alphabar_t.unsqueeze(-1)
        
        # Sample noise
        epsilon = torch.randn_like(x0)
        
        # Forward diffusion equation
        xt = sqrt_alphabar_t * x0 + sqrt_one_minus_alphabar_t * epsilon
        
        return xt, epsilon

    def _predict_x0_from_eps(
        self, 
        xt: torch.Tensor, 
        t_idx: torch.Tensor, 
        eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        Args:
            xt: Noisy image [B, C, H, W]
            t_idx: Timestep indices (0-indexed) [B]
            eps_pred: Predicted noise [B, C, H, W]
            
        Returns:
            x0_pred: Predicted clean image
        """
        device = xt.device
        B = xt.shape[0]
        
        # Get coefficients
        sqrt_alphabar = self.sqrt_alphabar[t_idx].to(device).view(B, 1, 1, 1)
        sqrt_one_minus_alphabar = self.sqrt_one_minus_alphabar[t_idx].to(device).view(B, 1, 1, 1)
        
        
        x0_pred = (xt - sqrt_one_minus_alphabar * eps_pred) / (sqrt_alphabar + 1e-8)
        
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        return x0_pred

    def _ddpm_posterior(
        self,
        x0_pred: torch.Tensor,
        xt: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DDPM posterior distribution parameters.
        
        q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, β̃_t I)
        
        μ̃_t = (sqrt(αbar_{t-1}) * β_t / (1 - αbar_t)) * x_0 
             + (sqrt(α_t) * (1 - αbar_{t-1}) / (1 - αbar_t)) * x_t
        
        Args:
            x0_pred: Predicted x_0
            xt: Current noisy image
            t_idx: Timestep indices (0-indexed)
            
        Returns:
            posterior_mean, posterior_variance
        """
        device = xt.device
        B = xt.shape[0]
        
        coef1 = self.posterior_mean_coef1[t_idx].to(device).view(B, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t_idx].to(device).view(B, 1, 1, 1)
        variance = self.posterior_variance[t_idx].to(device).view(B, 1, 1, 1)
        
        posterior_mean = coef1 * x0_pred + coef2 * xt
        
        return posterior_mean, variance

    def TaDiff_inverse(
        self, 
        net: nn.Module,
        x: torch.Tensor,
        intv: Optional[List[torch.Tensor]] = None,
        treat_cond: Optional[List[torch.Tensor]] = None,
        i_tg: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        start_t: Optional[int] = None,
        step_mask: int = 10,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDPM-style inverse diffusion for TaDiff.
        
        Args:
            net: The denoising network
            x: Input tensor [B, 12, H, W] (4 sessions × 3 modalities)
            intv: List of day interval tensors for each session
            treat_cond: List of treatment code tensors
            i_tg: Target session indices [B]
            steps: Number of sampling steps
            start_t: Starting timestep
            step_mask: Number of steps to use for mask averaging
            device: Target device
            
        Returns:
            x0_final: Reconstructed target session image [B, 3, H, W]
            mask_pred: Predicted segmentation masks [B, 4, H, W]
        """
        net.eval()
        
        # Set defaults
        start_t = int(start_t) if start_t is not None else self.T
        steps = int(steps) if steps is not None else self.T
        
        B, C, H, W = x.shape
        x = x.to(device)
        
        # Initialize target indices
        if i_tg is None:
            i_tg = -torch.ones((B,), dtype=torch.long, device=device)
        else:
            i_tg = i_tg.to(device).long()
        
        # Handle negative indices
        i_tg_positive = torch.where(i_tg < 0, 4 + i_tg, i_tg)
        
        # Initialize mask accumulator
        mask_accum = torch.zeros((B, 4, H, W), device=device)
        
        # Calculate mask averaging weights
        T_m = min(step_mask, steps)
        alphabar_weights = self.alphabar[:T_m].to(device)
        w_p = alphabar_weights / alphabar_weights.sum()
        
        
        for timestep in range(start_t, start_t - steps, -1):
            
            t_idx = timestep - 1
            
            # Get noise schedule values
            alpha_t = self.alpha[t_idx].to(device)
            alphabar_t = self.alphabar[t_idx].to(device)
            beta_t = self.beta[t_idx].to(device)
            
            # Prepare noise for stochastic sampling
            if timestep > 1:
                z = torch.randn((B, 3, H, W), device=device)
                alphabar_prev = self.alphabar[t_idx - 1].to(device)
                # Posterior variance: β̃_t = β_t * (1 - αbar_{t-1}) / (1 - αbar_t)
                beta_tilde = beta_t * (1.0 - alphabar_prev) / (1.0 - alphabar_t)
                beta_tilde = torch.clamp(beta_tilde, min=1e-20)
            else:
                z = torch.zeros((B, 3, H, W), device=device)
                beta_tilde = torch.tensor(0.0, device=device)
            
            
            t_tensor = torch.full((B,), timestep, device=device, dtype=torch.float32)
            
            # Network forward pass
            with torch.no_grad():
                pred = net(x, t_tensor, intv_t=intv, treat_code=treat_cond, i_tg=i_tg)
                eps_pred = pred[:, 4:7, :, :]  # Predicted noise (image channels)
                mask_pred = pred[:, 0:4, :, :]  # Predicted masks
            
            # Extract current noisy target from x
            # x shape: [B, 12, H, W] -> reshape to [B, 4, 3, H, W]
            x_sessions = x.view(B, 4, 3, H, W)
            
            # Get target session's noisy image
            xt_list = []
            for b in range(B):
                session_idx = i_tg_positive[b].item()
                xt_list.append(x_sessions[b, session_idx])
            xt = torch.stack(xt_list, dim=0)  # [B, 3, H, W]
            
            
            sqrt_alphabar_t = torch.sqrt(alphabar_t)
            sqrt_one_minus_alphabar_t = torch.sqrt(1.0 - alphabar_t)
            
            x0_pred = (xt - sqrt_one_minus_alphabar_t * eps_pred) / (sqrt_alphabar_t + 1e-8)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)  # Clamp for stability
            
            # DDPM reverse step
            # x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-αbar_t)) * ε_θ) + sqrt(β̃_t) * z
            sqrt_alpha_t = torch.sqrt(alpha_t)
            coef_eps = beta_t / sqrt_one_minus_alphabar_t
            
            x_prev = (1.0 / sqrt_alpha_t) * (xt - coef_eps * eps_pred) + torch.sqrt(beta_tilde) * z
            
            # Update x with the new denoised estimate
            x_sessions = x.view(B, 4, 3, H, W).clone()
            for b in range(B):
                session_idx = i_tg_positive[b].item()
                x_sessions[b, session_idx] = x_prev[b]
            x = x_sessions.view(B, 12, H, W)
            
            # Accumulate mask predictions (weighted by noise level)
            if t_idx < T_m:
                mask_accum = mask_accum + mask_pred * w_p[t_idx]
        
        # Final x0 prediction
        x_sessions = x.view(B, 4, 3, H, W)
        x0_final = torch.stack([x_sessions[b, i_tg_positive[b].item()] for b in range(B)], dim=0)
        
        return x0_final, mask_accum

    def ddim_inverse(
        self,
        net: nn.Module,
        x: torch.Tensor,
        intv: Optional[List[torch.Tensor]] = None,
        treat_cond: Optional[List[torch.Tensor]] = None,
        i_tg: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        start_t: Optional[int] = None,
        step_mask: int = 10,
        eta: float = 0.0,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM-style deterministic/stochastic inverse diffusion.
        
        
        Args:
            net: The denoising network
            x: Input tensor [B, 12, H, W]
            intv: List of day interval tensors
            treat_cond: List of treatment code tensors
            i_tg: Target session indices
            steps: Number of sampling steps
            start_t: Starting timestep (not used, kept for API compatibility)
            step_mask: Steps for mask averaging
            eta: Stochasticity (0 = deterministic DDIM, 1 = DDPM)
            device: Target device
            
        Returns:
            x0_final: Reconstructed image
            mask_pred: Predicted masks
        """
        net.eval()
        
        steps = steps if steps is not None else self.T
        B, C, H, W = x.shape
        x = x.to(device)
        
        # Initialize target indices
        if i_tg is None:
            i_tg = -torch.ones((B,), dtype=torch.long, device=device)
        else:
            i_tg = i_tg.to(device).long()
        
        i_tg_positive = torch.where(i_tg < 0, 4 + i_tg, i_tg)
        
        
        # Training uses t ∈ [1, T], so inference must too
        times = torch.linspace(self.T, 1, steps + 1, device=device).long()
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))
        
        # Initialize accumulators
        mask_accum = torch.zeros((B, 4, H, W), device=device)
        
        # Mask averaging weights
        T_m = min(step_mask, steps)
        alphabar_weights = self.alphabar[:T_m].to(device)
        w_p = alphabar_weights / alphabar_weights.sum()
        
        x0_final = None
        
        for t_curr, t_next in time_pairs:
            
            t_curr_idx = t_curr - 1
            
            # Get noise schedule values (0-indexed array access)
            alphabar_t = self.alphabar[t_curr_idx].to(device)
            sqrt_alphabar_t = torch.sqrt(alphabar_t)
            sqrt_one_minus_alphabar_t = torch.sqrt(1.0 - alphabar_t)
            
            # Pass 1-indexed timestep to model (matches training)
            t_batch = torch.full((B,), t_curr, device=device, dtype=torch.float32)
            
            # Network prediction
            with torch.no_grad():
                pred = net(x, t_batch, intv_t=intv, treat_code=treat_cond, i_tg=i_tg)
                eps_pred = pred[:, 4:7, :, :]
                mask_pred = pred[:, 0:4, :, :]
            
            # Extract current noisy target
            x_sessions = x.view(B, 4, 3, H, W)
            xt = torch.stack([x_sessions[b, i_tg_positive[b].item()] for b in range(B)], dim=0)
            
            
            x0_pred = (xt - sqrt_one_minus_alphabar_t * eps_pred) / (sqrt_alphabar_t + 1e-8)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            
            if t_next >= 1:
                t_next_idx = t_next - 1  # Convert to 0-indexed for array access
                alphabar_t_next = self.alphabar[t_next_idx].to(device)
                sqrt_alphabar_t_next = torch.sqrt(alphabar_t_next)
                sqrt_one_minus_alphabar_t_next = torch.sqrt(1.0 - alphabar_t_next)
                
                # DDIM sigma calculation
                if eta > 0:
                    # σ = η * sqrt((1 - αbar_{t-1})/(1 - αbar_t)) * sqrt(1 - αbar_t/αbar_{t-1})
                    sigma = eta * torch.sqrt(
                        (1.0 - alphabar_t_next) / (1.0 - alphabar_t + 1e-8)
                    ) * torch.sqrt(
                        1.0 - alphabar_t / (alphabar_t_next + 1e-8)
                    )
                    sigma = torch.clamp(sigma, min=0.0)
                    noise = torch.randn_like(xt)
                else:
                    sigma = 0.0
                    noise = 0.0
                
                
                dir_coef = torch.sqrt(torch.clamp(1.0 - alphabar_t_next - sigma**2, min=0.0))
                
                x_next = sqrt_alphabar_t_next * x0_pred + dir_coef * eps_pred
                if eta > 0:
                    x_next = x_next + sigma * noise
            else:
                # Final step: return x_0 prediction
                x_next = x0_pred
            
            # Update x
            x_sessions = x.view(B, 4, 3, H, W).clone()
            for b in range(B):
                session_idx = i_tg_positive[b].item()
                x_sessions[b, session_idx] = x_next[b]
            x = x_sessions.view(B, 12, H, W)
            
            # Accumulate masks (use 0-indexed for weights array)
            
            if t_curr_idx < T_m:
                mask_accum = mask_accum + mask_pred * w_p[t_curr_idx]
            
            x0_final = x_next
        
        return x0_final, mask_accum

    def dpm_solver_plus_plus_inverse(
        self,
        net: nn.Module,
        x: torch.Tensor,
        intv: Optional[List[torch.Tensor]] = None,
        treat_cond: Optional[List[torch.Tensor]] = None,
        i_tg: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        step_mask: int = 10,
        start_t: Optional[int] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DPM-Solver++ inverse (simplified 2nd order).
        
        Note: This is a simplified implementation. For production use,
        consider using the full DPM-Solver++ with proper order handling.
        
        For now, this falls back to DDIM with eta=0 for stability.
        """
        # DPM-Solver++ is complex; fall back to DDIM for reliability
        return self.ddim_inverse(
            net=net,
            x=x,
            intv=intv,
            treat_cond=treat_cond,
            i_tg=i_tg,
            steps=steps,
            step_mask=step_mask,
            eta=0.0,
            device=device
        )
