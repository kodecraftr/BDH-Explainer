"""
TaDiff-DiT: Treatment-aware Diffusion Transformer Architecture
CORRECTED VERSION

Key Fixes Applied:
1. BDH Attention: ELU+1 instead of ReLU, correct normalization
2. FinalLayer: Xavier initialization instead of zeros
3. HybridPatchEmbed: CNN stem for spatial locality
4. Proper weight initialization throughout

Author: TaDiff-Net Project (Corrected)
"""

from abc import abstractmethod
import math
from typing import Optional, Tuple, List

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Handle imports for both standalone and src.net module contexts
try:
    from src.net.utils import (
        checkpoint,
        conv_nd,
        linear,
        normalization,
        timestep_embedding,
        FourierFeatures,
    )
except ModuleNotFoundError:
    try:
        from utils_net import (
            checkpoint,
            conv_nd,
            linear,
            normalization,
            timestep_embedding,
            FourierFeatures,
        )
    except ModuleNotFoundError:
        # Inline fallbacks for standalone testing
        def checkpoint(func, inputs, params, flag):
            if flag:
                return th.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
            return func(*inputs)
        
        def conv_nd(dims, *args, **kwargs):
            if dims == 2:
                return nn.Conv2d(*args, **kwargs)
            raise ValueError(f"Unsupported dims: {dims}")
        
        def linear(*args, **kwargs):
            return nn.Linear(*args, **kwargs)
        
        def normalization(channels, g=32):
            return nn.GroupNorm(min(g, channels), channels)
        
        def timestep_embedding(timesteps, dim, max_period=10000):
            half = dim // 2
            freqs = th.exp(
                -math.log(max_period) * th.arange(0, half, dtype=th.float32, device=timesteps.device) / half
            )
            args = timesteps[:, None].float() * freqs[None]
            return th.cat([th.cos(args), th.sin(args)], dim=-1)
        
        class FourierFeatures(nn.Module):
            def __init__(self, in_features, out_features, std=0.2):
                super().__init__()
                assert out_features % 2 == 0
                self.weight = nn.Parameter(th.randn([out_features // 2, in_features]) * std)
            
            def forward(self, x):
                f = 2 * math.pi * x @ self.weight.T
                return th.cat([f.cos(), f.sin()], dim=-1)


# =============================================================================
# Utility Functions
# =============================================================================

def modulate(x: th.Tensor, shift: th.Tensor, scale: th.Tensor) -> th.Tensor:
    """Apply adaptive layer normalization modulation."""
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


def unpatchify(x: th.Tensor, patch_size: int, height: int, width: int, out_channels: int) -> th.Tensor:
    """Reshape sequence of patch embeddings back to image."""
    h = height // patch_size
    w = width // patch_size
    
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, out_channels)
    x = th.einsum('bhwpqc->bchpwq', x)
    x = x.reshape(x.shape[0], out_channels, height, width)
    return x


# =============================================================================
# FIXED: Hybrid Patch Embedding with CNN Stem
# =============================================================================

class HybridPatchEmbed(nn.Module):
    """
    FIXED: Convolutional stem before patching to preserve local structure.
    
    The pure ViT-style patching loses spatial locality which is critical
    for medical image reconstruction. This CNN stem captures local features
    (edges, textures, boundaries) before tokenization.
    
    Architecture: Dynamically builds CNN stem based on patch_size.
    - patch_size=8: 3 stride-2 convs (8x downsampling)
    - patch_size=16: 4 stride-2 convs (16x downsampling)
    """
    
    def __init__(
        self,
        image_size: int = 192,
        patch_size: int = 16,
        in_channels: int = 12,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Determine number of stride-2 stages needed
        # patch_size = 2^num_stages
        import math
        num_stages = int(math.log2(patch_size))
        assert 2 ** num_stages == patch_size, f"patch_size must be power of 2, got {patch_size}"
        
        # Build channel progression
        if num_stages == 3:  # patch_size=8
            stem_channels = [in_channels, embed_dim // 4, embed_dim // 2, embed_dim]
        elif num_stages == 4:  # patch_size=16
            stem_channels = [in_channels, embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]
        else:  # General case
            # Exponentially grow channels
            stem_channels = [in_channels]
            for i in range(num_stages):
                stem_channels.append(embed_dim // (2 ** (num_stages - i - 1)))
        
        # Build CNN stem dynamically
        layers = []
        for i in range(num_stages):
            # First layer uses larger kernel
            kernel_size = 7 if i == 0 else 3
            layers.extend([
                nn.Conv2d(stem_channels[i], stem_channels[i+1], 
                         kernel_size=kernel_size, stride=2, 
                         padding=kernel_size//2, bias=bias),
                nn.GroupNorm(min(32, stem_channels[i+1]//8), stem_channels[i+1]),
                nn.GELU(),
            ])
        
        self.stem = nn.Sequential(*layers)
        
        # FIXED: Proper weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # FIXED: Xavier/Kaiming initialization instead of default
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: [B, C, H, W] input image
            
        Returns:
            [B, N, D] patch tokens where N = (H/patch_size) * (W/patch_size)
        """
        # Apply CNN stem (downsamples by patch_size)
        x = self.stem(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        
        # Flatten to sequence: [B, D, h, w] -> [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class PatchEmbed(nn.Module):
    """
    Standard Conv2d-based patch embedding (kept for backward compatibility).
    
    For better results, use HybridPatchEmbed instead.
    """
    def __init__(
        self,
        image_size: int = 192,
        patch_size: int = 16,
        in_channels: int = 12,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        assert image_size % patch_size == 0
        
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        
        # FIXED: Proper initialization
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.shape[0], -1))
        if bias:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============================================================================
# Positional Embedding
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """2D sinusoidal positional embeddings for patch sequences."""
    
    def __init__(self, embed_dim: int, grid_size: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        pos_embed = self._get_2d_sincos_pos_embed(embed_dim, grid_size, temperature)
        self.register_buffer('pos_embed', th.from_numpy(pos_embed).float().unsqueeze(0))
    
    def _get_2d_sincos_pos_embed(self, embed_dim: int, grid_size: int, temperature: float) -> np.ndarray:
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
        
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid, temperature)
        return pos_embed
    
    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid: np.ndarray, temperature: float) -> np.ndarray:
        assert embed_dim % 2 == 0
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], temperature)
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], temperature)
        return np.concatenate([emb_h, emb_w], axis=1)
    
    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: np.ndarray, temperature: float) -> np.ndarray:
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega = 1.0 / (temperature ** (omega / (embed_dim / 2)))
        
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.pos_embed


# =============================================================================
# FIXED: BDH Attention with ELU+1 and Correct Normalization
# =============================================================================

class BDH_Attention(nn.Module):
    """
    Baby Dragon Hatchling (BDH) Linear Attention - CORRECTED VERSION
    
    FIXES APPLIED:
    1. Changed ReLU to ELU+1 for positivity without dead neurons
    2. Fixed normalization: Output = (Q @ (K^T @ V)) / (Q @ K_sum)
    3. Added proper scaling for numerical stability
    4. Correct einsum operations
    
    Linear Attention achieves O(N) complexity instead of O(N²) by computing
    K^T @ V first (associative property).
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        expansion_factor: Factor to expand the hidden dimension
        qkv_bias: Whether to use bias in projections
        attn_drop: Dropout rate (not used in linear attention)
        proj_drop: Output projection dropout
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        expansion_factor: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expanded_dim = dim * expansion_factor
        self.expanded_head_dim = self.expanded_dim // num_heads
        
        # FIXED: Scaling factor for numerical stability
        self.scale = self.expanded_head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, self.expanded_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.expanded_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.expanded_dim, bias=qkv_bias)
        
        self.out_proj = nn.Linear(self.expanded_dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # FIXED: Proper weight initialization
        self._init_weights()
    
    def _init_weights(self):
        # FIXED: Xavier initialization for all projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def _feature_map(self, x: th.Tensor) -> th.Tensor:
        """
        FIXED: ELU + 1 feature map instead of ReLU.
        
        Why this matters:
        - ReLU creates "dead neurons" (outputs exactly 0 for negative inputs)
        - When Q or K rows are all zeros, division by zero occurs
        - ELU + 1 is always positive (range: (0, ∞)) with smooth gradients
        
        ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        ELU(x) + 1 shifts the range to (0, ∞)
        """
        return F.elu(x) + 1.0
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: [B, N, C] input tokens
            
        Returns:
            [B, N, C] output tokens
        """
        B, N, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, N, expanded_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # FIXED: Apply ELU+1 feature map for positivity
        q = self._feature_map(q)
        k = self._feature_map(k)
        
        # Reshape for multi-head attention
        # [B, N, expanded_dim] -> [B, N, num_heads, expanded_head_dim] -> [B, num_heads, N, expanded_head_dim]
        q = q.view(B, N, self.num_heads, self.expanded_head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.expanded_head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.expanded_head_dim).transpose(1, 2)
        
        # Apply scaling to Q
        q = q * self.scale
        
        # =================================================================
        # FIXED: Linear Attention with correct normalization
        # =================================================================
        
        # Step 1: Compute K^T @ V (the "memory" matrix)
        # [B, H, N, D] @ [B, H, D, N] would be O(N²) - WRONG ORDER
        # [B, H, D, N]^T @ [B, H, N, D] -> [B, H, D, D] - CORRECT ORDER (O(N))
        # Using einsum: 'bhnd,bhne->bhde' means K^T @ V
        kv = th.einsum('bhnd,bhne->bhde', k, v)  # [B, H, D, D]
        
        # Step 2: Compute Q @ (K^T @ V)
        # [B, H, N, D] @ [B, H, D, D] -> [B, H, N, D]
        qkv = th.einsum('bhnd,bhde->bhne', q, kv)  # [B, H, N, D]
        
        # Step 3: FIXED: Correct normalization
        # Denominator: sum of K over sequence dimension, then dot with Q
        # k_sum: [B, H, D] (sum over N dimension)
        k_sum = k.sum(dim=2)  # [B, H, D]
        
        # normalizer: Q @ k_sum -> [B, H, N]
        # Each query position gets normalized by its dot product with the sum of all keys
        normalizer = th.einsum('bhnd,bhd->bhn', q, k_sum)  # [B, H, N]
        
        # FIXED: Clamp for numerical stability (prevent division by zero)
        normalizer = normalizer.unsqueeze(-1).clamp(min=1e-6)  # [B, H, N, 1]
        
        # Normalize
        out = qkv / normalizer
        
        # =================================================================
        # Reshape and project output
        # =================================================================
        
        # [B, H, N, D] -> [B, N, H, D] -> [B, N, H*D]
        out = out.transpose(1, 2).reshape(B, N, self.expanded_dim)
        
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


# =============================================================================
# MLP (Feed-Forward Network)
# =============================================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # FIXED: Proper initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# DiT Block with Adaptive Layer Normalization
# =============================================================================

class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with BDH Attention and AdaLN.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_checkpoint: bool = False,
        bdh_expansion: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Layer norms (elementwise_affine=False for adaLN)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # FIXED: Using corrected BDH Attention
        self.attn = BDH_Attention(
            dim=hidden_size,
            num_heads=num_heads,
            expansion_factor=bdh_expansion,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        
        # adaLN modulation: outputs 6 values (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * hidden_size, bias=True)
        )
        
        # FIXED: Initialize adaLN weights to zero for stability
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        
        # CRITICAL FIX: Initialize biases so that:
        # - shift1, shift2 = 0 (no shift initially)
        # - scale1, scale2 = 0 (since modulate does x * (1 + scale), scale=0 means multiply by 1)
        # - gate1, gate2 = 1 (gates MUST be 1 to allow gradient flow!)
        # Bias layout: [shift1, scale1, gate1, shift2, scale2, gate2] each of size hidden_size
        bias = th.zeros(6 * hidden_size)
        # Set gate1 (positions 2*hidden_size to 3*hidden_size) to 1.0
        bias[2 * hidden_size : 3 * hidden_size] = 1.0
        # Set gate2 (positions 5*hidden_size to 6*hidden_size) to 1.0
        bias[5 * hidden_size : 6 * hidden_size] = 1.0
        self.adaLN_modulation[-1].bias.data = bias
    
    def forward(self, x: th.Tensor, c: th.Tensor) -> th.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, c), self.parameters(), self.use_checkpoint)
        return self._forward(x, c)
    
    def _forward(self, x: th.Tensor, c: th.Tensor) -> th.Tensor:
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        
        # Attention block with adaLN
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift1, scale1)
        x = x + gate1.unsqueeze(1) * self.attn(x_mod)
        
        # MLP block with adaLN
        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(x_mod)
        
        return x


# =============================================================================
# FIXED: Final Layer with Proper Initialization
# =============================================================================

class FinalLayer(nn.Module):
    """
    Final layer that converts transformer output back to image space.
    
    FIXED: Uses Xavier initialization instead of zeros for the projection weight.
    """
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        condition_dim: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_size, bias=True)
        )
        
        self.proj = nn.Linear(hidden_size, patch_size ** 2 * out_channels, bias=True)
        
        # FIXED: adaLN initialized to identity (zeros is correct here)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        
        # FIXED: Output projection uses XAVIER initialization with proper gain
        # gain=0.02 was 50x too small, causing vanishing initial outputs
        # Standard gain for linear layers is 1.0 (or sqrt(2) for ReLU)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)  # Bias can be zero
    
    def forward(self, x: th.Tensor, c: th.Tensor, height: int, width: int) -> th.Tensor:
        shift_scale = self.adaLN_modulation(c)
        shift, scale = shift_scale.chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        
        x = self.proj(x)
        x = unpatchify(x, self.patch_size, height, width, self.out_channels)
        return x


# =============================================================================
# Main TaDiff-DiT Model
# =============================================================================

class TaDiff_DiT(nn.Module):
    """
    Treatment-aware Diffusion Transformer with BDH Attention - CORRECTED VERSION
    
    Fixes applied:
    1. HybridPatchEmbed: CNN stem for spatial locality
    2. BDH_Attention: ELU+1 activation, correct normalization
    3. FinalLayer: Xavier initialization
    4. Proper weight initialization throughout
    """
    
    def __init__(
        self,
        image_size: int = 192,
        in_channels: int = 12,
        out_channels: int = 7,
        model_channels: int = 128,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        use_hybrid_embed: bool = True,  # NEW: Use CNN stem by default
        bdh_layers: int = 4,
        bdh_expansion: int = 2,
        # Legacy parameters (kept for compatibility)
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (8, 16),
        channel_mult: tuple = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        
        # Store configuration
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # =================================================================
        # Conditioning Embeddings
        # =================================================================
        
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels * 2),
            nn.SiLU(),
            linear(model_channels * 2, model_channels),
        )
        
        self.days_embed = nn.Sequential(
            linear(model_channels, model_channels * 2),
            nn.SiLU(),
            linear(model_channels * 2, model_channels),
        )
        
        self.treats_embed = nn.Sequential(
            FourierFeatures(1, model_channels),
            linear(model_channels, model_channels * 2),
            nn.SiLU(),
            linear(model_channels * 2, model_channels),
        )
        
        self.all_time_day_dim = model_channels * 4
        
        # Project conditioning to transformer hidden dimension
        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.all_time_day_dim, hidden_size),
        )
        
        # =================================================================
        # CRITICAL FIX: Direct timestep → conditioning projection
        # =================================================================
        # Without this, the diffusion timestep is buried inside 1 of 4 slots
        # in the conditioning vector and must be disentangled by a single
        # linear layer. This severely dilutes the noise-level signal that
        # the model needs to predict the correct amount of noise at each t.
        # Standard DiT adds the timestep embedding directly to the condition.
        self.time_to_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, hidden_size),
        )
        
        # =================================================================
        # FIXED: Patch Embedding (use hybrid by default)
        # =================================================================
        
        if use_hybrid_embed:
            self.patch_embed = HybridPatchEmbed(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=hidden_size,
            )
        else:
            self.patch_embed = PatchEmbed(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=hidden_size,
            )
        
        self.pos_embed = SinusoidalPositionEmbeddings(
            embed_dim=hidden_size,
            grid_size=self.grid_size,
        )
        
        # =================================================================
        # Transformer Blocks (with corrected BDH)
        # =================================================================
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                condition_dim=hidden_size,
                mlp_ratio=mlp_ratio,
                drop=dropout,
                attn_drop=dropout,
                use_checkpoint=use_checkpoint,
                bdh_expansion=bdh_expansion,
            )
            for _ in range(depth)
        ])
        
        # =================================================================
        # FIXED: Final Layer with proper initialization
        # =================================================================
        
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            condition_dim=hidden_size,
        )
        
        # =================================================================
        # CRITICAL FIX: Skip Connection CNN Branch
        # =================================================================
        # The transformer is good at global attention but lacks direct
        # pixel-to-pixel correspondence needed for noise prediction.
        # This CNN branch provides a direct path from input to output,
        # similar to U-Net skip connections.
        
        self.skip_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )
        
        # Learnable weight to balance skip and transformer outputs.
        # We apply sigmoid(skip_weight), so 0.0 -> 0.5 (equal contribution).
        self.skip_weight = nn.Parameter(th.tensor(0.0))
        
        # Initialize all weights properly
        self._initialize_weights()
        
        # CRITICAL: Re-initialize DiTBlock gates to 1.0 (must be done AFTER _initialize_weights)
        self._initialize_gates()
    
    def _initialize_weights(self):
        """Apply proper weight initialization."""
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # FIXED: Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        
        # Re-initialize the conditioning projections
        for embed in [self.time_embed, self.days_embed]:
            for layer in embed:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # FIX: Problem 3.2 — Re-apply zero-init to adaLN modulation layers
        # These MUST be zeros for identity initialization (DiT paper requirement)
        # The _basic_init above overwrote them with Xavier, so we must restore zeros
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        
        # Keep FinalLayer projection at standard Xavier gain so transformer
        # branch has a meaningful signal early in training.
        nn.init.xavier_uniform_(self.final_layer.proj.weight, gain=1.0)
        nn.init.zeros_(self.final_layer.proj.bias)
        
        # =================================================================
        # CRITICAL FIX: Zero-init skip branch output convolution
        # =================================================================
        # The skip branch has Kaiming-inited convolutions that produce
        # non-trivial output at initialization (~0.3 magnitude), while
        # with a tiny-gain FinalLayer and positive skip_weight init,
        # the skip branch can dominate early optimization.
        #
        # Since the skip branch has NO timestep conditioning, it learns
        # a noise-level-independent mapping that looks like noise.
        # The transformer never gets a chance to learn because the skip
        # branch converges first to a bad local minimum.
        #
        # Fix: zero-init the last conv so both branches start at zero.
        # This matches the DiT paper's zero-init strategy and lets the
        # transformer and skip branch learn simultaneously.
        nn.init.zeros_(self.skip_branch[-1].weight)
        nn.init.zeros_(self.skip_branch[-1].bias)
    
    def _initialize_gates(self):
        """
        CRITICAL FIX: Initialize adaLN gates to 1.0 so gradient can flow.
        
        The adaLN modulation outputs [shift1, scale1, gate1, shift2, scale2, gate2].
        - shift and scale should be 0 (identity transform)
        - gate1 and gate2 MUST be 1.0 to allow attention/MLP outputs through
        
        Without this, the model predicts near-zero and cannot learn.
        """
        for block in self.blocks:
            hidden = block.hidden_size
            bias = block.adaLN_modulation[-1].bias.data
            
            # Layout: [shift1, scale1, gate1, shift2, scale2, gate2]
            # Set gate1 (index 2) and gate2 (index 5) to 1.0
            bias[2 * hidden : 3 * hidden] = 1.0  # gate1
            bias[5 * hidden : 6 * hidden] = 1.0  # gate2
    
    def convert_to_fp16(self):
        self.dtype = th.float16
        self.half()
    
    def convert_to_fp32(self):
        self.dtype = th.float32
        self.float()
    
    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        intv_t: Optional[List[th.Tensor]] = None,
        treat_code: Optional[List[th.Tensor]] = None,
        i_tg: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 12, H, W] (4 sessions × 3 modalities)
            timesteps: Diffusion timesteps [B]
            intv_t: List of 4 day interval tensors, each [B]
            treat_code: List of 4 treatment code tensors, each [B]
            i_tg: Target session indices [B]
            
        Returns:
            Output tensor [B, 7, H, W] (4 masks + 3 image channels)
        """
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        device = x.device
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if emb.shape[0] == 1 and B > 1:
            emb = emb.repeat(B, 1)
        
        # Target indices
        if i_tg is None:
            i_tg = -th.ones(B, dtype=th.int8, device=device)
        
        # Day embeddings for each session
        d_embs = [timestep_embedding(days.to(device), self.model_channels) for days in intv_t]
        days_feat = [self.days_embed(d) for d in d_embs]
        
        # Treatment embeddings for each session
        treat_feat = [self.treats_embed((t.to(device)[:, None] + 1) * 10) for t in treat_code]
        
        # Combine treatment and day features
        treat_day_sum = th.cat([
            (t + d).unsqueeze(1) for t, d in zip(treat_feat, days_feat)
        ], dim=1)  # [B, 4, model_channels]
        
        # Get target session features
        i_tg_positive = th.where(i_tg < 0, 4 + i_tg, i_tg).long()
        # FIX: Use consistent indexing for both B==1 and B>1 cases
        # Previous code used tensor indexing which caused shape issues
        target = th.stack([
            treat_day_sum[i, i_tg_positive[i].item(), :] for i in range(B)
        ], dim=0)  # [B, model_channels]
        
        # Compute relative features
        treat_day_diff = treat_day_sum - target.unsqueeze(1)
        middle_emb = emb + target
        
        for i in range(B):
            idx = i_tg_positive[i].item()
            treat_day_diff[i, idx, :] = treat_day_diff[i, idx, :] + middle_emb[i]
        
        treat_day_diff = treat_day_diff.view(B, -1)  # [B, 4 * model_channels]
        
        # Project conditioning to hidden size
        # CRITICAL FIX: Add direct timestep projection so adaLN layers
        # receive a strong, undiluted noise-level signal at every block.
        condition = self.condition_proj(treat_day_diff) + self.time_to_cond(emb)  # [B, hidden_size]
        
        # =================================================================
        # CRITICAL FIX: Compute skip connection BEFORE patchifying
        # =================================================================
        # This gives the model a direct pixel-to-pixel path for noise prediction
        x_input = x.type(self.dtype)
        skip_out = self.skip_branch(x_input)  # [B, out_channels, H, W]
        
        # Patch embedding
        x = self.patch_embed(x_input)  # [B, N, hidden_size]
        x = self.pos_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, condition)
        
        # Final layer
        transformer_out = self.final_layer(x, condition, H, W)
        transformer_out = transformer_out.type(self.dtype)
        
        # =================================================================
        # Combine transformer output with skip connection
        # =================================================================
        # skip_weight is learnable; with 0.0 init sigmoid(skip_weight)=0.5
        # This allows the model to learn the optimal balance
        w = th.sigmoid(self.skip_weight)  # Constrain to [0, 1]
        output = w * skip_out + (1 - w) * transformer_out
        
        return output


# Alias for backward compatibility
TaDiff_Net = TaDiff_DiT
