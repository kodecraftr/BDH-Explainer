"""
Various utilities for neural networks.
Enhanced with Linear Self-Attention for efficient long-range modeling.
"""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LinearSelfAttention(nn.Module):
    """
    Efficient Linear Self-Attention using the kernel trick.
    
    Implements: Attention(Q, K, V) = φ(Q) (φ(K)^T V) / (φ(Q) Σφ(K)^T)
    where φ(x) = elu(x) + 1.0 (ensures non-negative values)
    
    This achieves O(N) complexity instead of O(N²) for standard attention,
    making it suitable for high-resolution medical images and long sequences.
    
    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.0)
        
    Input Shape:
        x: (Batch, Sequence_Length, Dim) - Token format
        
    Output Shape:
        (Batch, Sequence_Length, Dim)
        
    Note:
        IMPORTANT: dim must be divisible by num_heads. For BDH integration,
        ensure that (model_channels * channel_mult[-1] * bdh_expansion) 
        is divisible by num_heads. Typical safe configs:
        - model_channels=32, channel_mult=(1,2,4,8), num_heads=4 -> 256 channels, OK
        - model_channels=64, channel_mult=(1,2,4,8), num_heads=8 -> 512 channels, OK
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"LinearSelfAttention: dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Check your model_channels, channel_mult, and bdh_expansion configuration."
            )
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
    
    @staticmethod
    def feature_map(x):
        """
        Feature map φ(x) = elu(x) + 1.0
        Ensures non-negative values for valid probability interpretation.
        """
        return F.elu(x) + 1.0
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C) where T is sequence length
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, T, head_dim)
        
        # Apply feature map (kernel trick)
        q = self.feature_map(q) * self.scale
        k = self.feature_map(k)
        
        # Linear attention: O(N) complexity
        # Compute K^T V first: (B, num_heads, head_dim, head_dim)
        kv = th.einsum('bhnd,bhne->bhde', k, v)
        
        # Compute Q @ (K^T V): (B, num_heads, T, head_dim)
        qkv_out = th.einsum('bhnd,bhde->bhne', q, kv)
        
        # Compute normalization: sum of K for each query position
        # k_sum: (B, num_heads, 1, head_dim)
        k_sum = k.sum(dim=2, keepdim=True)
        
        # Denominator: Q @ k_sum^T -> (B, num_heads, T, 1)
        normalizer = th.einsum('bhnd,bhkd->bhnk', q, k_sum)
        normalizer = normalizer.clamp(min=1e-6)  # Numerical stability
        
        # Normalize
        out = qkv_out / normalizer
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class LinearSelfAttention2D(nn.Module):
    """
    Linear Self-Attention that directly handles 2D spatial inputs.
    
    This is a convenience wrapper that handles the spatial flattening
    internally, accepting (B, C, H, W) format directly.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.attn = LinearSelfAttention(channels, num_heads, dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        
        # Apply attention
        x = self.attn(x)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        return x


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=0.2):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(th.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return th.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention3D(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttention3D, self).__init__()
        self.input_dim = input_dim
        self.W = nn.Linear(input_dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.pool3d = nn.AdaptiveAvgPool3d((None, 1, 1))
        
    def forward(self, x):
        """
        input:
            x: size (N, D, T, H, W), N: batch size, T: sequence length, D: Hidden dimension
            x: size (N, T, D, 1, 1) reshape from (N, D, T, 1, 1) after pooling
        attention_weight:
            att_w : size (N, T, 1)
            att_w : size (N, 1, T, 1, 1) # reshape to (N, 1, T, 1, 1) for broadcasting
        
        return:
            x: size (N, D, T, H, W)
        """
        x_c = self.pool3d(x).squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()
        att_w = self.softmax(self.W(x_c))
        att_w = att_w.unsqueeze(1).unsqueeze(-1)
        return x * att_w


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, g=32):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(g, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])
