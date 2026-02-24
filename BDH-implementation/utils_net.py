"""
Neural Network Utilities for DIT-BDH
"""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=0.2):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(th.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return th.cat([f.cos(), f.sin()], dim=-1)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")


def normalization(channels, g=32):
    return GroupNorm32(min(g, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(0, half, dtype=th.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        return th.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
    return func(*inputs)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
