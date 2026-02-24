#!/usr/bin/env python3
"""
Embedding processing functions for dimensionality reduction and normalization
"""

import torch
import numpy as np
import math

def reduce_dimensions(embeddings, target_dim=32):
    """
    Reduce embedding dimensions from 256 to target_dim (default 32).
    Uses evenly spaced sampling across the original dimensions.
    
    Args:
        embeddings: torch.Tensor of shape [num_tokens, 256]
        target_dim: int, target dimension size (default 32)
        
    Returns:
        torch.Tensor of shape [num_tokens, target_dim]
    """
    if embeddings.size(-1) < target_dim:
        raise ValueError(f"Target dimension {target_dim} is larger than input dimension {embeddings.size(-1)}")
    
    # Calculate evenly spaced indices
    original_dim = embeddings.size(-1)
    indices = torch.linspace(0, original_dim - 1, target_dim, dtype=torch.long)
    
    # Extract the selected dimensions
    reduced_embeddings = embeddings[..., indices]
    
    return reduced_embeddings

def normalize_embeddings(embeddings, method='min_max'):
    """
    Normalize embeddings between 0 and 1, then round to 4 decimal places.
    
    Args:
        embeddings: torch.Tensor of any shape
        method: str, normalization method ('min_max', 'token_wise', 'global')
                - 'min_max': Global min-max normalization across all values
                - 'token_wise': Normalize each token's embedding independently 
                - 'global': Global normalization across all dimensions
                
    Returns:
        torch.Tensor of same shape, normalized between 0 and 1, rounded to 4 decimal places
    """
    if method == 'min_max':
        # Global min-max normalization
        min_val = embeddings.min()
        max_val = embeddings.max()
        
        if max_val - min_val == 0:
            # Handle case where all values are the same
            normalized = torch.zeros_like(embeddings)
        else:
            normalized = (embeddings - min_val) / (max_val - min_val)
        
    elif method == 'token_wise':
        # Normalize each token independently
        normalized = torch.zeros_like(embeddings)
        for i in range(embeddings.size(0)):
            token_embed = embeddings[i]
            min_val = token_embed.min()
            max_val = token_embed.max()
            
            if max_val - min_val == 0:
                normalized[i] = torch.zeros_like(token_embed)
            else:
                normalized[i] = (token_embed - min_val) / (max_val - min_val)
                
    elif method == 'global':
        # Global normalization (same as min_max)
        min_val = embeddings.min()
        max_val = embeddings.max()
        normalized = (embeddings - min_val) / (max_val - min_val)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Round to 4 decimal places
    normalized = torch.round(normalized * 10000) / 10000
    
    return normalized

def process_embeddings(embeddings, target_dim=32, norm_method='min_max'):
    """
    Complete embedding processing pipeline: reduce dimensions and normalize.
    
    Args:
        embeddings: torch.Tensor of shape [num_tokens, 256]
        target_dim: int, target dimension size (default 32)
        norm_method: str, normalization method (default 'min_max')
        
    Returns:
        torch.Tensor of shape [num_tokens, target_dim], normalized between 0 and 1
    """
    # Step 1: Reduce dimensions
    reduced = reduce_dimensions(embeddings, target_dim)
    
    # Step 2: Normalize
    normalized = normalize_embeddings(reduced, norm_method)
    
    return normalized

def process_embeddings_with_info(embeddings, target_dim=32, norm_method='min_max'):
    """
    Process embeddings and return processing information.
    
    Args:
        embeddings: torch.Tensor of shape [num_tokens, 256]
        target_dim: int, target dimension size (default 32)
        norm_method: str, normalization method (default 'min_max')
        
    Returns:
        dict: {
            'processed_embeddings': torch.Tensor,
            'original_shape': tuple,
            'processed_shape': tuple,
            'dimension_indices': torch.Tensor,
            'original_min': float,
            'original_max': float,
            'processed_min': float,
            'processed_max': float
        }
    """
    original_shape = embeddings.shape
    
    # Calculate dimension indices used
    original_dim = embeddings.size(-1)
    dimension_indices = torch.linspace(0, original_dim - 1, target_dim, dtype=torch.long)
    
    # Store original stats
    original_min = embeddings.min().item()
    original_max = embeddings.max().item()
    
    # Process embeddings
    processed = process_embeddings(embeddings, target_dim, norm_method)
    
    # Get processed stats
    processed_min = processed.min().item()
    processed_max = processed.max().item()
    
    return {
        'processed_embeddings': processed,
        'original_shape': original_shape,
        'processed_shape': processed.shape,
        'dimension_indices': dimension_indices,
        'original_min': original_min,
        'original_max': original_max,
        'processed_min': processed_min,
        'processed_max': processed_max,
        'normalization_method': norm_method
    }

def get_freqs(n, theta=2**16, dtype=torch.float32):
    """
    Generate frequencies for RoPE.
    
    Args:
        n: Dimension size
        theta: Base frequency (default 2^16)
        dtype: Data type
        
    Returns:
        torch.Tensor: Frequency values
    """
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))

def apply_rope(embeddings):
    """
    Apply Rotary Position Embedding (RoPE) to embedding matrix.
    
    Args:
        embeddings: torch.Tensor of shape [num_tokens, embedding_dim]
        
    Returns:
        torch.Tensor: RoPE-transformed embeddings of same shape
    """
    num_tokens, embedding_dim = embeddings.shape
    
    # Generate frequencies
    freqs = get_freqs(embedding_dim).view(1, 1, embedding_dim)
    
    # Calculate phases for each position
    positions = torch.arange(0, num_tokens, dtype=torch.float32).view(num_tokens, 1, 1)
    phases = positions * freqs
    
    # Convert phases to angles
    phases = (phases % 1) * (2 * math.pi)
    phases_cos = torch.cos(phases)
    phases_sin = torch.sin(phases)
    
    # Reshape embeddings for rotation: [num_tokens, 1, embedding_dim]
    emb_reshaped = embeddings.unsqueeze(1)
    
    # Create rotated version (swap and negate alternate dimensions)
    emb_rot = torch.stack((-emb_reshaped[..., 1::2], emb_reshaped[..., ::2]), dim=-1).view(*emb_reshaped.size())
    
    # Apply rotation
    rope_embeddings = (emb_reshaped * phases_cos + emb_rot * phases_sin).squeeze(1)
    
    return rope_embeddings