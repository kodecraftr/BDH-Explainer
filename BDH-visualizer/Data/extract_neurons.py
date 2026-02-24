#!/usr/bin/env python3
"""
Neuron Activation Extractor
Extracts neuron activations from BDH model for all layers and heads
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os


def forward_with_neuron_capture(model, config, idx, capture_layer=1):
    """
    Forward pass that captures neuron activations for a specific layer.
    
    Args:
        model: BDH model instance
        config: BDHConfig instance
        idx: Input token tensor
        capture_layer: Layer number to capture activations from (1-indexed)
        
    Returns:
        dict: Contains logits, activations, and configuration info
    """
    B, T = idx.size()
    D = config.n_embd
    nh = config.n_head
    N = D * config.mlp_internal_dim_multiplier // nh
    
    activations = {
        'x_sparse': None,
        'y_sparse': None,
        'xy_sparse': None,
        'attention_output': None,
    }
    
    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    
    for level in range(config.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        
        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)
        
        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        
        xy_sparse = x_sparse * y_sparse
        
        yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder)
        y = model.ln(yMLP)
        
        x = model.ln(x + y)
        
        if level + 1 == capture_layer:
            activations['x_sparse'] = x_sparse.detach().cpu()
            activations['y_sparse'] = y_sparse.detach().cpu()
            activations['xy_sparse'] = xy_sparse.detach().cpu()
            activations['attention_output'] = yKV.detach().cpu()
    
    logits = x.view(B, T, D) @ model.lm_head
    
    return {
        'logits': logits,
        'activations': activations,
        'config': {'B': B, 'T': T, 'D': D, 'nh': nh, 'N': N}
    }


def save_neurons_for_head(filepath, text, token_strs, layer_num, head_num, activations, config, top_n=50):
    """
    Save top N neuron activations for a specific head.
    
    Args:
        filepath: Output JSON file path
        text: Input text
        token_strs: List of token strings
        layer_num: Layer number (1-indexed)
        head_num: Head number (0-indexed)
        activations: Dictionary of activation tensors
        config: Configuration dictionary with B, T, D, nh, N
        top_n: Number of top neurons to save (default: 50)
    """
    x_sparse = activations['x_sparse'][0]
    T = config['T']
    N = config['N']
    
    token_idx = T - 1
    
    # Collect all active neurons with their values
    active_neurons = []
    for neuron_idx in range(N):
        val = x_sparse[head_num, token_idx, neuron_idx].item()
        if val > 0.0:
            active_neurons.append((neuron_idx, val))
    
    # Sort by value (highest first) and take top N
    active_neurons.sort(key=lambda x: x[1], reverse=True)
    top_neurons = active_neurons[:top_n]
    
    data = {
        'layer': layer_num,
        'head': head_num + 1,
        'neurons': [{'neuron_number': int(neuron_idx), 'value': float(val)} 
                    for neuron_idx, val in top_neurons]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_all_neurons(model, config, encoder, text, base_dir):
    """
    Extract neuron activations for all layers and heads.
    
    Args:
        model: BDH model instance
        config: BDHConfig instance
        encoder: Tokenizer encoder
        text: Input text string
        base_dir: Base directory to save neuron JSON files
        
    Returns:
        str: Path to base directory with saved files
    """
    # Tokenize
    token_ids = encoder.encode(text)
    token_strs = [encoder.decode([t]) for t in token_ids]
    idx = torch.tensor([token_ids], dtype=torch.long)
    
    print("[Neurons] Extracting neuron activations...")
    
    for layer_num in range(1, config.n_layer + 1):
        with torch.no_grad():
            result = forward_with_neuron_capture(model, config, idx, capture_layer=layer_num)
        
        activations = result['activations']
        config_dict = result['config']
        
        layer_dir = os.path.join(base_dir, f"layer{layer_num}")
        os.makedirs(layer_dir, exist_ok=True)
        
        for head_num in range(config_dict['nh']):
            head_file = os.path.join(layer_dir, f"head{head_num + 1}.json")
            save_neurons_for_head(
                head_file, text, token_strs, layer_num, head_num,
                activations, config_dict
            )
        
        print(f"  ✓ Layer {layer_num} complete ({config_dict['nh']} heads)")
    
    return base_dir
