#!/usr/bin/env python3
"""
Model and Tokenizer Loader
Handles BDH model and tokenizer initialization
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.BDH_model.run import BDH, BDHConfig
import torch


def load_model(model_path=None):
    """
    Load the BDH model from checkpoint.
    
    Args:
        model_path: Path to the BDH model checkpoint. If None, uses default checkpoint.
        
    Returns:
        tuple: (model, config)
    """
    if model_path is None:
        # Use default path relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "Models", "BDH_model", "checkpoints", "ckpt_5000.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: {model_path} not found.")
    
    print("Loading BDH model...")
    
    import __main__
    __main__.BDH = BDH
    __main__.BDHConfig = BDHConfig
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_args' in checkpoint:
            config = checkpoint['model_args']
        else:
            config = BDHConfig()
        
        config.device = 'cpu'
        model = BDH(config)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ Model loaded successfully!")
        
        return model, config
        
    finally:
        if hasattr(__main__, 'BDH'):
            delattr(__main__, 'BDH')
        if hasattr(__main__, 'BDHConfig'):
            delattr(__main__, 'BDHConfig')


def setup_tokenizer():
    """
    Setup the tiktoken tokenizer.
    
    Returns:
        tokenizer encoder
    """
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("gpt2")
        print("✓ Tokenizer loaded.")
        return encoder
    except ImportError:
        print("Please install tiktoken: pip install tiktoken")
        sys.exit(1)
