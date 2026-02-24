#!/usr/bin/env python3
"""
Logits Extractor
Extracts raw logits from BDH model
"""

import torch
import json


def save_logits(filepath, text, logits):
    """
    Save raw logits to JSON file.
    
    Args:
        filepath: Output JSON file path
        text: Input text string
        logits: Raw logits tensor for all vocabulary tokens
    """
    logits_np = logits.cpu().numpy()
    
    data = {
        'input_text': text,
        'vocab_size': len(logits_np),
        'logits': [{'token_id': int(token_id), 'logit': float(logit_val)} 
                   for token_id, logit_val in enumerate(logits_np)]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_logits(text, logits, filepath):
    """
    Extract logits and save to JSON.
    
    Args:
        text: Input text string
        logits: Raw logits tensor (typically from predictions extraction)
        filepath: Output JSON file path
    """
    print("[Logits] Saving logits...")
    save_logits(filepath, text, logits)
    print(f"  ✓ Logits saved: {filepath}")
