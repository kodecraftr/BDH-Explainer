#!/usr/bin/env python3
"""
Predictions Extractor
Extracts next token predictions with probabilities from BDH model
"""

import torch
from torch.nn import functional as F
import json


def get_predictions(model, encoder, text, temperature=0.8, top_k=20):
    """
    Get next token predictions and raw logits.
    
    Args:
        model: BDH model instance
        encoder: Tokenizer encoder
        text: Input text string
        temperature: Temperature for probability scaling (default: 0.8)
        top_k: Number of top predictions to return (default: 20)
        
    Returns:
        tuple: (predictions list, raw logits tensor)
            predictions: List of dicts with rank, token, token_id, probability
            raw_logits: Tensor of raw logits before temperature scaling
    """
    # Tokenize
    token_ids = encoder.encode(text)
    idx = torch.tensor([token_ids], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(idx)
        logits_last = logits[:, -1, :]  # Get last token logits
        
        # Store raw logits before temperature scaling
        raw_logits = logits_last.clone()
        
        # Apply temperature
        scaled_logits = logits_last / temperature
        
        if top_k is not None:
            v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(scaled_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, probs.size(-1)))
        
        predictions = []
        for i, (prob, idx_val) in enumerate(zip(top_probs, top_indices)):
            token = encoder.decode([idx_val.item()])
            predictions.append({
                'rank': i + 1,
                'token': token,
                'token_id': idx_val.item(),
                'probability': prob.item()
            })
        
        return predictions, raw_logits[0]


def save_predictions(filepath, text, predictions, temperature, top_k):
    """
    Save predictions to JSON file.
    
    Args:
        filepath: Output JSON file path
        text: Input text string
        predictions: List of prediction dictionaries
        temperature: Temperature parameter used
        top_k: Top-K parameter used
    """
    data = {
        'input_text': text,
        'temperature': temperature,
        'top_k': top_k,
        'predictions': predictions
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_predictions(model, encoder, text, filepath, temperature=0.8, top_k=20):
    """
    Extract predictions and save to JSON.
    
    Args:
        model: BDH model instance
        encoder: Tokenizer encoder
        text: Input text string
        filepath: Output JSON file path
        temperature: Temperature for sampling (default: 0.8)
        top_k: Top-K for sampling (default: 20)
        
    Returns:
        tuple: (predictions list, raw logits tensor)
    """
    print("[Predictions] Getting predictions...")
    predictions, raw_logits = get_predictions(model, encoder, text, temperature, top_k)
    
    save_predictions(filepath, text, predictions, temperature, top_k)
    print(f"  ✓ Top prediction: '{predictions[0]['token']}' (prob: {predictions[0]['probability']:.4f})")
    print(f"  ✓ Predictions saved: {filepath}")
    
    return predictions, raw_logits
