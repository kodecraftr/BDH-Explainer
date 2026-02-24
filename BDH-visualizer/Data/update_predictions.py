#!/usr/bin/env python3
"""
Update Predictions from Existing Logits
Recalculates predictions with different temperature/top-K without re-running the model
"""

import os
import json
import torch
from torch.nn import functional as F


def load_logits_from_json(filepath):
    """
    Load raw logits from JSON file.
    
    Args:
        filepath: Path to logits.json
        
    Returns:
        tuple: (input_text, logits_tensor)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Logits file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    input_text = data['input_text']
    logits = [item['logit'] for item in data['logits']]
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    
    return input_text, logits_tensor


def calculate_predictions(logits, temperature=0.8, top_k=20):
    """
    Calculate predictions from raw logits.
    
    Args:
        logits: Raw logits tensor
        temperature: Temperature for sampling
        top_k: Top-K for sampling
        
    Returns:
        list: Predictions with rank, token_id, probability
    """
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None:
        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        scaled_logits[scaled_logits < v[-1]] = -float('Inf')
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
    
    predictions = []
    for i, (prob, idx_val) in enumerate(zip(top_probs, top_indices)):
        predictions.append({
            'rank': i + 1,
            'token_id': idx_val.item(),
            'probability': prob.item()
        })
    
    return predictions


def save_predictions(filepath, input_text, predictions, temperature, top_k):
    """
    Save predictions to JSON file.
    
    Args:
        filepath: Output JSON file path
        input_text: Input text string
        predictions: List of prediction dictionaries
        temperature: Temperature parameter used
        top_k: Top-K parameter used
    """
    # Setup tokenizer to decode tokens
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Warning: tiktoken not available, token text will not be decoded")
        encoder = None
    
    predictions_with_text = []
    for pred in predictions:
        if encoder:
            token_text = encoder.decode([pred['token_id']])
        else:
            token_text = f"<token_{pred['token_id']}>"
        
        predictions_with_text.append({
            'rank': pred['rank'],
            'token': token_text,
            'token_id': pred['token_id'],
            'probability': pred['probability']
        })
    
    data = {
        'input_text': input_text,
        'temperature': temperature,
        'top_k': top_k,
        'predictions': predictions_with_text
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_predictions(logits_file, predictions_file, temperature, top_k):
    """
    Update predictions from existing logits.
    
    Args:
        logits_file: Path to logits.json
        predictions_file: Path to save updated predictions.json
        temperature: New temperature value
        top_k: New top-K value
        
    Returns:
        list: Updated predictions
    """
    print(f"Loading logits from: {logits_file}")
    input_text, logits = load_logits_from_json(logits_file)
    print(f"✓ Loaded logits for: '{input_text}'")
    
    print(f"\nRecalculating predictions with:")
    print(f"  Temperature: {temperature}")
    print(f"  Top-K: {top_k}")
    
    predictions = calculate_predictions(logits, temperature, top_k)
    
    save_predictions(predictions_file, input_text, predictions, temperature, top_k)
    print(f"\n✓ Updated predictions saved: {predictions_file}")
    print(f"  Top prediction: Token ID {predictions[0]['token_id']} (prob: {predictions[0]['probability']:.4f})")
    
    return predictions


def main():
    """Interactive main function."""
    print("=" * 60)
    print("BDH Predictions Updater")
    print("=" * 60)
    print("Recalculates predictions using existing logits")
    print("No model inference needed - very fast!")
    print()
    
    # Use relative path from current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "infer")
    logits_file = os.path.join(base_dir, "logits.json")
    predictions_file = os.path.join(base_dir, "predictions.json")
    
    # Check if logits file exists
    if not os.path.exists(logits_file):
        print(f"Error: Logits file not found at {logits_file}")
        print("Please run demo.py first to generate logits.")
        return
    
    # Load to show current input
    input_text, _ = load_logits_from_json(logits_file)
    print(f"Current input text: '{input_text}'")
    print()
    
    while True:
        print("-" * 60)
        choice = input("Update predictions? (y/n or 'quit'): ").strip().lower()
        
        if choice in ['quit', 'exit', 'q', 'n']:
            print("Goodbye!")
            break
        
        if choice != 'y':
            continue
        
        temperature_input = input("Enter temperature (0.1-2.0, default 0.8): ").strip()
        try:
            temperature = float(temperature_input) if temperature_input else 0.8
            temperature = max(0.1, min(2.0, temperature))
        except ValueError:
            print("Invalid temperature, using 0.8")
            temperature = 0.8
        
        top_k_input = input("Enter top-K (1-100, default 20): ").strip()
        try:
            top_k = int(top_k_input) if top_k_input else 20
            top_k = max(1, min(100, top_k))
        except ValueError:
            print("Invalid top-K, using 20")
            top_k = 20
        
        try:
            update_predictions(logits_file, predictions_file, temperature, top_k)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
