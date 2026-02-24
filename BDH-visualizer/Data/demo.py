#!/usr/bin/env python3
"""
Interactive BDH Complete Data Extractor
Generates: neuron activations (all layers/heads), embeddings, and predictions
"""

import os

# Import extraction modules
from model_loader import load_model, setup_tokenizer
from extract_neurons import extract_all_neurons
from extract_embeddings import extract_embeddings_simple
from extract_predictions import extract_predictions
from extract_logits import extract_logits


def process_text(model, config, encoder, text, temperature=0.8, top_k=20):
    """Process text and save all data."""
    print(f"\n{'='*60}")
    print(f"Processing: '{text}'")
    print('='*60)
    
    # Tokenize for display
    token_ids = encoder.encode(text)
    token_strs = [encoder.decode([t]) for t in token_ids]
    
    print(f"Tokens: {len(token_strs)}")
    print(f"Token list: {token_strs}")
    
    # Base directory - use relative path from current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "infer")
    os.makedirs(base_dir, exist_ok=True)
    
    # [1/4] Get predictions and logits
    print("\n[1/4] Getting predictions...")
    pred_file = os.path.join(base_dir, "predictions.json")
    predictions, raw_logits = extract_predictions(model, encoder, text, pred_file, temperature, top_k)
    
    logits_file = os.path.join(base_dir, "logits.json")
    extract_logits(text, raw_logits, logits_file)
    
    # [2/4] Get embeddings
    print("[2/4] Extracting embeddings...")
    emb_file = os.path.join(base_dir, "embeddings.json")
    token_ids, embeddings = extract_embeddings_simple(model, encoder, text, emb_file)
    
    # [3/4] Extract neurons
    print("[3/4] Extracting neuron activations...")
    extract_all_neurons(model, config, encoder, text, base_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ ALL DATA SAVED TO: {base_dir}")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  • logits.json")
    print(f"  • predictions.json")
    print(f"  • embeddings.json")
    print(f"  • layer1/ to layer{config.n_layer}/ ({config.n_head} heads each)")
    print(f"\nSummary:")
    print(f"  Input: '{text}'")
    print(f"  Tokens: {len(token_strs)}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-K: {top_k}")
    print(f"  Total JSON files: {3 + config.n_layer * config.n_head} (3 + {config.n_layer} layers × {config.n_head} heads)")


def main():
    """Main interactive function."""
    print("=" * 60)
    print("BDH Complete Data Extractor")
    print("=" * 60)
    print("Generates: Neuron activations, embeddings, and predictions")
    print("All data saved to: Data/infer/ (JSON format)")
    print()
    
    # Load model and tokenizer once
    model, config = load_model()
    encoder = setup_tokenizer()
    
    while True:
        print("\n" + "-" * 60)
        text = input("Enter text (or 'quit' to exit): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        temperature_input = input("Enter temperature (0.1-2.0, default 0.8): ").strip()
        try:
            temperature = float(temperature_input) if temperature_input else 0.8
            temperature = max(0.1, min(2.0, temperature))
        except ValueError:
            temperature = 0.8
        
        top_k_input = input("Enter top-K (1-100, default 20): ").strip()
        try:
            top_k = int(top_k_input) if top_k_input else 20
            top_k = max(1, min(100, top_k))
        except ValueError:
            top_k = 20
        
        try:
            process_text(model, config, encoder, text, temperature, top_k)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
