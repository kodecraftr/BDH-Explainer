import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.BDH_model.run import BDH, BDHConfig
import torch
import json
import numpy as np
from datetime import datetime

# Import the embedding processing functions
from embedding_processing import process_embeddings_with_info, apply_rope

class BDHEmbeddingExtractor:
    def __init__(self, model_path=None):
        """
        Initialize the BDH embedding extractor.
        
        Args:
            model_path: Path to the BDH model checkpoint. If None, uses default checkpoint.
        """
        if model_path is None:
            # Use default path relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, "Models", "BDH_model", "checkpoints", "ckpt_5000.pt")
        """
        Initialize the BDH embedding extractor.
        
        Args:
            model_path: Path to the BDH model checkpoint
        """
        self.model_path = model_path
        self.model = None
        self.config = None
        self.encoder = None
        self.load_model()
        self.setup_tokenizer()
    
    @staticmethod
    def validate_parameters(temperature, top_k):
        """
        Validate and clamp temperature and top_k parameters to acceptable ranges.
        
        Args:
            temperature: Temperature value (will be clamped to 0.1-2.0)
            top_k: Top-K value (will be clamped to 1-100)
            
        Returns:
            tuple: (validated_temperature, validated_top_k)
        """
        # Clamp temperature to reasonable range
        temp = max(0.1, min(2.0, float(temperature)))
        
        # Clamp top_k to reasonable range  
        k = max(1, min(100, int(top_k)))
        
        return temp, k
    
    @staticmethod
    def get_parameter_info():
        """
        Get information about temperature and top_k parameters.
        
        Returns:
            dict: Parameter information and recommended ranges
        """
        return {
            'temperature': {
                'range': (0.1, 2.0),
                'default': 0.8,
                'description': 'Controls randomness in sampling (0.1=focused, 2.0=creative)'
            },
            'top_k': {
                'range': (1, 100), 
                'default': 20,
                'description': 'Limits sampling to top K most likely tokens (1=deterministic, 100=diverse)'
            }
        }
    
    def load_model(self):
        """Load the BDH model from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Error: {self.model_path} not found.")
        
        print("Loading BDH model...")
        
        # Import classes in the correct way to handle pickle loading
        import sys
        original_main = sys.modules.get('__main__')
        
        try:
            # Temporarily add BDH classes to __main__ to handle pickle loading
            import __main__
            __main__.BDH = BDH
            __main__.BDHConfig = BDHConfig
            
            # Now try to load the checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract config and model state
            if 'model_args' in checkpoint:
                self.config = checkpoint['model_args']
            else:
                # Fallback to default config
                self.config = BDHConfig()
            
            # Ensure device is set to CPU
            self.config.device = 'cpu'
            
            # Initialize model
            self.model = BDH(self.config)
            
            # Load model state
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"Model loaded. Embedding dimension: {self.config.n_embd}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying alternative approach...")
            
            # Alternative: Create model with default config and try manual loading
            self.config = BDHConfig()
            self.config.device = 'cpu'
            self.model = BDH(self.config)
            print("Created model with default config. Note: This may not match the trained checkpoint.")
            self.model.eval()
            print(f"Model initialized with default config. Embedding dimension: {self.config.n_embd}")
            
        finally:
            # Clean up main module modifications
            if hasattr(__main__, 'BDH'):
                delattr(__main__, 'BDH')
            if hasattr(__main__, 'BDHConfig'):
                delattr(__main__, 'BDHConfig')
    
    def setup_tokenizer(self):
        """Setup the tiktoken tokenizer."""
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding("gpt2")
            print("Tokenizer loaded.")
        except ImportError:
            print("Please install tiktoken: pip install tiktoken")
            sys.exit(1)
    
    def tokenize_text(self, text):
        """
        Tokenize input text and return token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            list: Token IDs
        """
        token_ids = self.encoder.encode(text)
        return token_ids

    def get_embedding_vectors(self, token_ids):
        """
        Extract embedding vectors for given token IDs from the BDH model.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            torch.Tensor: Embedding vectors of shape [num_tokens, embedding_dim]
        """
        with torch.no_grad():
            # Convert token IDs to tensor
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            # Get embeddings from the model's embedding layer
            embeddings = self.model.embed(token_tensor)
        return embeddings
    
    def predict_top_k_tokens(self, text, temperature=0.8, top_k=20):
        """
        Predict top-K next tokens with their probabilities for given text.
        
        Args:
            text: Input text string
            temperature: Temperature for probability scaling
            top_k: Number of top tokens to return
            
        Returns:
            list: List of dicts with 'token_id', 'token_text', 'probability'
        """
        # Tokenize the input text
        input_ids = self.encoder.encode(text)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass through model
            logits = self.model(input_tensor)
            
            # Get logits for the last token position
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_actual = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k_actual)
            else:
                top_k_logits = next_token_logits
                top_k_indices = torch.arange(len(next_token_logits))
            
            # Convert to probabilities
            probabilities = torch.softmax(top_k_logits, dim=-1)
            
            # Create result list
            predictions = []
            for i, (token_idx, prob) in enumerate(zip(top_k_indices, probabilities)):
                token_id = token_idx.item()
                token_text = self.encoder.decode([token_id])
                predictions.append({
                    'rank': i + 1,
                    'token_id': token_id,
                    'token_text': token_text,
                    'probability': round(prob.item(), 4)
                })
                
        return predictions
    
    def save_to_json(self, text, token_ids, original_embeddings, processed_info, rope_processed_info, top_k_predictions, filename=None, temperature=0.8, top_k=20):
        """
        Save processed 32-dimensional embeddings (both original and RoPE) and top-K predictions to JSON.
        
        Each token will have both original and rope embeddings in a single object.
        Includes token_ids and words arrays for easy access.
        
        Args:
            text: Original input text
            token_ids: List of token IDs
            original_embeddings: Original embedding vectors tensor [num_tokens, 256] (for metadata only)
            processed_info: Dictionary from process_embeddings_with_info() for original embeddings
            rope_processed_info: Dictionary from process_embeddings_with_info() for RoPE embeddings
            top_k_predictions: List of top-K token predictions with probabilities
            filename: Optional custom filename for JSON
            temperature: Temperature parameter used
            top_k: Top-K parameter used
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embeddings_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Get processed embeddings
        processed_embeddings = processed_info['processed_embeddings']
        processed_embeddings_np = processed_embeddings.numpy()
        
        rope_processed_embeddings = rope_processed_info['processed_embeddings']
        rope_processed_embeddings_np = rope_processed_embeddings.numpy()
        
        # Decode tokens back to text for reference
        decoded_tokens = [self.encoder.decode([tid]) for tid in token_ids]
        
        # Build data structure
        data = {
            'input_text': text,
            'token_ids': token_ids,
            'words': decoded_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'tokens': [],
            'predictions': top_k_predictions
        }
        
        # Add combined embeddings (both original and rope for each token)
        for i, (token_id, token_text, original_embedding_32d, rope_embedding_32d) in enumerate(zip(token_ids, decoded_tokens, processed_embeddings_np, rope_processed_embeddings_np)):
            original_embedding_rounded = [round(float(val), 4) for val in original_embedding_32d]
            rope_embedding_rounded = [round(float(val), 4) for val in rope_embedding_32d]
            data['tokens'].append({
                'token_index': i,
                'token_id': int(token_id),
                'token_text': token_text,
                'original_embedding': original_embedding_rounded,
                'rope_embedding': rope_embedding_rounded
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Embeddings and predictions saved to: {filepath}")
        return filepath
    
    def process_user_input(self, text, save_json=True, json_filename=None, target_dim=32, norm_method='min_max', temperature=0.8, top_k=20):
        """
        Process text input: tokenize, extract embeddings, reduce to 32D, normalize, and save to JSON.
        
        Args:
            text: Input text string
            save_json: Whether to save results to JSON
            json_filename: Optional custom filename for JSON
            target_dim: Target embedding dimension (default 32)
            norm_method: Normalization method (default 'min_max')
            temperature: Temperature for model sampling (0.1-2.0, default 0.8)
            top_k: Top-K for model sampling (1-100, default 20)
            
        Returns:
            dict: Results containing tokens, original embeddings, processed embeddings, and JSON file path
        """
        # Validate parameters
        temperature, top_k = self.validate_parameters(temperature, top_k)
        
        # Tokenize
        token_ids = self.tokenize_text(text)
        
        # Extract 256D embeddings
        original_embeddings = self.get_embedding_vectors(token_ids)
        
        # Apply RoPE transformation to original embeddings
        rope_embeddings = apply_rope(original_embeddings)
        
        # Process original embeddings to 32D and normalize
        processed_info = process_embeddings_with_info(
            original_embeddings, 
            target_dim=target_dim, 
            norm_method=norm_method
        )
        processed_embeddings = processed_info['processed_embeddings']
        
        # Process RoPE embeddings to 32D and normalize
        rope_processed_info = process_embeddings_with_info(
            rope_embeddings,
            target_dim=target_dim,
            norm_method=norm_method
        )
        rope_processed_embeddings = rope_processed_info['processed_embeddings']
        
        # Get token text
        decoded_tokens = [self.encoder.decode([tid]) for tid in token_ids]
        
        # Get top-K predictions with probabilities
        top_k_predictions = self.predict_top_k_tokens(text, temperature, top_k)
        
        results = {
            'text': text,
            'token_ids': token_ids,
            'original_embeddings': original_embeddings,
            'rope_embeddings': rope_embeddings,
            'processed_embeddings': processed_embeddings,
            'rope_processed_embeddings': rope_processed_embeddings,
            'processed_info': processed_info,
            'rope_processed_info': rope_processed_info,
            'decoded_tokens': decoded_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_k_predictions': top_k_predictions
        }
        
        # Save to JSON
        if save_json:
            json_path = self.save_to_json(text, token_ids, original_embeddings, processed_info, rope_processed_info, top_k_predictions, json_filename, temperature, top_k)
            results['json_file'] = json_path
        
        return results

def main():
    """Interactive main function for user input."""
    print("BDH Model Embedding Extractor")
    print("=" * 40)
    
    # Initialize the extractor
    try:
        extractor = BDHEmbeddingExtractor()
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        return
    
    while True:
        print("\nOptions:")
        print("1. Enter text to process")
        print("2. Process text from command line argument")
        print("3. Quit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            text = input("\nEnter your text: ")
            if text.strip():
                json_name = input("Enter JSON filename (or press Enter for auto-generated name): ").strip()
                json_name = json_name if json_name else None
                extractor.process_user_input(text, save_json=True, json_filename=json_name)
            else:
                print("Empty text provided!")
                
        elif choice == '2':
            if len(sys.argv) > 1:
                text = ' '.join(sys.argv[1:])
                extractor.process_user_input(text, save_json=True)
            else:
                print("No command line argument provided!")
                
        elif choice == '3':
            print("Goodbye!")
            break
            
        else:
            print("Invalid option! Please choose 1, 2, or 3.")

def save_embeddings_simple(filepath, text, token_ids, embeddings, encoder):
    """
    Simple function to save embedding vectors to JSON (reduced to 30D, includes RoPE).
    Used by demo.py for consistent interface.
    
    Each token will have both original and rope embeddings in a single object.
    Includes token_ids and words arrays for easy access.
    
    Args:
        filepath: Output JSON file path
        text: Input text string
        token_ids: List of token IDs
        embeddings: Original embedding tensor [num_tokens, embedding_dim]
        encoder: Tokenizer encoder
    """
    import numpy as np
    
    embeddings_np = embeddings.cpu().numpy()
    
    # Apply RoPE transformation
    rope_embeddings = apply_rope(embeddings)
    rope_embeddings_np = rope_embeddings.cpu().numpy()
    
    # Decode tokens to get text representation
    decoded_tokens = [encoder.decode([tid]) for tid in token_ids]
    
    # Take first 30 dimensions (simple slice from 256D)
    embeddings_reduced = embeddings_np[:, :30]
    rope_embeddings_reduced = rope_embeddings_np[:, :30]
    
    # Normalize to 0-1 range (min-max normalization per token)
    embeddings_normalized = np.zeros_like(embeddings_reduced)
    for i in range(embeddings_reduced.shape[0]):
        min_val = embeddings_reduced[i].min()
        max_val = embeddings_reduced[i].max()
        if max_val - min_val > 1e-8:  # Avoid division by zero
            embeddings_normalized[i] = (embeddings_reduced[i] - min_val) / (max_val - min_val)
        else:
            embeddings_normalized[i] = 0.5  # If all values same, set to middle
    
    # Normalize RoPE embeddings
    rope_embeddings_normalized = np.zeros_like(rope_embeddings_reduced)
    for i in range(rope_embeddings_reduced.shape[0]):
        min_val = rope_embeddings_reduced[i].min()
        max_val = rope_embeddings_reduced[i].max()
        if max_val - min_val > 1e-8:
            rope_embeddings_normalized[i] = (rope_embeddings_reduced[i] - min_val) / (max_val - min_val)
        else:
            rope_embeddings_normalized[i] = 0.5
    
    # Build data structure
    data = {
        'input_text': text,
        'token_ids': token_ids,
        'words': decoded_tokens,
        'tokens': []
    }
    
    # Add combined embeddings (both original and rope for each token)
    for token_text, token_id, original_embedding, rope_embedding in zip(decoded_tokens, token_ids, embeddings_normalized, rope_embeddings_normalized):
        data['tokens'].append({
            'token_text': token_text,
            'token_id': int(token_id),
            'original_embedding': [round(float(val), 4) for val in original_embedding],
            'rope_embedding': [round(float(val), 4) for val in rope_embedding]
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_embeddings_simple(model, encoder, text, filepath):
    """
    Extract embeddings and save to JSON. Simple interface for demo.py.
    
    Args:
        model: BDH model instance
        encoder: Tokenizer encoder
        text: Input text string
        filepath: Output JSON file path
        
    Returns:
        tuple: (token_ids, embeddings)
    """
    print("[Embeddings] Extracting embeddings...")
    
    # Get embeddings
    token_ids = encoder.encode(text)
    with torch.no_grad():
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        embeddings = model.embed(token_tensor)
    
    print(f"  ✓ Embeddings extracted: {embeddings.shape}")
    
    # Save to JSON
    save_embeddings_simple(filepath, text, token_ids, embeddings, encoder)
    print(f"  ✓ Embeddings saved: {filepath}")
    
    return token_ids, embeddings


if __name__ == "__main__":
    main()