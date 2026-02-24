import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Ensure we can import from the Models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from Models.BDH_model.run import BDH, BDHConfig
except ImportError:
    # Fallback/Mock for development if model files are missing
    print("Warning: Could not import BDH model. ensuring Models/BDH_model/run.py exists.")
    BDH = None
    BDHConfig = None

class BDHModelHandler:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default path based on your file structure
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "Models", "BDH_model", "checkpoints", "ckpt_5000.pt")
        
        self.model_path = model_path
        self.model = None
        self.config = None
        self.encoder = None
        self.device = 'cpu'
        
        self.load_model()
        self.setup_tokenizer()

    def load_model(self):
        """Loads the BDH model safely."""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model checkpoint not found at {self.model_path}. Using random init for testing.")
            self.config = BDHConfig() if BDHConfig else None
            if BDH:
                self.model = BDH(self.config)
                self.model.eval()
            return

        print(f"Loading BDH model from {self.model_path}...")
        try:
            # Trick to handle the pickle import issue mentioned in your scripts
            import __main__
            __main__.BDH = BDH
            __main__.BDHConfig = BDHConfig

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if 'model_args' in checkpoint:
                self.config = checkpoint['model_args']
            else:
                self.config = BDHConfig()
            
            self.config.device = self.device
            self.model = BDH(self.config)
            
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to random init
            self.config = BDHConfig()
            self.model = BDH(self.config)
            self.model.eval()
        finally:
            # Clean up __main__ to avoid side effects
            if hasattr(__main__, 'BDH'): delattr(__main__, 'BDH')
            if hasattr(__main__, 'BDHConfig'): delattr(__main__, 'BDHConfig')

    def setup_tokenizer(self):
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding("gpt2")
        except ImportError:
            print("Error: tiktoken not installed.")
            self.encoder = None

    def tokenize(self, text):
        if not self.encoder:
            raise ImportError("tiktoken not installed")
        ids = self.encoder.encode(text, allowed_special={'<|endoftext|>'})
        return torch.tensor([ids], dtype=torch.long)

    def predict(self, text, temperature=0.8, top_k=20):
        """
        Generates next token predictions.
        Incorporates logic from recalculate_predictions.py directly into inference.
        """
        idx = self.tokenize(text)
        with torch.no_grad():
            logits = self.model(idx)
            # Focus on the last token's logits
            last_token_logits = logits[:, -1, :] 
            
            # Apply temperature
            scaled_logits = last_token_logits / temperature
            
            # Apply Top-K
            if top_k is not None:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], k=min(top_k, probs.size(-1)))
            
            predictions = []
            for i, (prob, idx_val) in enumerate(zip(top_probs, top_indices)):
                token = self.encoder.decode([idx_val.item()])
                predictions.append({
                    'rank': i + 1,
                    'token': token,
                    'token_id': idx_val.item(),
                    'probability': float(prob.item())
                })
            
            return predictions

    def get_embeddings(self, text, target_dim=32):
        """
        Extracts and normalizes embeddings.
        Implements logic from demo.py's save_embeddings method.
        """
        idx = self.tokenize(text)
        token_ids = idx[0].tolist()
        
        with torch.no_grad():
            # Get raw embeddings [1, T, 256]
            embeddings = self.model.embed(idx)
            embeddings = embeddings.squeeze(0) # [T, 256]
            
        embeddings_np = embeddings.cpu().numpy()
        decoded_tokens = [self.encoder.decode([tid]) for tid in token_ids]
        
        # 1. Dimensionality Reduction (Slice first N dimensions)
        # Using target_dim=30 or 32 as requested
        embeddings_reduced = embeddings_np[:, :target_dim]
        
        # 2. Min-Max Normalization per token
        normalized_embeddings = []
        for i in range(embeddings_reduced.shape[0]):
            min_val = embeddings_reduced[i].min()
            max_val = embeddings_reduced[i].max()
            if max_val - min_val > 1e-8:
                norm_row = (embeddings_reduced[i] - min_val) / (max_val - min_val)
            else:
                norm_row = np.full_like(embeddings_reduced[i], 0.5)
            normalized_embeddings.append(norm_row.tolist())
            
        return {
            "tokens": decoded_tokens,
            "token_ids": token_ids,
            "embeddings": normalized_embeddings
        }

    def capture_neurons(self, text, layer_idx):
        """
        Captures neuron activations for a specific layer.
        Ported from extract_neuron_values.py -> forward_with_neuron_capture
        """
        idx = self.tokenize(text)
        B, T = idx.size()
        
        # We need these config values for reshaping
        D = self.config.n_embd
        nh = self.config.n_head
        N = D * self.config.mlp_internal_dim_multiplier // nh
        
        captured_data = {}
        
        # Forward pass (manual implementation to access intermediates)
        x = self.model.embed(idx).unsqueeze(1)
        x = self.model.ln(x)
        
        # Loop through layers
        # Note: layer_idx is 1-based in your scripts, keeping that convention
        target_layer_index = layer_idx - 1 
        
        for level in range(self.config.n_layer):
            # 1. Expansion
            x_latent = x @ self.model.encoder
            x_sparse = F.relu(x_latent)
            
            # 2. Attention
            yKV = self.model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.model.ln(yKV)
            
            # 3. Second Expansion
            y_latent = yKV @ self.model.encoder_v
            y_sparse = F.relu(y_latent)
            
            # 4. Gating
            xy_sparse = x_sparse * y_sparse
            
            # 5. Contraction
            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.model.decoder)
            y = self.model.ln(yMLP)
            
            # Residual
            x = self.model.ln(x + y)
            
            # Capture if this is the target layer
            if level == target_layer_index:
                # Convert tensors to lists for JSON serialization
                # x_sparse shape: [B, nh, T, N]
                # We usually care about the activations for the *last token* in visualization,
                # but let's return the full structure for the last token only to save bandwidth.
                
                # Squeezing Batch dim (B=1) -> [nh, T, N]
                activations = x_sparse[0].detach().cpu()
                
                # Get activations for the last token only: [nh, N]
                last_token_acts = activations[:, -1, :]
                
                captured_data = {
                    "layer": layer_idx,
                    "heads": nh,
                    "neurons_per_head": N,
                    "activations": last_token_acts.tolist() # List of lists [head][neuron_val]
                }
                break # We found our layer, no need to compute the rest
                
        return captured_data