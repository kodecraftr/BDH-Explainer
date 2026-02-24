import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
import os
import argparse

# -----------------------------------------------------------------------------
# BDH Classes (Must act as the blueprint)
# -----------------------------------------------------------------------------
@dataclass
class BDHConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    dropout: float = 0.0
    block_size: int = 256
    device: str = 'cpu'

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.register_buffer("freqs", self.get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))

    @staticmethod
    def get_freqs(n, theta, dtype):
        def quantize(t, q=2):
            return (t / q).floor() * q
        return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        _, _, T, _ = Q.size()
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR 
        scores = (QR @ KR.transpose(-2, -1)).tril(diagonal=-1)
        return scores @ V

class BDH(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = D * config.mlp_internal_dim_multiplier // nh
        self.embed = nn.Embedding(config.vocab_size, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))
        self.attn = Attention(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        B, T = idx.size()
        D = self.config.n_embd
        nh = self.config.n_head
        N = D * self.config.mlp_internal_dim_multiplier // nh
        
        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        
        for level in range(self.config.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder)
            y = self.ln(yMLP)
            x = self.ln(x + y)
        return x.view(B, T, D) @ self.lm_head

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    exit()

def generate(model, prompt, max_tokens=150, temperature=0.8, top_k=20):
    print(f"\n[Prompt]: {prompt}")
    print(f"[Config]: Temp={temperature}, Top-K={top_k}")
    print("[Story]: ", end="", flush=True)
    
    ids = enc.encode(prompt)
    
    # Handle empty prompt
    if len(ids) == 0:
        print("(empty prompt - using default starting token)")
        ids = [enc.encode(" ")[0]]  # Use a space token as starting point
    
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / temperature # Scale by temp
            
            # Optional: Top-K Sampling (Keeps only the K most likely next words)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
            
            # Stop token check (optional, if dataset has <|endoftext|>)
            if next_token.item() == 50256: 
                break

            word = enc.decode([next_token.item()])
            print(word, end="", flush=True)
    print("\n\n[End]")

if __name__ == "__main__":
    # Settings
    MODEL_PATH = "checkpoints/ckpt_5000.pt" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
    else:
        print(f"Loading model...")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        config = checkpoint['model_args']
        config.device = 'cpu'
        
        model = BDH(config)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        while True:
            text = input("\nEnter prompt (or 'q' to quit): ")
            if text.lower() == 'q': break
            
            # You can tweak these!
            # Temp 0.7 = Focused
            # Temp 1.0 = Creative
            generate(model, text, temperature=0.7, top_k=50)