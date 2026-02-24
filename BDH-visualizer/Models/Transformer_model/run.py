import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
import os

# -----------------------------------------------------------------------------
# Transformer Architecture (Must match training script)
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    block_size: int = 256
    dropout: float = 0.0
    device: str = 'cpu'

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.gelu    = nn.GELU()
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Install tiktoken: pip install tiktoken")
        exit()

    checkpoint_path = "checkpoints/transformer_final.pt" # Or 'checkpoints/ckpt_5000.pt'

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
    else:
        print(f"Loading {checkpoint_path}...")
        
        # Load with weights_only=False fix
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        config = checkpoint['model_args']
        config.device = 'cpu'
        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        print(f"Model Loaded. Final Loss: {checkpoint.get('loss', 'N/A')}")
        
        while True:
            text = input("\n[Prompt] (q to quit): ")
            if text == 'q': break
            
            ids = enc.encode(text)
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            
            print("[Generating] ", end="", flush=True)
            with torch.no_grad():
                for _ in range(150):
                    logits = model(x)
                    logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x, next_token), dim=1)
                    print(enc.decode([next_token.item()]), end="", flush=True)
                    if next_token.item() == 50256: break
            print("\n")