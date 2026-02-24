import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 1. Setup & Drive Mount (Optional for Colab)
# -----------------------------------------------------------------------------
def setup_directories():
    """Setup data directories. Works both on Colab and local systems."""
    # Try to detect if running in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        default_data_dir = '/content/drive/My Drive/BDH_Data'
        default_out_dir = '/content/drive/My Drive/BDH_Data/checkpoint_transformer'
        print("Running in Google Colab")
    except ImportError:
        # Not in Colab - use local paths relative to project
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        default_data_dir = os.path.join(project_root, 'data')
        default_out_dir = os.path.join(current_dir, 'checkpoints')
        print("Running locally")
    
    # Allow override via environment variables
    DATA_DIR = os.environ.get('BDH_DATA_DIR', default_data_dir)
    OUT_DIR = os.environ.get('BDH_OUT_DIR', default_out_dir)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    
    return DATA_DIR, OUT_DIR

DATA_DIR, OUT_DIR = setup_directories()

# -----------------------------------------------------------------------------
# 2. Standard Transformer Architecture (GPT-2 Style)
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_embd: int = 256     # Matched to BDH
    n_head: int = 4       # Matched to BDH
    block_size: int = 256 # Matched to BDH
    dropout: float = 0.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        # Standard Attention (Scaled Dot Product)
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
        
        # Weight tying (Standard GPT practice)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------
# 3. Data Loader
# -----------------------------------------------------------------------------
def get_batch(split, config):
    filename = os.path.join(DATA_DIR, f'{split}.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    if config.device == 'cuda':
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------
config = GPTConfig()
model = GPT(config)
model.to(config.device)

max_iters = 5000
eval_interval = 100
save_interval = 100
batch_size = 32
learning_rate = 3e-4 # Standard for this size

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting TRANSFORMER training on device: {config.device}")
print(f"Saving checkpoints to: {OUT_DIR}")

model.train()
for iter_num in range(max_iters):
    xb, yb = get_batch('train', config)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter_num % 10 == 0:
        print(f"Iter {iter_num}: Loss {loss.item():.4f}")
        
    if iter_num > 0 and iter_num % save_interval == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': config,
            'iter_num': iter_num,
            'loss': loss.item(),
        }
        ckpt_path = os.path.join(OUT_DIR, f'ckpt_{iter_num}.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"--> Saved checkpoint to {ckpt_path}")

# Save Final
final_path = os.path.join(OUT_DIR, 'transformer_final.pt')
torch.save(checkpoint, final_path)
print("Transformer Training Complete.")