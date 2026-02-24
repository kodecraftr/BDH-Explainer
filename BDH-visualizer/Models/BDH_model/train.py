import os
import math
import time
import inspect
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
        default_out_dir = '/content/drive/My Drive/BDH_Data/checkpoints'
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
# 2. BDH Model Architecture (Based on your "Baby Dragon" docs )
# -----------------------------------------------------------------------------
@dataclass
class BDHConfig:
    vocab_size: int = 50257 # Default for GPT-2 tokenizer, adjust if using different
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    dropout: float = 0.0
    block_size: int = 256 # Context length (TinyStories is usually small)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        # N is the sparse neuron dimension (hidden width)
        N = config.mlp_internal_dim_multiplier * D // nh

        # RoPE frequencies buffer
        self.register_buffer("freqs", get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        # Apply Rotary Positional Embeddings
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        # Q, K shape: [B, nh, T, N]
        _, _, T, _ = Q.size()

        # Calculate rotary phases based on time T
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)) * self.freqs

        # Apply RoPE to Query and Key (injecting position info)
        QR = self.rope(r_phases, Q)
        KR = QR # In BDH, K is often symmetric to Q in the parallel formulation

        # Linear Attention / Hebbian Update Score
        scores = (QR @ KR.transpose(-2, -1)).tril(diagonal=-1)

        return scores @ V

class BDH(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = D * config.mlp_internal_dim_multiplier // nh

        # Learnable Weights
        self.embed = nn.Embedding(config.vocab_size, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        # Matrices for the "Expansion" and "Contraction" phases
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        self.attn = Attention(config)
        self.drop = nn.Dropout(config.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # 1. Embedding (No positional embeddings added here, handled in RoPE)
        x = self.embed(idx).unsqueeze(1) # [B, 1, T, D]
        x = self.ln(x)

        for level in range(C.n_layer):
            # Phase 1: Expansion (Create Sparse Neurons)
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent) # Sparsity induced by ReLU

            # Phase 2: Interaction (Hebbian Memory / Attention)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            # Phase 3: Value Projection
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)

            # Phase 4: Gating (Hebbian "Fire Together" Logic)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            # Phase 5: Compression (Back to embedding dim)
            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder)

            y = self.ln(yMLP)
            x = self.ln(x + y) # Residual connection

        # Output Head
        logits = x.view(B, T, D) @ self.lm_head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# -----------------------------------------------------------------------------
# 3. Data Loader (For .bin files)
# -----------------------------------------------------------------------------
def get_batch(split, config):
    # We recreate the memmap every batch to avoid memory leaks in Colab
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
config = BDHConfig()
model = BDH(config)
model.to(config.device)

# Hyperparameters
max_iters = 3000 # Adjust as needed
eval_interval = 100
save_interval = 100 # Requirement: Save every 100 iterations
batch_size = 30 # Adjust based on GPU memory (T4 vs A100)
learning_rate = 3e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting training on device: {config.device}")
print(f"Saving checkpoints to: {OUT_DIR}")

model.train()
for iter_num in range(max_iters):
    # 1. Get Batch
    xb, yb = get_batch('train', config)

    # 2. Forward Pass
    logits, loss = model(xb, yb)

    # 3. Backward Pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 4. Logging & Saving
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
        # Save filename includes iteration number
        ckpt_path = os.path.join(OUT_DIR, f'ckpt_{iter_num}.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"--> Saved checkpoint to {ckpt_path}")

# Save Final Model
final_path = os.path.join(OUT_DIR, 'bdh_final.pt')
torch.save(checkpoint, final_path)
print("Training Complete. Final model saved.")