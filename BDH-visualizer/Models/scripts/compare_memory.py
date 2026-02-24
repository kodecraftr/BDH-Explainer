#!/usr/bin/env python3
"""
compare_memory.py — Compare BDH vs Transformer *context memory* (long-range recall).

This script measures how well each model "remembers" earlier tokens in the sequence
by computing per-position cross-entropy loss. A model with better memory will have
lower loss at later positions because it can use earlier context effectively.

Outputs:
  - memory_comparison.csv: per-position loss for each model
  - memory_comparison.png: visualization of loss vs position
"""
import os
import sys
import argparse
import csv
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Pickle shims for configs saved under __main__ during training
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


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    block_size: int = 256
    dropout: float = 0.0
    device: str = 'cpu'


# Workspace root (two levels up from Models/scripts)
WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if WS_ROOT not in sys.path:
    sys.path.append(WS_ROOT)

# Import models
from Models.BDH_model.run import BDH  # type: ignore
from Models.Transformer_model.run import GPT  # type: ignore


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt


def init_model(model_type: str, ckpt_path: str, device: torch.device):
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt['model_args']
    try:
        config.device = str(device)
    except Exception:
        if isinstance(config, dict):
            config['device'] = str(device)

    if model_type == 'bdh':
        model = BDH(config)
    elif model_type == 'transformer':
        model = GPT(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    vocab_size = getattr(config, 'vocab_size', 50257) if not isinstance(config, dict) else config.get('vocab_size', 50257)
    block_size = getattr(config, 'block_size', 256) if not isinstance(config, dict) else config.get('block_size', 256)

    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(device)
    return model, vocab_size, block_size


def get_test_sequences(vocab_size: int, seq_len: int, num_samples: int, device: torch.device) -> torch.Tensor:
    """Generate random token sequences for testing."""
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    return torch.randint(low=0, high=vocab_size, size=(num_samples, seq_len), dtype=torch.long, device=device)


def create_copy_memory_task(vocab_size: int, seq_len: int, num_samples: int, device: torch.device, gap: int = 50) -> tuple:
    """
    Create a COPY task: the model must copy tokens from the beginning to the end.
    
    Structure: [pattern tokens] [filler tokens] [pattern tokens repeated]
    
    A model with better memory should predict the repeated pattern more accurately.
    
    Returns: (sequences, pattern_length, gap_start, repeat_start)
    """
    torch.manual_seed(42)
    
    # Use a smaller subset of vocab to make the task clearer
    pattern_vocab = min(1000, vocab_size)
    filler_token = 50256  # <|endoftext|> as filler/delimiter
    
    pattern_len = min(20, (seq_len - gap) // 2)
    
    sequences = []
    for _ in range(num_samples):
        # Generate pattern to copy
        pattern = torch.randint(0, pattern_vocab, (pattern_len,), device=device)
        
        # Filler tokens (gap)
        filler = torch.full((gap,), filler_token, dtype=torch.long, device=device)
        
        # Repeat the pattern
        repeated = pattern.clone()
        
        # Combine: [pattern][filler][repeated pattern]
        seq = torch.cat([pattern, filler, repeated])
        
        # Pad or truncate to seq_len
        if len(seq) < seq_len:
            pad = torch.full((seq_len - len(seq),), filler_token, dtype=torch.long, device=device)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:seq_len]
        
        sequences.append(seq)
    
    sequences = torch.stack(sequences)
    repeat_start = pattern_len + gap
    
    return sequences, pattern_len, pattern_len, repeat_start


def create_key_retrieval_task(vocab_size: int, seq_len: int, num_samples: int, device: torch.device) -> tuple:
    """
    Create a KEY-VALUE retrieval task.
    
    Structure: [key1:value1] [key2:value2] ... [filler] [query_key] -> should predict query_value
    
    Tests if the model can retrieve a value associated with a key seen earlier.
    
    Returns: (sequences, query_positions)
    """
    torch.manual_seed(42)
    
    num_pairs = 5
    key_vocab = 100  # Keys are tokens 0-99
    value_vocab = 100  # Values are tokens 100-199
    delimiter = 50256
    
    sequences = []
    query_positions = []
    
    for _ in range(num_samples):
        parts = []
        keys = torch.randperm(key_vocab)[:num_pairs]
        values = torch.randint(100, 100 + value_vocab, (num_pairs,))
        
        # Create key-value pairs
        for i in range(num_pairs):
            parts.append(keys[i].unsqueeze(0))
            parts.append(values[i].unsqueeze(0))
            parts.append(torch.tensor([delimiter], device=device))
        
        # Add filler
        filler_len = max(10, seq_len - len(parts) * 3 - 4)
        filler = torch.full((filler_len,), delimiter, dtype=torch.long, device=device)
        parts.append(filler)
        
        # Query: repeat a random key, model should predict corresponding value
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        parts.append(keys[query_idx].unsqueeze(0))
        parts.append(values[query_idx].unsqueeze(0))  # Target
        
        seq = torch.cat([p.to(device) for p in parts])
        
        # Pad or truncate
        if len(seq) < seq_len:
            pad = torch.full((seq_len - len(seq),), delimiter, dtype=torch.long, device=device)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:seq_len]
        
        sequences.append(seq)
        query_positions.append(len(seq) - 2)  # Position where value should be predicted
    
    return torch.stack(sequences), query_positions


def compute_per_position_loss(model: nn.Module, sequences: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss at each position in the sequence.
    Returns tensor of shape (seq_len - 1,) with average loss at each position.
    """
    B, T = sequences.shape
    
    with torch.no_grad():
        logits = model(sequences)  # (B, T, vocab_size)
    
    # Shift: predict position i+1 from position i
    logits_shifted = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    targets = sequences[:, 1:].contiguous()           # (B, T-1)
    
    # Compute per-token loss
    B, T_minus_1, V = logits_shifted.shape
    logits_flat = logits_shifted.view(B * T_minus_1, V)
    targets_flat = targets.view(B * T_minus_1)
    
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B * (T-1),)
    loss_per_token = loss_per_token.view(B, T_minus_1)  # (B, T-1)
    
    # Average across batch for each position
    per_position_loss = loss_per_token.mean(dim=0)  # (T-1,)
    
    return per_position_loss


def run_memory_benchmark(output_dir: str, device: torch.device, num_samples: int = 32):
    """
    Compare context memory of BDH vs Transformer using structured memory tasks.
    """
    rows: List[Dict[str, Any]] = []

    # Locate checkpoints
    bdh_ckpt = os.path.join(WS_ROOT, 'Models', 'BDH_model', 'checkpoints', 'bdh_final.pt')
    if not os.path.exists(bdh_ckpt):
        fallback = os.path.join(WS_ROOT, 'Models', 'BDH_model', 'checkpoints', 'ckpt_5000.pt')
        if os.path.exists(fallback):
            bdh_ckpt = fallback
        else:
            raise FileNotFoundError('BDH checkpoint not found')

    tf_ckpt = os.path.join(WS_ROOT, 'Models', 'Transformer_model', 'checkpoints', 'transformer_final.pt')
    if not os.path.exists(tf_ckpt):
        fallback = os.path.join(WS_ROOT, 'Models', 'Transformer_model', 'checkpoints', 'ckpt_5000.pt')
        if os.path.exists(fallback):
            tf_ckpt = fallback
        else:
            raise FileNotFoundError('Transformer checkpoint not found')

    print("Loading models...")
    bdh_model, bdh_vocab, bdh_block = init_model('bdh', bdh_ckpt, device)
    tf_model, tf_vocab, tf_block = init_model('transformer', tf_ckpt, device)

    # Use minimum block size
    seq_len = min(bdh_block, tf_block)
    print(f"Testing with sequence length: {seq_len}")

    # =========================================================================
    # TEST 1: Per-position loss on random sequences (baseline)
    # =========================================================================
    print("\n=== Test 1: Per-Position Loss (Random Sequences) ===")
    sequences_bdh = get_test_sequences(bdh_vocab, seq_len, num_samples, device)
    sequences_tf = get_test_sequences(tf_vocab, seq_len, num_samples, device)

    bdh_losses_random = compute_per_position_loss(bdh_model, sequences_bdh).cpu().tolist()
    tf_losses_random = compute_per_position_loss(tf_model, sequences_tf).cpu().tolist()

    for pos in range(len(bdh_losses_random)):
        rows.append({'test': 'random', 'model': 'BDH', 'position': pos + 1, 'loss': bdh_losses_random[pos]})
        rows.append({'test': 'random', 'model': 'Transformer', 'position': pos + 1, 'loss': tf_losses_random[pos]})

    # =========================================================================
    # TEST 2: Copy Task - tests long-range memory explicitly
    # =========================================================================
    print("\n=== Test 2: Copy Task (Long-Range Memory) ===")
    gaps = [20, 50, 100, 150]  # Different memory distances
    copy_results = []
    
    for gap in gaps:
        if gap + 40 > seq_len:
            continue
            
        copy_seqs_bdh, pattern_len, gap_start, repeat_start = create_copy_memory_task(
            bdh_vocab, seq_len, num_samples, device, gap=gap
        )
        copy_seqs_tf, _, _, _ = create_copy_memory_task(
            tf_vocab, seq_len, num_samples, device, gap=gap
        )
        
        # Compute loss only on the repeated pattern portion (where memory matters)
        bdh_copy_loss = compute_per_position_loss(bdh_model, copy_seqs_bdh).cpu()
        tf_copy_loss = compute_per_position_loss(tf_model, copy_seqs_tf).cpu()
        
        # Extract loss for the repeat section
        repeat_end = min(repeat_start + pattern_len, len(bdh_copy_loss))
        if repeat_start < len(bdh_copy_loss):
            bdh_repeat_loss = bdh_copy_loss[repeat_start:repeat_end].mean().item()
            tf_repeat_loss = tf_copy_loss[repeat_start:repeat_end].mean().item()
        else:
            bdh_repeat_loss = float('nan')
            tf_repeat_loss = float('nan')
        
        copy_results.append({
            'gap': gap,
            'bdh_loss': bdh_repeat_loss,
            'tf_loss': tf_repeat_loss,
        })
        
        rows.append({'test': f'copy_gap{gap}', 'model': 'BDH', 'position': gap, 'loss': bdh_repeat_loss})
        rows.append({'test': f'copy_gap{gap}', 'model': 'Transformer', 'position': gap, 'loss': tf_repeat_loss})
        
        print(f"  Gap={gap}: BDH={bdh_repeat_loss:.4f}, Transformer={tf_repeat_loss:.4f}")

    # =========================================================================
    # TEST 3: Attention pattern analysis - measure how much each position attends to early tokens
    # =========================================================================
    print("\n=== Test 3: Late vs Early Position Loss Ratio ===")
    # Compare loss at early positions (1-50) vs late positions (last 50)
    early_bdh = sum(bdh_losses_random[:50]) / 50 if len(bdh_losses_random) >= 50 else sum(bdh_losses_random) / len(bdh_losses_random)
    early_tf = sum(tf_losses_random[:50]) / 50 if len(tf_losses_random) >= 50 else sum(tf_losses_random) / len(tf_losses_random)
    late_bdh = sum(bdh_losses_random[-50:]) / 50 if len(bdh_losses_random) >= 50 else sum(bdh_losses_random[-len(bdh_losses_random)//2:]) / (len(bdh_losses_random)//2)
    late_tf = sum(tf_losses_random[-50:]) / 50 if len(tf_losses_random) >= 50 else sum(tf_losses_random[-len(tf_losses_random)//2:]) / (len(tf_losses_random)//2)

    # Lower ratio = loss stays lower at late positions = better memory
    bdh_ratio = late_bdh / early_bdh if early_bdh > 0 else float('inf')
    tf_ratio = late_tf / early_tf if early_tf > 0 else float('inf')

    rows.append({'test': 'loss_ratio', 'model': 'BDH', 'position': 0, 'loss': bdh_ratio})
    rows.append({'test': 'loss_ratio', 'model': 'Transformer', 'position': 0, 'loss': tf_ratio})

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'memory_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['test', 'model', 'position', 'loss'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    # Generate visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Per-position loss (random)
        ax1 = axes[0, 0]
        positions = list(range(1, len(bdh_losses_random) + 1))
        ax1.plot(positions, bdh_losses_random, label='BDH', alpha=0.7, linewidth=1)
        ax1.plot(positions, tf_losses_random, label='Transformer', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Per-Position Loss (Random Sequences)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Smoothed comparison
        ax2 = axes[0, 1]
        window = 15
        if len(bdh_losses_random) > window:
            bdh_smooth = np.convolve(bdh_losses_random, np.ones(window)/window, mode='valid')
            tf_smooth = np.convolve(tf_losses_random, np.ones(window)/window, mode='valid')
            smooth_pos = list(range(window, len(bdh_losses_random) + 1))
            ax2.plot(smooth_pos, bdh_smooth, label='BDH', linewidth=2)
            ax2.plot(smooth_pos, tf_smooth, label='Transformer', linewidth=2)
            ax2.set_xlabel('Token Position')
            ax2.set_ylabel('Smoothed Loss (window=15)')
            ax2.set_title('Smoothed Per-Position Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Copy task results
        ax3 = axes[1, 0]
        if copy_results:
            gaps_plot = [r['gap'] for r in copy_results]
            bdh_copy = [r['bdh_loss'] for r in copy_results]
            tf_copy = [r['tf_loss'] for r in copy_results]
            x = np.arange(len(gaps_plot))
            width = 0.35
            ax3.bar(x - width/2, bdh_copy, width, label='BDH', alpha=0.8)
            ax3.bar(x + width/2, tf_copy, width, label='Transformer', alpha=0.8)
            ax3.set_xlabel('Gap Distance (tokens)')
            ax3.set_ylabel('Loss on Repeated Pattern')
            ax3.set_title('Copy Task: Loss at Different Memory Distances')
            ax3.set_xticks(x)
            ax3.set_xticklabels(gaps_plot)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Summary stats
        ax4 = axes[1, 1]
        labels = ['Avg Loss\n(All Pos)', 'Late Loss\n(Last 50)', 'Early Loss\n(First 50)', 'Late/Early\nRatio']
        avg_bdh = sum(bdh_losses_random) / len(bdh_losses_random)
        avg_tf = sum(tf_losses_random) / len(tf_losses_random)
        bdh_vals = [avg_bdh, late_bdh, early_bdh, bdh_ratio]
        tf_vals = [avg_tf, late_tf, early_tf, tf_ratio]
        x = np.arange(len(labels))
        width = 0.35
        ax4.bar(x - width/2, bdh_vals, width, label='BDH', alpha=0.8)
        ax4.bar(x + width/2, tf_vals, width, label='Transformer', alpha=0.8)
        ax4.set_ylabel('Value')
        ax4.set_title('Summary Statistics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(output_dir, 'memory_comparison.png')
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved plot: {png_path}")
    except ImportError:
        print("matplotlib not available, skipping plot generation")

    # Print summary
    print("\n" + "="*60)
    print("=== CONTEXT MEMORY COMPARISON SUMMARY ===")
    print("="*60)
    avg_bdh = sum(bdh_losses_random) / len(bdh_losses_random)
    avg_tf = sum(tf_losses_random) / len(tf_losses_random)
    print(f"\n1. Average Loss (random sequences):")
    print(f"   BDH:         {avg_bdh:.4f}")
    print(f"   Transformer: {avg_tf:.4f}")
    print(f"   Winner: {'BDH' if avg_bdh < avg_tf else 'Transformer'} (lower is better)")
    
    print(f"\n2. Late-Position Loss (last 50 tokens):")
    print(f"   BDH:         {late_bdh:.4f}")
    print(f"   Transformer: {late_tf:.4f}")
    print(f"   Winner: {'BDH' if late_bdh < late_tf else 'Transformer'} (lower is better)")
    
    print(f"\n3. Late/Early Ratio (lower = better memory retention):")
    print(f"   BDH:         {bdh_ratio:.4f}")
    print(f"   Transformer: {tf_ratio:.4f}")
    print(f"   Winner: {'BDH' if bdh_ratio < tf_ratio else 'Transformer'}")
    
    if copy_results:
        print(f"\n4. Copy Task (explicit memory test):")
        for r in copy_results:
            winner = 'BDH' if r['bdh_loss'] < r['tf_loss'] else 'Transformer'
            print(f"   Gap {r['gap']:3d}: BDH={r['bdh_loss']:.4f}, TF={r['tf_loss']:.4f} -> {winner}")
    
    print("\n" + "="*60)
    print("NOTE: These models were trained on TinyStories, not memory tasks.")
    print("Random sequence loss tests language modeling, not true recall.")
    print("The Copy Task specifically tests long-range memory capability.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compare BDH vs Transformer context memory (long-range recall).')
    default_out = os.path.join(WS_ROOT, 'Models', 'comparison', 'memory')
    parser.add_argument('--output_dir', type=str, default=default_out)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test sequences')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    run_memory_benchmark(args.output_dir, device, args.num_samples)


if __name__ == '__main__':
    main()
