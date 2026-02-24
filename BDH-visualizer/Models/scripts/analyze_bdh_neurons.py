import os
import csv
import argparse
from typing import List, Tuple, Dict

import torch
from dataclasses import dataclass


# Define a minimal BDHConfig to satisfy safe unpickling of trusted checkpoints
@dataclass
class BDHConfig:
    pass

# Allowlist BDHConfig for safe globals if the API is available
try:
    import torch.serialization as _ts
    if hasattr(_ts, "add_safe_globals"):
        _ts.add_safe_globals([BDHConfig])
except Exception:
    # If not available, proceed; we'll still try weights_only=True
    pass


def list_checkpoints(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.startswith("ckpt_") and name.endswith(".pt"):
            files.append(os.path.join(dir_path, name))
    return sorted(files, key=lambda p: int(os.path.basename(p)[5:-3]))


def topk_indices(t: torch.Tensor, k: int) -> torch.Tensor:
    # returns indices of top-k by absolute value along last dim
    vals = t.abs()
    _, idx = torch.topk(vals, k=min(k, vals.shape[-1]), dim=-1)
    return idx


def analyze_checkpoint(ckpt_path: str, out_dir: str, k_in: int = 5, k_out: int = 5, top_neighbors: int = 3) -> None:
    device = torch.device("cpu")
    # Load only tensor weights to avoid unpickling Python classes (e.g., BDHConfig)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)  # support direct state_dict or full checkpoint

    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format for {ckpt_path}")

    # Required params
    enc = state["encoder"]  # [nh, D, N]
    dec = state["decoder"]  # [nh*N, D]

    nh, D, N = enc.shape

    # Build per-head input inverted index based on top-k input dims per neuron
    # For each head h: map input dim d -> list of neuron ids j sharing that dim in their top-k
    input_inverted: List[Dict[int, List[int]]] = [{i: [] for i in range(D)} for _ in range(nh)]

    # Precompute top-k input dims per neuron per head
    top_in_dims: List[List[List[int]]] = [[[] for _ in range(N)] for _ in range(nh)]
    for h in range(nh):
        W = enc[h]  # [D, N]
        idx = topk_indices(W.transpose(0, 1), k_in)  # [N, k_in] over D
        for j in range(N):
            dims = idx[j].tolist()
            top_in_dims[h][j] = dims
            for d in dims:
                input_inverted[h][d].append(j)

    # Precompute top-k output dims per neuron from decoder rows
    # Global neuron index: g = h*N + j
    top_out_dims: List[List[List[int]]] = [[[] for _ in range(N)] for _ in range(nh)]
    for h in range(nh):
        start = h * N
        end = start + N
        rows = dec[start:end]  # [N, D]
        idx = topk_indices(rows, k_out)  # [N, k_out] over D
        for j in range(N):
            top_out_dims[h][j] = idx[j].tolist()

    # Compute per-neuron stats and neighbor relations via shared top-k input dims
    # Output CSV per checkpoint
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(ckpt_path)
    out_csv = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_neurons.csv")

    # Prepare writer
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "checkpoint",
            "head",
            "neuron",
            "encoder_l2",
            "encoder_max_abs",
            "decoder_l2",
            "top_in_dims",
            "top_out_dims",
            "neighbors_input",  # format: h:neuron:shared_count|...
        ])

        for h in range(nh):
            W = enc[h]  # [D, N]
            W_abs = W.abs()
            # l2 over D for each neuron (column)
            enc_l2 = torch.linalg.vector_norm(W, dim=0)  # [N]
            enc_max = torch.max(W_abs, dim=0).values  # [N]

            # decoder l2 per neuron row
            start = h * N
            rows = dec[start:start + N]  # [N, D]
            dec_l2 = torch.linalg.vector_norm(rows, dim=1)  # [N]

            # neighbors via input inverted index
            inverted = input_inverted[h]

            for j in range(N):
                dims = top_in_dims[h][j]
                # candidate neighbors: union of neurons sharing any of these dims
                candidates: Dict[int, int] = {}
                for d in dims:
                    for other in inverted[d]:
                        if other == j:
                            continue
                        candidates[other] = candidates.get(other, 0) + 1

                # select top neighbors by shared dim count
                if candidates:
                    sorted_neighbors = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
                    topN = sorted_neighbors[:top_neighbors]
                    neighbors_str = "|".join([f"{h}:{other}:{cnt}" for other, cnt in topN])
                else:
                    neighbors_str = ""

                w.writerow([
                    base,
                    h,
                    j,
                    float(enc_l2[j].item()),
                    float(enc_max[j].item()),
                    float(dec_l2[j].item()),
                    ",".join(map(str, dims)),
                    ",".join(map(str, top_out_dims[h][j])),
                    neighbors_str,
                ])


def main():
    parser = argparse.ArgumentParser(description="Analyze BDH neurons across checkpoints")
    parser.add_argument("--bdh_dir", default=os.path.join("Models", "BDH_model"), help="Path to BDH_model directory")
    parser.add_argument("--out_dir", default=os.path.join("Models", "BDH_model", "visuals", "neuron_analysis"), help="Output directory for CSVs")
    parser.add_argument("--k_in", type=int, default=5, help="Top-K input dims per neuron")
    parser.add_argument("--k_out", type=int, default=5, help="Top-K output dims per neuron")
    parser.add_argument("--neighbors", type=int, default=3, help="Neighbors per neuron from shared input dims")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of checkpoints to process (0 = all)")
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.bdh_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    checkpoints = list_checkpoints(ckpt_dir)
    if args.limit and args.limit > 0:
        checkpoints = checkpoints[:args.limit]

    for path in checkpoints:
        print(f"Analyzing {path}...")
        analyze_checkpoint(path, args.out_dir, k_in=args.k_in, k_out=args.k_out, top_neighbors=args.neighbors)
    print("Done.")


if __name__ == "__main__":
    main()
