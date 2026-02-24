import os
import csv
import argparse
from typing import Dict, List

import torch
from dataclasses import dataclass


# Minimal GPTConfig to satisfy safe unpickling when needed
@dataclass
class GPTConfig:
    pass

try:
    import torch.serialization as _ts
    if hasattr(_ts, "add_safe_globals"):
        _ts.add_safe_globals([GPTConfig])
except Exception:
    pass


def list_checkpoints(dir_path: str, include_final: bool = True) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.startswith("ckpt_") and name.endswith(".pt"):
            files.append(os.path.join(dir_path, name))
    files = sorted(files, key=lambda p: int(os.path.basename(p)[5:-3]))
    if include_final:
        final_path = os.path.join(os.path.dirname(dir_path), "transformer_final.pt")
        if os.path.exists(final_path):
            files.append(final_path)
    return files


def topk_indices_row(mat: torch.Tensor, k: int) -> torch.Tensor:
    # mat: [rows, cols]; returns indices of top-k by abs per row
    vals = mat.abs()
    _, idx = torch.topk(vals, k=min(k, vals.shape[-1]), dim=-1)
    return idx


def topk_indices_col(mat: torch.Tensor, k: int) -> torch.Tensor:
    # mat: [rows, cols]; returns indices of top-k by abs per column
    vals = mat.abs()
    _, idx = torch.topk(vals, k=min(k, vals.shape[0]), dim=0)
    return idx


def analyze_checkpoint(ckpt_path: str, out_dir: str, k_in: int = 5, k_out: int = 5, top_neighbors: int = 3) -> None:
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format for {ckpt_path}")

    # Collect MLP weights per block
    # c_fc.weight shape: [4*C, C]
    # c_proj.weight shape: [C, 4*C]
    # Keys are like: 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_proj.weight'
    mlp_fc: Dict[int, torch.Tensor] = {}
    mlp_proj: Dict[int, torch.Tensor] = {}
    for k, v in state.items():
        if not k.endswith(".weight"):
            continue
        if ".mlp.c_fc.weight" in k:
            # extract block index
            try:
                blk = int(k.split("transformer.h.")[1].split(".")[0])
                mlp_fc[blk] = v.cpu()
            except Exception:
                pass
        elif ".mlp.c_proj.weight" in k:
            try:
                blk = int(k.split("transformer.h.")[1].split(".")[0])
                mlp_proj[blk] = v.cpu()
            except Exception:
                pass

    if not mlp_fc or not mlp_proj:
        raise RuntimeError("No MLP weights found in checkpoint")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(ckpt_path)
    out_csv = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_mlp_neurons.csv")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "checkpoint",
            "block",
            "neuron",
            "fc_l2",
            "fc_max_abs",
            "proj_l2",
            "top_in_dims",
            "top_out_dims",
            "neighbors_input",
        ])

        for blk in sorted(mlp_fc.keys()):
            fc = mlp_fc[blk]  # [4C, C]
            proj = mlp_proj[blk]  # [C, 4C]
            rows = fc.shape[0]
            C = fc.shape[1]
            assert proj.shape == (C, rows), "MLP dims mismatch"

            # Precompute top-k input dims per neuron (row of fc)
            top_in = topk_indices_row(fc, k_in)  # [rows, k_in]
            # Precompute top-k output dims per neuron (column of proj)
            top_out = topk_indices_col(proj, k_out)  # [C, rows] indices per col; we need per neuron columns

            # l2 norms
            fc_l2 = torch.linalg.vector_norm(fc, dim=1)  # [rows]
            fc_max = torch.max(fc.abs(), dim=1).values   # [rows]
            proj_l2 = torch.linalg.vector_norm(proj, dim=0)  # [rows]

            # Build input inverted index for neighbor relations within block
            inverted: Dict[int, List[int]] = {i: [] for i in range(C)}
            for j in range(rows):
                for d in top_in[j].tolist():
                    inverted[d].append(j)

            for j in range(rows):
                dims_in = top_in[j].tolist()
                # top_out per neuron j: look at column j of proj -> use top_out[:, j]
                dims_out = top_out[:, j].tolist()

                # neighbors via shared input dims
                candidates: Dict[int, int] = {}
                for d in dims_in:
                    for other in inverted[d]:
                        if other == j:
                            continue
                        candidates[other] = candidates.get(other, 0) + 1
                if candidates:
                    sorted_neighbors = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
                    topN = sorted_neighbors[:top_neighbors]
                    neighbors_str = "|".join([f"{blk}:{other}:{cnt}" for other, cnt in topN])
                else:
                    neighbors_str = ""

                w.writerow([
                    base,
                    blk,
                    j,
                    float(fc_l2[j].item()),
                    float(fc_max[j].item()),
                    float(proj_l2[j].item()),
                    ",".join(map(str, dims_in)),
                    ",".join(map(str, dims_out)),
                    neighbors_str,
                ])


def main():
    parser = argparse.ArgumentParser(description="Analyze Transformer MLP neurons across checkpoints")
    parser.add_argument("--tx_dir", default=os.path.join("Models", "Transformer_model"), help="Path to Transformer_model directory")
    parser.add_argument("--out_dir", default=os.path.join("Models", "Transformer_model", "visuals", "neuron_analysis"), help="Output directory for CSVs")
    parser.add_argument("--k_in", type=int, default=5, help="Top-K input dims per neuron (c_fc)")
    parser.add_argument("--k_out", type=int, default=5, help="Top-K output dims per neuron (c_proj)")
    parser.add_argument("--neighbors", type=int, default=3, help="Neighbors per neuron from shared input dims")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of checkpoints to process (0 = all)")
    parser.add_argument("--file", default="", help="Analyze a single checkpoint file path (overrides tx_dir)")
    args = parser.parse_args()

    if args.file:
        checkpoints = [args.file]
    else:
        ckpt_dir = os.path.join(args.tx_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        checkpoints = list_checkpoints(ckpt_dir, include_final=True)
        if args.limit and args.limit > 0:
            checkpoints = checkpoints[:args.limit]

    for path in checkpoints:
        print(f"Analyzing {path}...")
        analyze_checkpoint(path, args.out_dir, k_in=args.k_in, k_out=args.k_out, top_neighbors=args.neighbors)
    print("Done.")


if __name__ == "__main__":
    main()
