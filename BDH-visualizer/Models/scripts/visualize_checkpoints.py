import argparse
import os
import re
from typing import List, Tuple, Dict, Any

import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Minimal config stubs to satisfy checkpoint unpickling
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


def extract_step_from_filename(filename: str) -> int:
    m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(filename))
    return int(m.group(1)) if m else -1


def load_checkpoint(path: str) -> Dict[str, Any]:
    try:
        # We explicitly set weights_only=False to allow loading checkpoints
        # that may include non-tensor objects (e.g., config dataclasses) saved
        # alongside model/optimizer state. These files are local/trusted.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt if isinstance(ckpt, dict) else {"model": ckpt}
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return {}


def model_param_l2_norm(ckpt: Dict[str, Any]) -> float:
    state = None
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None:
        # Might be a raw state_dict already
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            return float("nan")

    total = 0.0
    for tensor in state.values():
        if isinstance(tensor, torch.Tensor):
            total += float(torch.norm(tensor.float(), p=2).item())
    return total


def collect_metrics(checkpoint_paths: List[str]) -> Dict[str, List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    norms: List[float] = []

    for path in checkpoint_paths:
        step = extract_step_from_filename(path)
        ckpt = load_checkpoint(path)
        loss = ckpt.get("loss", float("nan"))
        norm = model_param_l2_norm(ckpt)

        steps.append(step)
        losses.append(loss)
        norms.append(norm)

    return {"step": steps, "loss": losses, "param_l2": norms}


def plot_series(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_csv(metrics: Dict[str, List[float]], out_path: str):
    import csv
    headers = list(metrics.keys())
    rows = zip(*(metrics[k] for k in headers))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def visualize_dir(checkpoint_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and f.startswith("ckpt_")
    ]
    pts.sort(key=lambda p: extract_step_from_filename(p))

    if not pts:
        print(f"No .pt checkpoints found in {checkpoint_dir}")
        return

    metrics = collect_metrics(pts)

    # Save CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    write_csv(metrics, csv_path)
    print(f"Wrote metrics CSV: {csv_path}")

    # Loss vs step if available
    if any(not (isinstance(v, float) and (v != v)) for v in metrics["loss"]):  # check for any non-NaN
        plot_series(
            metrics["step"], metrics["loss"],
            title="Training Loss vs. Iteration",
            xlabel="Iteration", ylabel="Loss",
            out_path=os.path.join(out_dir, "loss_vs_iter.png"),
        )
        print(f"Saved plot: {os.path.join(out_dir, 'loss_vs_iter.png')}")
    else:
        print("Loss not present in checkpoints; skipping loss plot.")

    # Param L2 norm vs step
    plot_series(
        metrics["step"], metrics["param_l2"],
        title="Parameter L2 Norm vs. Iteration",
        xlabel="Iteration", ylabel="Param L2 Norm",
        out_path=os.path.join(out_dir, "param_norm_vs_iter.png"),
    )
    print(f"Saved plot: {os.path.join(out_dir, 'param_norm_vs_iter.png')}")


def main():
    parser = argparse.ArgumentParser(description="Visualize .pt checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing .pt checkpoints")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save visuals")
    args = parser.parse_args()

    visualize_dir(args.checkpoint_dir, args.out_dir)


if __name__ == "__main__":
    main()
