import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_metrics_csv(path: str) -> Dict[str, List[float]]:
    """Read a metrics.csv with headers like: step, loss, param_l2."""
    metrics: Dict[str, List[float]] = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for h in headers:
            metrics[h] = []
        for row in reader:
            for h, v in zip(headers, row):
                try:
                    metrics[h].append(float(v))
                except ValueError:
                    metrics[h].append(float("nan"))
    return metrics


def _clean_xy(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    cx, cy = [], []
    for x, y in zip(xs, ys):
        if isinstance(y, float) and not math.isnan(y):
            cx.append(x)
            cy.append(y)
    return cx, cy


def plot_compare(
    x1: List[float], y1: List[float], label1: str,
    x2: List[float], y2: List[float], label2: str,
    title: str, xlabel: str, ylabel: str, out_path: str,
):
    plt.figure(figsize=(9, 5.5))
    plt.plot(x1, y1, marker="o", linewidth=1.6, label=label1)
    plt.plot(x2, y2, marker="s", linewidth=1.6, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare BDH and Transformer training metrics and plot overlays.")
    parser.add_argument(
        "--bdh-metrics",
        type=str,
        default=os.path.join("Models", "BDH_model", "visuals", "metrics.csv"),
        help="Path to BDH metrics.csv",
    )
    parser.add_argument(
        "--transformer-metrics",
        type=str,
        default=os.path.join("Models", "Transformer_model", "visuals", "metrics.csv"),
        help="Path to Transformer metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("Models", "visuals", "comparison"),
        help="Directory to save comparison plots",
    )
    args = parser.parse_args()

    bdh = read_metrics_csv(args.bdh_metrics)
    tfm = read_metrics_csv(args.transformer_metrics)

    bdh_steps = bdh.get("step", [])
    tfm_steps = tfm.get("step", [])

    # Loss comparison (if present)
    if "loss" in bdh and "loss" in tfm:
        bx, by = _clean_xy(bdh_steps, bdh["loss"]) 
        tx, ty = _clean_xy(tfm_steps, tfm["loss"]) 
        if by and ty:
            plot_compare(
                bx, by, "BDH",
                tx, ty, "Transformer",
                title="Training Loss vs Iteration (BDH vs Transformer)",
                xlabel="Iteration",
                ylabel="Loss",
                out_path=os.path.join(args.out_dir, "loss_vs_iter_bdh_vs_transformer.png"),
            )
            print("Saved loss comparison plot.")
        else:
            print("Loss values missing or all NaN; skipping loss comparison plot.")
    else:
        print("Loss column not found in one or both metrics; skipping loss plot.")

    # Parameter L2 norm comparison
    if "param_l2" in bdh and "param_l2" in tfm:
        bx, by = _clean_xy(bdh_steps, bdh["param_l2"]) 
        tx, ty = _clean_xy(tfm_steps, tfm["param_l2"]) 
        if by and ty:
            plot_compare(
                bx, by, "BDH",
                tx, ty, "Transformer",
                title="Parameter L2 Norm vs Iteration (BDH vs Transformer)",
                xlabel="Iteration",
                ylabel="Param L2 Norm",
                out_path=os.path.join(args.out_dir, "param_norm_vs_iter_bdh_vs_transformer.png"),
            )
            print("Saved parameter norm comparison plot.")
        else:
            print("Param L2 values missing or all NaN; skipping param norm comparison plot.")
    else:
        print("param_l2 column not found in one or both metrics; skipping param norm plot.")

    # Optionally, save the individual metrics CSVs into the out dir for convenience
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        bdh_copy = os.path.join(args.out_dir, "bdh_metrics.csv")
        tfm_copy = os.path.join(args.out_dir, "transformer_metrics.csv")
        if os.path.abspath(args.bdh_metrics) != os.path.abspath(bdh_copy):
            with open(args.bdh_metrics, "r") as src, open(bdh_copy, "w") as dst:
                dst.write(src.read())
        if os.path.abspath(args.transformer_metrics) != os.path.abspath(tfm_copy):
            with open(args.transformer_metrics, "r") as src, open(tfm_copy, "w") as dst:
                dst.write(src.read())
        print(f"Copied metrics to {args.out_dir} for reference.")
    except Exception as e:
        print(f"Failed to copy metrics: {e}")


if __name__ == "__main__":
    main()
