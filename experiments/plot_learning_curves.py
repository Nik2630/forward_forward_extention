from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/ablations.csv")
    parser.add_argument("--metric", type=str, default="test_error_pct")
    parser.add_argument("--hue", type=str, default="run_name")
    parser.add_argument("--output", type=str, default="results/figures/learning_curves.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    needed = {"epoch", args.metric, args.hue}
    missing = needed.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for plotting: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in df.groupby(args.hue):
        grouped = group.groupby("epoch", as_index=False)[args.metric].mean()
        ax.plot(grouped["epoch"], grouped[args.metric], marker="o", label=str(name))

    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.metric)
    ax.set_title(f"Learning curves grouped by {args.hue}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote figure to {out}")


if __name__ == "__main__":
    main()
