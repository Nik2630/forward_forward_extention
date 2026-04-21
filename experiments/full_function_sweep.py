from __future__ import annotations

import argparse
import subprocess
import sys

from src.goodness.factory import available_goodness_names
from src.models.activations import available_activation_names


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset", type=int, default=4000)
    parser.add_argument("--jitter-pixels", type=int, default=0)
    parser.add_argument("--goodnesses", type=str, default="all")
    parser.add_argument("--activations", type=str, default="all")
    parser.add_argument("--output", type=str, default="results/logs/ablation_full_functions.csv")
    args = parser.parse_args()

    goodnesses = available_goodness_names() if args.goodnesses == "all" else _parse_csv_list(args.goodnesses)
    activations = available_activation_names() if args.activations == "all" else _parse_csv_list(args.activations)

    print(f"Running sweep with {len(goodnesses)} goodness functions and {len(activations)} activations")
    total = len(goodnesses) * len(activations)
    done = 0

    for goodness_name in goodnesses:
        goodness_kwargs = ""
        if goodness_name == "topk_abs":
            goodness_kwargs = "k=10"
        elif goodness_name == "huber_sum":
            goodness_kwargs = "delta=1.0"
        elif goodness_name == "l2_norm":
            goodness_kwargs = "eps=1e-8"
        elif goodness_name == "softplus_sum":
            goodness_kwargs = "beta=1.0,threshold=20.0"
        elif goodness_name == "exp_mean":
            goodness_kwargs = "clamp=30.0"

        for activation_name in activations:
            done += 1
            run_name = f"full_{goodness_name}__{activation_name}"
            cmd = [
                sys.executable,
                "experiments/baseline_mnist.py",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--hidden-dim",
                str(args.hidden_dim),
                "--layers",
                str(args.layers),
                "--lr",
                str(args.lr),
                "--threshold",
                str(args.threshold),
                "--seed",
                str(args.seed),
                "--goodness-name",
                goodness_name,
                "--activation",
                activation_name,
                "--norm-mode",
                "auto",
                "--jitter-pixels",
                str(args.jitter_pixels),
                "--train-subset",
                str(args.train_subset),
                "--run-name",
                run_name,
                "--output",
                args.output,
            ]
            if goodness_kwargs:
                cmd.extend(["--goodness-kwargs", goodness_kwargs])

            print(f"[{done}/{total}] Running {run_name}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
