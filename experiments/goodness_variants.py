from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--jitter-pixels", type=int, default=0)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for seed in seeds:
        variants = [
            ("squared", "auto", "results/logs/ablation_goodness_squared.csv"),
            ("unsquared", "auto", "results/logs/ablation_goodness_unsquared.csv"),
        ]
        for goodness, norm_mode, out in variants:
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
                str(seed),
                "--goodness",
                goodness,
                "--norm-mode",
                norm_mode,
                "--activation",
                "relu",
                "--jitter-pixels",
                str(args.jitter_pixels),
                "--train-subset",
                str(args.train_subset),
                "--run-name",
                f"goodness_{goodness}",
                "--output",
                out,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
