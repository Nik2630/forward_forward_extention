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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--goodness", type=str, default="squared", choices=["squared", "unsquared"])
    parser.add_argument("--norm-mode", type=str, default="auto", choices=["auto", "l2", "l1"])
    parser.add_argument("--student-df", type=float, default=2.0)
    parser.add_argument("--student-scale", type=float, default=1.0)
    parser.add_argument("--jitter-pixels", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="activation")
    parser.add_argument("--output", type=str, default="results/logs/activation_experiment.csv")
    args = parser.parse_args()

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
        "--activation",
        args.activation,
        "--goodness",
        args.goodness,
        "--norm-mode",
        args.norm_mode,
        "--student-df",
        str(args.student_df),
        "--student-scale",
        str(args.student_scale),
        "--jitter-pixels",
        str(args.jitter_pixels),
        "--train-subset",
        str(args.train_subset),
        "--run-name",
        args.run_name,
        "--output",
        args.output,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
