from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/ablations_best.csv")
    parser.add_argument("--group-by", type=str, default="run_name")
    parser.add_argument("--output", type=str, default="results/summary_stats.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.group_by not in df.columns:
        raise ValueError(f"group-by column '{args.group_by}' not present in input")

    agg = (
        df.groupby(args.group_by, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_error=("test_error_pct", "mean"),
            std_test_error=("test_error_pct", "std"),
            mean_epoch_time_sec=("epoch_time_sec", "mean"),
            mean_convergence_epoch=("convergence_epoch", "mean"),
        )
        .reset_index()
        .sort_values(by="mean_test_error", ascending=True)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(f"wrote summary stats to {out}")


if __name__ == "__main__":
    main()
