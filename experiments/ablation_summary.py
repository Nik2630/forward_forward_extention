from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=str, default="results/logs")
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument("--output", type=str, default="results/ablations.csv")
    parser.add_argument("--best-output", type=str, default="results/ablations_best.csv")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    frames = []
    for csv_path in sorted(logs_dir.glob(args.pattern)):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        df["source"] = csv_path.name
        frames.append(df)

    if not frames:
        print("No log CSV files found.")
        return

    merged = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}")

    required = {"run_name", "seed", "test_error_pct", "epoch", "elapsed_sec"}
    if required.issubset(set(merged.columns)):
        grouped = []
        for (run_name, seed), group in merged.groupby(["run_name", "seed"], dropna=False):
            idx = group["test_error_pct"].idxmin()
            best = group.loc[idx].copy()
            best["best_epoch"] = int(best["epoch"])
            grouped.append(best)
        best_df = pd.DataFrame(grouped)
        best_df.to_csv(args.best_output, index=False)
        print(f"Wrote {len(best_df)} best rows to {args.best_output}")


if __name__ == "__main__":
    main()
