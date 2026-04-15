from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["run_name", "seed", "test_error_pct"])
    out = out[out["run_name"].astype(str).str.len() > 0]
    return out


def _best_per_run_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_name, seed), group in df.groupby(["run_name", "seed"], dropna=False):
        idx = group["test_error_pct"].idxmin()
        row = group.loc[idx].copy()
        row["best_epoch"] = int(row.get("epoch", 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_convergence_vs_error(best_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for run_name, group in best_df.groupby("run_name", dropna=False):
        ax.scatter(
            group["convergence_epoch"].fillna(0),
            group["test_error_pct"],
            label=str(run_name),
            alpha=0.85,
            s=45,
        )

    ax.set_xlabel("Convergence epoch (0 means not reached threshold)")
    ax.set_ylabel("Best test error (%)")
    ax.set_title("Convergence speed vs final error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)


def _write_markdown(summary_df: pd.DataFrame, output_path: Path, figure_path: str) -> None:
    lines = []
    lines.append("# Final Ablation Analysis")
    lines.append("")
    lines.append("## Convergence Vs Error")
    lines.append(f"![Convergence vs Error]({figure_path})")
    lines.append("")
    lines.append("## Aggregated Results")
    lines.append("")
    lines.append("| run_name | seeds | mean_test_error | std_test_error | mean_convergence_epoch | mean_epoch_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        std_value = float(row["std_test_error"]) if pd.notna(row["std_test_error"]) else 0.0
        lines.append(
            "| {run_name} | {seeds} | {mean_test_error:.2f} | {std_test_error:.2f} | {mean_convergence_epoch:.2f} | {mean_epoch_time_sec:.2f} |".format(
                run_name=row["run_name"],
                seeds=int(row["seeds"]),
                mean_test_error=float(row["mean_test_error"]),
                std_test_error=std_value,
                mean_convergence_epoch=float(row["mean_convergence_epoch"]),
                mean_epoch_time_sec=float(row["mean_epoch_time_sec"]),
            )
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    if len(summary_df) > 0:
        best = summary_df.iloc[0]
        converged = summary_df[summary_df["mean_convergence_epoch"] > 0]
        fastest = converged.sort_values("mean_convergence_epoch").iloc[0] if len(converged) > 0 else None
        lines.append(
            "- Best mean test error run: **{name}** with {err:.2f}%".format(
                name=best["run_name"], err=float(best["mean_test_error"])
            )
        )
        if fastest is not None:
            lines.append(
                "- Fastest converged run: **{name}** at mean epoch {ep:.2f}".format(
                    name=fastest["run_name"], ep=float(fastest["mean_convergence_epoch"])
                )
            )
        else:
            lines.append("- No run crossed the convergence threshold in the evaluated epoch budget.")
        lines.append(
            "- This file is auto-generated from ablation logs and can be reproduced with one command."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/ablations.csv")
    parser.add_argument("--summary-output", type=str, default="results/summary_stats.csv")
    parser.add_argument("--figure-output", type=str, default="results/figures/convergence_vs_error.png")
    parser.add_argument("--markdown-output", type=str, default="results/final_analysis.md")
    parser.add_argument("--convergence-threshold", type=float, default=25.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    req = {"run_name", "seed", "epoch", "test_error_pct", "epoch_time_sec"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {sorted(missing)}")

    df = _clean_rows(df)
    if len(df) == 0:
        raise ValueError("No valid rows found after filtering for run_name/seed/test_error_pct")
    threshold = float(args.convergence_threshold)
    conv = []
    for (run_name, seed), group in df.groupby(["run_name", "seed"], dropna=False):
        reached = group[group["test_error_pct"] <= threshold]
        conv.append(((run_name, seed), int(reached["epoch"].min()) if len(reached) > 0 else 0))
    conv_map = {k: v for k, v in conv}
    df["convergence_epoch"] = [conv_map[(r, s)] for r, s in zip(df["run_name"], df["seed"])]

    best_df = _best_per_run_seed(df)
    summary = (
        best_df.groupby("run_name", dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_error=("test_error_pct", "mean"),
            std_test_error=("test_error_pct", "std"),
            mean_epoch_time_sec=("epoch_time_sec", "mean"),
            mean_convergence_epoch=("convergence_epoch", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_error")
    )

    summary_out = Path(args.summary_output)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    fig_out = Path(args.figure_output)
    _plot_convergence_vs_error(best_df, fig_out)

    md_out = Path(args.markdown_output)
    figure_rel = str(Path(args.figure_output))
    _write_markdown(summary, md_out, figure_rel)

    print(f"wrote summary to {summary_out}")
    print(f"wrote figure to {fig_out}")
    print(f"wrote markdown analysis to {md_out}")


if __name__ == "__main__":
    main()
