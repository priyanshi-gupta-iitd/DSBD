#!/usr/bin/env python3
"""Aggregate constrained DSBD metrics CSV into a 2x2 summary (+ optional plots)."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "constraints"]
    for keys, g in df.groupby(group_cols):
        method, constraints = keys
        wall = g["wall_time_s"].sum()
        proposed = g["tokens_proposed"].sum()
        useful = g["n_useful"].sum()
        useful_c = g["n_useful_correct"].sum()
        rows.append({
            "method": method,
            "constraints": constraints,
            "n_examples": len(g),
            "exec_acc_mean": g["exec_acc"].mean() if "exec_acc" in g else np.nan,
            "executable_rate": g["executable"].mean() if "executable" in g else np.nan,
            "throughput": (proposed / wall) if wall > 0 else 0.0,
            "goodput": (useful / wall) if wall > 0 else 0.0,
            "goodput_correct": (useful_c / wall) if wall > 0 else 0.0,
            "useful_frac": (useful / proposed) if proposed > 0 else 0.0,
            "acc_len_mean": g["acc_len_mean"].mean() if "acc_len_mean" in g else np.nan,
            "acc_rate_mean": g["acc_rate"].mean() if "acc_rate" in g else np.nan,
            "xgrammar_rejects": g["xgrammar_rejects"].sum() if "xgrammar_rejects" in g else 0,
            "z3_rejects": g["z3_rejects"].sum() if "z3_rejects" in g else 0,
            "constraint_rejects": g["tokens_constraint_rejected"].sum() if "tokens_constraint_rejected" in g else 0,
            "xgrammar_time_s": g["xgrammar_time_s"].sum() if "xgrammar_time_s" in g else 0.0,
            "z3_time_s": g["z3_time_s"].sum() if "z3_time_s" in g else 0.0,
            "wall_time_s": wall,
        })
    return pd.DataFrame(rows).sort_values(["method", "constraints"])


def maybe_plot(summary: pd.DataFrame, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    dsbd = summary[summary["method"] == "dsbd"].copy()
    if dsbd.empty:
        dsbd = summary

    # goodput vs accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in dsbd.iterrows():
        ax.scatter(r["exec_acc_mean"], r["goodput"], s=80, label=r["constraints"])
        ax.annotate(r["constraints"], (r["exec_acc_mean"], r["goodput"]))
    ax.set_xlabel("Exec accuracy (mean)")
    ax.set_ylabel("Goodput (useful tok/s)")
    ax.set_title("Constrained DSBD: goodput vs accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "goodput_vs_accuracy.png"), dpi=150)
    plt.close(fig)

    # bar: throughput vs goodput
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(dsbd))
    w = 0.35
    ax.bar(x - w / 2, dsbd["throughput"], w, label="throughput")
    ax.bar(x + w / 2, dsbd["goodput"], w, label="goodput")
    ax.set_xticks(x)
    ax.set_xticklabels(dsbd["constraints"])
    ax.set_ylabel("tokens / s")
    ax.set_title("Throughput vs goodput")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_goodput.png"), dpi=150)
    plt.close(fig)

    # reject rates
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, dsbd["xgrammar_rejects"], w, label="xgrammar rejects")
    ax.bar(x + w / 2, dsbd["z3_rejects"], w, label="z3 rejects")
    ax.set_xticks(x)
    ax.set_xticklabels(dsbd["constraints"])
    ax.set_ylabel("reject count")
    ax.set_title("Constraint force-rejects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "constraint_rejects.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote plots to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, default="logs/constraint_metrics.csv")
    ap.add_argument("--out_csv", type=str, default="logs/constraint_summary.csv")
    ap.add_argument("--plot_dir", type=str, default="logs/constraint_plots")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.metrics_csv):
        raise SystemExit(f"missing metrics csv: {args.metrics_csv}")

    df = pd.read_csv(args.metrics_csv)
    summary = summarize(df)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    if "executable" in df.columns:
        print(f"\nexecutable_rate overall: {df['executable'].mean():.3f}")
        print("goodput is 0 on rows where executable=0 (SQL did not run).")
    if "exec_error" in df.columns:
        print("\nTop exec_error reasons (non-executable rows):")
        bad = df[df.get("executable", 1) == 0] if "executable" in df.columns else df
        print(bad["exec_error"].fillna("").value_counts().head(15).to_string())
    if not args.no_plot:
        maybe_plot(summary, args.plot_dir)


if __name__ == "__main__":
    main()
