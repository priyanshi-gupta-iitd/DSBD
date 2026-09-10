#!/usr/bin/env python3
"""Aggregate constrained DSBD metrics CSV into a 2x2 summary (+ optional plots)."""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import numpy as np
import pandas as pd


def load_metrics_csv(path: str) -> pd.DataFrame:
    """Load metrics even if older rows lack exec_error (30 vs 31 columns)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"empty metrics csv: {path}")
        raw_rows: List[List[str]] = [row for row in reader if row]

    max_len = max((len(r) for r in raw_rows), default=len(header))
    if "exec_error" not in header and max_len > len(header):
        if "executable" in header:
            i = header.index("executable") + 1
        else:
            i = len(header)
        header = header[:i] + ["exec_error"] + header[i:]

    aligned = []
    for row in raw_rows:
        if len(row) < len(header):
            # old row: insert blank exec_error after executable
            if "exec_error" in header and len(row) == len(header) - 1:
                pos = header.index("exec_error")
                row = row[:pos] + [""] + row[pos:]
            else:
                row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        aligned.append(row)

    df = pd.DataFrame(aligned, columns=header)
    numeric = [
        "example_idx", "committed_tokens", "tokens_proposed", "tokens_accepted",
        "tokens_constraint_rejected", "xgrammar_rejects", "z3_rejects",
        "wall_time_s", "xgrammar_time_s", "z3_time_s", "throughput", "main_throughput",
        "goodput", "goodput_correct", "useful_frac", "n_useful", "n_useful_correct",
        "executable", "exec_correct", "exec_acc", "acc_len_mean", "acc_rate",
        "width", "gamma", "w_thres", "min_w", "extra_sample_cnt",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Backfill main_throughput for older CSVs: committed tokens / wall (target output only)
    if "main_throughput" not in df.columns or df["main_throughput"].isna().all():
        wall = df["wall_time_s"].replace(0, np.nan)
        df["main_throughput"] = df["committed_tokens"] / wall
        df["main_throughput"] = df["main_throughput"].fillna(0.0)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "constraints"]
    if "width" in df.columns and df["width"].nunique(dropna=True) > 1:
        group_cols = ["method", "constraints", "width"]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        wall = g["wall_time_s"].sum()
        proposed = g["tokens_proposed"].sum()
        committed = g["committed_tokens"].sum() if "committed_tokens" in g else 0.0
        useful = g["n_useful"].sum()
        useful_c = g["n_useful_correct"].sum()
        rec.update({
            "n_examples": len(g),
            "exec_acc_mean": g["exec_acc"].mean() if "exec_acc" in g else np.nan,
            "executable_rate": g["executable"].mean() if "executable" in g else np.nan,
            "throughput": (proposed / wall) if wall > 0 else 0.0,
            "main_throughput": (committed / wall) if wall > 0 else 0.0,
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
        if "width" not in rec and "width" in g.columns:
            rec["width"] = g["width"].iloc[0]
        rows.append(rec)
    sort_cols = [c for c in ["method", "constraints", "width"] if c in rows[0]] if rows else ["method"]
    return pd.DataFrame(rows).sort_values(sort_cols)


def _fmt(v, kind="f"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "nan"
    if kind == "int":
        return f"{int(round(float(v)))}"
    if kind == "acc":
        return f"{float(v):.3f}"
    if abs(float(v)) >= 1000:
        return f"{float(v):.0f}"
    if abs(float(v)) >= 10:
        return f"{float(v):.2f}"
    return f"{float(v):.3f}"


def _pad_ylim(ax, values, frac=0.18):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return
    ymax = max(vals)
    ymin = min(0.0, min(vals))
    pad = max(abs(ymax) * frac, 0.08 if abs(ymax) < 2 else 1.0)
    ax.set_ylim(ymin, ymax + pad)


def _label_bars(ax, bars, kind="f"):
    heights = []
    for bar in bars:
        h = bar.get_height()
        heights.append(h)
        if not np.isfinite(h):
            continue
        ax.annotate(
            _fmt(h, kind),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return heights


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

    multi_width = "width" in dsbd.columns and dsbd["width"].nunique(dropna=True) > 1
    if multi_width:
        plot_2x2 = summary.copy()
        med = dsbd["width"].median()
        keep = (plot_2x2["method"] != "dsbd") | (plot_2x2["width"] == med)
        plot_2x2 = plot_2x2[keep]
    else:
        plot_2x2 = summary.copy()
    _mode_order = {"none": 0, "xgrammar": 1, "z3": 2, "both": 3}
    _method_order = {"ar_target": 0, "sd": 1, "dsbd": 2}

    def _series_label(r):
        m = {"ar_target": "AR", "dsbd": "DSBD", "sd": "SD"}.get(str(r["method"]), str(r["method"]))
        return f"{m}/{r['constraints']}"

    plot_2x2 = plot_2x2.assign(
        _method_rank=plot_2x2["method"].map(lambda m: _method_order.get(str(m), 99)),
        _mode_rank=plot_2x2["constraints"].map(lambda c: _mode_order.get(str(c), 99)),
    ).sort_values(["_method_rank", "_mode_rank"]).drop(columns=["_method_rank", "_mode_rank"]).reset_index(drop=True)
    labels = [_series_label(r) for _, r in plot_2x2.iterrows()]
    mixed = plot_2x2["method"].nunique() > 1

    # goodput vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for (_, r), lab in zip(plot_2x2.iterrows(), labels):
        ax.scatter(r["exec_acc_mean"], r["goodput"], s=90, label=lab)
        ax.annotate(
            f"{lab}\nacc={_fmt(r['exec_acc_mean'], 'acc')}\ngp={_fmt(r['goodput'])}",
            (r["exec_acc_mean"], r["goodput"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel("Exec accuracy (mean fraction of gold-matching SQL)")
    ax.set_ylabel("Goodput (useful committed tokens / s)")
    title = "Accuracy vs goodput (AR vs DSBD)" if mixed else "Constrained DSBD: goodput vs accuracy"
    if multi_width and len(dsbd):
        title += f" (DSBD width={int(dsbd['width'].median())})"
    ax.set_title(title)
    ax.legend(title="method/constraints")
    ax.grid(True, alpha=0.3)
    xs = plot_2x2["exec_acc_mean"].astype(float)
    ys = plot_2x2["goodput"].astype(float)
    if len(xs):
        ax.set_xlim(min(0.0, float(xs.min()) - 0.05), float(xs.max()) + 0.14)
        ax.set_ylim(min(0.0, float(ys.min()) - 0.2), float(ys.max()) * 1.18 + 0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "goodput_vs_accuracy.png"), dpi=150)
    plt.close(fig)

    if "goodput_correct" in plot_2x2.columns:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for (_, r), lab in zip(plot_2x2.iterrows(), labels):
            ax.scatter(r["exec_acc_mean"], r["goodput_correct"], s=90, label=lab)
            ax.annotate(
                f"{lab}\nacc={_fmt(r['exec_acc_mean'], 'acc')}\ngp*={_fmt(r['goodput_correct'])}",
                (r["exec_acc_mean"], r["goodput_correct"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
        ax.set_xlabel("Exec accuracy (mean fraction of gold-matching SQL)")
        ax.set_ylabel("Goodput-correct (gold-matching tokens / s)")
        ax.set_title("Accuracy vs goodput-correct")
        ax.legend(title="method/constraints")
        ax.grid(True, alpha=0.3)
        ys = plot_2x2["goodput_correct"].astype(float)
        if len(xs):
            ax.set_xlim(min(0.0, float(xs.min()) - 0.05), float(xs.max()) + 0.14)
            ax.set_ylim(min(0.0, float(ys.min()) - 0.2), float(ys.max()) * 1.18 + 0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "goodput_correct_vs_accuracy.png"), dpi=150)
        plt.close(fig)

    # bar: all-proposed throughput vs main (committed) throughput
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(plot_2x2)), 5))
    x = np.arange(len(plot_2x2))
    w = 0.35
    b1 = ax.bar(x - w / 2, plot_2x2["throughput"], w, label="throughput (all proposed tok/s)")
    b2 = ax.bar(
        x + w / 2,
        plot_2x2["main_throughput"],
        w,
        label="main throughput (committed / wall)",
    )
    heights = _label_bars(ax, b1) + _label_bars(ax, b2)
    _pad_ylim(ax, heights)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Decoder / constraint mode")
    ax.set_ylabel("tokens / s")
    ax.set_title("All-proposed vs main-model throughput")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_main.png"), dpi=150)
    plt.close(fig)

    # bar: main throughput vs goodput
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(plot_2x2)), 5))
    b1 = ax.bar(x - w / 2, plot_2x2["main_throughput"], w, label="main throughput (committed tok/s)")
    b2 = ax.bar(x + w / 2, plot_2x2["goodput"], w, label="goodput (useful tok/s)")
    heights = _label_bars(ax, b1) + _label_bars(ax, b2)
    _pad_ylim(ax, heights)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Decoder / constraint mode")
    ax.set_ylabel("tokens / s")
    ax.set_title("Main throughput vs goodput")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_goodput.png"), dpi=150)
    plt.close(fig)

    # reject rates
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(plot_2x2)), 5))
    b1 = ax.bar(x - w / 2, plot_2x2["xgrammar_rejects"], w, label="xgrammar rejects")
    b2 = ax.bar(x + w / 2, plot_2x2["z3_rejects"], w, label="z3 rejects")
    heights = _label_bars(ax, b1, kind="int") + _label_bars(ax, b2, kind="int")
    _pad_ylim(ax, heights)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Decoder / constraint mode")
    ax.set_ylabel("Force-reject count (tokens)")
    ax.set_title("Constraint force-rejects by mode")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "constraint_rejects.png"), dpi=150)
    plt.close(fig)

    if multi_width:
        def _width_line(ycol, ylabel, title, fname, kind="f"):
            fig, ax = plt.subplots(figsize=(8, 5))
            modes = sorted(
                dsbd["constraints"].unique(),
                key=lambda c: _mode_order.get(str(c), 99),
            )
            for mode in modes:
                g = dsbd[dsbd["constraints"] == mode].sort_values("width")
                ax.plot(g["width"], g[ycol], marker="o", label=str(mode))
                for _, r in g.iterrows():
                    ax.annotate(
                        _fmt(r[ycol], kind),
                        (r["width"], r[ycol]),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=8,
                    )
            ax.set_xlabel("Beam width (num_beams)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(title="constraints")
            ax.grid(True, alpha=0.3)
            _pad_ylim(ax, dsbd[ycol].tolist())
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)

        _width_line(
            "goodput",
            "Goodput (useful committed tokens / s)",
            "Goodput vs beam width",
            "goodput_vs_width.png",
        )
        _width_line(
            "exec_acc_mean",
            "Exec accuracy (mean)",
            "Accuracy vs beam width",
            "accuracy_vs_width.png",
            kind="acc",
        )
        _width_line(
            "acc_len_mean",
            "Mean accepted draft length (tokens)",
            "Accept length vs beam width",
            "acclen_vs_width.png",
        )
        _width_line(
            "throughput",
            "Throughput (all proposed tokens / s)",
            "Throughput vs beam width",
            "throughput_vs_width.png",
        )

    print(f"Wrote plots to {out_dir}")


def _print_paired_flips(df: pd.DataFrame):
    need = {"method", "constraints", "example_idx", "exec_acc"}
    if not need.issubset(df.columns):
        return
    pairs = [
        ("dsbd", "none", "dsbd", "both"),
        ("ar_target", "none", "ar_target", "both"),
        ("ar_target", "none", "dsbd", "none"),
        ("ar_target", "both", "dsbd", "both"),
    ]
    print("\nPaired exec_acc flips (same example_idx):")
    any_pair = False
    for m1, c1, m2, c2 in pairs:
        a = df[(df["method"] == m1) & (df["constraints"] == c1)].drop_duplicates("example_idx")
        b = df[(df["method"] == m2) & (df["constraints"] == c2)].drop_duplicates("example_idx")
        a = a.set_index("example_idx")["exec_acc"]
        b = b.set_index("example_idx")["exec_acc"]
        common = a.index.intersection(b.index)
        if len(common) < 3:
            continue
        any_pair = True
        aa = a.loc[common] >= 1.0
        bb = b.loc[common] >= 1.0
        wr = int((~aa & bb).sum())
        rw = int((aa & ~bb).sum())
        print(
            f"  {m1}/{c1} -> {m2}/{c2}: n={len(common)} "
            f"wrong→right={wr} right→wrong={rw} "
            f"acc {aa.mean():.3f}→{bb.mean():.3f}"
        )
    if not any_pair:
        print("  (need AR + DSBD rows on overlapping examples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, default="logs/constraint_metrics.csv")
    ap.add_argument("--out_csv", type=str, default="logs/constraint_summary.csv")
    ap.add_argument("--plot_dir", type=str, default="logs/constraint_plots")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.metrics_csv):
        raise SystemExit(f"missing metrics csv: {args.metrics_csv}")

    df = load_metrics_csv(args.metrics_csv)
    summary = summarize(df)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    _print_paired_flips(df)
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
