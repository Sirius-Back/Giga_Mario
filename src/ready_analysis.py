#!/usr/bin/env python3
"""Exploratory plots for ready/ (data_ready) window groups.

Groups
------
- non-coding: kind == non_coding in non_coding.csv
- large coding: gene windows matching large_genes.csv (CDS > 130 kb crops)
- normal coding: remaining gene windows

Outputs (default: output/ready_analysis/)
-----------------------------------------
- group_counts.csv / group_counts_by_genome.csv
- barplot_group_counts.pdf|.png
- barplot_group_counts_by_genome.pdf|.png
- density_gc_by_group.pdf|.png
- density_length_by_group.pdf|.png
- summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Okabe–Ito (colorblind-safe); distinct linestyles for grayscale
GROUP_ORDER = ["non-coding", "normal coding", "large coding"]
GROUP_COLORS = {
    "non-coding": "#0072B2",
    "normal coding": "#009E73",
    "large coding": "#E69F00",
}
GROUP_LINESTYLES = {
    "non-coding": "-",
    "normal coding": "--",
    "large coding": "-.",
}

SEED = 42
DPI = 300


def apply_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 10,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_grouped(ready_dir: Path) -> pd.DataFrame:
    nc_path = ready_dir / "non_coding.csv"
    lg_path = ready_dir / "large_genes.csv"
    if not nc_path.is_file():
        raise FileNotFoundError(f"Missing input: {nc_path}")
    if not lg_path.is_file():
        raise FileNotFoundError(f"Missing input: {lg_path}")

    nc = pd.read_csv(nc_path, sep="|")
    required = {
        "GeneOrID",
        "Chr",
        "Position_start",
        "Position_end",
        "Length",
        "GC",
        "kind",
        "Genome",
    }
    missing = required - set(nc.columns)
    if missing:
        raise ValueError(f"{nc_path} missing columns: {sorted(missing)}")
    if nc.empty:
        raise ValueError(f"{nc_path} is empty")

    kinds = set(nc["kind"].unique())
    if not {"gene", "non_coding"} <= kinds:
        raise ValueError(f"Unexpected kind values in {nc_path}: {sorted(kinds)}")

    lg = pd.read_csv(lg_path, sep="|")
    lg_req = {"Genome", "Gene", "Chr", "Window_start", "Window_end"}
    missing_lg = lg_req - set(lg.columns)
    if missing_lg:
        raise ValueError(f"{lg_path} missing columns: {sorted(missing_lg)}")

    large_keys = set(
        zip(
            lg["Genome"],
            lg["Gene"],
            lg["Chr"],
            lg["Window_start"],
            lg["Window_end"],
        )
    )
    gene_keys = list(
        zip(
            nc["Genome"],
            nc["GeneOrID"],
            nc["Chr"],
            nc["Position_start"],
            nc["Position_end"],
        )
    )
    is_large = [
        k in large_keys if kind == "gene" else False
        for k, kind in zip(gene_keys, nc["kind"])
    ]
    n_matched = int(sum(is_large))
    if n_matched != len(lg):
        raise ValueError(
            f"Large-gene join mismatch: matched {n_matched} windows, "
            f"expected {len(lg)} rows in {lg_path.name}"
        )

    out = nc.copy()
    out["group"] = np.where(
        out["kind"] == "non_coding",
        "non-coding",
        np.where(is_large, "large coding", "normal coding"),
    )
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)

    # Sanity: Length must be positive; GC in [0, 1]
    if (out["Length"] <= 0).any():
        raise ValueError("Non-positive Length values found")
    if ((out["GC"] < 0) | (out["GC"] > 1)).any():
        raise ValueError("GC outside [0, 1]")
    return out


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png", "svg"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    return paths


def plot_bar_counts(counts: pd.Series, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = np.arange(len(GROUP_ORDER))
    y = [int(counts.get(g, 0)) for g in GROUP_ORDER]
    bars = ax.bar(
        x,
        y,
        color=[GROUP_COLORS[g] for g in GROUP_ORDER],
        edgecolor="black",
        linewidth=0.6,
        width=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Number of windows")
    ax.set_xlabel("Group")
    ax.set_title("ready/ window counts by group")
    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, max(y) * 1.12 if y else 1)
    fig.tight_layout()
    paths = save_fig(fig, out_dir, "barplot_group_counts")
    plt.close(fig)
    return paths


def plot_bar_by_genome(
    df: pd.DataFrame, out_dir: Path
) -> tuple[list[Path], pd.DataFrame]:
    ct = (
        df.groupby(["Genome", "group"], observed=True)
        .size()
        .unstack("group")
        .reindex(columns=GROUP_ORDER)
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ct.plot(
        kind="bar",
        ax=ax,
        color=[GROUP_COLORS[g] for g in GROUP_ORDER],
        edgecolor="black",
        linewidth=0.4,
        width=0.85,
    )
    ax.set_ylabel("Number of windows")
    ax.set_xlabel("Genome")
    ax.set_title("ready/ window counts by genome and group")
    ax.legend(title="Group", frameon=False, loc="upper right")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    paths = save_fig(fig, out_dir, "barplot_group_counts_by_genome")
    plt.close(fig)
    return paths, ct


def plot_density(
    df: pd.DataFrame,
    column: str,
    xlabel: str,
    stem: str,
    out_dir: Path,
    *,
    clip: tuple[float, float] | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for g in GROUP_ORDER:
        sub = df.loc[df["group"] == g, column].dropna().to_numpy()
        if sub.size == 0:
            continue
        sns.kdeplot(
            sub,
            ax=ax,
            color=GROUP_COLORS[g],
            linestyle=GROUP_LINESTYLES[g],
            linewidth=1.8,
            label=f"{g} (n={sub.size:,})",
            clip=clip,
            bw_adjust=1.0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{xlabel} density by group")
    ax.legend(frameon=False, title="Group")
    fig.tight_layout()
    paths = save_fig(fig, out_dir, stem)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ready-dir",
        type=Path,
        default=Path("ready"),
        help="Path to ready/ (symlink to data_ready)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output/ready_analysis"),
        help="Output directory",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)  # reserved for future sampling; keeps seed explicit
    _ = rng

    apply_style()
    ready_dir = args.ready_dir.resolve()
    out_dir = args.outdir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_grouped(ready_dir)
    counts = df["group"].value_counts().reindex(GROUP_ORDER).astype(int)
    counts.to_csv(out_dir / "group_counts.csv", header=["count"])

    plot_bar_counts(counts, out_dir)
    _, by_genome = plot_bar_by_genome(df, out_dir)
    by_genome.to_csv(out_dir / "group_counts_by_genome.csv")

    plot_density(
        df,
        "GC",
        "GC fraction",
        "density_gc_by_group",
        out_dir,
        clip=(0.0, 1.0),
    )
    plot_density(
        df,
        "Length",
        "Window length (bp)",
        "density_length_by_group",
        out_dir,
        clip=(float(df["Length"].min()), float(df["Length"].max())),
    )

    summary = {
        "ready_dir": str(ready_dir),
        "outdir": str(out_dir),
        "seed": SEED,
        "n_total": int(len(df)),
        "counts": {g: int(counts[g]) for g in GROUP_ORDER},
        "gc_mean": {
            g: float(df.loc[df["group"] == g, "GC"].mean()) for g in GROUP_ORDER
        },
        "length_mean": {
            g: float(df.loc[df["group"] == g, "Length"].mean()) for g in GROUP_ORDER
        },
        "software": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "seaborn": sns.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "inputs": {
            "non_coding_csv": str(ready_dir / "non_coding.csv"),
            "large_genes_csv": str(ready_dir / "large_genes.csv"),
        },
        "group_definitions": {
            "non-coding": "kind == non_coding",
            "large coding": "gene window exact-matched to large_genes.csv (CDS > 130 kb)",
            "normal coding": "kind == gene and not large",
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote figures and tables to {out_dir}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
