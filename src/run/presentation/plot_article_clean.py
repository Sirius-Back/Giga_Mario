#!/usr/bin/env python3
"""Clean Nature-style article result + embedding-compare figures.

Writes PDF/SVG/PNG under ``figures/article/`` (overwrites curated stems):

* ``Fig01_best_checkpoint_spearman`` — horizontal test+ZSV Spearman by family
  (direct only; readable labels; no overlapping grouped bars)
* ``Fig05_embed_pairwise_rsa_pooled`` — LegNet pairwise RSA (centered cosine)
* ``Fig06_embed_pairwise_cka_pooled`` — LegNet pairwise linear CKA

Also writes ``Fig01_test_vs_zsv_scatter`` as an extra clarity panel.

Usage::

  python -m src.run.presentation.plot_article_clean
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

# Okabe–Ito
C_TEST = "#009E73"
C_ZSV = "#CC79A7"
C_TRAIN = "#0072B2"
C_EDGE = "#333333"
C_GRID = "#B0B0B0"
C_FACE = "#FFFFFF"

FAMILY_ORDER = ("legnet", "caduceus")


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _short_label(label: str) -> str:
    base = label.split(":", 1)[0]
    if base.startswith("legacy/"):
        base = base[len("legacy/") :]
    base = re.sub(r"_caduceus_", "_", base)
    base = re.sub(r"_legnet_", "_", base)
    base = re.sub(r"^run(\d+)_", r"r\1_", base)
    # drop redundant family tokens already in panel title
    base = re.sub(r"_(?:caduceus|legnet)$", "", base)
    return base


def _load_direct_spearman(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("metric") != "spearman":
                continue
            lab = r.get("label") or ""
            if not lab.endswith(":direct"):
                continue
            split = r.get("split") or ""
            if split not in {"test", "zsv", "train", "val"}:
                continue
            val = _finite(r.get("value"))
            if val is None:
                continue
            rows.append(
                {
                    "label": lab,
                    "short": _short_label(lab),
                    "family": (r.get("family") or "other").lower(),
                    "strategy": r.get("strategy") or "other",
                    "split": split,
                    "value": val,
                }
            )
    return rows


def _pivot_direct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by.setdefault(
            r["label"],
            {
                "label": r["label"],
                "short": r["short"],
                "family": r["family"],
                "strategy": r["strategy"],
            },
        )
        d[r["split"]] = r["value"]
    out = [d for d in by.values() if "test" in d]
    out.sort(key=lambda d: (-float(d["test"]), d["short"]))
    return out


def _setup_mpl(dpi: int = 300) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": C_FACE,
            "axes.facecolor": C_FACE,
            "axes.edgecolor": C_EDGE,
            "axes.labelcolor": C_EDGE,
            "xtick.color": C_EDGE,
            "ytick.color": C_EDGE,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _save(fig: Any, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in (".pdf", ".svg", ".png"):
        out = stem.with_suffix(ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=C_FACE)
        written.append(out)
    return written


def plot_spearman_clean(
    metrics_csv: Path,
    out_stem: Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    plt = _setup_mpl(dpi)
    pivoted = _pivot_direct(_load_direct_spearman(metrics_csv))
    if not pivoted:
        raise RuntimeError(f"no direct Spearman test rows in {metrics_csv}")

    # Height scales with number of bars
    families = [f for f in FAMILY_ORDER if any(d["family"] == f for d in pivoted)]
    # also include unknown families
    for d in pivoted:
        if d["family"] not in families:
            families.append(d["family"])

    n_rows = max(len([d for d in pivoted if d["family"] == f]) for f in families)
    fig_h = max(4.5, 0.28 * n_rows + 1.6)
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(5.2 * len(families), fig_h),
        sharex=False,
        constrained_layout=True,
    )
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        sub = [d for d in pivoted if d["family"] == fam]
        # keep high-test at top
        y = np.arange(len(sub))[::-1]
        labels = [d["short"] for d in sub][::-1]
        test = np.asarray([d["test"] for d in sub][::-1], dtype=float)
        zsv = np.asarray(
            [d["zsv"] if "zsv" in d else np.nan for d in sub][::-1], dtype=float
        )

        h = 0.38
        ax.barh(
            y + h / 2,
            test,
            height=h,
            color=C_TEST,
            edgecolor=C_EDGE,
            linewidth=0.4,
            label="test",
            zorder=3,
        )
        z_mask = np.isfinite(zsv)
        if z_mask.any():
            ax.barh(
                y[z_mask] - h / 2,
                zsv[z_mask],
                height=h,
                color=C_ZSV,
                edgecolor=C_EDGE,
                linewidth=0.4,
                label="zsv",
                zorder=3,
            )
        for yi, zv in zip(y, zsv):
            if not np.isfinite(zv):
                ax.plot(
                    [0.02],
                    [yi - h / 2],
                    marker="x",
                    color=C_GRID,
                    markersize=5,
                    zorder=4,
                )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Spearman ρ (best checkpoint)")
        ax.set_title(fam.capitalize())
        ax.set_xlim(0, 1.0)
        ax.axvline(0, color=C_EDGE, lw=0.6)
        ax.grid(axis="x", color=C_GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle(
        "Best-checkpoint Spearman (direct runs): test vs ZSV",
        fontsize=12,
        y=1.02,
    )
    written = _save(fig, out_stem, dpi)
    plt.close(fig)
    return written


def plot_test_vs_zsv_scatter(
    metrics_csv: Path,
    out_stem: Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    plt = _setup_mpl(dpi)
    pivoted = [
        d
        for d in _pivot_direct(_load_direct_spearman(metrics_csv))
        if "zsv" in d and _finite(d.get("zsv")) is not None
    ]
    fig, ax = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
    markers = {"legnet": "o", "caduceus": "s"}
    for fam in FAMILY_ORDER:
        sub = [d for d in pivoted if d["family"] == fam]
        if not sub:
            continue
        ax.scatter(
            [d["test"] for d in sub],
            [d["zsv"] for d in sub],
            c=C_TRAIN if fam == "legnet" else C_TEST,
            marker=markers.get(fam, "o"),
            s=42,
            edgecolors=C_EDGE,
            linewidths=0.4,
            label=fam,
            zorder=3,
        )
        for d in sub:
            ax.annotate(
                d["short"],
                (d["test"], d["zsv"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                color=C_EDGE,
                alpha=0.85,
            )
    lims = [0.0, 1.0]
    ax.plot(lims, lims, ls="--", color=C_GRID, lw=1, zorder=1)
    ax.set_xlim(0.2, 0.7)
    ax.set_ylim(0.2, 0.7)
    # expand if data outside
    xs = [d["test"] for d in pivoted]
    ys = [d["zsv"] for d in pivoted]
    if xs and ys:
        lo = min(min(xs), min(ys)) - 0.05
        hi = max(max(xs), max(ys)) + 0.05
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Test Spearman ρ")
    ax.set_ylabel("ZSV Spearman ρ")
    ax.set_title("Test vs ZSV (direct, best checkpoint)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    written = _save(fig, out_stem, dpi)
    plt.close(fig)
    return written


def _short_run(run: str) -> str:
    s = re.sub(r"^run(\d+)_legnet_", r"r\1_", run)
    s = re.sub(r"^run(\d+)_", r"r\1_", s)
    s = s.replace("pangenome_", "pg_")
    s = s.replace("paralogs_only", "para")
    s = s.replace("gc_kmeans_elbow", "gc")
    s = s.replace("mmseqs_id08", "mmseqs")
    s = s.replace("hashfrag", "hf")
    s = s.replace("kmer_", "k")
    s = s.replace("w0_100", "w0")
    s = s.replace("wm100_100", "wm")
    s = s.replace("vgae_stage1_k5", "vgae")
    s = s.replace("loo5/", "loo/")
    return s


def _matrix_from_pairwise(
    tsv: Path,
    *,
    layer: str,
    score: str,
) -> tuple[list[str], np.ndarray]:
    pairs: list[tuple[str, str, float]] = []
    labels: set[str] = set()
    with tsv.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r.get("layer") or "") != layer:
                continue
            if (r.get("id_role") or "all") not in {"all", ""}:
                continue
            a, b = r["run_a"], r["run_b"]
            v = _finite(r.get(score))
            if v is None:
                continue
            pairs.append((a, b, v))
            labels.add(a)
            labels.add(b)
    keys = sorted(labels)
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    mat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        mat[i, i] = 1.0 if score in {"rsa_centered_cosine", "cka_linear"} else 0.0
    for a, b, v in pairs:
        i, j = idx[a], idx[b]
        mat[i, j] = v
        mat[j, i] = v
    return keys, mat


def plot_embed_heatmap(
    tsv: Path,
    out_stem: Path,
    *,
    layer: str,
    score: str,
    title: str,
    dpi: int = 300,
    cmap: str = "viridis",
) -> list[Path]:
    plt = _setup_mpl(dpi)
    keys, mat = _matrix_from_pairwise(tsv, layer=layer, score=score)
    if not keys:
        raise RuntimeError(f"no pairwise rows for layer={layer} score={score}")
    short = [_short_run(k) for k in keys]
    n = len(keys)
    fig_w = max(6.5, 0.42 * n + 2.2)
    fig_h = max(5.5, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    finite = mat[np.isfinite(mat)]
    if score in {"rsa_centered_cosine", "cka_linear"}:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, ha="center", va="top", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(score.replace("_", " "), fontsize=9)
    written = _save(fig, out_stem, dpi)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Default: <root>/docs/best_models_compare_metrics.csv",
    )
    ap.add_argument(
        "--pairwise-tsv",
        type=Path,
        default=None,
        help="Default: <root>/results/embed_legnet/pairwise/pairwise_compare.tsv",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Default: <root>/figures/article",
    )
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    root = args.root.resolve()
    outdir = (args.outdir or (root / "figures" / "article")).resolve()
    metrics = (
        args.metrics_csv or (root / "docs" / "best_models_compare_metrics.csv")
    ).resolve()
    pairwise = (
        args.pairwise_tsv
        or (root / "results" / "embed_legnet" / "pairwise" / "pairwise_compare.tsv")
    ).resolve()
    if not metrics.is_file():
        raise SystemExit(f"missing metrics CSV: {metrics}")
    if not pairwise.is_file():
        raise SystemExit(f"missing pairwise TSV: {pairwise}")

    written: list[Path] = []
    written += plot_spearman_clean(
        metrics, outdir / "Fig01_best_checkpoint_spearman", dpi=args.dpi
    )
    written += plot_test_vs_zsv_scatter(
        metrics, outdir / "Fig01b_test_vs_zsv_scatter", dpi=args.dpi
    )
    written += plot_embed_heatmap(
        pairwise,
        outdir / "Fig05_embed_pairwise_rsa_pooled",
        layer="pooled",
        score="rsa_centered_cosine",
        title="LegNet embed pairwise RSA (pooled, centered cosine)",
        dpi=args.dpi,
        cmap="viridis",
    )
    written += plot_embed_heatmap(
        pairwise,
        outdir / "Fig06_embed_pairwise_cka_pooled",
        layer="pooled",
        score="cka_linear",
        title="LegNet embed pairwise CKA (pooled, linear)",
        dpi=args.dpi,
        cmap="magma",
    )

    # Keep a copy also under presentation for reuse
    pres = root / "figures" / "presentation"
    for stem in (
        "Fig01_best_checkpoint_spearman",
        "Fig01b_test_vs_zsv_scatter",
        "Fig05_embed_pairwise_rsa_pooled",
        "Fig06_embed_pairwise_cka_pooled",
    ):
        for ext in (".pdf", ".svg", ".png"):
            src = outdir / f"{stem}{ext}"
            if src.is_file():
                dst = pres / f"article_{stem}{ext}"
                dst.write_bytes(src.read_bytes())
                written.append(dst)

    for p in written:
        print(f"wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
