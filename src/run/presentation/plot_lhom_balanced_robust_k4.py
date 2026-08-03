"""Presentation: balanced vs robust L_hom for GCN-VAE / VAE / random / pangenome (k=4).

Style mirrors ``strategy_zsv_unified`` Pearson panel: human-readable row labels,
two marker shapes connected by a horizontal line (no annotation table).
Figure size 11.4 × 8.1 in.

Metrics:
  ○  balanced = size-weighted L_hom (``weighted``, √n)
  □  robust   = winsorized mean of sd/√n (``robust``)

Usage:
  python -m src.run.presentation.plot_lhom_balanced_robust_k4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

FIG_W = 11.4
FIG_H = 8.1

# Okabe–Ito-ish by method family
ROW_COLORS = {
    "random": "#000000",
    "pangenome": "#D55E00",
    "vae": "#0072B2",
    "gcn": "#009E73",
    "gat": "#E69F00",
    "sage": "#56B4E9",
    "gcl": "#CC79A7",
    "gcl_gat": "#882255",
    "appnp": "#44AA99",
    "gcnii": "#AA4499",
    "other": "#999999",
}


def _color_for(row: dict[str, Any]) -> str:
    key = str(row.get("family") or row.get("architecture") or "other").lower()
    if key in ROW_COLORS:
        return ROW_COLORS[key]
    lab = str(row.get("label") or "").lower()
    for k, c in ROW_COLORS.items():
        if k in lab:
            return c
    return ROW_COLORS["other"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") and r["status"] != "COMPLETED":
            continue
        ag = r.get("all_aggs") or {}
        if "weighted" not in ag or "robust" not in ag:
            continue
        out.append(
            {
                "label": str(r["label"]),
                "family": str(r.get("family") or "other"),
                "architecture": str(r.get("architecture") or ""),
                "balanced": float(ag["weighted"]["l_hom"]),
                "robust": float(ag["robust"]["l_hom"]),
                "source": str(r.get("source") or r.get("path") or ""),
            }
        )
    # more negative = better → sort ascending balanced
    out.sort(key=lambda x: (x["balanced"], x["robust"], x["label"]))
    return out


def plot_lhom(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    stem: str = "lhom_balanced_robust_k4",
    dpi: int = 300,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not rows:
        raise ValueError("no rows to plot")

    n = len(rows)
    y = np.arange(n, dtype=float)
    bal = np.asarray([r["balanced"] for r in rows], dtype=float)
    rob = np.asarray([r["robust"] for r in rows], dtype=float)
    colors = [_color_for(r) for r in rows]
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor="white")
    for i in range(n):
        c = colors[i]
        ax.plot([bal[i], rob[i]], [i, i], color=c, lw=1.35, alpha=0.9, zorder=2)
        ax.scatter(
            [bal[i]],
            [i],
            s=78,
            marker="o",
            facecolors=c,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.scatter(
            [rob[i]],
            [i],
            s=78,
            marker="s",
            facecolors=c,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(r"$L_{\mathrm{hom}}$  (lower is better)", fontsize=12)
    ax.set_title(
        "k=4 splits: balanced vs robust homology loss",
        fontsize=14,
        pad=12,
        color="#222222",
        fontweight="medium",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(True, axis="x", color="#D0D0D0", lw=0.7, alpha=0.95)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=10)

    xmin = float(min(bal.min(), rob.min()))
    xmax = float(max(bal.max(), rob.max()))
    pad = max(0.05, 0.08 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markersize=9,
            label="balanced (weighted √n)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markersize=9,
            label="robust (sd/√n, winsorized)",
        ),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=10)

    fig.text(
        0.5,
        0.015,
        "Rows sorted by balanced L_hom ↑ (more negative first). "
        "○ balanced · □ robust · line connects the pair. "
        "GCN-VAE / VAE use k=4 features; pangenome & random are panel splits.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.90, bottom=0.09)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{ext}"
        kw: dict[str, Any] = {"facecolor": "white"}
        if ext == "png":
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("figures/presentation/lhom_balanced_robust_k4_metrics.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("figures/presentation"))
    ap.add_argument("--stem", default="lhom_balanced_robust_k4")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    rows = load_rows(args.metrics_json)
    written = plot_lhom(rows, args.out_dir, stem=args.stem, dpi=int(args.dpi))
    print(f"rows={len(rows)}", flush=True)
    for r in rows:
        print(
            f"  {r['label']}: balanced={r['balanced']:.4f} robust={r['robust']:.4f}",
            flush=True,
        )
    for p in written:
        print(f"figure {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
