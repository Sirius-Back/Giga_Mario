"""Score EXISTING ``split.csv`` with balanced/robust L_hom and plot (11.4×8.1).

No training. Style: human-readable row labels; ○ balanced (weighted √n) and
□ robust connected by a line — same visual language as ``strategy_zsv_unified``
Pearson panel (without the annotation table).

Usage:
  python -m src.run.presentation.plot_lhom_balanced_robust_existing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.run.presentation.plot_lhom_balanced_robust_k4 import ROW_COLORS, _color_for

FIG_W = 11.4
FIG_H = 8.1

# label, family, path — existing splits only
DEFAULT_CANDIDATES: list[tuple[str, str, str]] = [
    ("Random", "random", "runs_unif/legnet/run2_legnet_random"),
    ("Pangenome", "pangenome", "runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100"),
    ("MLP-VAE (k=4)", "vae", "VAE/mlp_vae_kmer_k4_lossfix"),
    ("GCN-VAE", "gcn", "VGAE/stage1_region_k5_lossfix"),
    ("GAT-VAE", "gat", "VGAE/stage1_region_k5_gat_lossfix"),
    ("SAGE-VAE", "sage", "VGAE/stage1_region_k5_sage_lossfix"),
    ("GCL-VAE", "gcl", "VGAE/stage1_region_k5_gcl_lossfix"),
    ("GCL-GAT-VAE", "gcl_gat", "VGAE/stage1_region_k5_gcl_gat_lossfix"),
    ("APPNP-VAE", "appnp", "VGAE/stage1_region_k5_appnp_lossfix"),
    ("GCNII-VAE", "gcnii", "VGAE/stage1_region_k5_gcnii_lossfix"),
    ("GCN-VAE (struct)", "gcn", "VGAE/stage1_region_k5_structfeat_lossfix"),
    ("GCN-VAE (multi-k)", "gcn", "VGAE/stage1_region_k5_multik457_lossfix"),
    ("GCN-VAE (k=7)", "gcn", "VGAE/stage1_region_k7_lossfix"),
]


def collect_rows(candidates: list[tuple[str, str, Path]] | None = None) -> list[dict[str, Any]]:
    from src.pipeline.mem_guard import kill_cursor_indexers
    from src.run.run_id.eval_vgae_legacy_losses import eval_run

    if candidates is None:
        candidates = [(a, b, Path(c)) for a, b, c in DEFAULT_CANDIDATES]
    rows: list[dict[str, Any]] = []
    for label, family, path in candidates:
        path = Path(path)
        if not (path / "split.csv").is_file():
            print(f"SKIP missing {label} {path}", flush=True)
            continue
        kill_cursor_indexers(min_used_fraction=0.94)
        r = eval_run(path)
        if r.get("status") != "COMPLETED":
            print(f"SKIP {label} {r.get('status')} {r.get('reason')}", flush=True)
            continue
        w = float(r["all_aggs"]["weighted"]["l_hom"])
        rob = float(r["all_aggs"]["robust"]["l_hom"])
        print(f"{label:22} balanced={w:8.4f}  robust={rob:8.4f}", flush=True)
        rows.append(
            {
                "label": label,
                "family": family,
                "architecture": family,
                "source": str(path),
                "status": "COMPLETED",
                "balanced": w,
                "robust": rob,
                "all_aggs": r["all_aggs"],
                "n_regions": r.get("n_regions"),
            }
        )
    rows.sort(key=lambda x: (x["balanced"], x["robust"], x["label"]))
    return rows


def plot_rows(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    stem: str = "lhom_balanced_robust_existing",
    dpi: int = 300,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not rows:
        raise ValueError("no rows")
    # Keep caller sort order; plot −L_hom (higher is better visually).
    n = len(rows)
    y = np.arange(n, dtype=float)
    bal = -np.asarray([r["balanced"] for r in rows], dtype=float)
    rob = -np.asarray([r["robust"] for r in rows], dtype=float)
    colors = [_color_for(r) for r in rows]
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor="white")
    for i in range(n):
        c = colors[i]
        ax.plot([bal[i], rob[i]], [i, i], color=c, lw=1.35, alpha=0.9, zorder=2)
        ax.scatter(
            [bal[i]], [i], s=78, marker="o", facecolors=c,
            edgecolors="white", linewidths=0.7, zorder=3,
        )
        ax.scatter(
            [rob[i]], [i], s=78, marker="s", facecolors=c,
            edgecolors="white", linewidths=0.7, zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-L_{\mathrm{hom}}$  (higher is better)", fontsize=12)
    ax.set_title(
        "Balanced vs robust homology loss (existing splits)",
        fontsize=14, pad=12, color="#222222", fontweight="medium",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(True, axis="x", color="#D0D0D0", lw=0.7, alpha=0.95)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
    ax.tick_params(axis="y", length=0, pad=6, labelsize=15)
    xmin = float(min(bal.min(), rob.min()))
    xmax = float(max(bal.max(), rob.max()))
    pad = max(0.05, 0.08 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#333333",
                markeredgecolor="white", markersize=9,
                label="balanced (weighted √n)",
            ),
            Line2D(
                [0], [0], marker="s", color="w", markerfacecolor="#333333",
                markeredgecolor="white", markersize=9,
                label="robust (sd/√n, winsorized)",
            ),
        ],
        loc="lower right", frameon=False, fontsize=10,
    )
    fig.text(
        0.5, 0.015,
        "Existing split.csv only (no retrain). Rows sorted by balanced L_hom "
        "(same order; plotted as −L_hom). ○ balanced · □ robust · line connects.",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )
    fig.subplots_adjust(left=0.30, right=0.97, top=0.90, bottom=0.09)
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
    ap.add_argument("--out-dir", type=Path, default=Path("figures/presentation"))
    ap.add_argument("--stem", default="lhom_balanced_robust_existing")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("figures/presentation/lhom_balanced_robust_existing_metrics.json"),
    )
    ap.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Reuse metrics JSON (skip re-scoring)",
    )
    args = ap.parse_args(argv)

    if args.from_json is not None and Path(args.from_json).is_file():
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        rows = data["rows"] if isinstance(data, dict) else data
        for r in rows:
            if "balanced" not in r and r.get("all_aggs"):
                r["balanced"] = float(r["all_aggs"]["weighted"]["l_hom"])
                r["robust"] = float(r["all_aggs"]["robust"]["l_hom"])
        rows = [r for r in rows if r.get("status") == "COMPLETED"]
        rows.sort(key=lambda x: (x["balanced"], x["robust"], x["label"]))
    else:
        rows = collect_rows()
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(
                {
                    "note": "Existing split.csv only. balanced=weighted, robust=winsorized sd/√n.",
                    "rows": rows,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"metrics {args.metrics_json}", flush=True)

    written = plot_rows(rows, args.out_dir, stem=args.stem, dpi=int(args.dpi))
    for p in written:
        print(f"figure {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
