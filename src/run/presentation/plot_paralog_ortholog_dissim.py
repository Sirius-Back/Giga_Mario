#!/usr/bin/env python3
"""Presentation: ranking split methods by D_hom_emb (paralog − ortholog distance).

Reads ``results/embed_legnet/homology_dissim/ranking.tsv`` (pooled layer).
Higher bars → paralogs more dissimilar relative to orthologs
(«Разделяем паралоги, объединяем ортологи»).

Usage:
  python -m src.run.presentation.plot_paralog_ortholog_dissim
  python -m src.run.presentation.plot_paralog_ortholog_dissim --lang en
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Okabe–Ito
COLOR_BAR = "#0072B2"
COLOR_ERR = "#333333"
COLOR_ZERO = "#666666"
COLOR_ANNOT = "#333333"
COLOR_PARA = "#D55E00"
COLOR_ORTHO = "#009E73"


def _load_ranking(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ranking TSV missing: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty ranking TSV: {path}")
    return rows


def _aggregate_by_split_method(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Mean ``D_hom_emb`` across LOO folds that share ``split_method``."""
    from collections import defaultdict

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (r.get("split_method") or r.get("run") or "").strip()
        buckets[key].append(r)
    out: list[dict[str, str]] = []
    for method, grp in buckets.items():
        d_vals = np.asarray([float(g["D_hom_emb"]) for g in grp], dtype=np.float64)
        sem_o = np.asarray(
            [float(g.get("sem_d_ortho") or "nan") for g in grp], dtype=np.float64
        )
        sem_p = np.asarray(
            [float(g.get("sem_d_para") or "nan") for g in grp], dtype=np.float64
        )
        # Within-fold SEM of difference, then SEM across folds if n>1
        err_within = np.sqrt(np.nan_to_num(sem_o) ** 2 + np.nan_to_num(sem_p) ** 2)
        if d_vals.size > 1:
            sem_across = float(np.std(d_vals, ddof=1) / np.sqrt(d_vals.size))
            err = float(np.sqrt(np.mean(err_within**2) + sem_across**2))
        else:
            err = float(err_within[0]) if err_within.size else float("nan")
        seed = dict(grp[0])
        seed["split_method"] = method
        seed["D_hom_emb"] = str(float(np.mean(d_vals)))
        seed["sem_d_ortho"] = str(err / np.sqrt(2.0))
        seed["sem_d_para"] = str(err / np.sqrt(2.0))
        seed["n_folds"] = str(len(grp))
        out.append(seed)
    out.sort(
        key=lambda d: (
            -float(d["D_hom_emb"])
            if np.isfinite(float(d["D_hom_emb"]))
            else 1e9,
            str(d.get("split_method", "")),
        )
    )
    return out


def _labels(lang: str) -> dict[str, str]:
    if lang == "en":
        return {
            "xlabel": "Split method",
            "ylabel": r"$D_{\mathrm{hom}}^{\mathrm{emb}}=\overline{d}_{\mathrm{para}}-\overline{d}_{\mathrm{ortho}}$",
            "title": "Paralog dissimilarity − ortholog cohesion (LegNet pooled)",
            "takeaway": (
                "Higher is better: embeddings separate paralogs while keeping orthologs close.\n"
                "Centered-cosine distance within Compara groups; train-fit centering."
            ),
        }
    return {
        "xlabel": "Метод сплита",
        "ylabel": r"$D_{\mathrm{hom}}^{\mathrm{emb}}=\overline{d}_{\mathrm{para}}-\overline{d}_{\mathrm{ortho}}$",
        "title": "Непохожесть паралогов − сплочённость ортологов (LegNet pooled)",
        "takeaway": (
            "Выше — лучше: эмбеддинги разделяют паралоги и сближают ортологи.\n"
            "Centered-cosine distance внутри Compara-групп; центрирование по train."
        ),
    }


def plot(
    ranking_tsv: Path,
    out_dir: Path,
    *,
    lang: str = "ru",
    stem: str = "paralog_ortholog_dissim",
    top_n: int | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _aggregate_by_split_method(_load_ranking(ranking_tsv))
    # Already sorted by D_hom_emb desc
    if top_n is not None and top_n > 0:
        rows = rows[: int(top_n)]

    methods = [r.get("split_method") or r.get("run", "") for r in rows]
    d_hom = np.asarray([float(r["D_hom_emb"]) for r in rows], dtype=np.float64)
    # SEM of difference ≈ sqrt(sem_o^2 + sem_p^2) under independence
    sem_o = np.asarray(
        [float(r.get("sem_d_ortho") or "nan") for r in rows], dtype=np.float64
    )
    sem_p = np.asarray(
        [float(r.get("sem_d_para") or "nan") for r in rows], dtype=np.float64
    )
    err = np.sqrt(np.nan_to_num(sem_o) ** 2 + np.nan_to_num(sem_p) ** 2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = _labels(lang)

    fig_w = max(8.0, 0.55 * len(methods) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(methods), dtype=np.float64)
    bars = ax.bar(
        x,
        d_hom,
        width=0.72,
        color=COLOR_BAR,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.errorbar(
        x,
        d_hom,
        yerr=err,
        fmt="none",
        ecolor=COLOR_ERR,
        elinewidth=1.2,
        capsize=3.0,
        zorder=4,
    )
    ax.axhline(0.0, color=COLOR_ZERO, ls="--", lw=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=90, ha="center", va="top", fontsize=9.5)
    ax.set_ylabel(L["ylabel"], fontsize=13)
    ax.set_xlabel(L["xlabel"], fontsize=13)
    ax.set_title(L["title"], fontsize=13.5, pad=10)
    ax.tick_params(axis="y", labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(True, axis="y", which="major", color="#B0B0B0", alpha=0.28, lw=0.8)
    ax.set_axisbelow(True)

    # Subtle legend for formula terms
    ax.plot([], [], color=COLOR_PARA, lw=3, label=r"$\overline{d}_{\mathrm{para}}$↑")
    ax.plot([], [], color=COLOR_ORTHO, lw=3, label=r"$\overline{d}_{\mathrm{ortho}}$↓")
    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
    )

    # Annotate best bar
    if len(d_hom):
        i_best = int(np.nanargmax(d_hom))
        bars[i_best].set_color("#56B4E9")
        ax.annotate(
            f"{d_hom[i_best]:.3f}",
            xy=(x[i_best], d_hom[i_best]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR_ANNOT,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.01,
        L["takeaway"],
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=COLOR_ANNOT,
        style="italic",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))

    written: list[Path] = []
    for ext in ("svg", "pdf", "png"):
        path = out_dir / f"{stem}_{lang}.{ext}"
        kw: dict = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ranking",
        type=Path,
        default=Path("results/embed_legnet/homology_dissim/ranking.tsv"),
        help="Path to ranking.tsv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/presentation"),
        help="Output directory for SVG/PDF/PNG",
    )
    p.add_argument(
        "--lang",
        choices=("ru", "en", "both"),
        default="ru",
        help="Annotation language (default: ru)",
    )
    p.add_argument("--stem", default="paralog_ortholog_dissim")
    p.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="If >0, plot only top-N methods by D_hom_emb",
    )
    args = p.parse_args()
    langs = ("ru", "en") if args.lang == "both" else (args.lang,)
    top_n = int(args.top_n) if int(args.top_n) > 0 else None
    for lang in langs:
        paths = plot(
            args.ranking,
            args.out_dir,
            lang=lang,
            stem=args.stem,
            top_n=top_n,
        )
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
