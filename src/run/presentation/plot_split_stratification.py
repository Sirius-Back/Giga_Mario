#!/usr/bin/env python3
"""Presentation: split stratification vs Compara OG/PG + link to D_hom_emb.

Four-panel figure (maximally informative):

  A  mean sd_random for orthogroups vs paragroups (paired bars)
  B  L_hom = mean(sd_para) − mean(sd_ortho)  (more negative → OG clumped, PG closer to random)
  C  scatter sd_ortho vs sd_para (geometry of fold assignment)
  D  scatter fold L_hom vs embedding D_hom_emb (two different questions)

Inputs (default):
  - ``runs_unif/splits/legnet_*/{othologs,paralogs}.csv``
  - ``results/embed_legnet/homology_dissim/ranking.tsv`` (optional overlay)

Usage:
  python -m src.run.presentation.plot_split_stratification --lang both
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Okabe–Ito
COLOR_ORTHO = "#0072B2"
COLOR_PARA = "#D55E00"
COLOR_LHOM = "#009E73"
COLOR_DHOM = "#CC79A7"
COLOR_ANNOT = "#333333"
COLOR_ZERO = "#666666"
COLOR_GRID = "#B0B0B0"

FAMILY_COLORS = {
    "random": "#000000",
    "composition": "#E69F00",
    "sequence_cluster": "#56B4E9",
    "pangenome": "#009E73",
    "homology_aware": "#D55E00",
    "chromosome": "#0072B2",
    "other": "#999999",
}

RUN_RE = re.compile(r"^run\d+_legnet_(.+)$")
SKIP_DIR_RE = re.compile(r"BAD|ARCHIVED", re.I)


@dataclass
class StratRow:
    split_method: str
    run: str
    audit_dir: str
    family: str
    mean_sd_ortho: float
    mean_sd_para: float
    sem_sd_ortho: float
    sem_sd_para: float
    L_hom: float
    n_og: int
    n_pg: int
    D_hom_emb: float
    n_dhom_folds: int


def _mean_sem_sd(path: Path) -> tuple[float, float, int]:
    vals: list[float] = []
    with path.open(encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            try:
                vals.append(float(parts[-1]))
            except ValueError:
                continue
    if not vals:
        return float("nan"), float("nan"), 0
    a = np.asarray(vals, dtype=np.float64)
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem, int(a.size)


def split_method_from_audit_dir(name: str) -> tuple[str, str]:
    """Return (run_key, split_method) from ``legnet_runX_...`` dir name."""
    s = name
    if s.startswith("legnet_"):
        s = s[len("legnet_") :]
    if s == "run5_hashfrag":
        return "run5_legnet_hashfrag", "hashfrag"
    if s.startswith("run15_blastp") or s == "run15_blastp":
        return "run15_blastp", "blastp"
    m = RUN_RE.match(s)
    if m:
        return s, m.group(1)
    # fallback: strip leading runN_
    m2 = re.match(r"run\d+_(?:legnet_)?(.+)$", s)
    if m2:
        return s, m2.group(1)
    return s, s


def family_of(method: str) -> str:
    m = method.lower()
    if m == "random":
        return "random"
    if m.startswith("gc") or m.startswith("kmer"):
        return "composition"
    if m.startswith("pangenome"):
        return "pangenome"
    if m in {"hashfrag", "mmseqs_id08", "blastp"} or m.startswith("mmseqs"):
        return "sequence_cluster"
    if m.startswith("paralogs") or m.startswith("vgae") or m.startswith("gcn"):
        return "homology_aware"
    if m.startswith("loco"):
        return "chromosome"
    return "other"


def short_label(method: str) -> str:
    """Compact tick label."""
    return (
        method.replace("pangenome_", "pg_")
        .replace("wm100_100", "wm")
        .replace("w0_100", "w0")
        .replace("gc_kmeans_elbow", "gc")
        .replace("paralogs_only", "para_only")
        .replace("vgae_stage1_k5", "vgae_k5")
        .replace("mmseqs_id08", "mmseqs")
    )


def load_dhom_by_method(ranking_tsv: Path) -> dict[str, tuple[float, int]]:
    """split_method → (mean D_hom_emb, n folds)."""
    if not ranking_tsv.is_file():
        return {}
    buckets: dict[str, list[float]] = defaultdict(list)
    with ranking_tsv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if str(row.get("layer", "")) != "pooled":
                # ranking.tsv is already pooled-only, but be safe
                if "layer" in row and row["layer"] != "pooled":
                    continue
            sm = (row.get("split_method") or "").strip()
            if not sm:
                continue
            buckets[sm].append(float(row["D_hom_emb"]))
    return {k: (float(np.mean(v)), len(v)) for k, v in buckets.items()}


def collect_stratification(
    splits_root: Path,
    *,
    ranking_tsv: Path | None = None,
    model_prefix: str = "legnet_",
) -> list[StratRow]:
    splits_root = Path(splits_root)
    dhom = load_dhom_by_method(Path(ranking_tsv)) if ranking_tsv else {}
    rows: list[StratRow] = []
    for d in sorted(splits_root.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(model_prefix):
            continue
        if SKIP_DIR_RE.search(d.name):
            continue
        o, p = d / "othologs.csv", d / "paralogs.csv"
        if not (o.is_file() and p.is_file()):
            continue
        run, method = split_method_from_audit_dir(d.name)
        mo, seo, no = _mean_sem_sd(o)
        mp, sep, np_ = _mean_sem_sd(p)
        dh, nfold = dhom.get(method, (float("nan"), 0))
        rows.append(
            StratRow(
                split_method=method,
                run=run,
                audit_dir=str(d),
                family=family_of(method),
                mean_sd_ortho=mo,
                mean_sd_para=mp,
                sem_sd_ortho=seo,
                sem_sd_para=sep,
                L_hom=float(mp - mo),
                n_og=no,
                n_pg=np_,
                D_hom_emb=dh,
                n_dhom_folds=nfold,
            )
        )
    # sort by L_hom ascending (more negative first = stronger OG clump / PG~random)
    rows.sort(key=lambda r: (r.L_hom, r.split_method))
    return rows


def write_summary_tsv(rows: list[StratRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(asdict(rows[0]).keys()) if rows else [
        "split_method",
        "run",
        "L_hom",
        "mean_sd_ortho",
        "mean_sd_para",
        "D_hom_emb",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def _labels(lang: str) -> dict[str, str]:
    if lang == "en":
        return {
            "suptitle": "Split stratification by Compara orthologs / paralogs",
            "A_title": "A. Fold assignment: mean sd_random",
            "B_title": "B. L_hom = mean(sd_para) − mean(sd_ortho)",
            "C_title": "C. Ortho vs para sd_random",
            "D_title": "D. Fold L_hom vs embedding D_hom_emb",
            "ylabel_sd": r"mean $sd_{\mathrm{random}}$",
            "ylabel_lhom": r"$L_{\mathrm{hom}}$ (fold)",
            "xlabel_method": "Split method",
            "sd_ortho": "orthogroups",
            "sd_para": "paragroups",
            "xlabel_so": r"mean $sd$ (orthogroups)",
            "ylabel_sp": r"mean $sd$ (paragroups)",
            "xlabel_lhom": r"$L_{\mathrm{hom}}$ (fold roles)",
            "ylabel_dhom": r"$D_{\mathrm{hom}}^{\mathrm{emb}}$ (LegNet pooled)",
            "note_b": "more negative → OG clumped, PG closer to random fractions",
            "note_d": "different questions: assignment ≠ representation",
            "takeaway": (
                "Stratification = how OG/PG sit in train/val/test (panels A–C).\n"
                "D_hom_emb (panel D) is how the trained model embeds homologs — not the same axis."
            ),
            "family_legend": "family",
            "no_dhom": "no embed score",
        }
    return {
        "suptitle": "Стратификация сплитов по Compara ортологам / паралогам",
        "A_title": "A. Назначение в фолды: mean sd_random",
        "B_title": "B. L_hom = mean(sd_para) − mean(sd_ortho)",
        "C_title": "C. sd_random ортологи vs паралоги",
        "D_title": "D. Fold L_hom vs эмбеддинги D_hom_emb",
        "ylabel_sd": r"mean $sd_{\mathrm{random}}$",
        "ylabel_lhom": r"$L_{\mathrm{hom}}$ (fold)",
        "xlabel_method": "Метод сплита",
        "sd_ortho": "ортогруппы",
        "sd_para": "парагруппы",
        "xlabel_so": r"mean $sd$ (ортогруппы)",
        "ylabel_sp": r"mean $sd$ (парагруппы)",
        "xlabel_lhom": r"$L_{\mathrm{hom}}$ (роли fold)",
        "ylabel_dhom": r"$D_{\mathrm{hom}}^{\mathrm{emb}}$ (LegNet pooled)",
        "note_b": "отрицательнее → OG скучены, PG ближе к случайным долям",
        "note_d": "разные вопросы: assignment ≠ representation",
        "takeaway": (
            "Стратификация — как OG/PG лежат в train/val/test (панели A–C).\n"
            "D_hom_emb (панель D) — как обученная модель представляет гомологи: другая ось."
        ),
        "family_legend": "семейство",
        "no_dhom": "нет embed-оценки",
    }


def plot(
    rows: list[StratRow],
    out_dir: Path,
    *,
    lang: str = "ru",
    stem: str = "split_stratification",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not rows:
        raise ValueError("no stratification rows to plot")

    L = _labels(lang)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [r.split_method for r in rows]
    labels = [short_label(m) for m in methods]
    x = np.arange(len(rows), dtype=np.float64)
    so = np.asarray([r.mean_sd_ortho for r in rows], dtype=np.float64)
    sp = np.asarray([r.mean_sd_para for r in rows], dtype=np.float64)
    seo = np.asarray([r.sem_sd_ortho for r in rows], dtype=np.float64)
    sep = np.asarray([r.sem_sd_para for r in rows], dtype=np.float64)
    lhom = np.asarray([r.L_hom for r in rows], dtype=np.float64)
    dhom = np.asarray([r.D_hom_emb for r in rows], dtype=np.float64)
    fams = [r.family for r in rows]
    facecolors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in fams]

    fig = plt.figure(figsize=(14.5, 10.2), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        hspace=0.38,
        wspace=0.28,
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ----- A: paired bars -----
    w = 0.38
    ax_a.bar(
        x - w / 2,
        so,
        width=w,
        color=COLOR_ORTHO,
        label=L["sd_ortho"],
        zorder=3,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_a.bar(
        x + w / 2,
        sp,
        width=w,
        color=COLOR_PARA,
        label=L["sd_para"],
        zorder=3,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_a.errorbar(
        x - w / 2, so, yerr=seo, fmt="none", ecolor=COLOR_ANNOT, elinewidth=0.9, capsize=2, zorder=4
    )
    ax_a.errorbar(
        x + w / 2, sp, yerr=sep, fmt="none", ecolor=COLOR_ANNOT, elinewidth=0.9, capsize=2, zorder=4
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=8)
    ax_a.set_ylabel(L["ylabel_sd"], fontsize=11)
    ax_a.set_title(L["A_title"], fontsize=12, loc="left", pad=6)
    ax_a.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax_a.grid(True, axis="y", color=COLOR_GRID, alpha=0.35, lw=0.7)
    ax_a.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_a.spines[spine].set_visible(False)

    # ----- B: L_hom bars colored by family -----
    ax_b.axhline(0.0, color=COLOR_ZERO, ls="--", lw=1.1, zorder=2)
    bars = ax_b.bar(
        x,
        lhom,
        width=0.72,
        color=facecolors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=8)
    ax_b.set_ylabel(L["ylabel_lhom"], fontsize=11)
    ax_b.set_title(L["B_title"], fontsize=12, loc="left", pad=6)
    ax_b.text(
        0.02,
        0.98,
        L["note_b"],
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        style="italic",
        color=COLOR_ANNOT,
    )
    # mark best (most negative)
    i_best = int(np.nanargmin(lhom))
    ax_b.annotate(
        f"{lhom[i_best]:+.2f}",
        xy=(x[i_best], lhom[i_best]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=facecolors[i_best],
    )
    ax_b.grid(True, axis="y", color=COLOR_GRID, alpha=0.35, lw=0.7)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)
    # family legend
    present = []
    for fam, col in FAMILY_COLORS.items():
        if fam in fams:
            present.append(Patch(facecolor=col, edgecolor="white", label=fam))
    ax_b.legend(
        handles=present,
        title=L["family_legend"],
        fontsize=8,
        title_fontsize=8,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
    )

    # ----- C: scatter sd_o vs sd_p -----
    for r, lab, col in zip(rows, labels, facecolors):
        ax_c.scatter(
            r.mean_sd_ortho,
            r.mean_sd_para,
            s=70,
            c=col,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax_c.annotate(
            lab,
            (r.mean_sd_ortho, r.mean_sd_para),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7.5,
            color=COLOR_ANNOT,
        )
    # reference: equal sd
    lim_lo = min(float(np.nanmin(so)), float(np.nanmin(sp))) * 0.92
    lim_hi = max(float(np.nanmax(so)), float(np.nanmax(sp))) * 1.05
    ax_c.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls=":", color=COLOR_ZERO, lw=1.2, zorder=1)
    # quadrant hint: high sd_o, low sd_p = desired for L_hom minimize
    ax_c.axvspan(
        float(np.median(so)),
        lim_hi,
        ymin=0,
        ymax=(float(np.median(sp)) - lim_lo) / max(lim_hi - lim_lo, 1e-9),
        color=COLOR_LHOM,
        alpha=0.06,
        zorder=0,
    )
    ax_c.set_xlim(lim_lo, lim_hi)
    ax_c.set_ylim(lim_lo, lim_hi * 0.55 + lim_lo * 0.45)  # para usually lower
    # better y-lim from data
    ax_c.set_ylim(float(np.nanmin(sp)) * 0.9, float(np.nanmax(sp)) * 1.08)
    ax_c.set_xlabel(L["xlabel_so"], fontsize=11)
    ax_c.set_ylabel(L["ylabel_sp"], fontsize=11)
    ax_c.set_title(L["C_title"], fontsize=12, loc="left", pad=6)
    ax_c.grid(True, color=COLOR_GRID, alpha=0.3, lw=0.7)
    for spine in ("top", "right"):
        ax_c.spines[spine].set_visible(False)

    # ----- D: L_hom vs D_hom_emb -----
    has_d = np.isfinite(dhom)
    for r, lab, col, ok in zip(rows, labels, facecolors, has_d):
        if not ok:
            ax_d.scatter(
                r.L_hom,
                0.0,
                s=40,
                marker="x",
                c=COLOR_ZERO,
                zorder=2,
                alpha=0.5,
            )
            continue
        ax_d.scatter(
            r.L_hom,
            r.D_hom_emb,
            s=80,
            c=col,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax_d.annotate(
            lab,
            (r.L_hom, r.D_hom_emb),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7.5,
            color=COLOR_ANNOT,
        )
    if has_d.any():
        # Spearman annotation
        from scipy.stats import spearmanr

        rho, pval = spearmanr(lhom[has_d], dhom[has_d])
        ax_d.text(
            0.98,
            0.98,
            f"Spearman ρ = {rho:.2f}\n(n = {int(has_d.sum())})",
            transform=ax_d.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=COLOR_ANNOT,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.95,
            ),
        )
    ax_d.axvline(0.0, color=COLOR_ZERO, ls="--", lw=1.0, zorder=1)
    ax_d.set_xlabel(L["xlabel_lhom"], fontsize=11)
    ax_d.set_ylabel(L["ylabel_dhom"], fontsize=11)
    ax_d.set_title(L["D_title"], fontsize=12, loc="left", pad=6)
    ax_d.text(
        0.02,
        0.02,
        L["note_d"],
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color=COLOR_ANNOT,
    )
    ax_d.grid(True, color=COLOR_GRID, alpha=0.3, lw=0.7)
    for spine in ("top", "right"):
        ax_d.spines[spine].set_visible(False)

    fig.suptitle(L["suptitle"], fontsize=14, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.015,
        L["takeaway"],
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=COLOR_ANNOT,
        style="italic",
    )

    written: list[Path] = []
    for ext in ("svg", "pdf", "png"):
        path = out_dir / f"{stem}_{lang}.{ext}"
        kw: dict[str, Any] = {"facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    _ = bars  # silence lint
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--splits-root",
        type=Path,
        default=Path("runs_unif/splits"),
    )
    p.add_argument(
        "--ranking",
        type=Path,
        default=Path("results/embed_legnet/homology_dissim/ranking.tsv"),
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/splits_stratification/summary.tsv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/presentation"),
    )
    p.add_argument("--lang", choices=("ru", "en", "both"), default="ru")
    p.add_argument("--stem", default="split_stratification")
    args = p.parse_args(argv)

    rows = collect_stratification(
        args.splits_root,
        ranking_tsv=args.ranking if Path(args.ranking).is_file() else None,
    )
    summary = write_summary_tsv(rows, args.summary_out)
    meta = {
        "n_methods": len(rows),
        "summary_tsv": str(summary),
        "formula_L_hom": "mean(sd_para) - mean(sd_ortho)",
        "note": "More negative L_hom ≈ orthogroups clumped in folds, paragroups nearer random",
    }
    meta_path = Path(args.summary_out).with_name("manifest.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[strat] summary → {summary} (n={len(rows)})", flush=True)

    langs = ("ru", "en") if args.lang == "both" else (args.lang,)
    for lang in langs:
        paths = plot(rows, args.out_dir, lang=lang, stem=args.stem)
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
