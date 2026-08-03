#!/usr/bin/env python3
"""Separate Caduceus / LegNet presentation panels (strategy table + Pearson + violins).

Each family gets its own 22.7∶9 figure, left → right:

1. Annotation table: ``strategy | strategy_params`` (no run column)
2. Train + ZSV Pearson as paired markers (circle / square) with a connecting line
3. Ortholog ``log10(sd_random+1)`` violins
4. Paralog ``log10(sd_random+1)`` violins

Rows sorted by strategy order × ZSV Pearson (desc). LOO folds are mean-aggregated.
Top-3 ZSV within each family are annotated to 3 decimals.

Usage:
  python -m src.run.presentation.plot_strategy_zsv_panel --refresh-metrics
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

STRATEGY_ORDER = (
    "paralogs_only",
    "random",
    "loco",
    "gc",
    "kmer",
    "mmseqs",
    "hashfrag",
    "blastp",
    "pangenome",
    "gcn",
    "vgae",
    "other",
)
STRATEGY_COLORS = {
    "paralogs_only": "#CC79A7",
    "random": "#000000",
    "loco": "#882255",
    "gc": "#E69F00",
    "kmer": "#009E73",
    "mmseqs": "#F0E442",
    "hashfrag": "#56B4E9",
    "blastp": "#0072B2",
    "pangenome": "#D55E00",
    "gcn": "#AA4499",
    "vgae": "#44AA99",
    "other": "#999999",
}

SUPERSEDED_SPLIT_DIRS = {
    "caduceus_run16_hashfrag_caduceus",
    "legnet_run5_hashfrag",
}
LEGACY_DIR_MARKERS = ("_BAD_", "BAD_random", "_legacy", "_LEGACY", "ARCHIVED")

FIG_W = 22.7
FIG_H = 9.0
KDE_GRID = 96
MAX_GROUPS_KDE = 2500
RNG_SEED = 42


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _run_id(label: str) -> int:
    m = re.search(r"(?:^|/|_)run(\d+)(?:_|$|:|/)", label)
    return int(m.group(1)) if m else 10**9


def _family_from_label(label: str, family_col: str | None) -> str:
    if family_col:
        f = str(family_col).lower()
        if f.startswith("c"):
            return "caduceus"
        if f.startswith("l"):
            return "legnet"
    low = label.lower()
    if "caduceus" in low:
        return "caduceus"
    if "legnet" in low:
        return "legnet"
    return "unknown"


def _strip_stage(label: str) -> str:
    return label.split(":", 1)[0]


def _base_run_key(label: str) -> str:
    """Drop ``/foldN`` so LOO folds aggregate."""
    base = _strip_stage(label)
    return re.sub(r"/fold\d+$", "", base)


def _run_short_from_label(label: str) -> str:
    base = _base_run_key(label)
    if base.startswith("legacy/"):
        base = base[len("legacy/") :]
    # legacy naming: run12_4mer_caduceus / run14_7mer_caduceus
    m = re.match(r"run(\d+)_(\d+)mer_caduceus$", base)
    if m:
        return f"run{m.group(1)}_kmer_k{m.group(2)}"
    m = re.match(r"run(\d+)_pangenome_CDS_caduceus$", base)
    if m:
        return f"run{m.group(1)}_pangenome_CDS"
    m = re.match(r"run(\d+)_(?:caduceus|legnet)_(.+)$", base)
    if m:
        return f"run{m.group(1)}_{m.group(2)}"
    if re.fullmatch(r"run5", base):
        return "run5_hashfrag"
    m = re.match(r"run(\d+)_hashfrag_caduceus$", base)
    if m:
        return f"run{m.group(1)}_hashfrag"
    m = re.match(r"run(\d+)_blastp_legnet$", base)
    if m:
        return f"run{m.group(1)}_blastp"
    m = re.match(r"run(\d+)_(.+)$", base)
    if m:
        rest = re.sub(r"^(?:caduceus|legnet)_", "", m.group(2))
        return f"run{m.group(1)}_{rest}"
    return base


def _strategy_family(run_short: str) -> str:
    s = run_short.lower()
    if "paralog" in s:
        return "paralogs_only"
    if "blast" in s:
        return "blastp"
    if "mmseq" in s:
        return "mmseqs"
    if "hashfrag" in s:
        return "hashfrag"
    if "gcn" in s:
        return "gcn"
    if "vgae" in s:
        return "vgae"
    if re.search(r"(?:^|_)loco(?:_|$)", s):
        return "loco"
    if "pangenome" in s:
        return "pangenome"
    if "kmer" in s or re.search(r"(?:^|_)k\d+(?:_|$)", s):
        return "kmer"
    if "gc_" in s or "gc-" in s or "kmeans" in s:
        return "gc"
    if "random" in s:
        return "random"
    return "other"


def _strategy_params(run_short: str) -> tuple[str, str]:
    strategy = _strategy_family(run_short)
    rest = re.sub(r"^run\d+_", "", run_short)
    patterns = {
        "paralogs_only": r"^paralogs_only_?",
        "blastp": r"^blastp_?",
        "mmseqs": r"^mmseqs_?",
        "hashfrag": r"^hashfrag_?",
        "gcn": r"^gcn_?",
        "vgae": r"^vgae_?",
        "loco": r"^loco_?",
        "pangenome": r"^pangenome_?",
        "kmer": r"^kmer_?",
        "gc": r"^gc_?",
        "random": r"^random_?",
    }
    params = re.sub(patterns.get(strategy, r"^"), "", rest, count=1, flags=re.I).strip("_")
    if strategy == "kmer" and not params:
        m = re.search(r"(k\d+.*)$", rest)
        params = m.group(1) if m else rest
    if not params:
        params = "—"
    return strategy, params


def _strategy_rank(strategy: str) -> int:
    try:
        return STRATEGY_ORDER.index(strategy)
    except ValueError:
        return len(STRATEGY_ORDER)


def discover_split_map(splits_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not splits_root.is_dir():
        return out
    for d in sorted(splits_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name in SUPERSEDED_SPLIT_DIRS or any(m in name for m in LEGACY_DIR_MARKERS):
            continue
        if not (d / "othologs.csv").is_file() or not (d / "paralogs.csv").is_file():
            continue
        family = "unknown"
        run_label = name
        for pref in ("caduceus_", "legnet_"):
            if name.startswith(pref):
                family = pref[:-1]
                run_label = name[len(pref) :]
                break
        run_short = re.sub(r"^run(\d+)_caduceus_", r"run\1_", run_label)
        run_short = re.sub(r"^run(\d+)_legnet_", r"run\1_", run_short)
        # legacy-style outdirs: run14_7mer_caduceus → run14_kmer_k7
        m = re.match(r"run(\d+)_(\d+)mer(?:_caduceus)?$", run_short)
        if m:
            run_short = f"run{m.group(1)}_kmer_k{m.group(2)}"
        m = re.match(r"run(\d+)_pangenome_CDS(?:_caduceus)?$", run_short)
        if m:
            run_short = f"run{m.group(1)}_pangenome_CDS"
        out.setdefault(f"{family}:{run_short}", d)
    return out


def _resolve_split_dir(
    family: str, run_short: str, split_map: dict[str, Path]
) -> Path | None:
    key = f"{family}:{run_short}"
    if key in split_map:
        return split_map[key]
    rid = _run_id(run_short)
    strat = _strategy_family(run_short)
    for k, p in split_map.items():
        fam, short = k.split(":", 1)
        if fam != family:
            continue
        if _run_id(short) != rid:
            continue
        if _strategy_family(short) == strat:
            return p
    return None


def load_pearson_rows(
    metrics_csv: Path, *, include_adv: bool = False
) -> list[dict[str, Any]]:
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics CSV missing: {metrics_csv}")
    # label → metrics; keep fold-level then aggregate
    by_label: dict[str, dict[str, Any]] = {}
    with metrics_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("metric") != "pearson":
                continue
            label = str(row.get("label") or "")
            if not include_adv and not label.endswith(":direct"):
                continue
            if any(m in label for m in ("ARCHIVED", "BAD_", "FAILED")):
                continue
            split = str(row.get("split") or "")
            if split not in {"train", "zsv"}:
                continue
            val = _finite(row.get("value"))
            if val is None:
                continue
            family = _family_from_label(label, row.get("family"))
            slot = by_label.setdefault(
                label,
                {
                    "label": label,
                    "family": family,
                    "train_pearson": None,
                    "zsv_pearson": None,
                    "train_dir": row.get("train_dir") or "",
                },
            )
            if split == "train":
                slot["train_pearson"] = val
            else:
                slot["zsv_pearson"] = val

    # Aggregate LOO folds → one row per run
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for slot in by_label.values():
        key = (slot["family"], _base_run_key(slot["label"]))
        buckets.setdefault(key, []).append(slot)

    rows: list[dict[str, Any]] = []
    for (family, base), grp in buckets.items():
        # prefer a synthetic label without fold
        label = f"{base}:direct" if not base.startswith("legacy/") else f"{base}:direct"
        trains = [g["train_pearson"] for g in grp if g["train_pearson"] is not None]
        zsvs = [g["zsv_pearson"] for g in grp if g["zsv_pearson"] is not None]
        run_short = _run_short_from_label(label)
        strategy, params = _strategy_params(run_short)
        rows.append(
            {
                "label": label,
                "family": family,
                "run_short": run_short,
                "run_id": _run_id(label),
                "strategy": strategy,
                "strategy_params": params,
                "train_pearson": float(np.mean(trains)) if trains else None,
                "zsv_pearson": float(np.mean(zsvs)) if zsvs else None,
                "n_folds": len(grp),
                "train_dir": grp[0].get("train_dir") or "",
            }
        )

    rows.sort(
        key=lambda r: (
            _strategy_rank(r["strategy"]),
            -(r["zsv_pearson"] if r["zsv_pearson"] is not None else -1e9),
            r["run_id"],
            r["label"],
        )
    )
    # Prefer runs_unif over legacy when same family+strategy+params (e.g. hashfrag run16)
    seen_keys: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    # non-legacy first
    ordered = sorted(rows, key=lambda r: (1 if str(r["label"]).startswith("legacy/") else 0))
    for r in ordered:
        key = (r["family"], r["strategy"], r["strategy_params"])
        # allow multiple pangenome/kmer param variants; only collapse exact param match
        if key in seen_keys and str(r["label"]).startswith("legacy/"):
            continue
        # also collapse same run_id within family+strategy when one is legacy
        rid_key = (r["family"], r["strategy"], r["run_id"])
        if str(r["label"]).startswith("legacy/") and any(
            (x["family"], x["strategy"], x["run_id"]) == rid_key
            and not str(x["label"]).startswith("legacy/")
            for x in deduped
        ):
            continue
        seen_keys.add(key)
        deduped.append(r)
    deduped.sort(
        key=lambda r: (
            _strategy_rank(r["strategy"]),
            -(r["zsv_pearson"] if r["zsv_pearson"] is not None else -1e9),
            r["run_id"],
            r["label"],
        )
    )
    return deduped


def _load_sd_values(path: Path, *, max_groups: int, rng: np.random.Generator) -> np.ndarray:
    vals: list[float] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            v = _finite(row.get("sd_random"))
            if v is not None:
                vals.append(v)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size > max_groups:
        idx = rng.choice(arr.size, size=max_groups, replace=False)
        arr = arr[idx]
    return arr


def attach_sd(
    rows: list[dict[str, Any]],
    splits_root: Path,
    *,
    max_groups: int = MAX_GROUPS_KDE,
) -> list[dict[str, Any]]:
    split_map = discover_split_map(splits_root)
    rng = np.random.default_rng(RNG_SEED)
    for r in rows:
        d = _resolve_split_dir(r["family"], r["run_short"], split_map)
        r["split_dir"] = str(d) if d else ""
        if d is None:
            r["sd_ortho"] = np.asarray([], dtype=np.float64)
            r["sd_para"] = np.asarray([], dtype=np.float64)
            continue
        r["sd_ortho"] = _load_sd_values(
            d / "othologs.csv", max_groups=max_groups, rng=rng
        )
        r["sd_para"] = _load_sd_values(
            d / "paralogs.csv", max_groups=max_groups, rng=rng
        )
    return rows


def _kde_curve(vals: np.ndarray, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, x_max, KDE_GRID)
    if vals.size < 5:
        return grid, np.zeros_like(grid)
    from scipy.stats import gaussian_kde

    if np.unique(vals).size < 2:
        dens = np.exp(
            -0.5 * ((grid - float(vals.mean())) / max(x_max * 0.02, 1e-6)) ** 2
        )
    else:
        dens = gaussian_kde(vals)(grid)
    m = float(dens.max()) if dens.size else 0.0
    if m > 0:
        dens = dens / m
    return grid, dens.astype(np.float64)


def write_table_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "family",
        "strategy",
        "strategy_params",
        "label",
        "n_folds",
        "train_pearson",
        "zsv_pearson",
        "split_dir",
        "n_sd_ortho",
        "n_sd_para",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            w.writerow(
                {
                    "rank": i,
                    "family": r["family"],
                    "strategy": r["strategy"],
                    "strategy_params": r["strategy_params"],
                    "label": r["label"],
                    "n_folds": r.get("n_folds", 1),
                    "train_pearson": ""
                    if r["train_pearson"] is None
                    else f"{r['train_pearson']:.6f}",
                    "zsv_pearson": ""
                    if r["zsv_pearson"] is None
                    else f"{r['zsv_pearson']:.6f}",
                    "split_dir": r.get("split_dir", ""),
                    "n_sd_ortho": int(np.asarray(r.get("sd_ortho", [])).size),
                    "n_sd_para": int(np.asarray(r.get("sd_para", [])).size),
                }
            )


def _text_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r_, g_, b_ = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = (0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_) / 255.0
    return "#333333" if lum > 0.65 else hex_color


def _draw_family_row(
    fig,
    gs_row,
    rows: list[dict[str, Any]],
    *,
    family_title: str,
    pearson_xlim: tuple[float, float],
    x_max_o: float,
    x_max_p: float,
) -> None:
    import matplotlib.pyplot as plt

    n = len(rows)
    if n == 0:
        ax = fig.add_subplot(gs_row[0, :])
        ax.axis("off")
        ax.text(0.5, 0.5, f"{family_title}: no runs", ha="center", va="center")
        return

    y = np.arange(n, dtype=float)
    colors = [STRATEGY_COLORS.get(r["strategy"], "#999999") for r in rows]
    train = np.asarray(
        [np.nan if r["train_pearson"] is None else r["train_pearson"] for r in rows],
        dtype=float,
    )
    zsv = np.asarray(
        [np.nan if r["zsv_pearson"] is None else r["zsv_pearson"] for r in rows],
        dtype=float,
    )

    ax_tab = fig.add_subplot(gs_row[0, 0])
    ax_pr = fig.add_subplot(gs_row[0, 1])
    ax_o = fig.add_subplot(gs_row[0, 2], sharey=ax_pr)
    ax_p = fig.add_subplot(gs_row[0, 3], sharey=ax_pr)

    # table: strategy | strategy_params
    ax_tab.set_xlim(0, 2)
    ax_tab.set_ylim(-0.5, n - 0.5)
    ax_tab.invert_yaxis()
    ax_tab.axis("off")
    for j, h in enumerate(("strategy", "strategy_params")):
        ax_tab.text(
            (j + 0.5) / 2.0,
            1.03,
            h,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#222222",
            transform=ax_tab.transAxes,
            clip_on=False,
        )
    for i, r in enumerate(rows):
        c = _text_color(colors[i])
        for j, text in enumerate((r["strategy"], r["strategy_params"])):
            shown = text if len(text) <= 22 else text[:20] + "…"
            ax_tab.text(
                j + 0.04,
                i,
                shown,
                ha="left",
                va="center",
                fontsize=12,
                color=c,
            )
        if i > 0 and rows[i]["strategy"] != rows[i - 1]["strategy"]:
            ax_tab.axhline(i - 0.5, color="#DDDDDD", lw=0.55, zorder=0)
    ax_tab.set_title(family_title, fontsize=14, pad=10, loc="left", fontweight="medium")

    def _style(ax, title: str, xlabel: str) -> None:
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="y", length=0, labelleft=False)
        ax.tick_params(axis="x", labelsize=7.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#888888")
        ax.grid(True, axis="x", color="#D0D0D0", lw=0.65, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()

    # paired Pearson points + line
    for i in range(n):
        t = train[i]
        z = zsv[i]
        c = colors[i]
        if np.isfinite(t) and np.isfinite(z):
            ax_pr.plot([t, z], [i, i], color=c, lw=1.15, alpha=0.85, zorder=2)
        if np.isfinite(t):
            ax_pr.scatter(
                [t],
                [i],
                s=34,
                marker="o",
                facecolors=c,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label="train" if i == 0 else None,
            )
        if np.isfinite(z):
            ax_pr.scatter(
                [z],
                [i],
                s=34,
                marker="s",
                facecolors=c,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label="ZSV" if i == 0 else None,
            )

    # top-3 ZSV annotations within this family
    ranked = sorted(
        ((i, float(zsv[i])) for i in range(n) if np.isfinite(zsv[i])),
        key=lambda t: -t[1],
    )[:3]
    for rank_i, (i, val) in enumerate(ranked, start=1):
        ax_pr.annotate(
            f"{val:.3f}",
            xy=(val, i),
            xytext=(-10, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=11,
            color="#222222",
            fontweight="bold",
            zorder=5,
        )

    ax_pr.set_xlim(*pearson_xlim)
    _style(ax_pr, "Train ○  ·  ZSV □", "Pearson r")
    handles, labels = ax_pr.get_legend_handles_labels()
    if handles:
        ax_pr.legend(
            handles[:2],
            labels[:2],
            loc="lower right",
            fontsize=7,
            frameon=False,
            handletextpad=0.3,
            borderpad=0.2,
        )

    def _draw_violins(ax, key: str, x_max: float, title: str) -> None:
        half = 0.38
        for i, r in enumerate(rows):
            raw = np.asarray(r.get(key, []), dtype=float)
            if raw.size == 0:
                continue
            vals = np.log10(raw + 1.0)
            grid, dens = _kde_curve(vals, x_max)
            if dens.max() <= 0:
                continue
            ax.fill_between(
                grid,
                i - half * dens,
                i + half * dens,
                color=colors[i],
                alpha=0.28,
                linewidth=0,
                zorder=2,
            )
            ax.plot(grid, i - half * dens, color=colors[i], lw=0.8, zorder=3)
            ax.plot(grid, i + half * dens, color=colors[i], lw=0.8, zorder=3)
            med = float(np.median(vals))
            ax.plot([med, med], [i - 0.26, i + 0.26], color="#333333", lw=1.0, zorder=4)
        ax.set_xlim(0.0, x_max)
        _style(ax, title, r"$\log_{10}(sd_{\mathrm{random}}+1)$")

    _draw_violins(ax_o, "sd_ortho", x_max_o, "Orthologs")
    _draw_violins(ax_p, "sd_para", x_max_p, "Paralogs")


def plot_panel(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    stem: str = "strategy_zsv_unified",
    dpi: int = 300,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if not rows:
        raise ValueError("no rows to plot")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    families = (
        ("caduceus", "Caduceus", f"{stem}_caduceus"),
        ("legnet", "LegNet", f"{stem}_legnet"),
    )
    for fam_key, title, fam_stem in families:
        fam_rows = [r for r in rows if r["family"] == fam_key]
        if not fam_rows:
            print(f"WARN no rows for {fam_key}", flush=True)
            continue

        all_p = [
            v
            for r in fam_rows
            for v in (r["train_pearson"], r["zsv_pearson"])
            if v is not None
        ]
        pmin = min(all_p) if all_p else 0.0
        pmax = max(all_p) if all_p else 1.0
        pearson_xlim = (min(0.0, pmin - 0.03), max(1.0, pmax + 0.08))

        def _xmax(key: str, _rows: list[dict[str, Any]] = fam_rows) -> float:
            vals = []
            for r in _rows:
                raw = np.asarray(r.get(key, []), dtype=float)
                if raw.size:
                    vals.append(np.log10(raw + 1.0))
            if not vals:
                return 1.0
            cat = np.concatenate(vals)
            return float(max(1.0, np.percentile(cat, 99.5) * 1.05))

        x_max_o = _xmax("sd_ortho")
        x_max_p = _xmax("sd_para")

        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor="white")
        gs = GridSpec(
            1,
            4,
            figure=fig,
            width_ratios=[0.72, 1.0, 1.0, 1.0],
            wspace=0.08,
            left=0.02,
            right=0.99,
            top=0.90,
            bottom=0.08,
        )
        _draw_family_row(
            fig,
            gs,
            fam_rows,
            family_title=title,
            pearson_xlim=pearson_xlim,
            x_max_o=x_max_o,
            x_max_p=x_max_p,
        )
        fig.suptitle(
            f"{title}: best-checkpoint Pearson and split homology stratification",
            fontsize=14,
            y=0.97,
            fontweight="medium",
            color="#222222",
        )
        fig.text(
            0.5,
            0.015,
            "Rows: strategy order × ZSV Pearson ↓; LOO = mean over folds. "
            "○ train Pearson, □ ZSV Pearson (line connects). "
            "Top-3 ZSV labeled. Violins: log10(sd_random+1), median tick.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        for ext in ("pdf", "svg", "png"):
            path = out_dir / f"{fam_stem}.{ext}"
            kw: dict[str, Any] = {"facecolor": "white"}
            if ext == "png":
                kw["dpi"] = dpi
            fig.savefig(path, **kw)
            written.append(path)
        plt.close(fig)

    # Keep a combined table; optional thin combined pointer file not needed
    return written


def refresh_metrics(csv_path: Path, md_path: Path) -> None:
    from src.pipeline.compare_best_models import (
        collect_rows,
        discover_train_dirs,
        write_csv,
        write_markdown,
    )

    dirs = discover_train_dirs()
    rows = collect_rows(dirs)
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("docs/best_models_compare_metrics.csv"),
    )
    ap.add_argument(
        "--metrics-md",
        type=Path,
        default=Path("docs/best_models_compare_report.md"),
    )
    ap.add_argument("--splits-root", type=Path, default=Path("runs_unif/splits"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/presentation"))
    ap.add_argument("--stem", default="strategy_zsv_unified")
    ap.add_argument("--table-tsv", type=Path, default=None)
    ap.add_argument("--include-adv", action="store_true")
    ap.add_argument("--refresh-metrics", action="store_true")
    ap.add_argument("--max-groups", type=int, default=MAX_GROUPS_KDE)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    if args.refresh_metrics:
        refresh_metrics(args.metrics_csv, args.metrics_md)
        print(f"refreshed {args.metrics_csv}", flush=True)

    rows = load_pearson_rows(args.metrics_csv, include_adv=bool(args.include_adv))
    rows = attach_sd(rows, args.splits_root, max_groups=int(args.max_groups))
    table_path = args.table_tsv or (args.out_dir / f"{args.stem}_table.tsv")
    write_table_tsv(rows, table_path)
    written = plot_panel(rows, args.out_dir, stem=args.stem, dpi=int(args.dpi))
    print(
        f"rows={len(rows)} cad={sum(r['family']=='caduceus' for r in rows)} "
        f"leg={sum(r['family']=='legnet' for r in rows)} table={table_path}",
        flush=True,
    )
    for p in written:
        print(f"figure {p}", flush=True)
    missing = [r for r in rows if not r.get("split_dir")]
    if missing:
        print(f"WARN {len(missing)} rows without split-check SD", flush=True)
        for r in missing:
            print(f"  missing SD: {r['label']}", flush=True)
    no_zsv = [r for r in rows if r["zsv_pearson"] is None]
    if no_zsv:
        print(f"WARN {len(no_zsv)} rows without ZSV pearson", flush=True)
        for r in no_zsv:
            print(f"  missing ZSV: {r['label']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
