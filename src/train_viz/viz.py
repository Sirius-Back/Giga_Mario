#!/usr/bin/env python3
"""Publication-quality training visualization (Nature Methods / NMI style).

Never uses matplotlib defaults. Exports PDF/SVG/PNG@600dpi, tables, and
visualization_config.yaml. See ../SKILL.md and visualization_config.yaml.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml  # type: ignore
except ImportError:  # minimal fallback
    yaml = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "visualization_config.yaml"

SPLIT_KEYS = (
    "train",
    "validation",
    "val",
    "test",
    "zero_shot",
    "zero-shot",
    "zero_shot_validation",
    "zero-shot-validation",
)
SCALAR_KEYS = ("elapsed_sec", "lr", "learning_rate", "grad_norm")
SPLIT_ORDER = [
    "train",
    "validation",
    "test",
    "zero_shot",
    "zero_shot_validation",
    "_run",
]

# Flat metric token for regex / naming
METRIC_TOKEN = (
    r"loss|accuracy|acc|f1|auc|auroc|auprc|precision|recall|"
    r"pearson|spearman|mse|rmse|mae|r2|r²|"
    r"gene[_-]?wise[_-]?pearson(?:_median)?|sample[_-]?wise[_-]?pearson(?:_median)?|"
    r"genewise_pearson_median|samplewise_pearson_median|"
    r"elapsed_sec|lr|learning_rate|grad_norm"
)
RUN_LEVEL_METRICS = frozenset({"elapsed_sec", "lr", "grad_norm"})


def _load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_CONFIG_PATH
    defaults: dict[str, Any] = {
        "dpi_png": 600,
        "figure_size_px": {"single": [1800, 1400], "double": [3600, 2400]},
        "font": {
            "family": ["Arial", "Helvetica", "DejaVu Sans"],
            "axis_label_pt": 9,
            "tick_pt": 8,
            "legend_pt": 8,
            "title_pt": 10,
        },
        "line_width": 2.2,
        "grid": {"which": "major", "color": "#B0B0B0", "alpha": 0.3},
        "palette": {
            "train": "#0072B2",
            "validation": "#E69F00",
            "test": "#009E73",
            "zero_shot": "#CC79A7",
            "_run": "#56B4E9",
            "models": [
                "#0072B2",
                "#E69F00",
                "#009E73",
                "#D55E00",
                "#CC79A7",
                "#56B4E9",
                "#F0E442",
                "#000000",
            ],
        },
        "linestyle": {
            "train": "solid",
            "validation": "dashed",
            "test": "dotted",
            "zero_shot": "dashdot",
            "_run": "solid",
        },
        "seed_aggregation": "mean_ci95",
        "ribbon_default": "ci95",
        "lowess_frac": 0.35,
        "metric_direction": {
            "lower_is_better": ["loss", "mse", "rmse", "mae", "elapsed_sec"],
            "higher_is_better": [
                "pearson",
                "spearman",
                "r2",
                "r²",
                "gene_wise_pearson",
                "sample_wise_pearson",
                "genewise_pearson_median",
                "samplewise_pearson_median",
                "accuracy",
                "f1",
                "auc",
                "auroc",
                "auprc",
                "precision",
                "recall",
            ],
        },
        "layout_grid": {
            1: [1, 1],
            2: [1, 2],
            3: [1, 3],
            4: [2, 2],
            5: [2, 3],
            6: [2, 3],
            7: [2, 4],
            8: [2, 4],
            9: [3, 3],
        },
        "qc": {"min_dpi": 600, "require_ci_when_multi_seed": True},
    }
    if path.is_file() and yaml is not None:
        loaded = yaml.safe_load(path.read_text()) or {}
        defaults.update({k: v for k, v in loaded.items() if v is not None})
        # deep-merge nested known keys
        for key in ("palette", "font", "grid", "figure_size_px", "metric_direction", "linestyle", "qc"):
            if key in loaded and isinstance(loaded[key], dict):
                base = defaults.get(key, {})
                if isinstance(base, dict):
                    merged = dict(base)
                    merged.update(loaded[key])
                    defaults[key] = merged
    return defaults


def figsize_inches(cfg: dict[str, Any], column: str = "double") -> tuple[float, float]:
    px = cfg["figure_size_px"].get(column, cfg["figure_size_px"]["double"])
    dpi = float(cfg["dpi_png"])
    return (px[0] / dpi, px[1] / dpi)


def apply_pub_style(cfg: dict[str, Any]) -> None:
    """Nature Methods / NMI — never matplotlib defaults."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fam = cfg["font"]["family"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(fam),
            "font.size": cfg["font"]["tick_pt"],
            "axes.labelsize": cfg["font"]["axis_label_pt"],
            "axes.titlesize": cfg["font"]["title_pt"],
            "axes.titleweight": "regular",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "savefig.dpi": cfg["dpi_png"],
            "xtick.labelsize": cfg["font"]["tick_pt"],
            "ytick.labelsize": cfg["font"]["tick_pt"],
            "legend.fontsize": cfg["font"]["legend_pt"],
            "legend.frameon": False,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": cfg["grid"]["color"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": cfg["grid"]["alpha"],
            "lines.linewidth": cfg["line_width"],
            "lines.markersize": 5.0,
            "axes.prop_cycle": mpl.cycler(color=cfg["palette"]["models"]),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def normalize_metric_name(name: str) -> str:
    n = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "acc": "accuracy",
        "auroc": "auc",
        "r²": "r2",
        "gene_wise_pearson": "gene_wise_pearson",
        "genewise_pearson": "gene_wise_pearson",
        "genewise_pearson_median": "genewise_pearson_median",
        "gene_wise_pearson_median": "genewise_pearson_median",
        "sample_wise_pearson": "sample_wise_pearson",
        "samplewise_pearson": "sample_wise_pearson",
        "samplewise_pearson_median": "samplewise_pearson_median",
        "sample_wise_pearson_median": "samplewise_pearson_median",
        "learning_rate": "lr",
    }
    return aliases.get(n, n)


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return bool(math.isfinite(float(v)))


def split_label(split: str) -> str:
    """Legend / axis-friendly split name."""
    if split == "_run":
        return "run"
    return split


def splits_for_metric(rows: list[dict[str, Any]], metric: str) -> list[str]:
    """Splits that have at least one finite value for metric (includes _run)."""
    m = normalize_metric_name(metric)
    present: list[str] = []
    for split in SPLIT_ORDER:
        for r in rows:
            if r.get("metric") != m or r.get("split") != split:
                continue
            try:
                if math.isfinite(float(r["value"])):
                    present.append(split)
                    break
            except (TypeError, ValueError):
                continue
    return present


def metrics_with_data(rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    """Drop metrics that have no finite values (avoids empty panels)."""
    return [m for m in metrics if splits_for_metric(rows, m)]


def is_lower_better(metric: str, cfg: dict[str, Any]) -> bool:
    m = normalize_metric_name(metric)
    low = {normalize_metric_name(x) for x in cfg["metric_direction"]["lower_is_better"]}
    high = {normalize_metric_name(x) for x in cfg["metric_direction"]["higher_is_better"]}
    if m in low:
        return True
    if m in high:
        return False
    # default: unknown metrics → higher is better except *loss*
    return "loss" in m or m.endswith("error")


def best_index(values: list[float], metric: str, cfg: dict[str, Any]) -> int | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    if is_lower_better(metric, cfg):
        return int(np.nanargmin(arr))
    return int(np.nanargmax(arr))


def _try_parse_mapping(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text or text[0] not in "{[":
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _has_epoch_metrics(obj: dict[str, Any]) -> bool:
    if "epoch" not in obj and "global_step" not in obj and "step" not in obj:
        return False
    if any(k in obj and isinstance(obj[k], dict) for k in SPLIT_KEYS):
        return True
    flat = re.compile(
        rf"^(train|validation|val|test|zero[_-]shot(?:[_-]validation)?)[_/]({METRIC_TOKEN})$",
        re.I,
    )
    return any(flat.match(str(k)) for k in obj)


def parse_log(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Log not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Log is empty: {path}")

    config: dict[str, Any] | None = None
    epochs: list[dict[str, Any]] = []
    buf: list[str] = []
    depth = 0
    collecting = False
    meta_keys = (
        "model_name",
        "epochs",
        "n_train",
        "seed",
        "data_root",
        "batch_size",
        "optimizer",
        "scheduler",
        "lr",
        "learning_rate",
        "fold",
        "species",
        "split",
    )

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()

            if collecting or (stripped.startswith("{") and not _try_parse_mapping(stripped)):
                if not collecting and stripped.startswith("{"):
                    collecting = True
                    buf = [line]
                    depth = stripped.count("{") - stripped.count("}")
                    continue
                if collecting:
                    buf.append(line)
                    depth += stripped.count("{") - stripped.count("}")
                    if depth <= 0:
                        blob = "\n".join(buf)
                        collecting = False
                        buf = []
                        try:
                            obj = json.loads(blob)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and "epoch" not in obj and any(
                            k in obj for k in meta_keys
                        ):
                            config = obj
                    continue

            obj = _try_parse_mapping(stripped)
            if obj is None:
                continue
            if _has_epoch_metrics(obj):
                epochs.append(obj)
            elif config is None and any(k in obj for k in meta_keys):
                config = obj

    if not epochs:
        raise ValueError(
            f"No epoch metric lines found in {path}. "
            "Expected dict/JSON lines with 'epoch' and train/validation/test blocks."
        )
    return config, epochs


def flatten_epochs(
    run_id: str,
    epochs: list[dict[str, Any]],
    *,
    model: str,
    seed: Any,
) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
    rows: list[dict[str, Any]] = []
    metrics: set[str] = set()
    splits: set[str] = set()
    nan_only_candidates: set[str] = set()
    flat_re = re.compile(
        rf"^(train|validation|val|test|zero[_-]shot(?:[_-]validation)?)"
        rf"[_/]({METRIC_TOKEN})$",
        re.I,
    )

    for obj in epochs:
        epoch = obj.get("epoch", obj.get("global_step", obj.get("step")))
        gstep = obj.get("global_step", obj.get("step"))
        base = {
            "run": run_id,
            "model": model,
            "seed": seed,
            "epoch": epoch,
            "global_step": gstep,
        }
        for sk in SPLIT_KEYS:
            block = obj.get(sk)
            if not isinstance(block, dict):
                continue
            split = "validation" if sk in ("val",) else sk.replace("-", "_")
            splits.add(split)
            for mk, mv in block.items():
                if mk == "n":
                    continue
                name = normalize_metric_name(str(mk))
                if isinstance(mv, float) and not math.isfinite(mv):
                    nan_only_candidates.add(name)
                    continue
                if not _is_finite_number(mv):
                    continue
                metrics.add(name)
                rows.append({**base, "split": split, "metric": name, "value": float(mv)})
        for sk in SCALAR_KEYS:
            if sk not in obj:
                continue
            name = normalize_metric_name(sk)
            if isinstance(obj[sk], float) and not math.isfinite(float(obj[sk])):
                nan_only_candidates.add(name)
                continue
            if not _is_finite_number(obj[sk]):
                continue
            metrics.add(name)
            splits.add("_run")
            rows.append({**base, "split": "_run", "metric": name, "value": float(obj[sk])})
        for k, v in obj.items():
            m = flat_re.match(str(k))
            if not m:
                continue
            name = normalize_metric_name(m.group(2))
            if isinstance(v, float) and not math.isfinite(v):
                nan_only_candidates.add(name)
                continue
            if not _is_finite_number(v):
                continue
            split = m.group(1).lower().replace("-", "_")
            if split == "val":
                split = "validation"
            metrics.add(name)
            splits.add(split)
            rows.append({**base, "split": split, "metric": name, "value": float(v)})

    nan_only = {m for m in nan_only_candidates if m not in metrics}
    return rows, metrics, splits, nan_only


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def order_metrics(metrics: set[str] | list[str], cfg: dict[str, Any]) -> list[str]:
    preferred = [
        "loss",
        "pearson",
        "spearman",
        "mse",
        "rmse",
        "mae",
        "r2",
        "gene_wise_pearson",
        "sample_wise_pearson",
        "genewise_pearson_median",
        "samplewise_pearson_median",
        "accuracy",
        "f1",
        "auc",
        "precision",
        "recall",
        "lr",
        "elapsed_sec",
        "grad_norm",
    ]
    # Exclude pure LR from multi-metric grids (own figure)
    skip_in_grid = {"lr", "learning_rate"}
    ms = {normalize_metric_name(m) for m in metrics} - skip_in_grid
    ordered = [m for m in preferred if m in ms]
    ordered += sorted(m for m in ms if m not in ordered)
    return ordered


def layout_for(n: int, cfg: dict[str, Any]) -> tuple[int, int]:
    grid = cfg.get("layout_grid") or {}
    # YAML may stringify keys
    key = n if n in grid else str(n)
    if key in grid:
        r, c = grid[key]
        return int(r), int(c)
    if n <= 9:
        # nearest preferred
        for k in (n, n + 1, n + 2):
            kk = k if k in grid else str(k)
            if kk in grid:
                r, c = grid[kk]
                return int(r), int(c)
    cols = 3
    rows = int(math.ceil(n / cols))
    return rows, cols


def split_color(split: str, cfg: dict[str, Any]) -> str:
    pal = cfg["palette"]
    if split in pal:
        return pal[split]
    if split.startswith("zero_shot"):
        return pal.get("zero_shot", pal["models"][0])
    return pal["models"][0]


def model_color(model_i: int, cfg: dict[str, Any]) -> str:
    colors = cfg["palette"]["models"]
    return colors[model_i % len(colors)]


def split_linestyle(split: str, cfg: dict[str, Any]) -> str:
    styles = cfg["linestyle"]
    if split in styles:
        return styles[split]
    if split.startswith("zero_shot"):
        return styles.get("zero_shot", "dashdot")
    return "solid"


def _lowess(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray | None:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore

        out = lowess(y, x, frac=frac, return_sorted=False)
        return np.asarray(out, dtype=float)
    except Exception:
        return None


def _series(
    rows: list[dict[str, Any]],
    *,
    model: str | None = None,
    seed: Any | None = None,
    run: str | None = None,
    split: str,
    metric: str,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for r in rows:
        if r["metric"] != metric or r["split"] != split:
            continue
        if model is not None and r.get("model") != model:
            continue
        if seed is not None and r.get("seed") != seed:
            continue
        if run is not None and r.get("run") != run:
            continue
        if r.get(x_key) is None:
            continue
        try:
            val = float(r["value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        pts.append((r[x_key], val))
    if not pts:
        return np.array([]), np.array([])
    pts.sort(key=lambda t: t[0])
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    return xs, ys


def aggregate_seeds(
    rows: list[dict[str, Any]],
    *,
    model: str,
    split: str,
    metric: str,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, mean, median, std, ci95_half (t~1.96*se)."""
    seeds = sorted({r["seed"] for r in rows if r.get("model") == model}, key=lambda s: str(s))
    # Align on union of x
    by_seed: dict[Any, dict[float, float]] = {}
    all_x: set[float] = set()
    for seed in seeds:
        xs, ys = _series(rows, model=model, seed=seed, split=split, metric=metric, x_key=x_key)
        by_seed[seed] = {float(x): float(y) for x, y in zip(xs, ys)}
        all_x.update(by_seed[seed].keys())
    if not all_x:
        empty = np.array([])
        return empty, empty, empty, empty, empty
    x = np.array(sorted(all_x), dtype=float)
    mat = []
    for seed in seeds:
        mat.append([by_seed[seed].get(float(xi), np.nan) for xi in x])
    M = np.asarray(mat, dtype=float)
    mean = np.nanmean(M, axis=0)
    median = np.nanmedian(M, axis=0)
    std = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean)
    n = np.sum(~np.isnan(M), axis=0).astype(float)
    se = np.divide(std, np.sqrt(np.maximum(n, 1.0)), out=np.zeros_like(std), where=n > 0)
    ci = 1.96 * se
    return x, mean, median, std, ci


def save_figure(fig: Any, stem: Path, dpi: int) -> list[Path]:
    """Always PDF + SVG + PNG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(path, dpi=dpi)
        else:
            fig.savefig(path)
        written.append(path)
    return written


def qc_check(
    *,
    dpi: int,
    cfg: dict[str, Any],
    n_seeds: int,
    ribbon: str,
    used_pub_style: bool,
) -> list[str]:
    errors: list[str] = []
    min_dpi = int(cfg.get("qc", {}).get("min_dpi", 600))
    if dpi < min_dpi:
        errors.append(f"low DPI ({dpi} < {min_dpi})")
    if not used_pub_style:
        errors.append("default matplotlib style")
    if n_seeds >= 2 and cfg.get("qc", {}).get("require_ci_when_multi_seed", True):
        if ribbon not in ("ci95", "std"):
            errors.append("missing confidence intervals / ribbons for multi-seed")
    return errors


class FigureIndex:
    def __init__(self) -> None:
        self.n = 0

    def next_stem(self, outdir: Path, name: str) -> Path:
        self.n += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "figure"
        return outdir / f"Figure_{self.n:02d}_{safe}"


def plot_learning_curves(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    x_key: str,
    title: str | None,
    column: str,
    ribbon: str,
    smooth: bool,
    patience: int | None,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    written: list[Path] = []
    models = sorted({r["model"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    multi_model = len(models) > 1
    metrics = metrics_with_data(rows, metrics)
    if not metrics:
        return written

    # --- Combined grid of metrics (single-model: color=split; multi: color=model, ls=split) ---
    # Paginate if >9
    pages: list[list[str]] = []
    if len(metrics) <= 9:
        pages = [metrics]
    else:
        for i in range(0, len(metrics), 9):
            pages.append(metrics[i : i + 9])

    for page_i, page_metrics in enumerate(pages):
        nrows, ncols = layout_for(len(page_metrics), cfg)
        fw, fh = figsize_inches(cfg, column)
        # scale height with rows for readability while keeping Nature proportions base
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fw, max(fh, fh * nrows / max(ncols, 1) * 0.85)),
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        flat = axes.ravel()
        for ax in flat[len(page_metrics) :]:
            ax.set_visible(False)

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for ax, metric in zip(flat, page_metrics):
            metric_splits = splits_for_metric(rows, metric)
            _draw_metric_panel(
                ax,
                rows,
                metric,
                models=models,
                splits=metric_splits,
                cfg=cfg,
                x_key=x_key,
                ribbon=ribbon,
                smooth=smooth,
                patience=patience,
                multi_model=multi_model,
                n_seeds=n_seeds,
            )
            h, lab = ax.get_legend_handles_labels()
            for handle, label in zip(h, lab):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=min(len(legend_labels), 4),
                fontsize=cfg["font"]["legend_pt"],
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )
        if title:
            suffix = f" ({page_i + 1}/{len(pages)})" if len(pages) > 1 else ""
            fig.suptitle(f"{title}{suffix}", fontsize=cfg["font"]["title_pt"], y=1.06)
        fig.tight_layout()
        stem = idx.next_stem(outdir, f"learning_curves_p{page_i + 1}" if len(pages) > 1 else "learning_curves")
        written.extend(save_figure(fig, stem, dpi))
        plt.close(fig)

    # --- Per-metric figures ---
    for metric in metrics:
        metric_splits = splits_for_metric(rows, metric)
        if not metric_splits:
            continue
        fw, fh = figsize_inches(cfg, "single")
        fig, ax = plt.subplots(1, 1, figsize=(fw * 1.15, fh * 1.05))
        _draw_metric_panel(
            ax,
            rows,
            metric,
            models=models,
            splits=metric_splits,
            cfg=cfg,
            x_key=x_key,
            ribbon=ribbon,
            smooth=smooth,
            patience=patience,
            multi_model=multi_model,
            n_seeds=n_seeds,
            show_legend=True,
        )
        ax.set_title(metric.replace("_", " "), fontsize=cfg["font"]["title_pt"])
        fig.tight_layout()
        stem = idx.next_stem(outdir, metric)
        written.extend(save_figure(fig, stem, dpi))
        plt.close(fig)

    return written


def _draw_metric_panel(
    ax: Any,
    rows: list[dict[str, Any]],
    metric: str,
    *,
    models: list[str],
    splits: list[str],
    cfg: dict[str, Any],
    x_key: str,
    ribbon: str,
    smooth: bool,
    patience: int | None,
    multi_model: bool,
    n_seeds: int,
    show_legend: bool = False,
) -> None:
    for mi, model in enumerate(models):
        for split in splits:
            if n_seeds >= 2:
                x, mean, _med, std, ci = aggregate_seeds(
                    rows, model=model, split=split, metric=metric, x_key=x_key
                )
                if x.size == 0:
                    continue
                color = model_color(mi, cfg) if multi_model else split_color(split, cfg)
                ls = split_linestyle(split, cfg) if multi_model else split_linestyle(split, cfg)
                label = f"{model}/{split_label(split)}" if multi_model else split_label(split)
                ax.plot(x, mean, color=color, linestyle=ls, linewidth=cfg["line_width"], label=label, zorder=3)
                if ribbon == "ci95":
                    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2, linewidth=0, zorder=2)
                elif ribbon == "std":
                    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=2)
                bi = best_index(list(mean), metric, cfg)
                if bi is not None:
                    ax.scatter([x[bi]], [mean[bi]], s=55, color=color, zorder=4, edgecolors="white", linewidths=0.6)
                    ax.annotate(
                        f"{mean[bi]:.3g}",
                        (x[bi], mean[bi]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=7,
                        color=color,
                    )
                    if patience is not None:
                        stop = min(float(x[bi] + patience), float(x[-1]))
                        ax.axvline(stop, color=color, linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
                if smooth:
                    sm = _lowess(x, mean, cfg.get("lowess_frac", 0.35))
                    if sm is not None:
                        ax.plot(x, sm, color=color, linestyle="-", linewidth=1.0, alpha=0.55, zorder=2)
            else:
                # single seed: raw curves
                seeds = list({r["seed"] for r in rows if r.get("model") == model})
                seed = seeds[0] if seeds else None
                xs, ys = _series(
                    rows, model=model, seed=seed, split=split, metric=metric, x_key=x_key
                )
                if xs.size == 0:
                    continue
                color = model_color(mi, cfg) if multi_model else split_color(split, cfg)
                ls = split_linestyle(split, cfg)
                label = f"{model}/{split_label(split)}" if multi_model else split_label(split)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=ls,
                    linewidth=cfg["line_width"],
                    label=label,
                    zorder=3,
                )
                bi = best_index(list(ys), metric, cfg)
                if bi is not None:
                    ax.scatter(
                        [xs[bi]],
                        [ys[bi]],
                        s=55,
                        color=color,
                        zorder=4,
                        edgecolors="white",
                        linewidths=0.6,
                    )
                    ax.annotate(
                        f"{ys[bi]:.3g}",
                        (xs[bi], ys[bi]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=7,
                        color=color,
                    )
                    if patience is not None:
                        stop = min(float(xs[bi] + patience), float(xs[-1]))
                        ax.axvline(stop, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
                if smooth:
                    sm = _lowess(xs, ys, cfg.get("lowess_frac", 0.35))
                    if sm is not None:
                        ax.plot(xs, sm, color=color, linestyle="-", linewidth=1.0, alpha=0.55)

    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    if show_legend:
        ax.legend(loc="best", fontsize=cfg["font"]["legend_pt"])


def plot_multimodel_split_isolated(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    x_key: str,
    ribbon: str,
    dpi: int,
) -> list[Path]:
    """Multi-model: one figure per metric × split; do not mix train/val."""
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return []
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    written: list[Path] = []
    splits = [s for s in ("train", "validation", "test") if any(r["split"] == s for r in rows)]
    n_seeds = len({r["seed"] for r in rows})
    fw, fh = figsize_inches(cfg, "single")

    for metric in metrics:
        for split in splits:
            fig, ax = plt.subplots(1, 1, figsize=(fw * 1.2, fh * 1.05))
            for mi, model in enumerate(models):
                color = model_color(mi, cfg)
                if n_seeds >= 2:
                    x, mean, _m, std, ci = aggregate_seeds(
                        rows, model=model, split=split, metric=metric, x_key=x_key
                    )
                    if x.size == 0:
                        continue
                    ax.plot(x, mean, color=color, linewidth=cfg["line_width"], label=model)
                    band = ci if ribbon == "ci95" else std
                    if ribbon in ("ci95", "std"):
                        ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.2)
                    bi = best_index(list(mean), metric, cfg)
                    if bi is not None:
                        ax.scatter([x[bi]], [mean[bi]], s=55, color=color, edgecolors="white", zorder=4)
                else:
                    xs, ys = _series(rows, model=model, split=split, metric=metric, x_key=x_key)
                    if xs.size == 0:
                        continue
                    ax.plot(xs, ys, color=color, linewidth=cfg["line_width"], label=model)
                    bi = best_index(list(ys), metric, cfg)
                    if bi is not None:
                        ax.scatter([xs[bi]], [ys[bi]], s=55, color=color, edgecolors="white", zorder=4)
            ax.set_xlabel(x_key.replace("_", " "))
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"{metric.replace('_', ' ')} — {split}", fontsize=cfg["font"]["title_pt"])
            ax.legend(loc="best", fontsize=cfg["font"]["legend_pt"])
            fig.tight_layout()
            stem = idx.next_stem(outdir, f"multimodel_{metric}_{split}")
            written.extend(save_figure(fig, stem, dpi))
            plt.close(fig)
    return written


def final_values_by_group(
    rows: list[dict[str, Any]], metric: str, split: str
) -> dict[str, list[float]]:
    """model -> list of last-epoch values across seeds."""
    out: dict[str, list[float]] = defaultdict(list)
    models = sorted({r["model"] for r in rows})
    for model in models:
        seeds = sorted({r["seed"] for r in rows if r["model"] == model}, key=lambda s: str(s))
        for seed in seeds:
            pts = [
                r
                for r in rows
                if r["model"] == model
                and r["seed"] == seed
                and r["metric"] == metric
                and r["split"] == split
                and r.get("epoch") is not None
            ]
            if not pts:
                continue
            last_ep = max(r["epoch"] for r in pts)
            vals = [r["value"] for r in pts if r["epoch"] == last_ep]
            if vals:
                out[model].append(float(vals[-1]))
    return dict(out)


def plot_final_performance(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    # Prefer validation for ranking; fallback test then train
    split = next(
        (s for s in ("validation", "test", "train") if any(r["split"] == s for r in rows)),
        None,
    )
    if split is None:
        return []
    preferred_final = {
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
        "gene_wise_pearson",
        "sample_wise_pearson",
        "loss",
        "accuracy",
        "auc",
        "f1",
    }
    plot_metrics = [
        m
        for m in metrics
        if m in preferred_final
        and m not in ("elapsed_sec", "lr", "grad_norm")
        and any(r["metric"] == m and r["split"] == split for r in rows)
    ]
    if not plot_metrics:
        # fall back to whatever exists on the ranking split
        plot_metrics = [
            m
            for m in metrics
            if m not in ("elapsed_sec", "lr", "grad_norm")
            and any(r["metric"] == m and r["split"] == split for r in rows)
        ]
    if not plot_metrics:
        return []

    written: list[Path] = []
    fw, fh = figsize_inches(cfg, "double")
    n = len(plot_metrics)
    nrows, ncols = layout_for(min(n, 9), cfg)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)

    for ax, metric in zip(flat, plot_metrics):
        groups = final_values_by_group(rows, metric, split)
        if not groups:
            continue
        # order best→worst
        stats = []
        for model, vals in groups.items():
            arr = np.asarray(vals, dtype=float)
            stats.append((model, float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) if len(arr) > 1 else 0.0), arr))
        reverse = not is_lower_better(metric, cfg)
        stats.sort(key=lambda t: t[1], reverse=reverse)
        names = [t[0] for t in stats]
        means = [t[1] for t in stats]
        errs = [t[2] for t in stats]
        y = np.arange(len(names))
        colors = [model_color(i, cfg) for i in range(len(names))]
        ax.barh(y, means, xerr=errs if any(e > 0 for e in errs) else None, color=colors, edgecolor="white", height=0.7)
        for yi, mval in zip(y, means):
            ax.text(mval, yi, f"  {mval:.3g}", va="center", ha="left", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} ({split})", fontsize=cfg["font"]["title_pt"])
        ax.invert_yaxis()

    fig.suptitle(f"Final performance — {split}", fontsize=cfg["font"]["title_pt"], y=1.02)
    fig.tight_layout()
    stem = idx.next_stem(outdir, "final_performance")
    written.extend(save_figure(fig, stem, dpi))
    plt.close(fig)
    return written


def plot_seed_variability(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    dpi: int,
) -> list[Path]:
    seeds = {r["seed"] for r in rows}
    if len(seeds) < 2:
        return []
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    split = next(
        (s for s in ("validation", "test", "train") if any(r["split"] == s for r in rows)),
        "validation",
    )
    metrics = [m for m in metrics if m not in ("elapsed_sec", "lr", "grad_norm")][:6]
    if not metrics:
        return []
    fw, fh = figsize_inches(cfg, "double")
    nrows, ncols = layout_for(len(metrics), cfg)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(metrics) :]:
        ax.set_visible(False)
    models = sorted({r["model"] for r in rows})
    for ax, metric in zip(flat, metrics):
        data = []
        labels = []
        for model in models:
            groups = final_values_by_group(rows, metric, split).get(model, [])
            if groups:
                data.append(groups)
                labels.append(model)
        if not data:
            continue
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#56B4E9")
            body.set_alpha(0.35)
        ax.boxplot(
            data,
            widths=0.25,
            showfliers=True,
            medianprops={"color": "#D55E00", "linewidth": 1.5},
            boxprops={"color": "#333333"},
            whiskerprops={"color": "#333333"},
            capprops={"color": "#333333"},
            flierprops={"marker": "o", "markersize": 3},
        )
        for i, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color="#0072B2", zorder=3, alpha=0.85)
            ax.scatter([i], [np.mean(vals)], marker="D", s=28, color="#000000", zorder=4, label="mean" if i == 1 else None)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} ({split})", fontsize=cfg["font"]["title_pt"])
    fig.suptitle("Seed variability (final epoch)", fontsize=cfg["font"]["title_pt"], y=1.02)
    fig.tight_layout()
    stem = idx.next_stem(outdir, "seed_variability")
    written = save_figure(fig, stem, dpi)
    plt.close(fig)
    return written


def plot_metric_correlation(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    dpi: int,
) -> list[Path]:
    # Need enough epochs
    epochs = sorted({r["epoch"] for r in rows if r.get("epoch") is not None})
    if len(epochs) < 3:
        return []
    metrics = [m for m in metrics if m not in ("lr", "grad_norm")]
    if len(metrics) < 2:
        return []
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    # Build matrix: for each epoch, mean across runs of each metric on validation (else train)
    split = next(
        (s for s in ("validation", "train", "test") if any(r["split"] == s for r in rows)),
        "train",
    )
    series = {}
    for metric in metrics:
        vals = []
        for ep in epochs:
            ep_vals = [
                r["value"]
                for r in rows
                if r["epoch"] == ep and r["metric"] == metric and r["split"] == split
            ]
            vals.append(float(np.mean(ep_vals)) if ep_vals else np.nan)
        series[metric] = np.asarray(vals, dtype=float)
    keys = list(series.keys())
    M = np.column_stack([series[k] for k in keys])
    # Spearman via rank pearson
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks

    n = len(keys)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(M[:, i]) & ~np.isnan(M[:, j])
            if mask.sum() < 3:
                c = np.nan
            else:
                ri = rankdata(M[mask, i])
                rj = rankdata(M[mask, j])
                c = float(np.corrcoef(ri, rj)[0, 1])
            corr[i, j] = corr[j, i] = c

    fw, fh = figsize_inches(cfg, "single")
    fig, ax = plt.subplots(1, 1, figsize=(max(fw, 0.45 * n + 1.5), max(fh, 0.45 * n + 1.5)))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [k.replace("_", " ") for k in keys]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if val == val:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="#111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Spearman metric correlation ({split})", fontsize=cfg["font"]["title_pt"])
    fig.tight_layout()
    stem = idx.next_stem(outdir, "metric_correlation")
    written = save_figure(fig, stem, dpi)
    plt.close(fig)
    return written


def plot_generalization_gap(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    x_key: str,
    dpi: int,
) -> list[Path]:
    if not any(r["split"] == "train" for r in rows) or not any(r["split"] == "validation" for r in rows):
        return []
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    metrics = [m for m in metrics if m not in ("lr", "elapsed_sec", "grad_norm")]
    if not metrics:
        return []
    models = sorted({r["model"] for r in rows})
    fw, fh = figsize_inches(cfg, "double")
    nrows, ncols = layout_for(min(len(metrics), 9), cfg)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(metrics) :]:
        ax.set_visible(False)
    for ax, metric in zip(flat, metrics):
        for mi, model in enumerate(models):
            # gap at each epoch: mean_val - mean_train (across seeds)
            x_v, mean_v, *_ = aggregate_seeds(
                rows, model=model, split="validation", metric=metric, x_key=x_key
            )
            x_t, mean_t, *_ = aggregate_seeds(
                rows, model=model, split="train", metric=metric, x_key=x_key
            )
            if x_v.size == 0 or x_t.size == 0:
                # try single series
                xs_v, ys_v = _series(rows, model=model, split="validation", metric=metric, x_key=x_key)
                xs_t, ys_t = _series(rows, model=model, split="train", metric=metric, x_key=x_key)
                common = sorted(set(xs_v.tolist()) & set(xs_t.tolist()))
                if not common:
                    continue
                map_v = dict(zip(xs_v.tolist(), ys_v.tolist()))
                map_t = dict(zip(xs_t.tolist(), ys_t.tolist()))
                x = np.array(common)
                gap = np.array([map_v[c] - map_t[c] for c in common])
            else:
                common = sorted(set(x_v.tolist()) & set(x_t.tolist()))
                map_v = dict(zip(x_v.tolist(), mean_v.tolist()))
                map_t = dict(zip(x_t.tolist(), mean_t.tolist()))
                x = np.array(common)
                gap = np.array([map_v[c] - map_t[c] for c in common])
            ax.plot(
                x,
                gap,
                color=model_color(mi, cfg),
                linewidth=cfg["line_width"],
                label=model if len(models) > 1 else "val − train",
            )
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel(f"gap ({metric.replace('_', ' ')})")
        ax.set_title(metric.replace("_", " "), fontsize=cfg["font"]["title_pt"])
        if len(models) > 1:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Generalization gap (validation − train)", fontsize=cfg["font"]["title_pt"], y=1.02)
    fig.tight_layout()
    stem = idx.next_stem(outdir, "generalization_gap")
    written = save_figure(fig, stem, dpi)
    plt.close(fig)
    return written


def plot_early_stopping(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    x_key: str,
    patience: int | None,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    metric = "loss" if any(r["metric"] == "loss" for r in rows) else order_metrics({r["metric"] for r in rows}, cfg)[0]
    split = "validation" if any(r["split"] == "validation" for r in rows) else "train"
    models = sorted({r["model"] for r in rows})
    fw, fh = figsize_inches(cfg, "single")
    fig, ax = plt.subplots(1, 1, figsize=(fw * 1.3, fh * 1.1))
    for mi, model in enumerate(models):
        xs, ys = _series(rows, model=model, split=split, metric=metric, x_key=x_key)
        if xs.size == 0:
            x, mean, *_ = aggregate_seeds(rows, model=model, split=split, metric=metric, x_key=x_key)
            xs, ys = x, mean
        if xs.size == 0:
            continue
        color = model_color(mi, cfg) if len(models) > 1 else split_color(split, cfg)
        ax.plot(xs, ys, color=color, linewidth=cfg["line_width"], label=f"{model} {split} {metric}")
        bi = best_index(list(ys), metric, cfg)
        if bi is not None:
            best_ep = xs[bi]
            ax.scatter([best_ep], [ys[bi]], s=70, color=color, zorder=4, edgecolors="white", label="best epoch")
            ax.axvline(best_ep, color=color, linestyle=":", linewidth=1.2, alpha=0.9)
            if patience is not None:
                stop = min(float(best_ep + patience), float(xs[-1]))
                ax.axvspan(best_ep, stop, color=color, alpha=0.12, label="patience interval")
                ax.axvline(stop, color=color, linestyle="--", linewidth=1.2, label="training stopped")
            ax.annotate(
                f"checkpoint@{best_ep:g}\n{ys[bi]:.3g}",
                (best_ep, ys[bi]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=7,
                color=color,
            )
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Early stopping / best checkpoint", fontsize=cfg["font"]["title_pt"])
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    stem = idx.next_stem(outdir, "early_stopping")
    written = save_figure(fig, stem, dpi)
    plt.close(fig)
    return written


def plot_learning_rate(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    x_key: str,
    dpi: int,
) -> list[Path]:
    if not any(r["metric"] == "lr" for r in rows):
        return []
    import matplotlib.pyplot as plt

    apply_pub_style(cfg)
    fw, fh = figsize_inches(cfg, "single")
    fig, ax = plt.subplots(1, 1, figsize=(fw * 1.2, fh))
    models = sorted({r["model"] for r in rows})
    for mi, model in enumerate(models):
        xs, ys = _series(rows, model=model, split="_run", metric="lr", x_key=x_key)
        if xs.size == 0:
            # sometimes logged under train
            xs, ys = _series(rows, model=model, split="train", metric="lr", x_key=x_key)
        if xs.size == 0:
            continue
        ax.plot(
            xs,
            ys,
            color=model_color(mi, cfg),
            linewidth=cfg["line_width"],
            label=model if len(models) > 1 else "lr",
        )
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule", fontsize=cfg["font"]["title_pt"])
    if len(models) > 1:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    stem = idx.next_stem(outdir, "learning_rate")
    written = save_figure(fig, stem, dpi)
    plt.close(fig)
    return written


def build_training_summary(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    patience: int | None,
    configs: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    runs = sorted({r["run"] for r in rows})
    for run in runs:
        run_rows = [r for r in rows if r["run"] == run]
        model = run_rows[0]["model"]
        seed = run_rows[0]["seed"]
        epochs = [r["epoch"] for r in run_rows if r.get("epoch") is not None]
        n_epochs = int(max(epochs) - min(epochs) + 1) if epochs else 0
        duration = None
        dur_pts = [r["value"] for r in run_rows if r["metric"] == "elapsed_sec"]
        if dur_pts:
            duration = float(max(dur_pts))
        # primary metric: loss on validation else train
        for metric in order_metrics({r["metric"] for r in run_rows}, cfg):
            for split in ("validation", "test", "train"):
                xs, ys = _series(run_rows, run=run, split=split, metric=metric, x_key="epoch")
                if xs.size == 0:
                    continue
                bi = best_index(list(ys), metric, cfg)
                if bi is None:
                    continue
                best_ep = float(xs[bi])
                best_val = float(ys[bi])
                final_val = float(ys[-1]) if not np.isnan(ys[-1]) else float("nan")
                early = best_ep + patience if patience is not None else None
                summary.append(
                    {
                        "run": run,
                        "model": model,
                        "seed": seed,
                        "split": split,
                        "metric": metric,
                        "best_epoch": best_ep,
                        "best_metric": best_val,
                        "final_metric": final_val,
                        "early_stopping_epoch": early,
                        "training_duration_sec": duration,
                        "number_of_epochs": n_epochs,
                        "model_name_meta": (configs.get(run) or {}).get("model_name"),
                        "batch_size": (configs.get(run) or {}).get("batch_size"),
                    }
                )
    return summary


def write_summary_md(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Training summary",
        "",
        "| run | model | seed | split | metric | best epoch | best | final | early stop | duration (s) | epochs |",
        "|-----|-------|------|-------|--------|------------|------|-------|------------|--------------|--------|",
    ]
    for r in summary:
        dur = "" if r["training_duration_sec"] is None else f"{r['training_duration_sec']:.3f}"
        early = "" if r["early_stopping_epoch"] is None else str(r["early_stopping_epoch"])
        lines.append(
            f"| {r['run']} | {r['model']} | {r['seed']} | {r['split']} | {r['metric']} | "
            f"{r['best_epoch']} | {r['best_metric']:.6g} | {r['final_metric']:.6g} | "
            f"{early} | {dur} | {r['number_of_epochs']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_manuscript(outdir: Path, written: list[Path]) -> list[Path]:
    ms = outdir / "manuscript"
    ms.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    # Prefer key figure stems
    keys = (
        "learning_curves",
        "loss",
        "final_performance",
        "generalization_gap",
        "early_stopping",
        "metric_correlation",
        "seed_variability",
    )
    for p in written:
        if p.suffix.lower() not in {".pdf", ".svg", ".png"}:
            continue
        if any(k in p.stem for k in keys):
            dest = ms / p.name
            shutil.copy2(p, dest)
            out.append(dest)
    return out


def _pick_log_file(directory: Path) -> Path | None:
    """Prefer caduceus-style train_metrics.jsonl inside a run or logs/ dir."""
    candidates = [
        directory / "train_metrics.jsonl",
        directory / "logs" / "train_metrics.jsonl",
        directory / "metrics.log",
        directory / "logs" / "metrics.log",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c
    # any *.jsonl / *.log with epoch content
    for pat in ("*.jsonl", "*.log", "logs/*.jsonl", "logs/*.log"):
        for hit in sorted(directory.glob(pat)):
            if hit.is_file() and hit.stat().st_size > 0:
                return hit
    return None


def resolve_one_input(spec: str | Path) -> Path:
    """Resolve a log file, glob hit, or run/logs directory to a single log file."""
    path = Path(spec)
    if any(ch in str(spec) for ch in "*?["):
        hits = (
            sorted(path.parent.glob(path.name))
            if path.is_absolute()
            else sorted(Path.cwd().glob(str(spec)))
        )
        files = [h for h in hits if h.is_file()]
        if not files:
            raise FileNotFoundError(f"No files matched glob: {spec}")
        return files[0]
    if path.is_file():
        return path
    if path.is_dir():
        picked = _pick_log_file(path)
        if picked is None:
            raise FileNotFoundError(
                f"No train_metrics.jsonl / metrics.log under directory: {path}"
            )
        return picked
    raise FileNotFoundError(f"Log input not found: {spec}")


def _pick_log_file(directory: Path) -> Path | None:
    """Prefer caduceus-style train_metrics.jsonl inside a run or logs/ dir."""
    candidates = [
        directory / "train_metrics.jsonl",
        directory / "logs" / "train_metrics.jsonl",
        directory / "metrics.log",
        directory / "logs" / "metrics.log",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c
    for pat in ("*.jsonl", "*.log", "logs/*.jsonl", "logs/*.log"):
        for hit in sorted(directory.glob(pat)):
            if hit.is_file() and hit.stat().st_size > 0:
                return hit
    return None


def resolve_one_input(spec: str | Path) -> Path:
    """Resolve a log file, glob hit, or run/logs directory to a single log file."""
    path = Path(spec)
    s = str(spec)
    if any(ch in s for ch in "*?["):
        hits = (
            sorted(path.parent.glob(path.name))
            if path.is_absolute()
            else sorted(Path.cwd().glob(s))
        )
        files = [h for h in hits if h.is_file()]
        if not files:
            raise FileNotFoundError(f"No files matched glob: {spec}")
        return files[0]
    if path.is_file():
        return path
    if path.is_dir():
        picked = _pick_log_file(path)
        if picked is None:
            raise FileNotFoundError(
                f"No train_metrics.jsonl / metrics.log under directory: {path}"
            )
        return picked
    raise FileNotFoundError(f"Log input not found: {spec}")


def resolve_logs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for p in patterns:
        path = Path(p)
        if any(ch in p for ch in "*?["):
            hits = (
                sorted(path.parent.glob(path.name))
                if path.is_absolute()
                else sorted(Path.cwd().glob(p))
            )
            for h in hits:
                try:
                    resolved = resolve_one_input(h) if h.is_dir() else h
                except FileNotFoundError:
                    continue
                if not resolved.is_file():
                    continue
                rp = resolved.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                paths.append(resolved)
        else:
            resolved = resolve_one_input(p)
            rp = resolved.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            paths.append(resolved)
    return paths


def resolve_model_inputs(
    models: list[str],
    *,
    runs_root: Path | None = None,
) -> list[tuple[Path, str]]:
    """Map model names or run dirs → (log_path, model_id)."""
    out: list[tuple[Path, str]] = []
    roots: list[Path] = []
    if runs_root is not None:
        roots.append(runs_root)
    roots.extend([Path("runs/caduceus"), Path("runs"), Path(".")])
    for m in models:
        p = Path(m)
        if p.exists():
            log = resolve_one_input(p)
            mid = p.name if p.is_dir() else p.stem
            if log.parent.name == "logs":
                mid = log.parent.parent.name
            out.append((log, mid))
            continue
        found = None
        for root in roots:
            for cand in (root / m, root / m / "logs"):
                if cand.exists():
                    found = cand
                    break
            if found is not None:
                break
        if found is None:
            raise FileNotFoundError(
                f"Model/run not found: {m!r} (tried under runs/caduceus/, runs/, .)"
            )
        log = resolve_one_input(found)
        out.append((log, m))
    return out


def infer_seed(label: str, config: dict[str, Any] | None, explicit: Any | None) -> Any:
    if explicit is not None:
        return explicit
    if config and "seed" in config:
        return config["seed"]
    m = re.search(r"(?:seed|s)[_-]?(\d+)", label, re.I)
    if m:
        return int(m.group(1))
    return 0


def infer_model(label: str, config: dict[str, Any] | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if config and config.get("model_name"):
        name = str(config["model_name"]).rstrip("/").split("/")[-1]
        return name
    if "__" in label:
        return label.split("__", 1)[0]
    return label


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "logs",
        nargs="*",
        help="Log file(s), globs, or run/log directories (optional if --models set)",
    )
    ap.add_argument(
        "--logs",
        dest="logs_opt",
        nargs="+",
        default=None,
        help="Same as positional logs (explicit flag)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more model run dirs/names to compare "
        "(e.g. runs/caduceus/smoke_M1). Resolves logs automatically.",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/caduceus"),
        help="Root for resolving --models short names",
    )
    ap.add_argument("-o", "--outdir", type=Path, default=Path("figures/train-viz"))
    ap.add_argument("--x", choices=("epoch", "global_step"), default="epoch")
    ap.add_argument("--title", default=None)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--model", action="append", default=[], help="Model id per log")
    ap.add_argument("--seed", action="append", default=[], help="Seed per log")
    ap.add_argument("--smooth", action="store_true", help="Add LOWESS overlay (raw kept)")
    ap.add_argument("--ribbon", choices=("ci95", "std", "none"), default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--column", choices=("single", "double"), default="double")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    dpi = int(args.dpi or cfg["dpi_png"])
    ribbon = args.ribbon or cfg.get("ribbon_default", "ci95")

    log_paths: list[Path] = []
    labels: list[str] = []
    models_arg = list(args.model)
    seeds_arg = list(args.seed)

    if args.models:
        pairs = resolve_model_inputs(args.models, runs_root=args.runs_root)
        for log, mid in pairs:
            log_paths.append(log)
            labels.append(mid)
            if len(models_arg) < len(log_paths):
                # fill model ids from --models when --model not given per-log
                pass
        if not models_arg:
            models_arg = [mid for _, mid in pairs]

    positional = list(args.logs_opt or []) + list(args.logs or [])
    if positional:
        resolved = resolve_logs(positional)
        existing = {x.resolve() for x in log_paths}
        for p in resolved:
            if p.resolve() in existing:
                continue
            log_paths.append(p)
            labels.append(p.stem if p.parent.name != "logs" else p.parent.parent.name)

    if args.label:
        labels = list(args.label)
        if len(labels) != len(log_paths):
            print("ERROR: --label count must match logs", file=sys.stderr)
            return 2
    elif len(labels) != len(log_paths):
        labels = [
            (p.parent.parent.name if p.parent.name == "logs" else p.stem)
            for p in log_paths
        ]

    if not log_paths:
        print(
            "ERROR: provide logs and/or --models (run dirs). "
            "Example: python -m src.train_viz --models runs/caduceus/smoke_M1 "
            "-o figures/train-viz",
            file=sys.stderr,
        )
        return 2

    if models_arg and len(models_arg) != len(log_paths):
        print("ERROR: --model count must match logs", file=sys.stderr)
        return 2
    if seeds_arg and len(seeds_arg) != len(log_paths):
        print("ERROR: --seed count must match logs", file=sys.stderr)
        return 2

    all_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_metrics: set[str] = set()
    all_nan_only: set[str] = set()
    configs: dict[str, dict[str, Any] | None] = {}
    summaries_txt: list[str] = []

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        config, epochs = parse_log(path)
        model = infer_model(label, config, models_arg[i] if models_arg else None)
        seed_raw = seeds_arg[i] if seeds_arg else None
        seed = infer_seed(label, config, int(seed_raw) if seed_raw is not None else None)
        rows, metrics, splits, nan_only = flatten_epochs(label, epochs, model=model, seed=seed)
        all_rows.extend(rows)
        all_metrics |= metrics
        all_nan_only |= nan_only
        configs[label] = config
        summaries_txt.append(
            f"- `{path}` → run `{label}` model=`{model}` seed=`{seed}`: "
            f"{len(epochs)} epochs; metrics={sorted(metrics)}; splits={sorted(splits)}"
        )
        if nan_only:
            summaries_txt.append(
                f"  (omitted all-NaN metrics: {sorted(nan_only)})"
            )

    n_seeds = len({r["seed"] for r in all_rows})
    n_models = len({r["model"] for r in all_rows})
    errors = qc_check(
        dpi=dpi, cfg=cfg, n_seeds=n_seeds, ribbon=ribbon, used_pub_style=True
    )
    if errors:
        print("ERROR: publication QC failed: " + "; ".join(errors), file=sys.stderr)
        return 3

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Persist config
    dest_cfg = args.outdir / "visualization_config.yaml"
    if args.config.is_file():
        shutil.copy2(args.config, dest_cfg)
    elif yaml is not None:
        dest_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        dest_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    fields = ["run", "model", "seed", "epoch", "global_step", "split", "metric", "value"]
    write_csv(all_rows, args.outdir / "train_metrics.csv", fields=fields)

    metrics_ordered = metrics_with_data(all_rows, order_metrics(all_metrics, cfg))
    omitted = sorted(set(order_metrics(all_metrics, cfg)) - set(metrics_ordered))
    if all_nan_only:
        print(
            "NOTE: omitted all-NaN metrics (no finite values to plot): "
            + ", ".join(sorted(all_nan_only)),
            file=sys.stderr,
        )
    if omitted:
        print(
            "NOTE: omitted metrics with no finite values: " + ", ".join(omitted),
            file=sys.stderr,
        )
    idx = FigureIndex()
    written: list[Path] = [dest_cfg, args.outdir / "train_metrics.csv"]

    import matplotlib

    matplotlib.use("Agg")

    written.extend(
        plot_learning_curves(
            all_rows,
            metrics_ordered,
            cfg,
            args.outdir,
            idx,
            x_key=args.x,
            title=args.title,
            column=args.column,
            ribbon=ribbon if n_seeds >= 2 else "none",
            smooth=args.smooth,
            patience=args.patience,
            dpi=dpi,
        )
    )
    if n_models >= 2:
        written.extend(
            plot_multimodel_split_isolated(
                all_rows,
                metrics_ordered,
                cfg,
                args.outdir,
                idx,
                x_key=args.x,
                ribbon=ribbon if n_seeds >= 2 else "none",
                dpi=dpi,
            )
        )
    written.extend(
        plot_final_performance(all_rows, metrics_ordered, cfg, args.outdir, idx, dpi=dpi)
    )
    written.extend(
        plot_seed_variability(all_rows, metrics_ordered, cfg, args.outdir, idx, dpi=dpi)
    )
    written.extend(
        plot_metric_correlation(all_rows, metrics_ordered, cfg, args.outdir, idx, dpi=dpi)
    )
    written.extend(
        plot_generalization_gap(
            all_rows, metrics_ordered, cfg, args.outdir, idx, x_key=args.x, dpi=dpi
        )
    )
    written.extend(
        plot_early_stopping(
            all_rows,
            cfg,
            args.outdir,
            idx,
            x_key=args.x,
            patience=args.patience,
            dpi=dpi,
        )
    )
    written.extend(
        plot_learning_rate(all_rows, cfg, args.outdir, idx, x_key=args.x, dpi=dpi)
    )

    summary = build_training_summary(all_rows, cfg, patience=args.patience, configs=configs)
    write_csv(
        summary,
        args.outdir / "training_summary.csv",
        fields=[
            "run",
            "model",
            "seed",
            "split",
            "metric",
            "best_epoch",
            "best_metric",
            "final_metric",
            "early_stopping_epoch",
            "training_duration_sec",
            "number_of_epochs",
            "model_name_meta",
            "batch_size",
        ],
    )
    write_summary_md(summary, args.outdir / "training_summary.md")
    written.extend(
        [args.outdir / "training_summary.csv", args.outdir / "training_summary.md"]
    )
    written.extend(copy_manuscript(args.outdir, written))

    print("Parsed logs:")
    for s in summaries_txt:
        print(s)
    print(f"models={n_models} seeds={n_seeds} metrics={metrics_ordered}")
    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
