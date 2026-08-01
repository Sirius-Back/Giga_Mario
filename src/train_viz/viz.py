#!/usr/bin/env python3
"""Publication-quality training visualization (Nature Methods / NMI style).

Renders via cnsplots (PDF/SVG/PNG@600dpi) and Altair (HTML + Vega-Lite).
Exports tables and visualization_config.yaml. See visualization_config.yaml.
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


def load_best_checkpoint_meta(run_dir: Path) -> dict[str, Any] | None:
    """Load ``best_model/best_meta.json`` (or final_model copy) if present."""
    run_dir = Path(run_dir)
    for rel in (
        Path("best_model") / "best_meta.json",
        Path("final_model") / "best_meta.json",
    ):
        path = run_dir / rel
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(meta, dict) and meta.get("epoch") is not None:
            return meta
    return None


def resolve_best_epoch_for_log(log_path: Path) -> float | None:
    """Infer selected best-checkpoint epoch from a train metrics log path."""
    log_path = Path(log_path)
    # …/logs/train_metrics*.jsonl → run dir is parent of logs/
    run_dir = log_path.parent.parent if log_path.parent.name == "logs" else log_path.parent
    meta = load_best_checkpoint_meta(run_dir)
    if meta is None:
        return None
    try:
        return float(meta["epoch"])
    except (TypeError, ValueError, KeyError):
        return None


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
            # Skip non-numeric x (e.g. epoch == "final" / ZSV rows in full jsonl).
            x = float(r[x_key])
            val = float(r["value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(val):
            continue
        pts.append((x, val))
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




def build_training_summary(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    patience: int | None,
    configs: dict[str, dict[str, Any] | None],
    best_epochs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    best_epochs = best_epochs or {}
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
        recorded_best = best_epochs.get(model)
        # primary metric: loss on validation else train
        for metric in order_metrics({r["metric"] for r in run_rows}, cfg):
            for split in ("validation", "test", "train"):
                xs, ys = _series(run_rows, run=run, split=split, metric=metric, x_key="epoch")
                if xs.size == 0:
                    continue
                if recorded_best is not None:
                    j = int(np.nanargmin(np.abs(xs.astype(float) - float(recorded_best))))
                    best_ep = float(xs[j])
                    best_val = float(ys[j])
                else:
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
                        "selected_final_checkpoint": recorded_best is not None,
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
    """Prefer epoch-only metrics when present, else full train_metrics.jsonl."""
    candidates = [
        # Epoch-only avoids epoch=="final" breaking numeric learning-curve x-axes;
        # ZSV is reattached from zero_shot_metrics.json in main().
        directory / "train_metrics_epochs.jsonl",
        directory / "logs" / "train_metrics_epochs.jsonl",
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
    ap.add_argument(
        "--max-epoch",
        type=float,
        default=None,
        help="Clip learning curves to epoch<=N (default: config compare_max_epoch when multi-model)",
    )
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
    all_metrics: set[str] = set()
    all_nan_only: set[str] = set()
    configs: dict[str, dict[str, Any] | None] = {}
    summaries_txt: list[str] = []
    best_epochs: dict[str, float] = {}

    from .plotting import load_zsv_rows

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        config, epochs = parse_log(path)
        model = infer_model(label, config, models_arg[i] if models_arg else None)
        seed_raw = seeds_arg[i] if seeds_arg else None
        seed = infer_seed(label, config, int(seed_raw) if seed_raw is not None else None)
        rows, metrics, splits, nan_only = flatten_epochs(label, epochs, model=model, seed=seed)
        # Attach ZSV point metrics when present beside the log (epochs jsonl omits final).
        train_dir = path.parent.parent if path.parent.name == "logs" else path.parent
        zsv_rows = load_zsv_rows(train_dir, model=model, seed=seed, run=label)
        if zsv_rows:
            rows.extend(zsv_rows)
            metrics |= {r["metric"] for r in zsv_rows}
            splits.add("zero_shot")
        all_rows.extend(rows)
        all_metrics |= metrics
        all_nan_only |= nan_only
        configs[label] = config
        be = resolve_best_epoch_for_log(path)
        if be is not None:
            best_epochs[model] = be
            summaries_txt.append(
                f"- `{path}` → run `{label}` model=`{model}` seed=`{seed}`: "
                f"{len(epochs)} epochs; metrics={sorted(metrics)}; splits={sorted(splits)}; "
                f"final/best_epoch={be:g}"
                + (f"; zsv_metrics={len(zsv_rows)}" if zsv_rows else "")
            )
        else:
            summaries_txt.append(
                f"- `{path}` → run `{label}` model=`{model}` seed=`{seed}`: "
                f"{len(epochs)} epochs; metrics={sorted(metrics)}; splits={sorted(splits)}"
                + (f"; zsv_metrics={len(zsv_rows)}" if zsv_rows else "")
            )
        if nan_only:
            summaries_txt.append(
                f"  (omitted all-NaN metrics: {sorted(nan_only)})"
            )

    n_seeds = len({r["seed"] for r in all_rows})
    n_models = len({r["model"] for r in all_rows})
    from .plotting import qc_check

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
    from .plotting import (
        FigureIndex,
        plot_early_stopping,
        plot_final_performance,
        plot_generalization_gap,
        plot_learning_curves,
        plot_learning_rate,
        plot_metric_correlation,
        plot_multimodel_split_isolated,
        plot_seed_variability,
        plot_zsv_barplots,
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
            best_epochs=best_epochs,
            max_epoch=args.max_epoch,
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
                max_epoch=args.max_epoch,
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
            best_epochs=best_epochs,
        )
    )
    written.extend(
        plot_learning_rate(all_rows, cfg, args.outdir, idx, x_key=args.x, dpi=dpi)
    )
    written.extend(plot_zsv_barplots(all_rows, cfg, args.outdir, idx, dpi=dpi))

    summary = build_training_summary(
        all_rows,
        cfg,
        patience=args.patience,
        configs=configs,
        best_epochs=best_epochs,
    )
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
            "selected_final_checkpoint",
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
