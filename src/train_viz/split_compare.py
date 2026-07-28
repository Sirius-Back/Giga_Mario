#!/usr/bin/env python3
"""Compare final train / validation / test / ZSV metrics (cnsplots + Altair).

Collects scalars from a completed train run directory and renders publication
bars + interactive Altair HTML for side-by-side split comparison.

Sources (first hit wins per field):
  - ``logs/zero_shot_metrics.json`` or ``zero_shot_metrics.json`` → zsv
  - ``metrics_summary.json`` → test (+ optional val)
  - ``logs/train_metrics.jsonl`` last finite epoch → train / validation
  - Lightning ``**/metrics.csv`` last epoch → train_loss / val_*

Example::

  conda run -n caduceus_env python -m src.train_viz.split_compare \\
    --run-dir run/run0/direct -o run/run0/direct/figures/split_compare
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .plotting import (
    FigureIndex,
    apply_pub_style,
    cns_layout_px,
    save_altair_chart,
    save_cns_figure,
)
from .viz import SPLIT_ORDER, _load_config, split_color, split_label

SPLIT_CANON = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
    "zero_shot": "zero_shot",
    "zero-shot": "zero_shot",
    "zero_shot_validation": "zero_shot",
    "zero-shot-validation": "zero_shot",
    "zsv": "zero_shot",
}

COMPARE_METRICS = (
    "loss",
    "pearson",
    "spearman",
    "mse",
    "rmse",
    "mae",
    "r2",
)


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _canon_split(name: str) -> str | None:
    return SPLIT_CANON.get(str(name).strip().lower())


def _add(
    rows: list[dict[str, Any]],
    *,
    split: str,
    metric: str,
    value: Any,
    model: str,
    source: str,
) -> None:
    x = _finite(value)
    if x is None:
        return
    rows.append(
        {
            "model": model,
            "split": split,
            "split_label": split_label(split),
            "metric": metric,
            "value": x,
            "source": source,
        }
    )


def _from_metrics_blob(
    blob: dict[str, Any],
    *,
    split: str,
    model: str,
    source: str,
    rows: list[dict[str, Any]],
) -> None:
    for key, raw in blob.items():
        if key in {"n", "predictions_path", "pred_aggregation"}:
            continue
        metric = "loss" if key in {"val_loss", "train_loss", "test_loss"} else key
        if metric.startswith("val_"):
            metric = metric[len("val_") :]
        if metric.startswith("train_"):
            metric = metric[len("train_") :]
        _add(rows, split=split, metric=metric, value=raw, model=model, source=source)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _collect_from_jsonl(path: Path, *, model: str, rows: list[dict[str, Any]]) -> None:
    if not path.is_file():
        return
    last_by_split: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("smoke"):
            continue
        for key, payload in rec.items():
            split = _canon_split(str(key))
            if split is None or not isinstance(payload, dict):
                continue
            last_by_split[split] = payload
    for split, payload in last_by_split.items():
        _from_metrics_blob(
            payload, split=split, model=model, source=str(path), rows=rows
        )


def _collect_from_lightning_csv(run_dir: Path, *, model: str, rows: list[dict[str, Any]]) -> None:
    csvs = sorted(run_dir.rglob("metrics.csv"))
    if not csvs:
        return
    path = csvs[0]
    with path.open(newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    if not table:
        return
    # Prefer highest epoch with values
    best_train: dict[str, float] = {}
    best_val: dict[str, float] = {}
    best_ep = -1
    for row in table:
        ep_raw = row.get("epoch")
        if ep_raw in (None, ""):
            continue
        try:
            ep = int(float(ep_raw))
        except ValueError:
            continue
        if ep < best_ep:
            continue
        if ep > best_ep:
            best_ep = ep
            best_train = {}
            best_val = {}
        if row.get("train_loss") not in (None, ""):
            v = _finite(row["train_loss"])
            if v is not None:
                best_train["loss"] = v
        for col, metric in (
            ("val_loss", "loss"),
            ("val_pearson", "pearson"),
            ("val_spearman", "spearman"),
            ("val_mse", "mse"),
            ("val_rmse", "rmse"),
            ("val_mae", "mae"),
        ):
            if row.get(col) not in (None, ""):
                v = _finite(row[col])
                if v is not None:
                    best_val[metric] = v
    for metric, value in best_train.items():
        _add(
            rows,
            split="train",
            metric=metric,
            value=value,
            model=model,
            source=str(path),
        )
    for metric, value in best_val.items():
        _add(
            rows,
            split="validation",
            metric=metric,
            value=value,
            model=model,
            source=str(path),
        )


def collect_split_metrics(
    run_dir: Path,
    *,
    model: str | None = None,
) -> pd.DataFrame:
    """Build a long-form table of final metrics by split for one run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir missing: {run_dir}")
    model_name = model or run_dir.name
    rows: list[dict[str, Any]] = []

    # Prefer structured summary artifacts
    summary = _load_json(run_dir / "metrics_summary.json")
    if summary:
        if isinstance(summary.get("test"), dict):
            _from_metrics_blob(
                summary["test"],
                split="test",
                model=model_name,
                source=str(run_dir / "metrics_summary.json"),
                rows=rows,
            )
        if isinstance(summary.get("validation"), dict):
            _from_metrics_blob(
                summary["validation"],
                split="validation",
                model=model_name,
                source=str(run_dir / "metrics_summary.json"),
                rows=rows,
            )
        if isinstance(summary.get("train"), dict):
            _from_metrics_blob(
                summary["train"],
                split="train",
                model=model_name,
                source=str(run_dir / "metrics_summary.json"),
                rows=rows,
            )

    zsv_path = run_dir / "logs" / "zero_shot_metrics.json"
    if not zsv_path.is_file():
        zsv_path = run_dir / "zero_shot_metrics.json"
    zsv = _load_json(zsv_path)
    if zsv:
        blob = zsv.get("metrics") if isinstance(zsv.get("metrics"), dict) else zsv
        if isinstance(blob, dict):
            _from_metrics_blob(
                blob,
                split="zero_shot",
                model=model_name,
                source=str(zsv_path),
                rows=rows,
            )

    _collect_from_jsonl(run_dir / "logs" / "train_metrics.jsonl", model=model_name, rows=rows)
    _collect_from_jsonl(run_dir / "train_metrics.jsonl", model=model_name, rows=rows)

    # Fill train/val gaps from Lightning CSV when jsonl was overwritten by smoke
    have = {(r["split"], r["metric"]) for r in rows}
    if ("train", "loss") not in have or ("validation", "loss") not in have:
        before = len(rows)
        _collect_from_lightning_csv(run_dir, model=model_name, rows=rows)
        # Drop duplicates keeping first (summary/json preferred over csv)
        seen: set[tuple[str, str]] = set()
        dedup: list[dict[str, Any]] = []
        for r in rows:
            key = (r["split"], r["metric"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        rows = dedup
        _ = before

    if not rows:
        raise ValueError(
            f"No split metrics found under {run_dir} "
            "(need metrics_summary.json, zero_shot_metrics.json, "
            "train_metrics.jsonl, and/or Lightning metrics.csv)"
        )
    df = pd.DataFrame(rows)
    # Stable split order
    order = [s for s in SPLIT_ORDER if s in set(df["split"])]
    df["split"] = pd.Categorical(df["split"], categories=order, ordered=True)
    df = df.sort_values(["metric", "split", "model"]).reset_index(drop=True)
    return df


def plot_split_comparison(
    df: pd.DataFrame,
    outdir: Path,
    *,
    cfg: dict[str, Any] | None = None,
    title: str = "Split comparison (train / val / test / ZSV)",
    dpi: int | None = None,
) -> list[Path]:
    """Render cnsplots PDF/SVG/PNG + Altair HTML/VL for split×metric bars."""
    import altair as alt
    import cnsplots as cns

    if df.empty:
        raise ValueError("empty metrics frame")
    cfg = cfg or _load_config()
    dpi = int(dpi or cfg.get("dpi_png", 600))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prefer shared regression metrics; keep any others present
    metrics = [m for m in COMPARE_METRICS if m in set(df["metric"])]
    extras = sorted(set(df["metric"]) - set(metrics))
    metrics.extend(extras)
    if not metrics:
        raise ValueError("no plottable metrics in frame")

    plot_df = df[df["metric"].isin(metrics)].copy()
    plot_df["split_label"] = plot_df["split"].map(lambda s: split_label(str(s)))

    # Colorblind-safe Okabe–Ito via project palette
    splits = [s for s in SPLIT_ORDER if s in set(plot_df["split"].astype(str))]
    palette = {s: split_color(s, cfg) for s in splits}

    written: list[Path] = []
    idx = FigureIndex()
    apply_pub_style(cfg, dpi=dpi)
    layout_w, layout_h = cns_layout_px(cfg, "double")
    n = min(len(metrics), 8)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    panel_w = max(120, int(layout_w / max(ncols, 1)))
    panel_h = max(100, int(layout_h / max(nrows, 1)))

    mp = cns.multipanel(
        max_width=int(layout_w),
        title=title,
        title_fontweight="regular",
    )
    for i, metric in enumerate(metrics[:8]):
        sub = plot_df[plot_df["metric"] == metric]
        if sub.empty:
            continue
        label = chr(ord("A") + i) if i < 26 else str(i + 1)
        mp.panel(label, width=panel_w, height=panel_h, pad_left=55, pad_top=14, margin_right=8)
        order = [split_label(s) for s in splits if split_label(s) in set(sub["split_label"])]
        colors = [palette[s] for s in splits if split_label(s) in order]
        ax = cns.barplot(
            data=sub,
            x="split_label",
            y="value",
            order=order,
            hue="split_label",
            hue_order=order,
            palette=colors,
            legend=False,
            add_tip=True,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        cns.setup_ax(ax)

    stem = idx.next_stem(outdir, "split_compare_train_val_test_zsv")
    written.extend(save_cns_figure(stem, dpi))

    # Altair: faceted interactive bars
    split_order_labels = [split_label(s) for s in splits]
    color_scale = alt.Scale(
        domain=split_order_labels,
        range=[palette[s] for s in splits],
    )
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("split_label:N", sort=split_order_labels, title="split"),
            y=alt.Y("value:Q", title="metric value"),
            color=alt.Color("split_label:N", scale=color_scale, legend=None),
            tooltip=[
                "model:N",
                "split_label:N",
                "metric:N",
                alt.Tooltip("value:Q", format=".4g"),
                "source:N",
            ],
            facet=alt.Facet("metric:N", columns=2, title=None),
        )
        .properties(title=title, width=220, height=160)
        .resolve_scale(y="independent")
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
    return written


def run_split_compare(
    run_dir: Path,
    outdir: Path,
    *,
    model: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Collect + plot + write CSV/JSON summary. Returns artifact manifest."""
    run_dir = Path(run_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = collect_split_metrics(run_dir, model=model)
    csv_path = outdir / "split_metrics_compare.csv"
    df.to_csv(csv_path, index=False)
    json_path = outdir / "split_metrics_compare.json"
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8"
    )
    figures = plot_split_comparison(
        df,
        outdir,
        title=title
        or f"Split comparison — {model or run_dir.name}",
    )
    manifest = {
        "run_dir": str(run_dir),
        "outdir": str(outdir),
        "csv": str(csv_path),
        "json": str(json_path),
        "figures": [str(p) for p in figures],
        "n_rows": int(len(df)),
        "splits": sorted({str(s) for s in df["split"]}),
        "metrics": sorted({str(m) for m in df["metric"]}),
    }
    (outdir / "split_compare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Train outdir (e.g. run/run0/direct)")
    p.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Figure outdir (default: <run-dir>/figures/split_compare)",
    )
    p.add_argument("--model", type=str, default=None, help="Legend/model name override")
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args(argv)
    out = args.outdir or (args.run_dir / "figures" / "split_compare")
    manifest = run_split_compare(
        args.run_dir, out, model=args.model, title=args.title
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
