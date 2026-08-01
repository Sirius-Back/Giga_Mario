#!/usr/bin/env python3
"""Compare best-checkpoint metrics across completed runs (barplots).

Collects train/val/test/zsv metrics from ``best_split_metrics.json`` (preferred)
or best-epoch jsonl / metrics_summary / ZSV json. Writes a long CSV and
publication barplots (one panel per metric; grouped by run × split).

Includes unified ``runs_unif`` stages plus selected legacy runs (hashfrag 5/16,
mmseqs 22, etc.).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.best_split_metrics import (
    caduceus_best_from_jsonl,
    detect_model_family,
    write_best_split_metrics,
)

# Explicit extras requested for cross-run comparison.
LEGACY_FOCUS = (
    Path("runs/run5/direct"),  # hashfrag LegNet
    Path("runs/run5/adversarial/train"),
    Path("runs/run16_hashfrag_caduceus/direct"),
    Path("runs/run16_hashfrag_caduceus/adversarial/train"),
)

PRIMARY_METRICS = (
    "spearman",
    "pearson",
    "mae",
    "mse",
    "rmse",
    "r2",
    "accuracy",  # classification / fold-class adv
)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _label_for(train_dir: Path) -> str:
    parts = train_dir.parts
    # …/runs_unif/<model>/<run>/(direct|adversarial/train)
    if "runs_unif" in parts:
        i = parts.index("runs_unif")
        run = parts[i + 2] if len(parts) > i + 2 else train_dir.name
        stage = "adv" if "adversarial" in parts else "direct"
        return f"{run}:{stage}"
    if "runs" in parts:
        i = parts.index("runs")
        run = parts[i + 1] if len(parts) > i + 1 else train_dir.name
        stage = "adv" if "adversarial" in parts else "direct"
        return f"legacy/{run}:{stage}"
    return str(train_dir)


def _strategy_tag(label: str) -> str:
    low = label.lower()
    if "hashfrag" in low or "/run5:" in low or low.startswith("legacy/run5:"):
        return "hashfrag"
    if "mmseqs" in low:
        return "mmseqs"
    if "pangenome" in low:
        return "pangenome"
    if "kmer" in low or "k2" in low or "k4" in low or "k7" in low:
        return "kmer"
    if "gc" in low:
        return "gc"
    if "random" in low:
        return "random"
    if "paralog" in low:
        return "paralogs"
    if "blastp" in low:
        return "blastp"
    return "other"


def discover_train_dirs(
    *,
    unif_root: Path = Path("runs_unif"),
    include_legacy_focus: bool = True,
) -> list[Path]:
    dirs: list[Path] = []
    if unif_root.is_dir():
        for model_dir in sorted(unif_root.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            for run in sorted(model_dir.iterdir()):
                if not run.is_dir() or any(
                    x in run.name for x in ("ARCHIVED", "FAILED", "BAD_", "adversarial_FAILED")
                ):
                    continue
                for tdir in (run / "direct", run / "adversarial" / "train"):
                    if (tdir / "best_model" / "best_meta.json").is_file():
                        dirs.append(tdir)
    if include_legacy_focus:
        for p in LEGACY_FOCUS:
            if (p / "best_model" / "best_meta.json").is_file():
                dirs.append(p)
    # mmseqs unif is already under runs_unif; ensure present
    mm = Path("runs_unif/legnet/run22_legnet_mmseqs_id08/direct")
    if (mm / "best_model" / "best_meta.json").is_file() and mm not in dirs:
        dirs.append(mm)
    # dedupe
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        rp = d.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(d)
    return out


def _ensure_best_payload(train_dir: Path) -> dict[str, Any] | None:
    existing = _load_json(train_dir / "best_split_metrics.json")
    if isinstance(existing, dict) and (
        existing.get("metrics_by_split") or existing.get("spearman")
    ):
        # Refresh caduceus if missing metrics_by_split
        if existing.get("metrics_by_split"):
            return existing
    family = detect_model_family(train_dir)
    if family == "caduceus":
        try:
            payload = caduceus_best_from_jsonl(train_dir)
            write_best_split_metrics(train_dir, payload)
            return payload
        except Exception as exc:  # noqa: BLE001
            print(f"WARN caduceus extract {train_dir}: {exc}", flush=True)
            return existing
    # LegNet without repredict: synthesize from summary + zsv + partial
    if existing:
        return existing
    summary = _load_json(train_dir / "metrics_summary.json") or {}
    by_split: dict[str, dict[str, float]] = {"train": {}, "val": {}, "test": {}, "zsv": {}}
    if isinstance(summary.get("test"), dict):
        for k, v in summary["test"].items():
            fv = _finite(v)
            if fv is not None and k not in {"predictions_path", "pred_aggregation", "error"}:
                by_split["test"][k] = fv
    zsv_path = train_dir / "logs" / "zero_shot_metrics.json"
    zsv = _load_json(zsv_path)
    if isinstance(zsv, dict) and not zsv.get("skipped"):
        block = zsv.get("metrics") if isinstance(zsv.get("metrics"), dict) else zsv
        if isinstance(block, dict):
            for k, v in block.items():
                fv = _finite(v)
                if fv is not None:
                    by_split["zsv"][k] = fv
    meta = _load_json(train_dir / "best_model" / "best_meta.json") or {}
    payload = {
        "model": "legnet",
        "source": "summary_partial_no_repredict",
        "best_epoch": meta.get("epoch"),
        "best_meta": meta,
        "spearman": {k: by_split[k].get("spearman") for k in by_split},
        "metrics_by_split": by_split,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "train/val Spearman pending best-ckpt repredict",
    }
    write_best_split_metrics(train_dir, payload)
    return payload


def collect_rows(train_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tdir in train_dirs:
        payload = _ensure_best_payload(tdir)
        if not payload:
            continue
        label = _label_for(tdir)
        family = str(payload.get("model") or detect_model_family(tdir))
        by_split = payload.get("metrics_by_split")
        if not isinstance(by_split, dict):
            # rebuild from spearman-only
            by_split = {
                k: ({"spearman": v} if v is not None else {})
                for k, v in (payload.get("spearman") or {}).items()
            }
        for split, metrics in by_split.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                fv = _finite(value)
                if fv is None:
                    continue
                rows.append(
                    {
                        "label": label,
                        "strategy": _strategy_tag(label),
                        "family": family,
                        "train_dir": str(tdir),
                        "best_epoch": payload.get("best_epoch"),
                        "source": payload.get("source"),
                        "split": "val" if split in {"validation", "val"} else split,
                        "metric": metric,
                        "value": fv,
                    }
                )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "strategy",
        "family",
        "split",
        "metric",
        "value",
        "best_epoch",
        "source",
        "train_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot_barplots(rows: list[dict[str, Any]], outdir: Path) -> list[Path]:
    """Publication barplots: facet metric × split groups by run label."""
    import math as _math

    import altair as alt
    import pandas as pd

    from src.train_viz.plotting import (
        FigureIndex,
        apply_pub_style,
        cns_layout_px,
        save_altair_chart,
        save_cns_figure,
    )
    from src.train_viz.viz import _load_config

    try:
        import cnsplots as cns
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cnsplots required for compare_best_models plots") from exc

    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    # Prefer primary metrics present in data
    metrics = [m for m in PRIMARY_METRICS if m in set(df["metric"])]
    if not metrics:
        metrics = sorted(df["metric"].unique())[:8]
    plot_df = df[df["metric"].isin(metrics)].copy()
    # stable label order: strategy then label
    plot_df = plot_df.sort_values(["strategy", "label", "split", "metric"])
    labels = list(dict.fromkeys(plot_df["label"].tolist()))
    splits = [s for s in ("train", "val", "test", "zsv") if s in set(plot_df["split"])]

    cfg = _load_config()
    dpi = int(cfg.get("dpi", 600) or 600)
    apply_pub_style(cfg, dpi=dpi)
    idx = FigureIndex()
    written: list[Path] = []

    # Okabe–Ito-ish split colors
    split_colors = {
        "train": "#0072B2",
        "val": "#E69F00",
        "test": "#009E73",
        "zsv": "#CC79A7",
    }

    layout_w, layout_h = cns_layout_px(cfg, "double")
    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(_math.ceil(n / ncols))
    panel_w = max(160, int(layout_w / max(ncols, 1)))
    panel_h = max(120, int(layout_h / max(nrows, 1)))

    mp = cns.multipanel(
        max_width=int(layout_w * 1.35),
        title="Best-checkpoint metrics by split",
        title_fontweight="regular",
    )
    for i, metric in enumerate(metrics):
        sub = plot_df[plot_df["metric"] == metric]
        if sub.empty:
            continue
        label = chr(ord("A") + i) if i < 26 else str(i + 1)
        mp.panel(label, width=panel_w, height=panel_h, pad_left=70, pad_top=14, margin_right=8)
        # grouped bars: x=label, hue=split
        ax = cns.barplot(
            data=sub,
            x="label",
            y="value",
            order=labels,
            hue="split",
            hue_order=splits,
            palette=[split_colors.get(s, "#999999") for s in splits],
            legend=(i == 0),
            add_tip=True,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        cns.setup_ax(ax)

    stem = idx.next_stem(outdir, "best_models_train_val_test_zsv")
    written.extend(save_cns_figure(stem, dpi))

    # Altair interactive
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=labels, title=None),
            y=alt.Y("value:Q", title="value"),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=splits,
                    range=[split_colors.get(s, "#999999") for s in splits],
                ),
                title="split",
            ),
            xOffset="split:N",
            tooltip=[
                "label:N",
                "strategy:N",
                "family:N",
                "split:N",
                "metric:N",
                alt.Tooltip("value:Q", format=".4g"),
                "source:N",
            ],
            facet=alt.Facet("metric:N", columns=2, title=None),
        )
        .properties(title="Best-checkpoint metrics", width=280, height=180)
        .resolve_scale(y="independent")
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))

    # Spearman's-only overview (often the headline)
    if "spearman" in metrics:
        sp = plot_df[plot_df["metric"] == "spearman"]
        mp2 = cns.multipanel(
            max_width=int(layout_w * 1.2),
            title="Best-checkpoint Spearman (train / val / test / ZSV)",
            title_fontweight="regular",
        )
        mp2.panel("A", width=int(layout_w * 1.1), height=int(layout_h * 0.9), pad_left=70, pad_top=14)
        ax = cns.barplot(
            data=sp,
            x="label",
            y="value",
            order=labels,
            hue="split",
            hue_order=splits,
            palette=[split_colors.get(s, "#999999") for s in splits],
            legend=True,
            add_tip=True,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Spearman ρ")
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        cns.setup_ax(ax)
        stem2 = idx.next_stem(outdir, "best_models_spearman")
        written.extend(save_cns_figure(stem2, dpi))
    return written


def write_markdown(rows: list[dict[str, Any]], path: Path, figure_paths: list[Path]) -> None:
    # pivot spearman for a compact table
    by: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        if r["metric"] != "spearman":
            continue
        key = (r["label"], r["strategy"])
        by.setdefault(key, {})[r["split"]] = r["value"]
    lines = [
        "# Best-checkpoint model comparison",
        "",
        f"Generation date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Values are from **best** checkpoints (`best_split_metrics.json` / best-epoch jsonl).",
        "LegNet train/val Spearman requires GPU repredict when still marked `summary_partial_no_repredict`.",
        "",
        "## Included focus splits",
        "",
        "- hashfrag: legacy `run5` (LegNet), `run16_hashfrag_caduceus`",
        "- mmseqs: `run22_legnet_mmseqs_id08` (unif run20/21 are pangenome, not mmseqs)",
        "- plus all unified runs with `best_model/`",
        "",
        "## Spearman (best)",
        "",
        "| run | strategy | train | val | test | zsv |",
        "|-----|----------|------:|----:|-----:|----:|",
    ]

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x:.4f}"

    for (label, strategy), splits in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(
            f"| `{label}` | {strategy} | {fmt(splits.get('train'))} | {fmt(splits.get('val'))} | "
            f"{fmt(splits.get('test'))} | {fmt(splits.get('zsv'))} |"
        )
    lines += ["", "## Figures", ""]
    for p in figure_paths:
        lines.append(f"- `{p}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("figures/best_models_compare"))
    ap.add_argument("--csv", type=Path, default=Path("docs/best_models_compare_metrics.csv"))
    ap.add_argument("--md", type=Path, default=Path("docs/best_models_compare_report.md"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    dirs = discover_train_dirs()
    rows = collect_rows(dirs)
    write_csv(rows, args.csv)
    figs: list[Path] = []
    if not args.no_plots:
        figs = plot_barplots(rows, args.outdir)
    write_markdown(rows, args.md, figs)
    print(f"rows={len(rows)} dirs={len(dirs)} csv={args.csv} md={args.md}", flush=True)
    for f in figs:
        print(f"figure {f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
