"""Publication training figures via cnsplots + Altair.

cnsplots renders journal PDF/SVG/PNG; Altair emits interactive HTML + Vega-Lite JSON
for the same long-form metric tables.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .viz import (
    SPLIT_ORDER,
    _lowess,
    _series,
    aggregate_seeds,
    best_index,
    final_values_by_group,
    is_lower_better,
    layout_for,
    metrics_with_data,
    model_color,
    normalize_metric_name,
    order_metrics,
    split_color,
    split_label,
    splits_for_metric,
)


def cns_layout_px(cfg: dict[str, Any], column: str = "double") -> tuple[int, int]:
    """Map export-pixel config to cnsplots layout pixels (journal CSS px)."""
    px = cfg["figure_size_px"].get(column, cfg["figure_size_px"]["double"])
    dpi = float(cfg.get("dpi_png", 600))
    # config stores raster export pixels @ dpi; cnsplots sizes are ~72 dpi layout units
    scale = 72.0 / max(dpi, 1.0)
    return max(100, int(px[0] * scale)), max(80, int(px[1] * scale))


class FigureIndex:
    def __init__(self) -> None:
        self.n = 0

    def next_stem(self, outdir: Path, name: str) -> Path:
        self.n += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "figure"
        return outdir / f"Figure_{self.n:02d}_{safe}"


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["run", "model", "seed", "epoch", "global_step", "split", "metric", "value"]
        )
    df = pd.DataFrame(rows)
    for col in ("epoch", "global_step", "value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["split_label"] = df["split"].map(lambda s: split_label(str(s)))
    df["series"] = df.apply(
        lambda r: f"{r['model']}/{split_label(str(r['split']))}",
        axis=1,
    )
    return df


def apply_pub_style(cfg: dict[str, Any], *, dpi: int | None = None) -> None:
    """Nature / Cell / Science styling through cnsplots (never raw mpl defaults)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cnsplots as cns

    models = list(cfg.get("palette", {}).get("models") or [])
    color_cycle = models if models else "OkabeIto"
    font_cfg = cfg.get("font", {})
    title_pt = font_cfg.get("title_pt", 10)
    legend_pt = font_cfg.get("legend_pt", 8)
    # Prefer fonts present on Linux servers; keep Arial/Helvetica as optional fallbacks.
    fam = ["DejaVu Sans", "Arial", "Helvetica"]
    for f in font_cfg.get("family") or []:
        if f not in fam:
            fam.append(f)

    cns.settings.font_family = "sans-serif"
    cns.settings.font_sans_serif = tuple(fam)
    cns.settings.axes_spines_top = False
    cns.settings.axes_spines_right = False
    cns.settings.legend_frameon = False
    cns.settings.title_fontsize = title_pt
    cns.settings.legend_fontsize = legend_pt
    if dpi is not None:
        cns.settings.figure_dpi = int(dpi)

    cns.setup_matplotlib(
        color_cycle=color_cycle,
        title_fontsize=title_pt,
        title_fontweight="regular",
        legend_fontsize=legend_pt,
        axes_linewidth=0.8,
    )
    # Project QC still requires high-res PNG; cnsplots default export DPI is lower.
    out_dpi = int(dpi or cfg.get("dpi_png", 600))
    plt.rcParams.update(
        {
            "savefig.dpi": out_dpi,
            "figure.dpi": min(out_dpi, 150),
            "axes.grid": True,
            "grid.color": cfg.get("grid", {}).get("color", "#B0B0B0"),
            "grid.alpha": cfg.get("grid", {}).get("alpha", 0.3),
            "grid.linewidth": 0.6,
            "lines.linewidth": cfg.get("line_width", 2.2),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # Keep a marker that publication style came from cnsplots.
    mpl.rcParams["axes.titleweight"] = "regular"


def save_cns_figure(stem: Path, dpi: int) -> list[Path]:
    """Save current cnsplots/matplotlib figure as PDF + SVG + PNG."""
    import matplotlib.pyplot as plt
    import cnsplots as cns

    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fig = plt.gcf()
    extras = [obj for obj in fig.legends] if getattr(fig, "legends", None) else []
    for ext in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{ext}")
        if ext == "png":
            plt.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.15,
                bbox_extra_artists=extras or None,
            )
        else:
            # Prefer matplotlib save so figure legends outside axes are kept.
            try:
                fig.savefig(
                    path,
                    bbox_inches="tight",
                    pad_inches=0.15,
                    bbox_extra_artists=extras or None,
                )
            except Exception:
                cns.savefig(path)
        written.append(path)
    plt.close("all")
    return written


def save_altair_chart(chart: Any, stem: Path) -> list[Path]:
    """Persist Altair chart as HTML + Vega-Lite JSON (+ PNG when vl-convert works)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    html = stem.with_suffix(".html")
    vl = stem.with_name(stem.name + ".vl.json")
    try:
        chart.save(str(html))
        written.append(html)
    except Exception as exc:  # noqa: BLE001 — keep static figures even if HTML fails
        (stem.parent / f"{stem.name}_altair_html_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
    try:
        spec = chart.to_dict()
        vl.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        written.append(vl)
    except Exception as exc:  # noqa: BLE001
        (stem.parent / f"{stem.name}_altair_json_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
    try:
        png = stem.with_suffix(".png")
        chart.save(str(png), scale_factor=2)
        written.append(png)
    except Exception:
        pass
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
        errors.append("default matplotlib style (cnsplots setup required)")
    if n_seeds >= 2 and cfg.get("qc", {}).get("require_ci_when_multi_seed", True):
        if ribbon not in ("ci95", "std"):
            errors.append("missing confidence intervals / ribbons for multi-seed")
    return errors


def _palette_for_splits(splits: list[str], cfg: dict[str, Any]) -> dict[str, str]:
    return {split_label(s): split_color(s, cfg) for s in splits}


def filter_rows_max_epoch(
    rows: list[dict[str, Any]],
    max_epoch: int | float | None,
    *,
    x_key: str = "epoch",
) -> list[dict[str, Any]]:
    """Keep rows with ``x_key <= max_epoch`` (non-finite / missing x kept out)."""
    if max_epoch is None:
        return rows
    out: list[dict[str, Any]] = []
    lim = float(max_epoch)
    for r in rows:
        raw = r.get(x_key)
        try:
            x = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and x <= lim + 1e-12:
            out.append(r)
    return out


def _model_palette(models: list[str], cfg: dict[str, Any]) -> dict[str, str]:
    return {m: model_color(i, cfg) for i, m in enumerate(models)}


def _facet_split_order(present: list[str], cfg: dict[str, Any]) -> list[str]:
    preferred = list(cfg.get("facet_splits") or ["train", "validation", "test"])
    ordered = [s for s in preferred if s in present]
    for s in present:
        if s not in ordered and s in ("train", "validation", "test"):
            ordered.append(s)
    return ordered or list(present)


def _dedupe_legend_entries(
    handles: list[Any], labels: list[str]
) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    out_h: list[Any] = []
    out_l: list[str] = []
    for h, lab in zip(handles, labels):
        if lab in seen or lab.startswith("_"):
            continue
        seen.add(lab)
        out_h.append(h)
        out_l.append(lab)
    return out_h, out_l


def place_legend_lower_right(
    fig: Any,
    axes: Any,
    *,
    cfg: dict[str, Any],
    position: str = "lower right",
) -> Any:
    """Figure-level matplotlib legend; remove axes legends to avoid overlap.

    ``position``:
      - ``\"lower right\"`` — bottom-right of the figure (faceted metric plots)
      - ``\"bottom\"`` — centered under panels (Figure_01 learning curves)
    """
    ax_list = list(np.atleast_1d(axes).ravel())
    handles: list[Any] = []
    labels: list[str] = []
    for ax in ax_list:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    handles, labels = _dedupe_legend_entries(handles, labels)
    if not handles:
        return None
    leg_cfg = cfg.get("legend") or {}
    ncol = 1
    thr = int(leg_cfg.get("ncol_auto_threshold", 6))
    if len(labels) >= thr:
        ncol = 2 if len(labels) < 12 else 3
    if position == "bottom":
        # Wide bottom strip under multipanel grid — does not cover curves.
        ncol = max(ncol, min(4, len(labels)))
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=bool(leg_cfg.get("frameon", True)),
            framealpha=float(leg_cfg.get("framealpha", 0.92)),
            fontsize=cfg.get("font", {}).get("legend_pt", 8),
            ncol=ncol,
            borderaxespad=0.4,
        )
        fig.subplots_adjust(bottom=0.18 if ncol <= 3 else 0.22)
        return legend
    bbox = leg_cfg.get("bbox_to_anchor", [0.99, 0.02])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
        bbox_t = (float(bbox[0]), float(bbox[1]))
    else:
        bbox_t = (0.99, 0.02)
    legend = fig.legend(
        handles,
        labels,
        loc=str(leg_cfg.get("loc", "lower right")),
        bbox_to_anchor=bbox_t,
        frameon=bool(leg_cfg.get("frameon", True)),
        framealpha=float(leg_cfg.get("framealpha", 0.92)),
        fontsize=cfg.get("font", {}).get("legend_pt", 8),
        ncol=ncol,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(bottom=0.20 if ncol >= 2 else 0.14, right=0.98)
    return legend


def _altair_faceted_by_split(
    df: pd.DataFrame,
    *,
    x_key: str,
    title: str,
    models: list[str],
    color_range: list[str],
    split_order: list[str],
) -> Any:
    """One color per model; columns = train / validation / test."""
    import altair as alt

    split_labels = [split_label(s) for s in split_order]
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(f"{x_key}:Q", title=x_key.replace("_", " ")),
            y=alt.Y("value:Q", title="value"),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=models, range=color_range),
                legend=alt.Legend(title="run"),
            ),
            tooltip=[
                alt.Tooltip(f"{x_key}:Q"),
                alt.Tooltip("value:Q", format=".4g"),
                alt.Tooltip("model:N"),
                alt.Tooltip("split_label:N"),
            ],
        )
        .properties(width=220, height=240)
        .facet(
            column=alt.Column(
                "split_label:N",
                title=None,
                sort=split_labels,
            ),
            title=title,
        )
        .resolve_scale(y="independent")
        .interactive()
    )
    return chart


def _altair_line(
    df: pd.DataFrame,
    *,
    x_key: str,
    title: str,
    color_field: str,
    color_domain: list[str],
    color_range: list[str],
    stroke_dash_field: str | None = None,
) -> Any:
    import altair as alt

    enc: dict[str, Any] = {
        "x": alt.X(f"{x_key}:Q", title=x_key.replace("_", " ")),
        "y": alt.Y("value:Q", title="value"),
        "color": alt.Color(
            f"{color_field}:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title=color_field),
        ),
        "tooltip": [
            alt.Tooltip(f"{x_key}:Q"),
            alt.Tooltip("value:Q", format=".4g"),
            alt.Tooltip(f"{color_field}:N"),
        ],
    }
    if stroke_dash_field and stroke_dash_field in df.columns:
        enc["strokeDash"] = alt.StrokeDash(f"{stroke_dash_field}:N")
    base = alt.Chart(df).mark_line(strokeWidth=2.2).encode(**enc)
    return base.properties(title=title, width=420, height=280).interactive()


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
    best_epochs: dict[str, float] | None = None,
    max_epoch: int | float | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import cnsplots as cns

    apply_pub_style(cfg, dpi=dpi)
    written: list[Path] = []
    if max_epoch is None and cfg.get("compare_max_epoch") is not None:
        # Only auto-apply config cap for multi-model combined figures.
        if len({r["model"] for r in rows}) > 1:
            max_epoch = cfg.get("compare_max_epoch")
    rows = filter_rows_max_epoch(rows, max_epoch, x_key=x_key)
    models = sorted({r["model"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    multi_model = len(models) > 1
    metrics = metrics_with_data(rows, metrics)
    if not metrics:
        return written
    best_epochs = best_epochs or {}
    model_pal = _model_palette(models, cfg)

    pages: list[list[str]] = []
    if len(metrics) <= 9:
        pages = [metrics]
    else:
        for i in range(0, len(metrics), 9):
            pages.append(metrics[i : i + 9])

    layout_w, layout_h = cns_layout_px(cfg, column)
    panel_w = max(120, int(layout_w / 3))
    panel_h = max(110, int(layout_h / 2))

    for page_i, page_metrics in enumerate(pages):
        apply_pub_style(cfg, dpi=dpi)
        page_title = title or "Learning curves"
        if len(pages) > 1:
            page_title = f"{page_title} ({page_i + 1}/{len(pages)})"
        if max_epoch is not None:
            page_title = f"{page_title} (epochs ≤ {int(max_epoch)})"
        mp = cns.multipanel(max_width=int(layout_w), title=page_title, title_fontweight="regular")
        page_axes: list[Any] = []
        for mi, metric in enumerate(page_metrics):
            metric_splits = splits_for_metric(rows, metric)
            df = _metric_long(
                rows,
                metric,
                models=models,
                splits=metric_splits,
                x_key=x_key,
                n_seeds=n_seeds,
                ribbon=ribbon,
            )
            label = chr(ord("A") + mi) if mi < 26 else str(mi + 1)
            mp.panel(
                label,
                width=panel_w,
                height=panel_h,
                pad_left=36,
                pad_top=14,
                margin_right=8,
                margin_bottom=8,
            )
            if df.empty:
                continue
            if multi_model:
                hue = "model"
                style = None
                palette = model_pal
            else:
                hue = "split_label"
                style = None
                palette = _palette_for_splits(metric_splits, cfg)
            err = ("ci", 95) if (n_seeds >= 2 and ribbon == "ci95") else (
                ("sd", 1) if (n_seeds >= 2 and ribbon == "std") else None
            )
            ax = cns.lineplot(
                data=df,
                x=x_key,
                y="value",
                hue=hue,
                style=style,
                palette=palette,
                hue_order=models if multi_model else None,
                errorbar=err,
                seed=42,
                legend=False,
                linewidth=cfg.get("line_width", 2.2),
            )
            ax.set_xlabel(x_key.replace("_", " "))
            ax.set_ylabel(metric.replace("_", " "))
            cns.setup_ax(ax)
            page_axes.append(ax)
            for model in models:
                mark_splits = (
                    ["validation"]
                    if "validation" in metric_splits
                    else list(metric_splits)
                )
                for split in mark_splits:
                    sub = df[(df["model"] == model) & (df["split"] == split)]
                    if sub.empty:
                        continue
                    grp = sub.groupby(x_key, as_index=False)["value"].mean()
                    color = (
                        model_pal[model]
                        if multi_model
                        else split_color(split, cfg)
                    )
                    _mark_selected_best(
                        ax,
                        grp,
                        x_key=x_key,
                        metric=metric,
                        cfg=cfg,
                        color=color,
                        best_epoch=best_epochs.get(model),
                        patience=patience,
                        annotate=(split == "validation" and not multi_model),
                    )
                    if smooth and len(grp) >= 4:
                        sm = _lowess(
                            grp[x_key].to_numpy(dtype=float),
                            grp["value"].to_numpy(dtype=float),
                            cfg.get("lowess_frac", 0.35),
                        )
                        if sm is not None:
                            ax.plot(
                                grp[x_key],
                                sm,
                                color=color,
                                linewidth=1.0,
                                alpha=0.55,
                                zorder=2,
                            )
                    break
        if page_axes:
            # Matplotlib only: bottom strip under the multipanel (Figure_01).
            place_legend_lower_right(plt.gcf(), page_axes, cfg=cfg, position="bottom")
        stem = idx.next_stem(
            outdir, f"learning_curves_p{page_i + 1}" if len(pages) > 1 else "learning_curves"
        )
        written.extend(save_cns_figure(stem, dpi))

        altair_frames = []
        for metric in page_metrics:
            df = _metric_long(
                rows,
                metric,
                models=models,
                splits=splits_for_metric(rows, metric),
                x_key=x_key,
                n_seeds=n_seeds,
                ribbon="none",
            )
            if not df.empty:
                d = df.copy()
                d["metric"] = metric
                altair_frames.append(d)
        if altair_frames:
            adf = pd.concat(altair_frames, ignore_index=True)
            import altair as alt

            if multi_model:
                domain = models
                crange = [model_pal[m] for m in models]
                color_field = "model"
            else:
                color_field = "split_label"
                domain = sorted(adf[color_field].unique())
                crange = [
                    split_color(next((s for s in SPLIT_ORDER if split_label(s) == d), "_run"), cfg)
                    for d in domain
                ]
            line = (
                alt.Chart(adf)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(f"{x_key}:Q", title=x_key.replace("_", " ")),
                    y=alt.Y("value:Q", title="value"),
                    color=alt.Color(
                        f"{color_field}:N",
                        scale=alt.Scale(domain=domain, range=crange),
                        legend=alt.Legend(title=color_field),
                    ),
                    tooltip=[x_key, "value", color_field, "metric"],
                )
            )
            chart = (
                line.properties(width=220, height=160)
                .facet(facet=alt.Facet("metric:N"), columns=3, title=page_title, data=adf)
                .interactive()
            )
            written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))

    # Per-metric figures (Figure_02_loss, …): multi-model → facets train/val/test
    for metric in metrics:
        metric_splits = splits_for_metric(rows, metric)
        if not metric_splits:
            continue
        facet_splits = _facet_split_order(metric_splits, cfg) if multi_model else metric_splits
        df = _metric_long(
            rows,
            metric,
            models=models,
            splits=facet_splits,
            x_key=x_key,
            n_seeds=n_seeds,
            ribbon=ribbon,
        )
        if df.empty:
            continue
        apply_pub_style(cfg, dpi=dpi)
        err = ("ci", 95) if (n_seeds >= 2 and ribbon == "ci95") else (
            ("sd", 1) if (n_seeds >= 2 and ribbon == "std") else None
        )
        sw, sh = cns_layout_px(cfg, "single" if not multi_model else "double")

        if multi_model:
            n_fac = max(len(facet_splits), 1)
            fig_w = max(6.0, (sw / 72.0) * max(n_fac, 1) * 0.55)
            fig_h = max(3.2, sh / 72.0 * 0.85)
            fig, axes = plt.subplots(
                1,
                n_fac,
                figsize=(fig_w, fig_h),
                sharey=False,
                squeeze=False,
            )
            axes_flat = list(axes.ravel())
            for ax, split in zip(axes_flat, facet_splits):
                sub = df[df["split"] == split]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                cns.lineplot(
                    data=sub,
                    x=x_key,
                    y="value",
                    hue="model",
                    hue_order=models,
                    palette=model_pal,
                    errorbar=err,
                    seed=42,
                    legend=False,
                    linewidth=cfg.get("line_width", 2.2),
                    ax=ax,
                )
                ax.set_title(split_label(split))
                ax.set_xlabel(x_key.replace("_", " "))
                ax.set_ylabel(metric.replace("_", " "))
                cns.setup_ax(ax)
                for model in models:
                    msub = sub[sub["model"] == model]
                    if msub.empty:
                        continue
                    grp = msub.groupby(x_key, as_index=False)["value"].mean()
                    _mark_selected_best(
                        ax,
                        grp,
                        x_key=x_key,
                        metric=metric,
                        cfg=cfg,
                        color=model_pal[model],
                        best_epoch=best_epochs.get(model),
                        patience=None,
                        annotate=False,
                    )
            fig.suptitle(metric.replace("_", " "), fontsize=cfg.get("font", {}).get("title_pt", 10))
            place_legend_lower_right(fig, axes_flat, cfg=cfg)
            stem = idx.next_stem(outdir, metric)
            written.extend(save_cns_figure(stem, dpi))
            chart = _altair_faceted_by_split(
                df,
                x_key=x_key,
                title=metric.replace("_", " "),
                models=models,
                color_range=[model_pal[m] for m in models],
                split_order=facet_splits,
            )
            written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
            continue

        cns.figure(width=int(sw * 1.15), height=int(sh * 1.05))
        palette = _palette_for_splits(metric_splits, cfg)
        ax = cns.lineplot(
            data=df,
            x=x_key,
            y="value",
            hue="split_label",
            palette=palette,
            errorbar=err,
            seed=42,
            linewidth=cfg.get("line_width", 2.2),
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel(metric.replace("_", " "))
        cns.setup_ax(ax)
        place_legend_lower_right(plt.gcf(), [ax], cfg=cfg)
        for model in models:
            mark_splits = (
                ["validation"] if "validation" in metric_splits else list(metric_splits)
            )
            for split in mark_splits:
                sub = df[(df["model"] == model) & (df["split"] == split)]
                if sub.empty:
                    continue
                grp = sub.groupby(x_key, as_index=False)["value"].mean()
                _mark_selected_best(
                    ax,
                    grp,
                    x_key=x_key,
                    metric=metric,
                    cfg=cfg,
                    color=split_color(split, cfg),
                    best_epoch=best_epochs.get(model),
                    patience=None,
                    annotate=True,
                )
                break
        stem = idx.next_stem(outdir, metric)
        written.extend(save_cns_figure(stem, dpi))
        domain = sorted(df["split_label"].unique())
        crange = [palette.get(d, "#000000") for d in domain]
        chart = _altair_line(
            df,
            x_key=x_key,
            title=metric.replace("_", " "),
            color_field="split_label",
            color_domain=domain,
            color_range=crange,
        )
        written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))

    return written


def _metric_long(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    models: list[str],
    splits: list[str],
    x_key: str,
    n_seeds: int,
    ribbon: str,
) -> pd.DataFrame:
    """Long-form table for one metric (raw seeds or mean±band for ribbons)."""
    records: list[dict[str, Any]] = []
    m = normalize_metric_name(metric)
    for model in models:
        for split in splits:
            if n_seeds >= 2 and ribbon in ("ci95", "std"):
                # Pass raw points so cns.lineplot/seaborn can CI-aggregate.
                for seed in sorted({r["seed"] for r in rows if r.get("model") == model}, key=str):
                    xs, ys = _series(
                        rows, model=model, seed=seed, split=split, metric=m, x_key=x_key
                    )
                    for x, y in zip(xs.tolist(), ys.tolist()):
                        if not math.isfinite(float(y)):
                            continue
                        records.append(
                            {
                                x_key: float(x),
                                "value": float(y),
                                "model": model,
                                "split": split,
                                "split_label": split_label(split),
                                "seed": seed,
                                "series": f"{model}/{split_label(split)}",
                                "metric": m,
                            }
                        )
            else:
                xs, ys = _series(rows, model=model, split=split, metric=m, x_key=x_key)
                if xs.size == 0:
                    x, mean, *_ = aggregate_seeds(
                        rows, model=model, split=split, metric=m, x_key=x_key
                    )
                    xs, ys = x, mean
                for x, y in zip(xs.tolist(), ys.tolist()):
                    if not math.isfinite(float(y)):
                        continue
                    records.append(
                        {
                            x_key: float(x),
                            "value": float(y),
                            "model": model,
                            "split": split,
                            "split_label": split_label(split),
                            "seed": None,
                            "series": f"{model}/{split_label(split)}",
                            "metric": m,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _mark_selected_best(
    ax: Any,
    grp: pd.DataFrame,
    *,
    x_key: str,
    metric: str,
    cfg: dict[str, Any],
    color: str,
    best_epoch: float | None,
    patience: int | None,
    annotate: bool = False,
) -> None:
    """Scatter the selected final/best checkpoint (or metric-inferred fallback)."""
    if grp.empty:
        return
    xb = yb = None
    label = "best"
    if best_epoch is not None and x_key == "epoch":
        # nearest epoch on the curve
        xs = grp[x_key].to_numpy(dtype=float)
        if xs.size:
            j = int(np.nanargmin(np.abs(xs - float(best_epoch))))
            xb = float(grp.iloc[j][x_key])
            yb = float(grp.iloc[j]["value"])
            label = "final/best"
    if xb is None:
        bi = best_index(list(grp["value"]), metric, cfg)
        if bi is None:
            return
        xb = float(grp.iloc[bi][x_key])
        yb = float(grp.iloc[bi]["value"])
    ax.scatter(
        [xb],
        [yb],
        s=90 if best_epoch is not None else 40,
        color=color,
        zorder=6,
        edgecolors="black",
        linewidths=0.8,
        marker="*",
    )
    if annotate or best_epoch is not None:
        ax.annotate(
            f"{label}@{xb:g}\n{yb:.3g}",
            (xb, yb),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            color=color,
        )
    if patience is not None:
        stop = min(xb + patience, float(grp[x_key].max()))
        ax.axvline(stop, color=color, linestyle="--", linewidth=1.0, alpha=0.7)



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
    max_epoch: int | float | None = None,
) -> list[Path]:
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return []
    import matplotlib.pyplot as plt
    import cnsplots as cns

    if max_epoch is None and cfg.get("compare_max_epoch") is not None:
        max_epoch = cfg.get("compare_max_epoch")
    rows = filter_rows_max_epoch(rows, max_epoch, x_key=x_key)
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return []

    apply_pub_style(cfg, dpi=dpi)
    written: list[Path] = []
    splits = [s for s in ("train", "validation", "test") if any(r["split"] == s for r in rows)]
    n_seeds = len({r["seed"] for r in rows})
    sw, sh = cns_layout_px(cfg, "single")
    palette = _model_palette(models, cfg)

    for metric in metrics:
        for split in splits:
            df = _metric_long(
                rows,
                metric,
                models=models,
                splits=[split],
                x_key=x_key,
                n_seeds=n_seeds,
                ribbon=ribbon,
            )
            if df.empty:
                continue
            apply_pub_style(cfg, dpi=dpi)
            cns.figure(width=int(sw * 1.2), height=int(sh * 1.05))
            err = ("ci", 95) if (n_seeds >= 2 and ribbon == "ci95") else (
                ("sd", 1) if (n_seeds >= 2 and ribbon == "std") else None
            )
            ax = cns.lineplot(
                data=df,
                x=x_key,
                y="value",
                hue="model",
                hue_order=models,
                palette=palette,
                errorbar=err,
                seed=42,
                legend=False,
                linewidth=cfg.get("line_width", 2.2),
            )
            title = f"{metric.replace('_', ' ')} — {split}"
            if max_epoch is not None:
                title = f"{title} (epochs ≤ {int(max_epoch)})"
            ax.set_title(title)
            ax.set_xlabel(x_key.replace("_", " "))
            ax.set_ylabel(metric.replace("_", " "))
            cns.setup_ax(ax)
            place_legend_lower_right(plt.gcf(), [ax], cfg=cfg)
            stem = idx.next_stem(outdir, f"multimodel_{metric}_{split}")
            written.extend(save_cns_figure(stem, dpi))
            chart = _altair_line(
                df,
                x_key=x_key,
                title=title,
                color_field="model",
                color_domain=models,
                color_range=[palette[m] for m in models],
            )
            written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
    return written


def plot_final_performance(
    rows: list[dict[str, Any]],
    metrics: list[str],
    cfg: dict[str, Any],
    outdir: Path,
    idx: FigureIndex,
    *,
    dpi: int,
) -> list[Path]:
    import cnsplots as cns
    import altair as alt

    apply_pub_style(cfg, dpi=dpi)
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
        plot_metrics = [
            m
            for m in metrics
            if m not in ("elapsed_sec", "lr", "grad_norm")
            and any(r["metric"] == m and r["split"] == split for r in rows)
        ]
    if not plot_metrics:
        return []

    records: list[dict[str, Any]] = []
    for metric in plot_metrics:
        groups = final_values_by_group(rows, metric, split)
        for model, vals in groups.items():
            for v in vals:
                records.append({"model": model, "metric": metric, "value": float(v), "split": split})
    if not records:
        return []
    df = pd.DataFrame(records)

    written: list[Path] = []
    layout_w, layout_h = cns_layout_px(cfg, "double")
    n = min(len(plot_metrics), 9)
    nrows, ncols = layout_for(n, cfg)
    panel_w = max(100, int(layout_w / max(ncols, 1)))
    panel_h = max(90, int(layout_h / max(nrows, 1)))
    apply_pub_style(cfg, dpi=dpi)
    mp = cns.multipanel(
        max_width=int(layout_w),
        title=f"Final performance — {split}",
        title_fontweight="regular",
    )
    for i, metric in enumerate(plot_metrics[:9]):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        # order best→worst by mean
        means = sub.groupby("model")["value"].mean().sort_values(
            ascending=is_lower_better(metric, cfg)
        )
        order = list(means.index)
        label = chr(ord("A") + i) if i < 26 else str(i + 1)
        mp.panel(label, width=panel_w, height=panel_h, pad_left=50, pad_top=12, margin_right=6)
        colors = [model_color(j, cfg) for j, _m in enumerate(order)]
        bar_kwargs: dict[str, Any] = dict(
            data=sub,
            x="value",
            y="model",
            order=order,
            add_tip=True,
            hue="model",
            hue_order=order,
            palette=colors,
            legend=False,
        )
        ax = cns.barplot(**bar_kwargs)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} ({split})")
        cns.setup_ax(ax)
    stem = idx.next_stem(outdir, "final_performance")
    written.extend(save_cns_figure(stem, dpi))

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("mean(value):Q", title="final value"),
            y=alt.Y("model:N", sort="-x"),
            color=alt.Color("model:N", legend=None),
            row=alt.Row("metric:N"),
            tooltip=["model", "metric", alt.Tooltip("mean(value):Q", format=".4g")],
        )
        .properties(title=f"Final performance — {split}", width=320, height=80)
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
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
    import cnsplots as cns
    import altair as alt

    apply_pub_style(cfg, dpi=dpi)
    split = next(
        (s for s in ("validation", "test", "train") if any(r["split"] == s for r in rows)),
        "validation",
    )
    metrics = [m for m in metrics if m not in ("elapsed_sec", "lr", "grad_norm")][:6]
    if not metrics:
        return []
    records: list[dict[str, Any]] = []
    models = sorted({r["model"] for r in rows})
    for metric in metrics:
        for model in models:
            for v in final_values_by_group(rows, metric, split).get(model, []):
                records.append({"model": model, "metric": metric, "value": float(v), "split": split})
    if not records:
        return []
    df = pd.DataFrame(records)
    layout_w, layout_h = cns_layout_px(cfg, "double")
    nrows, ncols = layout_for(len(metrics), cfg)
    panel_w = max(100, int(layout_w / max(ncols, 1)))
    panel_h = max(90, int(layout_h / max(nrows, 1)))
    apply_pub_style(cfg, dpi=dpi)
    mp = cns.multipanel(
        max_width=int(layout_w),
        title="Seed variability (final epoch)",
        title_fontweight="regular",
    )
    for i, metric in enumerate(metrics):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        label = chr(ord("A") + i) if i < 26 else str(i + 1)
        mp.panel(label, width=panel_w, height=panel_h, pad_left=40, pad_top=12)
        ax = cns.violinplot(data=sub, x="model", y="value", add_box=True, order=models)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} ({split})")
        cns.setup_ax(ax)
    stem = idx.next_stem(outdir, "seed_variability")
    written = save_cns_figure(stem, dpi)
    chart = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x="model:N",
            y=alt.Y("value:Q"),
            color="model:N",
            column="metric:N",
        )
        .properties(title="Seed variability", width=120, height=180)
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
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
    epochs = sorted({r["epoch"] for r in rows if r.get("epoch") is not None})
    if len(epochs) < 3:
        return []
    metrics = [m for m in metrics if m not in ("lr", "grad_norm")]
    if len(metrics) < 2:
        return []
    import cnsplots as cns
    import altair as alt
    import matplotlib.pyplot as plt

    apply_pub_style(cfg, dpi=dpi)
    split = next(
        (s for s in ("validation", "train", "test") if any(r["split"] == s for r in rows)),
        "train",
    )
    series: dict[str, np.ndarray] = {}
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

    sw, sh = cns_layout_px(cfg, "single")
    side = max(sw, sh, int(40 * n + 80))
    apply_pub_style(cfg, dpi=dpi)
    cns.figure(width=side, height=side)
    ax = plt.gca()
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
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Spearman metric correlation ({split})")
    cns.setup_ax(ax)
    stem = idx.next_stem(outdir, "metric_correlation")
    written = save_cns_figure(stem, dpi)

    long = [
        {"metric_i": keys[i], "metric_j": keys[j], "corr": float(corr[i, j])}
        for i in range(n)
        for j in range(n)
        if corr[i, j] == corr[i, j]
    ]
    cdf = pd.DataFrame(long)
    chart = (
        alt.Chart(cdf)
        .mark_rect()
        .encode(
            x=alt.X("metric_j:N", title=None),
            y=alt.Y("metric_i:N", title=None),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["metric_i", "metric_j", alt.Tooltip("corr:Q", format=".2f")],
        )
        .properties(title=f"Spearman metric correlation ({split})", width=280, height=280)
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
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
    import cnsplots as cns
    import altair as alt

    apply_pub_style(cfg, dpi=dpi)
    metrics = [m for m in metrics if m not in ("lr", "elapsed_sec", "grad_norm")]
    if not metrics:
        return []
    models = sorted({r["model"] for r in rows})
    records: list[dict[str, Any]] = []
    for metric in metrics:
        for model in models:
            x_v, mean_v, *_ = aggregate_seeds(
                rows, model=model, split="validation", metric=metric, x_key=x_key
            )
            x_t, mean_t, *_ = aggregate_seeds(
                rows, model=model, split="train", metric=metric, x_key=x_key
            )
            if x_v.size == 0 or x_t.size == 0:
                xs_v, ys_v = _series(rows, model=model, split="validation", metric=metric, x_key=x_key)
                xs_t, ys_t = _series(rows, model=model, split="train", metric=metric, x_key=x_key)
                common = sorted(set(xs_v.tolist()) & set(xs_t.tolist()))
                if not common:
                    continue
                map_v = dict(zip(xs_v.tolist(), ys_v.tolist()))
                map_t = dict(zip(xs_t.tolist(), ys_t.tolist()))
            else:
                common = sorted(set(x_v.tolist()) & set(x_t.tolist()))
                map_v = dict(zip(x_v.tolist(), mean_v.tolist()))
                map_t = dict(zip(x_t.tolist(), mean_t.tolist()))
            for c in common:
                records.append(
                    {
                        x_key: float(c),
                        "value": float(map_v[c] - map_t[c]),
                        "model": model,
                        "metric": metric,
                    }
                )
    if not records:
        return []
    df = pd.DataFrame(records)
    layout_w, layout_h = cns_layout_px(cfg, "double")
    n = min(len(metrics), 9)
    nrows, ncols = layout_for(n, cfg)
    panel_w = max(100, int(layout_w / max(ncols, 1)))
    panel_h = max(90, int(layout_h / max(nrows, 1)))
    apply_pub_style(cfg, dpi=dpi)
    mp = cns.multipanel(
        max_width=int(layout_w),
        title="Generalization gap (validation − train)",
        title_fontweight="regular",
    )
    palette = {m: model_color(i, cfg) for i, m in enumerate(models)}
    for i, metric in enumerate(metrics[:9]):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        label = chr(ord("A") + i) if i < 26 else str(i + 1)
        mp.panel(label, width=panel_w, height=panel_h, pad_left=40, pad_top=12)
        gap_kwargs: dict[str, Any] = dict(
            data=sub,
            x=x_key,
            y="value",
            errorbar=None,
            linewidth=cfg.get("line_width", 2.2),
            legend="auto" if (len(models) > 1 and i == 0) else False,
        )
        if len(models) > 1:
            gap_kwargs.update(hue="model", palette=palette)
        ax = cns.lineplot(**gap_kwargs)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_ylabel(f"gap ({metric.replace('_', ' ')})")
        ax.set_title(metric.replace("_", " "))
        cns.setup_ax(ax)
    stem = idx.next_stem(outdir, "generalization_gap")
    written = save_cns_figure(stem, dpi)
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=f"{x_key}:Q",
            y="value:Q",
            color="model:N",
            facet=alt.Facet("metric:N", columns=3),
            tooltip=[x_key, "value", "model", "metric"],
        )
        .properties(title="Generalization gap", width=200, height=140)
        .interactive()
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
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
    best_epochs: dict[str, float] | None = None,
) -> list[Path]:
    import cnsplots as cns
    import altair as alt

    apply_pub_style(cfg, dpi=dpi)
    metric = (
        "loss"
        if any(r["metric"] == "loss" for r in rows)
        else order_metrics({r["metric"] for r in rows}, cfg)[0]
    )
    split = "validation" if any(r["split"] == "validation" for r in rows) else "train"
    models = sorted({r["model"] for r in rows})
    best_epochs = best_epochs or {}
    df = _metric_long(
        rows,
        metric,
        models=models,
        splits=[split],
        x_key=x_key,
        n_seeds=1,
        ribbon="none",
    )
    if df.empty:
        return []
    sw, sh = cns_layout_px(cfg, "single")
    apply_pub_style(cfg, dpi=dpi)
    cns.figure(width=int(sw * 1.3), height=int(sh * 1.1))
    palette = {m: model_color(i, cfg) for i, m in enumerate(models)}
    line_kwargs: dict[str, Any] = dict(
        data=df,
        x=x_key,
        y="value",
        errorbar=None,
        linewidth=cfg.get("line_width", 2.2),
    )
    if len(models) > 1:
        line_kwargs.update(hue="model", palette=palette)
    ax = cns.lineplot(**line_kwargs)
    for mi, model in enumerate(models):
        sub = df[df["model"] == model].groupby(x_key, as_index=False)["value"].mean()
        if sub.empty:
            continue
        color = model_color(mi, cfg) if len(models) > 1 else split_color(split, cfg)
        _mark_selected_best(
            ax,
            sub,
            x_key=x_key,
            metric=metric,
            cfg=cfg,
            color=color,
            best_epoch=best_epochs.get(model),
            patience=patience,
            annotate=True,
        )
        be = best_epochs.get(model)
        if be is not None and x_key == "epoch":
            ax.axvline(float(be), color=color, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Early stopping / best checkpoint (final model)")
    cns.setup_ax(ax)
    stem = idx.next_stem(outdir, "early_stopping")
    written = save_cns_figure(stem, dpi)
    chart = _altair_line(
        df,
        x_key=x_key,
        title="Early stopping / best checkpoint (final model)",
        color_field="model" if len(models) > 1 else "split_label",
        color_domain=models if len(models) > 1 else [split_label(split)],
        color_range=[model_color(i, cfg) for i in range(len(models))]
        if len(models) > 1
        else [split_color(split, cfg)],
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
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
    import cnsplots as cns

    apply_pub_style(cfg, dpi=dpi)
    models = sorted({r["model"] for r in rows})
    frames = []
    for model in models:
        model_frames: list[dict[str, Any]] = []
        for split in ("_run", "train"):
            xs, ys = _series(rows, model=model, split=split, metric="lr", x_key=x_key)
            for x, y in zip(xs.tolist(), ys.tolist()):
                model_frames.append({x_key: float(x), "value": float(y), "model": model})
            if model_frames:
                break
        frames.extend(model_frames)
    if not frames:
        return []
    df = pd.DataFrame(frames)
    sw, sh = cns_layout_px(cfg, "single")
    apply_pub_style(cfg, dpi=dpi)
    cns.figure(width=int(sw * 1.2), height=int(sh))
    palette = {m: model_color(i, cfg) for i, m in enumerate(models)}
    line_kwargs: dict[str, Any] = dict(
        data=df,
        x=x_key,
        y="value",
        errorbar=None,
        linewidth=cfg.get("line_width", 2.2),
    )
    if len(models) > 1:
        line_kwargs.update(hue="model", palette=palette)
    ax = cns.lineplot(**line_kwargs)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    cns.setup_ax(ax)
    stem = idx.next_stem(outdir, "learning_rate")
    written = save_cns_figure(stem, dpi)
    chart = _altair_line(
        df.assign(split_label="lr"),
        x_key=x_key,
        title="Learning rate schedule",
        color_field="model" if len(models) > 1 else "split_label",
        color_domain=models if len(models) > 1 else ["lr"],
        color_range=[model_color(i, cfg) for i in range(len(models))]
        if len(models) > 1
        else ["#56B4E9"],
    )
    written.extend(save_altair_chart(chart, stem.with_name(stem.name + "_altair")))
    return written
