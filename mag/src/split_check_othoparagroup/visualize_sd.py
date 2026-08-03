"""Altair horizontal sd_random plots with left annotation table (aligned by run_id)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OKABE = [
    "#E69F00",  # gc
    "#56B4E9",  # hashfrag
    "#009E73",  # kmer
    "#F0E442",  # mmseqs
    "#0072B2",  # blastp
    "#D55E00",  # pangenome
    "#CC79A7",  # paralogs_only
    "#000000",  # random
    "#882255",  # loco
    "#44AA99",  # vgae
    "#AA4499",  # gcn
    "#999999",  # other
]
FAM_DOMAIN = [
    "gc",
    "hashfrag",
    "kmer",
    "mmseqs",
    "blastp",
    "pangenome",
    "paralogs_only",
    "random",
    "loco",
    "vgae",
    "gcn",
    "other",
]
# Prefer runs_unif-native outdirs over older runs/-sourced aliases
SUPERSEDED_OUTDIRS = {
    "caduceus_run16_hashfrag_caduceus",  # superseded by caduceus_run16_caduceus_hashfrag
    "legnet_run5_hashfrag",  # superseded by legnet_run5_legnet_hashfrag
}
ROW_PX = 28
HEADER_PX = 20
AXIS_PX = 30
LEGACY_DIR_MARKERS = ("_BAD_", "BAD_random", "_legacy", "_LEGACY")


def _parse_outdir_name(name: str) -> tuple[str, str]:
    for model in ("caduceus", "legnet"):
        prefix = model + "_"
        if name.startswith(prefix):
            return model, name[len(prefix) :]
    return "unknown", name


def _is_legacy_dir(name: str) -> bool:
    if name in SUPERSEDED_OUTDIRS:
        return True
    return any(m in name for m in LEGACY_DIR_MARKERS)


def _short_label(run_label: str) -> str:
    s = run_label
    s = re.sub(r"^run(\d+)_caduceus_", r"run\1_", s)
    s = re.sub(r"^run(\d+)_legnet_", r"run\1_", s)
    s = s.replace("_ARCHIVED_MEGA_20260731T224550Z", "_ARCHIVED")
    s = s.replace("_BAD_random_reassign_20260731", "_BAD")
    return s


def _run_id_from_label(label: str) -> int:
    m = re.search(r"(?:^|_)run(\d+)(?:_|$)", label)
    if not m:
        m = re.match(r"run(\d+)", label)
    return int(m.group(1)) if m else 10**9


def _order_by_run_id(labels: list[str]) -> list[str]:
    return sorted(set(labels), key=lambda s: (_run_id_from_label(s), s))


def _axis_labels_by_run_id(run_shorts: list[str]) -> dict[str, str]:
    ordered = _order_by_run_id(list(run_shorts))
    counts: dict[int, int] = {}
    out: dict[str, str] = {}
    for lab in ordered:
        rid = _run_id_from_label(lab)
        counts[rid] = counts.get(rid, 0) + 1
        n = counts[rid]
        out[lab] = f"run{rid}" if n == 1 else f"run{rid}_{n}"
    return out


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


def discover_split_dirs(splits_root: Path, *, include_legacy: bool = False) -> list[Path]:
    out = []
    for d in sorted(splits_root.iterdir()):
        if not d.is_dir():
            continue
        if not include_legacy and _is_legacy_dir(d.name):
            continue
        if (d / "othologs.csv").is_file() and (d / "paralogs.csv").is_file():
            out.append(d)
    return out


def load_long(splits_root: Path, table: str, *, include_legacy: bool = False) -> pd.DataFrame:
    fname = "othologs.csv" if table == "othologs" else "paralogs.csv"
    group_col = "orthogroup" if table == "othologs" else "paragroup"
    rows = []
    for d in discover_split_dirs(splits_root, include_legacy=include_legacy):
        model, run_label = _parse_outdir_name(d.name)
        if model == "unknown":
            continue
        df = pd.read_csv(d / fname, sep="|", dtype={group_col: str})
        run_short = _short_label(run_label)
        strategy, params = _strategy_params(run_short)
        part = pd.DataFrame(
            {
                "model": model,
                "run_dir": d.name,
                "run_label": run_label,
                "run_short": run_short,
                "run_id": _run_id_from_label(run_label),
                "strategy": strategy,
                "strategy_params": params,
                "group_id": df[group_col].astype(str),
                "sd_random": pd.to_numeric(df["sd_random"], errors="coerce"),
                "table": table,
            }
        )
        rows.append(part.dropna(subset=["sd_random"]))
    if not rows:
        raise FileNotFoundError(f"No {fname} under {splits_root}")
    return pd.concat(rows, ignore_index=True)


def _run_meta(sub: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    order_short = _order_by_run_id(sub["run_short"].tolist())
    axis_map = _axis_labels_by_run_id(order_short)
    order_axis = [axis_map[s] for s in order_short]
    rows = []
    for s in order_short:
        hit = sub.loc[sub["run_short"] == s].iloc[0]
        rows.append(
            {
                "run_short": s,
                "run_axis": axis_map[s],
                "run_id": int(hit["run_id"]),
                "strategy": hit["strategy"],
                "strategy_params": hit["strategy_params"],
                "model": hit["model"],
            }
        )
    return pd.DataFrame(rows), order_axis


def _color():
    import altair as alt

    return alt.Color(
        "strategy:N",
        scale=alt.Scale(domain=FAM_DOMAIN, range=OKABE),
        legend=None,
    )


def _annotation_band(meta: pd.DataFrame, order_axis: list[str], height: int):
    """Left table with shared categorical y (matches boxstrip / heatmap)."""
    import altair as alt

    pieces = []
    for field, col in (
        ("strategy", "strategy"),
        ("strategy_params", "strategy_params"),
        ("run", "run_axis"),
    ):
        pieces.append(meta[["run_axis", "strategy"]].assign(field=field, label=meta[col].astype(str)))
    long = pd.concat(pieces, ignore_index=True)
    return (
        alt.Chart(long)
        .mark_text(align="left", dx=6, fontSize=11, baseline="middle")
        .encode(
            y=alt.Y(
                "run_axis:N",
                sort=order_axis,
                title=None,
                axis=None,
                scale=alt.Scale(paddingInner=0, paddingOuter=0),
            ),
            x=alt.X(
                "field:N",
                sort=["strategy", "strategy_params", "run"],
                title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelPadding=4, ticks=False, domain=False),
            ),
            text="label:N",
            color=_color(),
            tooltip=["run_axis", "strategy", "label"],
        )
        .properties(width=320, height=height)
    )


def _annotation_vconcat(meta: pd.DataFrame, order_axis: list[str]):
    """Left table as row-vconcat (matches violin panel heights exactly)."""
    import altair as alt

    color = _color()
    hdr = (
        alt.Chart(
            pd.DataFrame(
                {
                    "field": ["strategy", "strategy_params", "run"],
                    "label": ["strategy", "strategy_params", "run"],
                }
            )
        )
        .mark_text(align="left", dx=6, fontSize=11, fontWeight="bold", baseline="middle")
        .encode(
            x=alt.X(
                "field:N",
                sort=["strategy", "strategy_params", "run"],
                axis=None,
            ),
            text="label:N",
        )
        .properties(width=320, height=HEADER_PX)
    )
    rows = []
    for ax in order_axis:
        m = meta.loc[meta["run_axis"] == ax]
        long = pd.concat(
            [
                m.assign(field="strategy", label=m["strategy"].astype(str)),
                m.assign(field="strategy_params", label=m["strategy_params"].astype(str)),
                m.assign(field="run", label=m["run_axis"].astype(str)),
            ],
            ignore_index=True,
        )
        rows.append(
            alt.Chart(long)
            .mark_text(align="left", dx=6, fontSize=11, baseline="middle")
            .encode(
                x=alt.X(
                    "field:N",
                    sort=["strategy", "strategy_params", "run"],
                    axis=None,
                ),
                text="label:N",
                color=color,
                tooltip=["strategy", "strategy_params", "run_axis"],
            )
            .properties(width=320, height=ROW_PX)
        )
    pad = (
        alt.Chart(pd.DataFrame({"t": [""]}))
        .mark_text(opacity=0)
        .encode(text="t:N")
        .properties(width=320, height=AXIS_PX)
    )
    return alt.vconcat(hdr, *rows, pad, spacing=0)


def _title_wrap(chart, title: str):
    import altair as alt

    return alt.vconcat(
        alt.Chart(pd.DataFrame({"t": [title]}))
        .mark_text(align="left", fontSize=14, fontWeight="bold", dx=0)
        .encode(text="t:N")
        .properties(height=24, width=720),
        chart,
        spacing=6,
    )


def _kde_violin_frame(sub: pd.DataFrame, order_axis: list[str], x_max: float) -> pd.DataFrame:
    """Precompute per-run KDE on a shared grid for band-aligned horizontal violins."""
    from scipy.stats import gaussian_kde

    grid = np.linspace(0.0, x_max, 128)
    idx = {a: i for i, a in enumerate(order_axis)}
    rows: list[dict] = []
    for ax in order_axis:
        part = sub.loc[sub["run_axis"] == ax]
        vals = part["sd_random"].to_numpy(dtype=float)
        strat = str(part["strategy"].iloc[0])
        if vals.size < 5:
            dens = np.zeros_like(grid)
        else:
            # clip tiny jitter for singular KDE
            if np.unique(vals).size < 2:
                dens = np.exp(-0.5 * ((grid - float(vals.mean())) / max(x_max * 0.02, 1e-6)) ** 2)
            else:
                dens = gaussian_kde(vals)(grid)
            m = float(dens.max()) if dens.size else 0.0
            dens = dens / m if m > 0 else dens
        y0 = float(idx[ax])
        for x, d in zip(grid, dens, strict=True):
            rows.append(
                {
                    "run_axis": ax,
                    "strategy": strat,
                    "sd_random": float(x),
                    "density": float(d),
                    "y_lo": y0 - 0.42 * float(d),
                    "y_hi": y0 + 0.42 * float(d),
                }
            )
    return pd.DataFrame(rows)


def _annotation_quant(meta: pd.DataFrame, order_axis: list[str], height: int):
    """Left table on the same quantitative y as KDE violins (0=top). No axes (shared chrome outside)."""
    import altair as alt

    idx = {a: float(i) for i, a in enumerate(order_axis)}
    pieces = []
    for field, col in (
        ("strategy", "strategy"),
        ("strategy_params", "strategy_params"),
        ("run", "run_axis"),
    ):
        pieces.append(
            meta.assign(
                field=field,
                label=meta[col].astype(str),
                y=meta["run_axis"].map(idx),
            )
        )
    long = pd.concat(pieces, ignore_index=True)
    n = len(order_axis)
    return (
        alt.Chart(long)
        .mark_text(align="left", dx=6, fontSize=11, baseline="middle")
        .encode(
            y=alt.Y(
                "y:Q",
                title=None,
                axis=None,
                scale=alt.Scale(domain=[-0.5, n - 0.5], reverse=True),
            ),
            x=alt.X(
                "field:N",
                sort=["strategy", "strategy_params", "run"],
                title=None,
                axis=None,
                scale=alt.Scale(paddingInner=0.15, paddingOuter=0.05),
            ),
            text="label:N",
            color=_color(),
            tooltip=["run_axis", "strategy", "label"],
        )
        .properties(width=360, height=height)
    )


def _violin_col_header(width: int = 360):
    import altair as alt

    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "field": ["strategy", "strategy_params", "run"],
                    "label": ["strategy", "strategy_params", "run"],
                }
            )
        )
        .mark_text(align="left", dx=6, fontSize=11, fontWeight="bold", baseline="middle")
        .encode(
            x=alt.X(
                "field:N",
                sort=["strategy", "strategy_params", "run"],
                axis=None,
                scale=alt.Scale(paddingInner=0.15, paddingOuter=0.05),
            ),
            text="label:N",
        )
        .properties(width=width, height=HEADER_PX)
    )


def plot_violin(df: pd.DataFrame, model: str, table: str, outdir: Path) -> list[Path]:
    import altair as alt
    from src.train_viz.plotting import save_altair_chart

    alt.data_transformers.disable_max_rows()
    sub = df[(df["model"] == model) & (df["table"] == table)].copy()
    if sub.empty:
        return []
    meta, order_axis = _run_meta(sub)
    axis_map = dict(zip(meta["run_short"], meta["run_axis"], strict=True))
    sub["run_axis"] = sub["run_short"].map(axis_map)

    n = len(order_axis)
    height = ROW_PX * n
    x_max = float(sub["sd_random"].quantile(0.995))
    color = _color()
    y_scale = alt.Scale(domain=[-0.5, n - 0.5], reverse=True)

    dens_df = _kde_violin_frame(sub, order_axis, x_max)
    # Row-wise vconcat(hconcat(...)) so axis chrome cannot vertically desync columns
    violin_body = (
        alt.Chart(dens_df)
        .mark_area(opacity=0.85, interpolate="monotone")
        .encode(
            x=alt.X("sd_random:Q", title=None, scale=alt.Scale(domain=[0, x_max]), axis=None),
            y=alt.Y("y_lo:Q", title=None, axis=None, scale=y_scale),
            y2="y_hi:Q",
            color=color,
            detail="run_axis:N",
            tooltip=["run_axis", "strategy", "sd_random"],
        )
        .properties(width=400, height=height)
    )
    # Draw x labels inside the reserved AXIS_PX band (axis=None) to avoid external padding
    xaxis = (
        alt.Chart(pd.DataFrame({"sd_random": np.linspace(0.0, x_max, 6), "y": 0.0}))
        .mark_text(fontSize=10, dy=0, baseline="middle")
        .encode(
            x=alt.X("sd_random:Q", title=None, scale=alt.Scale(domain=[0, x_max]), axis=None),
            y=alt.Y("y:Q", title=None, axis=None, scale=alt.Scale(domain=[-1, 1])),
            text=alt.Text("sd_random:Q", format=".0f"),
        )
        .properties(width=400, height=AXIS_PX)
    )
    x_title = (
        alt.Chart(pd.DataFrame({"t": ["sd_random"], "x": [0.5], "y": [0.0]}))
        .mark_text(fontSize=11, baseline="middle")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1, 1]), axis=None),
            text="t:N",
        )
        .properties(width=400, height=18)
    )
    hdr = _violin_col_header(360)
    hdr_spacer = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_point(opacity=0)
        .encode(x="x:Q")
        .properties(width=400, height=HEADER_PX)
    )
    left_pad = (
        alt.Chart(pd.DataFrame({"t": [""]}))
        .mark_text(opacity=0)
        .encode(text="t:N")
        .properties(width=360, height=AXIS_PX)
    )
    left_pad2 = (
        alt.Chart(pd.DataFrame({"t": [""]}))
        .mark_text(opacity=0)
        .encode(text="t:N")
        .properties(width=360, height=18)
    )
    ann_body = _annotation_quant(meta, order_axis, height)
    violin_block = alt.vconcat(
        alt.hconcat(hdr, hdr_spacer, spacing=8),
        alt.hconcat(ann_body, violin_body, spacing=8),
        alt.hconcat(left_pad, xaxis, spacing=8),
        alt.hconcat(left_pad2, x_title, spacing=8),
        spacing=0,
    )

    y = alt.Y(
        "run_axis:N",
        sort=order_axis,
        title=None,
        axis=None,
        scale=alt.Scale(paddingInner=0, paddingOuter=0),
    )
    parts = [
        g.sample(n=min(len(g), 2000), random_state=42)
        for _, g in sub.groupby("run_axis", sort=False)
    ]
    sample = pd.concat(parts, ignore_index=True) if parts else sub.iloc[0:0]
    box = (
        alt.Chart(sub)
        .mark_boxplot(extent="min-max", size=12, outliers=False)
        .encode(
            y=y,
            x=alt.X("sd_random:Q", title="sd_random"),
            color=color,
            tooltip=["run_axis", "strategy", "strategy_params", "run_short"],
        )
    )
    strip = (
        alt.Chart(sample)
        .mark_circle(size=9, opacity=0.12)
        .encode(y=y, x="sd_random:Q", color=color, tooltip=["run_axis", "strategy", "sd_random"])
    )
    boxstrip = (box + strip).properties(width=400, height=height)
    ann_band = _annotation_band(meta, order_axis, height)

    fig_v = _title_wrap(
        violin_block,
        f"{model} · {table} · sd_random violin (run ∼ sd)",
    )
    fig_b = _title_wrap(
        alt.hconcat(ann_band, boxstrip, spacing=8),
        f"{model} · {table} · sd_random box+strip (run ∼ sd)",
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    written += save_altair_chart(fig_v, outdir / f"Figure_01_{table}_sd_violin_facet_{model}")
    written += save_altair_chart(fig_b, outdir / f"Figure_02_{table}_sd_boxstrip_{model}")
    return written


def plot_corr_heatmap(df: pd.DataFrame, model: str, table: str, outdir: Path) -> list[Path]:
    import altair as alt
    from src.train_viz.plotting import save_altair_chart

    sub = df[(df["model"] == model) & (df["table"] == table)].copy()
    if sub.empty:
        return []
    meta, order_axis = _run_meta(sub)
    axis_map = dict(zip(meta["run_short"], meta["run_axis"], strict=True))
    sub["run_axis"] = sub["run_short"].map(axis_map)
    pivot = sub.pivot_table(index="group_id", columns="run_axis", values="sd_random", aggfunc="mean")
    pivot = pivot.dropna(thresh=2)
    cols = [c for c in order_axis if c in pivot.columns]
    if len(cols) < 2:
        return []
    corr = pivot[cols].corr(method="pearson")
    long = (
        corr.reset_index()
        .rename(columns={corr.index.name or "index": "run_a"})
        .melt(id_vars="run_a", var_name="run_b", value_name="pearson_r")
    )
    n = len(cols)
    height = ROW_PX * n
    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("run_a:N", sort=cols, title=None, axis=alt.Axis(labelAngle=0, labelPadding=4)),
            y=alt.Y(
                "run_b:N",
                sort=cols,
                title=None,
                axis=None,
                scale=alt.Scale(paddingInner=0, paddingOuter=0),
            ),
            color=alt.Color(
                "pearson_r:Q",
                scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                title="Pearson r",
            ),
            tooltip=["run_a:N", "run_b:N", alt.Tooltip("pearson_r:Q", format=".3f")],
        )
        .properties(width=ROW_PX * n, height=height)
    )
    labels = (
        alt.Chart(long)
        .mark_text(size=8)
        .encode(
            x=alt.X("run_a:N", sort=cols),
            y=alt.Y("run_b:N", sort=cols, axis=None),
            text=alt.Text("pearson_r:Q", format=".2f"),
            color=alt.condition("abs(datum.pearson_r) > 0.55", alt.value("white"), alt.value("black")),
        )
    )
    ann = _annotation_band(meta, cols, height)
    body = alt.hconcat(ann, heat + labels, spacing=8)
    fig = _title_wrap(body, f"{model} · {table} · sd_random correlation")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    corr_path = outdir / f"{table}_sd_corr_{model}.tsv"
    corr.loc[cols, cols].to_csv(corr_path, sep="\t", float_format="%.6f")
    written = save_altair_chart(fig, outdir / f"Figure_03_{table}_sd_corr_heatmap_{model}")
    written.append(corr_path)
    return written


def _ks_similarity_matrix(series_by_run: dict[str, np.ndarray], order: list[str]) -> pd.DataFrame:
    """Pairwise KS similarity = 1 − D (D = two-sample KS statistic)."""
    from scipy.stats import ks_2samp

    n = len(order)
    mat = np.eye(n, dtype=float)
    for i in range(n):
        a = series_by_run[order[i]]
        for j in range(i + 1, n):
            b = series_by_run[order[j]]
            d = float(ks_2samp(a, b, method="auto").statistic)
            sim = 1.0 - d
            mat[i, j] = sim
            mat[j, i] = sim
    return pd.DataFrame(mat, index=order, columns=order)


def plot_ks_heatmaps(df: pd.DataFrame, out_root: Path) -> list[Path]:
    """Two global heatmaps (othologs / paralogs) of KS distribution similarity."""
    import altair as alt
    from src.train_viz.plotting import save_altair_chart

    written: list[Path] = []
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for table in ("othologs", "paralogs"):
        sub = df[df["table"] == table].copy()
        if sub.empty:
            continue
        # Unique axis across models: model:run_axis
        meta_parts = []
        for model, g in sub.groupby("model", sort=True):
            meta_m, order_m = _run_meta(g)
            meta_m = meta_m.copy()
            meta_m["model"] = model
            meta_m["run_key"] = meta_m["model"] + ":" + meta_m["run_axis"]
            meta_parts.append(meta_m)
        meta = pd.concat(meta_parts, ignore_index=True)
        # order: by model then run_id
        meta = meta.sort_values(["model", "run_id", "run_axis"]).reset_index(drop=True)
        order = meta["run_key"].tolist()

        axis_map = {}
        for _, row in meta.iterrows():
            axis_map[(row["model"], row["run_short"])] = row["run_key"]
        sub["run_key"] = [
            axis_map[(m, s)] for m, s in zip(sub["model"], sub["run_short"], strict=True)
        ]

        series_by_run = {
            key: g["sd_random"].to_numpy(dtype=float)
            for key, g in sub.groupby("run_key", sort=False)
        }
        order = [k for k in order if k in series_by_run]
        if len(order) < 2:
            continue

        sim = _ks_similarity_matrix(series_by_run, order)
        long = (
            sim.reset_index()
            .rename(columns={"index": "run_a"})
            .melt(id_vars="run_a", var_name="run_b", value_name="ks_similarity")
        )
        n = len(order)
        height = ROW_PX * n
        # annotation meta with run_key as run_axis for shared encoder
        ann_meta = meta.loc[meta["run_key"].isin(order)].copy()
        ann_meta = ann_meta.set_index("run_key").loc[order].reset_index()
        ann_meta["run_axis"] = ann_meta["run_key"]
        # show short run label in run column, keep strategy cols
        ann_meta_plot = ann_meta.copy()
        ann_meta_plot["run_axis_label"] = ann_meta_plot["model"].str[0].str.upper() + ":" + ann_meta_plot[
            "run_axis"
        ].str.split(":").str[-1]
        # Use run_key on y; display labels via separate fields
        pieces = []
        for field, col in (
            ("strategy", "strategy"),
            ("strategy_params", "strategy_params"),
            ("run", "run_axis_label"),
        ):
            pieces.append(
                ann_meta_plot[["run_key", "strategy"]].assign(
                    field=field, label=ann_meta_plot[col].astype(str), run_axis=ann_meta_plot["run_key"]
                )
            )
        ann_long = pd.concat(pieces, ignore_index=True)
        ann = (
            alt.Chart(ann_long)
            .mark_text(align="left", dx=6, fontSize=10, baseline="middle")
            .encode(
                y=alt.Y(
                    "run_axis:N",
                    sort=order,
                    title=None,
                    axis=None,
                    scale=alt.Scale(paddingInner=0, paddingOuter=0),
                ),
                x=alt.X(
                    "field:N",
                    sort=["strategy", "strategy_params", "run"],
                    title=None,
                    axis=alt.Axis(orient="top", labelAngle=0, labelPadding=4, ticks=False, domain=False),
                ),
                text="label:N",
                color=_color(),
                tooltip=["run_axis", "strategy", "label"],
            )
            .properties(width=360, height=height)
        )
        heat = (
            alt.Chart(long)
            .mark_rect()
            .encode(
                x=alt.X("run_a:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=140)),
                y=alt.Y(
                    "run_b:N",
                    sort=order,
                    title=None,
                    axis=None,
                    scale=alt.Scale(paddingInner=0, paddingOuter=0),
                ),
                color=alt.Color(
                    "ks_similarity:Q",
                    scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                    title="KS sim (1−D)",
                ),
                tooltip=[
                    "run_a:N",
                    "run_b:N",
                    alt.Tooltip("ks_similarity:Q", format=".3f"),
                ],
            )
            .properties(width=max(ROW_PX * n, 280), height=height)
        )
        # labels only if not too many cells
        layers = heat
        if n <= 14:
            layers = heat + (
                alt.Chart(long)
                .mark_text(size=7)
                .encode(
                    x=alt.X("run_a:N", sort=order),
                    y=alt.Y("run_b:N", sort=order, axis=None),
                    text=alt.Text("ks_similarity:Q", format=".2f"),
                    color=alt.condition(
                        "datum.ks_similarity > 0.55",
                        alt.value("black"),
                        alt.value("white"),
                    ),
                )
            )
        fig = _title_wrap(
            alt.hconcat(ann, layers, spacing=8),
            f"all models · {table} · sd_random KS similarity (1−D)",
        )
        tsv = out_root / f"Figure_04_{table}_sd_ks_similarity.tsv"
        sim.to_csv(tsv, sep="\t", float_format="%.6f")
        written.append(tsv)
        written += save_altair_chart(fig, out_root / f"Figure_04_{table}_sd_ks_heatmap")
    return written


def run(splits_root: Path, out_root: Path) -> dict:
    splits_root = Path(splits_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [load_long(splits_root, "othologs"), load_long(splits_root, "paralogs")],
        ignore_index=True,
    )
    audit_path = out_root / "presence_audit.tsv"
    df.groupby(["model", "strategy", "run_dir"], as_index=False).size().to_csv(
        audit_path, sep="\t", index=False
    )
    written: list[str] = [str(audit_path)]
    for model in sorted(df["model"].unique()):
        mdir = out_root / model
        mdir.mkdir(parents=True, exist_ok=True)
        for table in ("othologs", "paralogs"):
            written += [str(p) for p in plot_violin(df, model, table, mdir)]
            written += [str(p) for p in plot_corr_heatmap(df, model, table, mdir)]
    written += [str(p) for p in plot_ks_heatmaps(df, out_root)]
    summary = {
        "splits_root": str(splits_root),
        "out_root": str(out_root),
        "n_rows": int(len(df)),
        "excluded_legacy": [d.name for d in splits_root.iterdir() if d.is_dir() and _is_legacy_dir(d.name)],
        "layout": "violin: shared quantitative y KDE; box/heatmap: band y; KS: 1-D global heatmaps",
        "written": written,
    }
    (out_root / "viz_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-root", type=Path, default=Path("runs_unif/splits"))
    p.add_argument("--outdir", type=Path, default=Path("runs_unif/splits/figures"))
    args = p.parse_args(argv)
    print(json.dumps(run(args.splits_root, args.outdir), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
