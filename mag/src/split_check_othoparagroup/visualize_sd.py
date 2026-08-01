"""Altair horizontal sd_random plots with left annotation table (aligned by run_id)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
FAM_DOMAIN = ["gc", "hashfrag", "kmer", "mmseqs", "pangenome", "paralogs_only", "random", "other"]
ROW_PX = 28


def _parse_outdir_name(name: str) -> tuple[str, str]:
    for model in ("caduceus", "legnet"):
        prefix = model + "_"
        if name.startswith(prefix):
            return model, name[len(prefix):]
    return "unknown", name


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
    if "mmseq" in s:
        return "mmseqs"
    if "hashfrag" in s:
        return "hashfrag"
    if "pangenome" in s:
        return "pangenome"
    if "kmer" in s or re.search(r"_k\d+", s):
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
        "mmseqs": r"^mmseqs_?",
        "hashfrag": r"^hashfrag_?",
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


def discover_split_dirs(splits_root: Path) -> list[Path]:
    out = []
    for d in sorted(splits_root.iterdir()):
        if d.is_dir() and (d / "othologs.csv").is_file() and (d / "paralogs.csv").is_file():
            out.append(d)
    return out


def load_long(splits_root: Path, table: str) -> pd.DataFrame:
    fname = "othologs.csv" if table == "othologs" else "paralogs.csv"
    group_col = "orthogroup" if table == "othologs" else "paragroup"
    rows = []
    for d in discover_split_dirs(splits_root):
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
            }
        )
    return pd.DataFrame(rows), order_axis


def _annotation_table(meta: pd.DataFrame, order_axis: list[str]):
    """Left table chart: x in {run, strategy, strategy_params}, y=run_axis."""
    import altair as alt

    pieces = []
    for field, col in (("run", "run_axis"), ("strategy", "strategy"), ("strategy_params", "strategy_params")):
        pieces.append(meta[["run_axis", "strategy"]].assign(field=field, label=meta[col].astype(str)))
    long = pd.concat(pieces, ignore_index=True)
    height = ROW_PX * len(order_axis)
    return (
        alt.Chart(long)
        .mark_text(align="left", dx=6, fontSize=11)
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
                sort=["run", "strategy", "strategy_params"],
                title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelPadding=4, ticks=False, domain=False),
            ),
            text="label:N",
            color=alt.Color(
                "strategy:N",
                scale=alt.Scale(domain=FAM_DOMAIN, range=OKABE),
                legend=None,
            ),
            tooltip=["run_axis", "strategy", "label"],
        )
        .properties(width=320, height=height)
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
    color = alt.Color("strategy:N", scale=alt.Scale(domain=FAM_DOMAIN, range=OKABE), legend=None)

    panels = []
    for i, axis_lab in enumerate(order_axis):
        part = sub[sub["run_axis"] == axis_lab]
        show_x = i == n - 1
        panels.append(
            alt.Chart(part)
            .transform_density(
                "sd_random",
                as_=["sd_random", "density"],
                groupby=["strategy"],
                extent=[0, x_max],
            )
            .mark_area(orient="vertical", opacity=0.85)
            .encode(
                x=alt.X(
                    "sd_random:Q",
                    title="sd_random" if show_x else None,
                    scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(title="sd_random") if show_x else None,
                ),
                y=alt.Y("density:Q", stack="center", title=None, axis=None),
                color=color,
            )
            .properties(width=400, height=ROW_PX)
        )
    violin = alt.vconcat(*panels, spacing=0)

    parts = [g.sample(n=min(len(g), 2000), random_state=42) for _, g in sub.groupby("run_axis", sort=False)]
    sample = pd.concat(parts, ignore_index=True) if parts else sub.iloc[0:0]
    y = alt.Y(
        "run_axis:N",
        sort=order_axis,
        title=None,
        axis=None,
        scale=alt.Scale(paddingInner=0, paddingOuter=0),
    )
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
    ann = _annotation_table(meta, order_axis)

    # Avoid .properties(title=...) on HConcat (Altair 6 schema bug); title on a wrap layer
    def titled(chart, title: str):
        return alt.vconcat(
            alt.Chart(pd.DataFrame({"t": [title]}))
            .mark_text(align="left", fontSize=14, fontWeight="bold", dx=0)
            .encode(text="t:N")
            .properties(height=24, width=720),
            chart,
            spacing=6,
        )

    fig_v = titled(alt.hconcat(ann, violin, spacing=8), f"{model} · {table} · sd_random violin (run ∼ sd)")
    fig_b = titled(alt.hconcat(ann, boxstrip, spacing=8), f"{model} · {table} · sd_random box+strip (run ∼ sd)")

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
    ann = _annotation_table(meta, cols)
    body = alt.hconcat(ann, heat + labels, spacing=8)
    fig = alt.vconcat(
        alt.Chart(pd.DataFrame({"t": [f"{model} · {table} · sd_random correlation"]}))
        .mark_text(align="left", fontSize=14, fontWeight="bold")
        .encode(text="t:N")
        .properties(height=24, width=720),
        body,
        spacing=6,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    corr_path = outdir / f"{table}_sd_corr_{model}.tsv"
    corr.loc[cols, cols].to_csv(corr_path, sep="\t", float_format="%.6f")
    written = save_altair_chart(fig, outdir / f"Figure_03_{table}_sd_corr_heatmap_{model}")
    written.append(corr_path)
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
    df.groupby(["model", "strategy", "run_dir"], as_index=False).size().to_csv(audit_path, sep="\t", index=False)
    written: list[str] = [str(audit_path)]
    for model in sorted(df["model"].unique()):
        mdir = out_root / model
        mdir.mkdir(parents=True, exist_ok=True)
        for table in ("othologs", "paralogs"):
            written += [str(p) for p in plot_violin(df, model, table, mdir)]
            written += [str(p) for p in plot_corr_heatmap(df, model, table, mdir)]
    summary = {
        "splits_root": str(splits_root),
        "out_root": str(out_root),
        "n_rows": int(len(df)),
        "layout": "hconcat(left_table[run,strategy,strategy_params], horizontal chart); y ordered by run_id",
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
