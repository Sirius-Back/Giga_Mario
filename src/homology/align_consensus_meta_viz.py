"""Meta-cluster consensus profiles: coverage filter, fine bins, ≤20 groups, ATG starts."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.homology.align_consensus import parse_fasta
from src.homology.align_consensus_viz import (
    RAW_COLS,
    SCOPES,
    _col,
    discover_metric_files,
    load_metrics_table,
)
from src.homology.visualize import _apply_cns_style

MIN_ALIGN_FRAC = 0.5
DEFAULT_N_BINS = 50
MAX_META_CLUSTERS = 20
TSS_REL_POS = 0.5  # LegNet CRS is TSS-centered (±100 bp)


def filter_well_aligned(df: pd.DataFrame, *, min_frac: float = MIN_ALIGN_FRAC) -> pd.DataFrame:
    """Drop columns with alignment coverage ``n_non_gap / n_seqs < min_frac``."""
    if "n_non_gap" not in df.columns or "n_seqs" not in df.columns:
        raise ValueError("metrics need n_non_gap and n_seqs for coverage filter")
    out = df.copy()
    out["align_frac"] = out["n_non_gap"] / out["n_seqs"].clip(lower=1)
    kept = out[out["align_frac"] >= float(min_frac)].copy()
    if kept.empty:
        raise ValueError(f"No positions with align_frac>={min_frac}")
    return kept


def add_bins_on_full_length(
    df: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Bin by relative position on the *full* alignment length (before filter).

    Expects ``aln_length`` = max position per cluster from the unfiltered table.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    out = df.copy()
    if "aln_length" not in out.columns:
        raise ValueError("aln_length required (full alignment length per cluster)")
    denom = (out["aln_length"] - 1).clip(lower=1)
    out["rel_pos"] = (out["position"] - 1) / denom
    out["pos_bin"] = np.minimum((out["rel_pos"] * n_bins).astype(int), n_bins - 1)
    return out


def per_cluster_bin_similarity(
    filtered: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_N_BINS,
    variant: str = "raw",
) -> pd.DataFrame:
    """Mean similarity per (cluster, pos_bin, scope) on well-aligned columns only."""
    rows: list[dict[str, Any]] = []
    for (cluster, pos_bin), g in filtered.groupby(["cluster", "pos_bin"], sort=False):
        for scope in SCOPES:
            col = _col(scope, variant)
            vals = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "cluster": cluster,
                    "pos_bin": int(pos_bin),
                    "scope": scope,
                    "similarity": float(vals.mean()),
                    "n_positions": int(len(vals)),
                    "rel_pos_mid": (int(pos_bin) + 0.5) / n_bins,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No per-cluster bin similarities after filter")
    return out


def profile_matrix(
    bin_sim: pd.DataFrame,
    *,
    n_bins: int,
    scopes: tuple[str, ...] = SCOPES,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Wide matrix: rows=cluster, cols=scope×bin similarities (NaN→column median)."""
    clusters = sorted(bin_sim["cluster"].unique())
    col_names: list[str] = []
    for scope in scopes:
        for b in range(n_bins):
            col_names.append(f"{scope}_bin{b:02d}")
    mat = np.full((len(clusters), len(col_names)), np.nan, dtype=float)
    idx = {c: i for i, c in enumerate(clusters)}
    jmap = {name: j for j, name in enumerate(col_names)}
    for _, row in bin_sim.iterrows():
        key = f"{row['scope']}_bin{int(row['pos_bin']):02d}"
        if key not in jmap:
            continue
        mat[idx[row["cluster"]], jmap[key]] = float(row["similarity"])
    # impute
    for j in range(mat.shape[1]):
        col = mat[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[~np.isfinite(col)] = med
        mat[:, j] = col
    wide = pd.DataFrame(mat, index=clusters, columns=col_names)
    wide.index.name = "cluster"
    return wide, mat, clusters


def choose_k_and_cluster(
    mat: np.ndarray,
    *,
    max_k: int = MAX_META_CLUSTERS,
    seed: int = 42,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """KMeans with k in 2..max_k chosen by silhouette (capped at n_samples-1)."""
    n = mat.shape[0]
    if n < 2:
        raise ValueError("Need >=2 clusters for meta-clustering")
    k_max = min(int(max_k), n - 1, 20)
    if k_max < 2:
        labels = np.zeros(n, dtype=int)
        return labels, 1, {"k": 1, "silhouette": None, "reason": "n<3"}

    scaler = StandardScaler()
    X = scaler.fit_transform(mat)
    best_k = 2
    best_score = -1.0
    best_labels = None
    scores: dict[str, float] = {}
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels, metric="euclidean"))
        scores[str(k)] = score
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        km = KMeans(n_clusters=2, random_state=seed, n_init=10)
        best_labels = km.fit_predict(X)
        best_k = 2
        best_score = float("nan")
    meta = {
        "k": int(best_k),
        "silhouette": best_score if np.isfinite(best_score) else None,
        "scores": scores,
        "max_k": k_max,
        "seed": seed,
    }
    return np.asarray(best_labels, dtype=int), int(best_k), meta


def find_atg_rel_positions(aln_path: Path) -> dict[str, Any]:
    """Map first ATG in each sequence to alignment-relative position; return summary."""
    records = parse_fasta(aln_path)
    if not records:
        raise ValueError(f"Empty alignment: {aln_path}")
    L = len(records[0].sequence)
    if L < 3:
        return {
            "cluster": aln_path.stem.replace(".aln", ""),
            "n_seqs": len(records),
            "n_with_atg": 0,
            "atg_rel_mean": float("nan"),
            "atg_rel_std": float("nan"),
            "atg_rel_q25": float("nan"),
            "atg_rel_q75": float("nan"),
            "aln_length": L,
        }
    rels: list[float] = []
    for rec in records:
        seq = rec.sequence.upper()
        # scan ungapped for first ATG, map to alignment index
        ungapped = []
        map_u2a: list[int] = []
        for i, c in enumerate(seq):
            if c != "-":
                ungapped.append(c)
                map_u2a.append(i)
        u = "".join(ungapped)
        idx = u.find("ATG")
        if idx < 0:
            continue
        aln_i = map_u2a[idx]
        rels.append(aln_i / max(L - 1, 1))
    cluster = aln_path.name.replace(".aln.fa", "").replace(".aln.fasta", "")
    if not rels:
        return {
            "cluster": cluster,
            "n_seqs": len(records),
            "n_with_atg": 0,
            "atg_rel_mean": float("nan"),
            "atg_rel_std": float("nan"),
            "atg_rel_q25": float("nan"),
            "atg_rel_q75": float("nan"),
            "aln_length": L,
        }
    arr = np.asarray(rels, dtype=float)
    return {
        "cluster": cluster,
        "n_seqs": len(records),
        "n_with_atg": int(len(arr)),
        "atg_rel_mean": float(arr.mean()),
        "atg_rel_std": float(arr.std(ddof=0)),
        "atg_rel_q25": float(np.quantile(arr, 0.25)),
        "atg_rel_q75": float(np.quantile(arr, 0.75)),
        "aln_length": L,
    }


def _atg_worker(path_s: str) -> dict[str, Any]:
    try:
        return find_atg_rel_positions(Path(path_s))
    except Exception as exc:  # noqa: BLE001
        return {
            "cluster": Path(path_s).name.replace(".aln.fa", ""),
            "n_seqs": 0,
            "n_with_atg": 0,
            "atg_rel_mean": float("nan"),
            "atg_rel_std": float("nan"),
            "atg_rel_q25": float("nan"),
            "atg_rel_q75": float("nan"),
            "aln_length": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_atg_table(
    aln_dir: Path | str,
    *,
    workers: int = 8,
    limit: int = 0,
) -> pd.DataFrame:
    aln_dir = Path(aln_dir)
    paths = sorted(aln_dir.glob("cluster_*.aln.fa"))
    if not paths:
        raise FileNotFoundError(f"No alignments under {aln_dir}")
    if limit > 0:
        paths = paths[:limit]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_atg_worker, str(p)) for p in paths]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return pd.DataFrame(rows)


def build_meta_tables(
    metrics_dir: Path | str,
    aln_dir: Path | str | None = None,
    *,
    limit: int = 0,
    n_bins: int = DEFAULT_N_BINS,
    min_align_frac: float = MIN_ALIGN_FRAC,
    max_k: int = MAX_META_CLUSTERS,
    seed: int = 42,
    atg_workers: int = 8,
) -> dict[str, Any]:
    """Filter → bin similarities → meta-clusters → ATG starts."""
    needed = [
        "cluster",
        "position",
        "n_seqs",
        "n_non_gap",
        "overall_consensus_rate",
        "orthologs_consensus_rate",
        "paralogs_consensus_rate",
    ]
    raw = load_metrics_table(metrics_dir, limit=limit, columns=needed)
    aln_len = raw.groupby("cluster")["position"].max().rename("aln_length")
    raw = raw.merge(aln_len, on="cluster", how="left")

    filtered = filter_well_aligned(raw, min_frac=min_align_frac)
    filtered = add_bins_on_full_length(filtered, n_bins=n_bins)
    bin_sim = per_cluster_bin_similarity(filtered, n_bins=n_bins, variant="raw")

    wide, mat, clusters = profile_matrix(bin_sim, n_bins=n_bins)
    labels, k, km_meta = choose_k_and_cluster(mat, max_k=max_k, seed=seed)
    assign = pd.DataFrame({"cluster": clusters, "meta_cluster": labels})
    bin_sim = bin_sim.merge(assign, on="cluster", how="left")

    # ATG
    if aln_dir is None:
        aln_dir = Path(metrics_dir).resolve().parent
    atg = collect_atg_table(aln_dir, workers=atg_workers, limit=limit)
    atg = atg.merge(assign, on="cluster", how="left")

    rows: list[dict[str, Any]] = []
    for mc, g in atg.groupby("meta_cluster"):
        vals = g["atg_rel_mean"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "meta_cluster": int(mc),
                "n_opg_with_atg": int(vals.size),
                "atg_rel_mean": float(vals.mean()),
                "atg_rel_std": float(vals.std(ddof=0)),
                "atg_rel_q25": float(np.quantile(vals, 0.25)),
                "atg_rel_q75": float(np.quantile(vals, 0.75)),
            }
        )
    atg_meta = pd.DataFrame(rows)

    filter_stats = {
        "n_positions_raw": int(len(raw)),
        "n_positions_kept": int(len(filtered)),
        "frac_kept": float(len(filtered) / max(len(raw), 1)),
        "min_align_frac": min_align_frac,
        "n_bins": n_bins,
        "n_opg": int(len(clusters)),
        "k_meta": k,
        "tss_rel_pos": TSS_REL_POS,
    }
    return {
        "bin_sim": bin_sim,
        "assign": assign,
        "wide": wide,
        "atg": atg,
        "atg_meta": atg_meta,
        "kmeans_meta": km_meta,
        "filter_stats": filter_stats,
        "n_bins": n_bins,
    }


def plot_meta_cnsplots(
    bin_sim: pd.DataFrame,
    atg_meta: pd.DataFrame,
    outdir: Path | str,
    *,
    n_bins: int,
    dpi: int = 300,
) -> list[Path]:
    """cnsplots: profile heat-ish lines + per-meta violin grids (sampled)."""
    import matplotlib.pyplot as plt
    import cnsplots as cns

    from src.train_viz.plotting import save_cns_figure

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _apply_cns_style(dpi)
    written: list[Path] = []

    # Mean profile per meta_cluster × scope
    prof = (
        bin_sim.groupby(["meta_cluster", "pos_bin", "scope"], as_index=False)["similarity"]
        .mean()
    )
    for scope in SCOPES:
        sub = prof[prof["scope"] == scope].copy()
        sub["meta_cluster"] = sub["meta_cluster"].astype(int).astype(str)
        cns.figure(width=480, height=300)
        ax = cns.lineplot(
            data=sub,
            x="pos_bin",
            y="similarity",
            hue="meta_cluster",
        )
        ax.set_xlabel("Relative-position bin")
        ax.set_ylabel("Mean similarity (align≥50%)")
        ax.set_title(f"Meta-cluster similarity profiles — {scope}")
        ax.axvline(TSS_REL_POS * n_bins - 0.5, color="#666666", ls="--", lw=0.8, label="TSS center")
        if not atg_meta.empty:
            for _, r in atg_meta.iterrows():
                x = float(r["atg_rel_mean"]) * n_bins - 0.5
                ax.axvline(x, color="#D55E00", alpha=0.25, lw=0.6)
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_10_meta_profile_{scope}", dpi))

    # One combined point+violin style: sample up to 8 meta-clusters for static readability
    metas = sorted(bin_sim["meta_cluster"].dropna().unique())
    show = metas[: min(8, len(metas))]
    sub = bin_sim[bin_sim["meta_cluster"].isin(show)].copy()
    sub["meta_cluster"] = sub["meta_cluster"].astype(int).astype(str)
    sub["pos_bin_s"] = sub["pos_bin"].astype(str)

    cns.figure(width=560, height=320)
    ax = cns.violinplot(
        data=sub,
        x="pos_bin_s",
        y="similarity",
        hue="scope",
        add_box=False,
    )
    ax.set_xlabel("Position bin")
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity by bin (subset of meta-clusters pooled)")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_11_similarity_violin_by_bin_pooled", dpi))

    # Per meta-cluster panels (static): mean± points
    for mc in metas:
        g = bin_sim[bin_sim["meta_cluster"] == mc]
        cns.figure(width=480, height=280)
        ax = cns.scatterplot(
            data=g,
            x="pos_bin",
            y="similarity",
            hue="scope",
            alpha=0.25,
            s=10,
        )
        means = g.groupby(["pos_bin", "scope"], as_index=False)["similarity"].mean()
        for scope, sg in means.groupby("scope"):
            ax.plot(sg["pos_bin"], sg["similarity"], lw=1.5, label=f"mean {scope}")
        ax.axvline(TSS_REL_POS * n_bins - 0.5, color="#666666", ls="--", lw=0.9)
        am = atg_meta[atg_meta["meta_cluster"] == mc]
        if not am.empty:
            r = am.iloc[0]
            x = float(r["atg_rel_mean"]) * n_bins - 0.5
            xerr = float(r["atg_rel_std"]) * n_bins if np.isfinite(r["atg_rel_std"]) else 0.0
            ax.errorbar(
                [x],
                [float(g["similarity"].median())],
                xerr=[xerr],
                fmt="D",
                color="#D55E00",
                ms=6,
                label="ATG mean±sd",
                zorder=5,
            )
        ax.set_xlabel("Position bin")
        ax.set_ylabel("Similarity (per OPG×bin)")
        ax.set_title(f"Meta-cluster {int(mc)} — points + means (TSS dashed, ATG diamond)")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_12_meta{int(mc):02d}_points", dpi))

    plt.close("all")
    return written


def plot_meta_altair(
    bin_sim: pd.DataFrame,
    atg_meta: pd.DataFrame,
    outdir: Path | str,
    *,
    n_bins: int,
) -> list[Path]:
    """Altair facet ``pos_bin ~ meta_cluster`` with point+boxplot (violin-like) by scope."""
    import altair as alt

    from src.train_viz.plotting import save_altair_chart

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alt.data_transformers.disable_max_rows()

    plot_df = bin_sim.copy()
    plot_df["meta_cluster"] = plot_df["meta_cluster"].astype(int)
    plot_df["pos_bin"] = plot_df["pos_bin"].astype(int)

    # Subsample OPG points if huge for HTML size
    if len(plot_df) > 120_000:
        plot_df = plot_df.sample(120_000, random_state=42)

    atg_df = atg_meta.copy()
    if not atg_df.empty:
        atg_df["atg_bin"] = atg_df["atg_rel_mean"] * n_bins - 0.5
        atg_df["tss_bin"] = TSS_REL_POS * n_bins - 0.5

    # Facet grid: rows=pos_bin (decimated if many), cols=meta_cluster
    # If too many bins, show every other bin in facet and a full non-faceted overview
    bins = sorted(plot_df["pos_bin"].unique())
    if len(bins) > 25:
        # keep ~25 facet rows: stride
        stride = int(np.ceil(len(bins) / 25))
        facet_bins = set(bins[::stride])
        facet_df = plot_df[plot_df["pos_bin"].isin(facet_bins)]
    else:
        facet_df = plot_df

    box = (
        alt.Chart(facet_df)
        .mark_boxplot(extent="min-max", size=8)
        .encode(
            x=alt.X("scope:N", title=None),
            y=alt.Y("similarity:Q", title="Similarity", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("scope:N", legend=None),
        )
    )
    pts = (
        alt.Chart(facet_df)
        .mark_circle(size=12, opacity=0.25)
        .encode(
            x=alt.X("scope:N", title=None),
            y=alt.Y("similarity:Q"),
            color=alt.Color("scope:N", title="Scope"),
            tooltip=["cluster", "scope", "similarity", "n_positions"],
        )
    )
    facet = (
        alt.layer(box, pts)
        .properties(width=70, height=90)
        .facet(
            row=alt.Row("pos_bin:O", title="Position bin"),
            column=alt.Column("meta_cluster:O", title="Meta-cluster"),
        )
        .resolve_scale(y="shared")
        .properties(title="Similarity by scope — facets: bin ~ meta-cluster (align≥50%)")
    )
    written.extend(save_altair_chart(facet, outdir / "Figure_13_facet_bin_by_metacluster_altair"))

    # Readable overview: one panel per meta-cluster (concat, not layered-facet)
    panels: list[Any] = []
    for mc in sorted(plot_df["meta_cluster"].unique()):
        sub = plot_df[plot_df["meta_cluster"] == mc]
        box = (
            alt.Chart(sub)
            .mark_boxplot(extent="min-max", outliers=False)
            .encode(
                x=alt.X("pos_bin:O", title="Position bin"),
                y=alt.Y("similarity:Q", title="Similarity", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("scope:N", title="Scope"),
            )
        )
        pts = (
            alt.Chart(sub)
            .mark_circle(size=10, opacity=0.2)
            .encode(
                x="pos_bin:O",
                y="similarity:Q",
                color="scope:N",
                tooltip=["cluster", "scope", "similarity"],
            )
        )
        layers_mc: list[Any] = [box, pts]
        am = atg_df[atg_df["meta_cluster"] == mc] if not atg_df.empty else atg_df
        tss_df = pd.DataFrame({"x": [TSS_REL_POS * n_bins - 0.5]})
        layers_mc.append(
            alt.Chart(tss_df).mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1.5).encode(x="x:Q")
        )
        if not am.empty and np.isfinite(am.iloc[0]["atg_rel_mean"]):
            r = am.iloc[0]
            iqr = pd.DataFrame(
                {
                    "x": [float(r["atg_rel_q25"]) * n_bins - 0.5],
                    "x2": [float(r["atg_rel_q75"]) * n_bins - 0.5],
                }
            )
            atg_pt = pd.DataFrame({"x": [float(r["atg_rel_mean"]) * n_bins - 0.5]})
            layers_mc.append(
                alt.Chart(iqr)
                .mark_rule(color="#D55E00", strokeWidth=6, opacity=0.3)
                .encode(x="x:Q", x2="x2:Q")
            )
            layers_mc.append(
                alt.Chart(atg_pt).mark_rule(color="#D55E00", strokeWidth=2).encode(x="x:Q")
            )
        panels.append(
            alt.layer(*layers_mc).properties(width=280, height=150, title=f"meta {int(mc)}")
        )
    panel = alt.concat(*panels, columns=4).properties(
        title="Similarity vs bin by meta-cluster (points+box; TSS dashed; ATG orange ± IQR)"
    )
    written.extend(save_altair_chart(panel, outdir / "Figure_14_metacluster_bin_profiles_altair"))

    # Point cloud + mean lines (per meta-cluster concat)
    means = (
        plot_df.groupby(["meta_cluster", "pos_bin", "scope"], as_index=False)["similarity"]
        .mean()
    )
    traj_panels: list[Any] = []
    for mc in sorted(plot_df["meta_cluster"].unique()):
        sub = plot_df[plot_df["meta_cluster"] == mc]
        msub = means[means["meta_cluster"] == mc]
        pts = (
            alt.Chart(sub)
            .mark_circle(size=8, opacity=0.15)
            .encode(
                x=alt.X("pos_bin:Q", title="Position bin"),
                y=alt.Y("similarity:Q", title="Similarity"),
                color="scope:N",
            )
        )
        line = (
            alt.Chart(msub)
            .mark_line()
            .encode(x="pos_bin:Q", y="similarity:Q", color="scope:N")
        )
        traj_panels.append(
            alt.layer(pts, line).properties(width=280, height=150, title=f"meta {int(mc)}")
        )
    layered = alt.concat(*traj_panels, columns=4).properties(
        title="Point + mean similarity trajectories by meta-cluster"
    )
    written.extend(save_altair_chart(layered, outdir / "Figure_15_metacluster_points_means_altair"))

    return written


def run_meta_viz(
    metrics_dir: Path | str,
    outdir: Path | str,
    *,
    aln_dir: Path | str | None = None,
    limit: int = 0,
    n_bins: int = DEFAULT_N_BINS,
    min_align_frac: float = MIN_ALIGN_FRAC,
    max_k: int = MAX_META_CLUSTERS,
    dpi: int = 300,
    seed: int = 42,
) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables = build_meta_tables(
        metrics_dir,
        aln_dir=aln_dir,
        limit=limit,
        n_bins=n_bins,
        min_align_frac=min_align_frac,
        max_k=max_k,
        seed=seed,
    )
    tables["bin_sim"].to_csv(outdir / "table_bin_similarity_meta.tsv", sep="\t", index=False)
    tables["assign"].to_csv(outdir / "table_meta_cluster_assign.tsv", sep="\t", index=False)
    tables["atg"].to_csv(outdir / "table_atg_rel_positions.tsv", sep="\t", index=False)
    tables["atg_meta"].to_csv(outdir / "table_atg_by_metacluster.tsv", sep="\t", index=False)
    tables["wide"].to_csv(outdir / "table_profile_matrix.tsv", sep="\t")
    (outdir / "meta_clustering_manifest.json").write_text(
        json.dumps(
            {"filter_stats": tables["filter_stats"], "kmeans": tables["kmeans_meta"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written = [
        outdir / "table_bin_similarity_meta.tsv",
        outdir / "table_meta_cluster_assign.tsv",
        outdir / "table_atg_rel_positions.tsv",
        outdir / "table_atg_by_metacluster.tsv",
        outdir / "table_profile_matrix.tsv",
        outdir / "meta_clustering_manifest.json",
    ]
    written.extend(
        plot_meta_cnsplots(
            tables["bin_sim"],
            tables["atg_meta"],
            outdir,
            n_bins=n_bins,
            dpi=dpi,
        )
    )
    written.extend(
        plot_meta_altair(tables["bin_sim"], tables["atg_meta"], outdir, n_bins=n_bins)
    )
    return written
