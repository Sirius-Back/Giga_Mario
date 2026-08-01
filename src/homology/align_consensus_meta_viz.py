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
MIN_META_CLUSTERS = 10
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
    min_k: int = MIN_META_CLUSTERS,
    max_k: int = MAX_META_CLUSTERS,
    fixed_k: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """KMeans with k chosen by silhouette inside ``[min_k, max_k]`` (default 10–20).

    Searching from k=2 often collapses to 2 coarse groups; the user-facing
    resolution target is 10–20 meta-clusters, so silhouette is evaluated only
    in that window (unless ``fixed_k`` is set).
    """
    n = mat.shape[0]
    if n < 2:
        raise ValueError("Need >=2 clusters for meta-clustering")

    scaler = StandardScaler()
    X = scaler.fit_transform(mat)

    if fixed_k is not None:
        k = int(fixed_k)
        if k < 2 or k > n - 1:
            raise ValueError(f"fixed_k={k} invalid for n={n}")
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        score = float(silhouette_score(X, labels, metric="euclidean")) if k >= 2 else float("nan")
        meta = {
            "k": k,
            "silhouette": score,
            "scores": {str(k): score},
            "min_k": int(min_k),
            "max_k": int(max_k),
            "fixed_k": k,
            "seed": seed,
        }
        return np.asarray(labels, dtype=int), k, meta

    k_max = min(int(max_k), n - 1)
    k_min = min(int(min_k), k_max)
    if k_min < 2:
        k_min = 2
    if k_max < k_min:
        raise ValueError(f"No valid k in [{min_k}, {max_k}] for n={n}")

    best_k = k_min
    best_score = -1.0
    best_labels = None
    scores: dict[str, float] = {}
    for k in range(k_min, k_max + 1):
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
        km = KMeans(n_clusters=k_min, random_state=seed, n_init=10)
        best_labels = km.fit_predict(X)
        best_k = k_min
        best_score = float("nan")
    meta = {
        "k": int(best_k),
        "silhouette": best_score if np.isfinite(best_score) else None,
        "scores": scores,
        "min_k": k_min,
        "max_k": k_max,
        "fixed_k": None,
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
    min_k: int = MIN_META_CLUSTERS,
    fixed_k: int | None = None,
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
    labels, k, km_meta = choose_k_and_cluster(
        mat, min_k=min_k, max_k=max_k, fixed_k=fixed_k, seed=seed
    )
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


def ortho_minus_para_profiles(bin_sim: pd.DataFrame) -> pd.DataFrame:
    """Mean (orthologs − paralogs) similarity per meta_cluster × pos_bin.

    For each OPG×bin with both scopes, take the paired difference, then average
    within the meta-cluster (same pairing logic as ``scope_bin_significance``).
    """
    wide = bin_sim.pivot_table(
        index=["cluster", "meta_cluster", "pos_bin"],
        columns="scope",
        values="similarity",
        aggfunc="mean",
    )
    empty_cols = [
        "meta_cluster",
        "pos_bin",
        "delta_ortho_minus_para",
        "n_opg",
        "delta_sd",
    ]
    if "orthologs" not in wide.columns or "paralogs" not in wide.columns:
        return pd.DataFrame(columns=empty_cols)
    paired = wide[["orthologs", "paralogs"]].dropna()
    if paired.empty:
        return pd.DataFrame(columns=empty_cols)
    delta = (paired["orthologs"] - paired["paralogs"]).rename("delta_ortho_minus_para")
    long = delta.reset_index()
    return (
        long.groupby(["meta_cluster", "pos_bin"], as_index=False)
        .agg(
            delta_ortho_minus_para=("delta_ortho_minus_para", "mean"),
            n_opg=("delta_ortho_minus_para", "size"),
            delta_sd=("delta_ortho_minus_para", "std"),
        )
        .sort_values(["meta_cluster", "pos_bin"])
        .reset_index(drop=True)
    )


def _minmax_1d(values: np.ndarray) -> np.ndarray:
    """Min–max scale to [0, 1]; flat profiles map to 0.5."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return out
    lo = float(np.min(arr[mask]))
    hi = float(np.max(arr[mask]))
    if hi <= lo:
        out[mask] = 0.5
        return out
    out[mask] = (arr[mask] - lo) / (hi - lo)
    return out


def ortho_minus_para_minmax_profiles(bin_sim: pd.DataFrame) -> pd.DataFrame:
    """Mean paired (minmax(orthologs) − minmax(paralogs)) per meta × bin.

    For each OPG, min–max scale the ortholog similarity profile across bins and
    (separately) the paralog profile, then take the per-bin difference on bins
    where both scopes are present; average within meta-cluster.
    """
    empty_cols = [
        "meta_cluster",
        "pos_bin",
        "delta_minmax_ortho_minus_para",
        "n_opg",
        "delta_sd",
    ]
    wide = bin_sim.pivot_table(
        index=["cluster", "meta_cluster", "pos_bin"],
        columns="scope",
        values="similarity",
        aggfunc="mean",
    )
    if "orthologs" not in wide.columns or "paralogs" not in wide.columns:
        return pd.DataFrame(columns=empty_cols)
    flat = wide.reset_index()
    rows: list[dict[str, Any]] = []
    for (cluster, mc), g in flat.groupby(["cluster", "meta_cluster"], sort=False):
        g = g.sort_values("pos_bin")
        o_scaled = _minmax_1d(g["orthologs"].to_numpy(dtype=float))
        p_scaled = _minmax_1d(g["paralogs"].to_numpy(dtype=float))
        both = np.isfinite(o_scaled) & np.isfinite(p_scaled)
        if not np.any(both):
            continue
        bins = g["pos_bin"].to_numpy(dtype=int)
        for pos_bin, d in zip(bins[both], (o_scaled - p_scaled)[both], strict=True):
            rows.append(
                {
                    "cluster": cluster,
                    "meta_cluster": int(mc),
                    "pos_bin": int(pos_bin),
                    "delta_minmax_ortho_minus_para": float(d),
                }
            )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    long = pd.DataFrame(rows)
    return (
        long.groupby(["meta_cluster", "pos_bin"], as_index=False)
        .agg(
            delta_minmax_ortho_minus_para=("delta_minmax_ortho_minus_para", "mean"),
            n_opg=("delta_minmax_ortho_minus_para", "size"),
            delta_sd=("delta_minmax_ortho_minus_para", "std"),
        )
        .sort_values(["meta_cluster", "pos_bin"])
        .reset_index(drop=True)
    )


def scope_bin_significance(
    bin_sim: pd.DataFrame,
    *,
    min_n: int = 5,
    fdr_q: float = 0.05,
) -> pd.DataFrame:
    """Per (meta_cluster, pos_bin): Wilcoxon signed-rank orthologs vs paralogs + BH-FDR.

    ``winner`` is ``paralogs`` / ``orthologs`` / ``none`` at q < ``fdr_q``
    (direction by median paralogs−orthologs).
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    rows: list[dict[str, Any]] = []
    for (mc, pos_bin), g in bin_sim.groupby(["meta_cluster", "pos_bin"], sort=False):
        wide = g.pivot_table(index="cluster", columns="scope", values="similarity", aggfunc="mean")
        if "orthologs" not in wide.columns or "paralogs" not in wide.columns:
            continue
        sub = wide[["orthologs", "paralogs"]].dropna()
        if len(sub) < min_n:
            continue
        diff = sub["paralogs"] - sub["orthologs"]
        if np.allclose(diff.to_numpy(dtype=float), 0.0):
            continue
        try:
            _stat, pval = stats.wilcoxon(
                sub["paralogs"].to_numpy(dtype=float),
                sub["orthologs"].to_numpy(dtype=float),
                alternative="two-sided",
                zero_method="wilcox",
            )
        except ValueError:
            continue
        rows.append(
            {
                "meta_cluster": int(mc),
                "pos_bin": int(pos_bin),
                "n_paired": int(len(sub)),
                "median_diff_para_minus_ortho": float(diff.median()),
                "pvalue": float(pval),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(q=pd.Series(dtype=float), winner=pd.Series(dtype=str))
    out["q"] = multipletests(out["pvalue"].to_numpy(dtype=float), method="fdr_bh")[1]
    winners: list[str] = []
    for _, r in out.iterrows():
        if float(r["q"]) >= fdr_q:
            winners.append("none")
        elif float(r["median_diff_para_minus_ortho"]) > 0:
            winners.append("paralogs")
        elif float(r["median_diff_para_minus_ortho"]) < 0:
            winners.append("orthologs")
        else:
            winners.append("none")
    out["winner"] = winners
    return out


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

    # Delta profile tables (figures drawn in Altair — see plot_meta_altair)
    delta_prof = ortho_minus_para_profiles(bin_sim)
    if not delta_prof.empty:
        delta_path = outdir / "table_meta_profile_ortho_minus_para.tsv"
        delta_prof.to_csv(delta_path, sep="\t", index=False)
        written.append(delta_path)
    delta_mm = ortho_minus_para_minmax_profiles(bin_sim)
    if not delta_mm.empty:
        mm_path = outdir / "table_meta_profile_ortho_minus_para_minmax.tsv"
        delta_mm.to_csv(mm_path, sep="\t", index=False)
        written.append(mm_path)

    # Violin by bin — sns treats x as strings; without order= sorts lexically ("10"<"2")
    metas = sorted(int(x) for x in bin_sim["meta_cluster"].dropna().unique())
    show = metas[: min(8, len(metas))]
    sub = bin_sim[bin_sim["meta_cluster"].isin(show)].copy()
    present_bins = sorted(int(b) for b in sub["pos_bin"].unique())
    bin_order = [str(b) for b in present_bins]
    sub["bin_lab"] = sub["pos_bin"].astype(int).astype(str)
    hue_order = [s for s in SCOPES if s in set(sub["scope"].astype(str))]

    cns.figure(width=560, height=320)
    ax = cns.violinplot(
        data=sub,
        x="bin_lab",
        y="similarity",
        hue="scope",
        order=bin_order,
        hue_order=hue_order or None,
        add_box=False,
    )
    ax.set_xlabel("Position bin (0→max, left→right)")
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity by bin (subset of meta-clusters pooled)")
    ticks = list(range(0, len(bin_order), max(1, len(bin_order) // 10)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([bin_order[i] for i in ticks], rotation=0)
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_11_similarity_violin_by_bin_pooled", dpi))

    # Per meta-cluster panels: ATG start marker at y=0
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
            sg = sg.sort_values("pos_bin")
            ax.plot(sg["pos_bin"], sg["similarity"], lw=1.5, label=f"mean {scope}")
        ax.axvline(TSS_REL_POS * n_bins - 0.5, color="#666666", ls="--", lw=0.9)
        am = atg_meta[atg_meta["meta_cluster"] == mc]
        if not am.empty:
            r = am.iloc[0]
            x = float(r["atg_rel_mean"]) * n_bins - 0.5
            xerr = float(r["atg_rel_std"]) * n_bins if np.isfinite(r["atg_rel_std"]) else 0.0
            ax.errorbar(
                [x],
                [0.0],
                xerr=[xerr],
                fmt="D",
                color="#D55E00",
                ms=7,
                capsize=3,
                label="ATG mean±sd (y=0)",
                zorder=6,
                clip_on=False,
            )
        ax.set_xlabel("Position bin")
        ax.set_ylabel("Similarity (per OPG×bin)")
        ax.set_title(f"Meta-cluster {int(mc)} — points + means (TSS dashed, ATG at y=0)")
        ax.set_ylim(0.0, 1.02)
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_12_meta{int(mc):02d}_points", dpi))

    plt.close("all")
    return written


def _annotate_figure13_borders_mpl(
    bin_sim: pd.DataFrame,
    contrast: pd.DataFrame,
    out_png: Path,
    *,
    n_bins: int,
    dpi: int = 300,
    max_facet_rows: int = 25,
) -> Path:
    """Draw Figure_13 as a bin×meta grid with colored spines for significant cells."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    border_colors = {"paralogs": "#E69F00", "orthologs": "#CC3311", "none": "#B0B0B0"}
    df = bin_sim.copy()
    df["meta_cluster"] = df["meta_cluster"].astype(int)
    df["pos_bin"] = df["pos_bin"].astype(int)
    bins = sorted(df["pos_bin"].unique())
    if len(bins) > max_facet_rows:
        stride = int(np.ceil(len(bins) / max_facet_rows))
        bins = bins[::stride]
    metas = sorted(df["meta_cluster"].unique())
    winner_lookup = {
        (int(r.meta_cluster), int(r.pos_bin)): str(r.winner)
        for _, r in contrast.iterrows()
    } if contrast is not None and not contrast.empty else {}

    nrows, ncols = len(bins), len(metas)
    fig_w = max(8.0, 1.15 * ncols)
    fig_h = max(6.0, 1.05 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    scope_order = ["full", "orthologs", "paralogs"]
    scope_pos = {s: i for i, s in enumerate(scope_order)}
    for i, pos_bin in enumerate(bins):
        for j, mc in enumerate(metas):
            ax = axes[i][j]
            cell = df[(df["pos_bin"] == pos_bin) & (df["meta_cluster"] == mc)]
            winner = winner_lookup.get((int(mc), int(pos_bin)), "none")
            stroke = border_colors.get(winner, "#B0B0B0")
            lw = 2.8 if winner != "none" else 0.7
            if not cell.empty:
                data = [
                    cell.loc[cell["scope"] == s, "similarity"].dropna().to_numpy(dtype=float)
                    for s in scope_order
                ]
                ax.boxplot(
                    data,
                    positions=list(range(len(scope_order))),
                    widths=0.55,
                    showfliers=False,
                    patch_artist=False,
                )
                rng = np.random.default_rng(42 + i * 100 + j)
                for s in scope_order:
                    vals = cell.loc[cell["scope"] == s, "similarity"].dropna().to_numpy(dtype=float)
                    if vals.size == 0:
                        continue
                    if vals.size > 80:
                        vals = rng.choice(vals, 80, replace=False)
                    jitter = rng.uniform(-0.15, 0.15, size=vals.size)
                    ax.scatter(
                        np.full(vals.size, scope_pos[s]) + jitter,
                        vals,
                        s=6,
                        alpha=0.25,
                        c={"full": "#0072B2", "orthologs": "#009E73", "paralogs": "#E69F00"}[s],
                        linewidths=0,
                    )
            ax.set_ylim(0, 1)
            ax.set_xticks(list(range(len(scope_order))))
            ax.set_xticklabels(["F", "O", "P"], fontsize=7)
            for spine in ax.spines.values():
                spine.set_color(stroke)
                spine.set_linewidth(lw)
            title = f"b{pos_bin}|m{mc}"
            if winner == "paralogs":
                title += " P↑"
            elif winner == "orthologs":
                title += " O↑"
            ax.set_title(title, fontsize=8, color=stroke if winner != "none" else "#333333")
            if j == 0:
                ax.set_ylabel(f"bin {pos_bin}", fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel(f"meta {mc}", fontsize=8)
    fig.suptitle(
        "Similarity by scope — bin × meta-cluster "
        "(orange border: paralogs↑; red: orthologs↑; Wilcoxon+BH q<0.05)",
        fontsize=10,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def _altair_meta_delta_profile(
    prof: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    y_title: str,
    n_bins: int,
    atg_meta: pd.DataFrame,
    stem: Path,
) -> list[Path]:
    """Line chart of meta-cluster delta profiles with legend outside the plot."""
    import altair as alt

    from src.train_viz.plotting import save_altair_chart

    if prof.empty or y_col not in prof.columns:
        return []
    df = prof.copy()
    df["meta_cluster"] = df["meta_cluster"].astype(int).astype(str)
    df["pos_bin"] = df["pos_bin"].astype(int)
    meta_order = sorted(df["meta_cluster"].unique(), key=int)

    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("pos_bin:Q", title="Relative-position bin"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color(
                "meta_cluster:N",
                title="Meta-cluster",
                sort=meta_order,
                legend=alt.Legend(
                    orient="right",
                    titleOrient="top",
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    columns=1,
                ),
            ),
            tooltip=["meta_cluster", "pos_bin", y_col, "n_opg"],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color="#444444", strokeWidth=1)
        .encode(y="y:Q")
    )
    tss = (
        alt.Chart(pd.DataFrame({"x": [TSS_REL_POS * n_bins - 0.5]}))
        .mark_rule(color="#666666", strokeDash=[5, 4], strokeWidth=1.2)
        .encode(x="x:Q")
    )
    layers: list[Any] = [zero, tss, lines]
    if not atg_meta.empty:
        atg_x = []
        for _, r in atg_meta.iterrows():
            x = float(r["atg_rel_mean"]) * n_bins - 0.5
            if np.isfinite(x):
                atg_x.append(x)
        if atg_x:
            layers.insert(
                2,
                alt.Chart(pd.DataFrame({"x": atg_x}))
                .mark_rule(color="#D55E00", opacity=0.25, strokeWidth=1)
                .encode(x="x:Q"),
            )
    chart = (
        alt.layer(*layers)
        .properties(width=560, height=340, title=title)
        .configure_legend(labelLimit=120, padding=8)
        .configure_view(strokeWidth=0)
    )
    return save_altair_chart(chart, stem)


def plot_meta_altair(
    bin_sim: pd.DataFrame,
    atg_meta: pd.DataFrame,
    outdir: Path | str,
    *,
    n_bins: int,
    contrast: pd.DataFrame | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Altair interactive charts + matplotlib Figure_13 with colored borders."""
    import altair as alt

    from src.train_viz.plotting import save_altair_chart

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alt.data_transformers.disable_max_rows()

    plot_df = bin_sim.copy()
    plot_df["meta_cluster"] = plot_df["meta_cluster"].astype(int)
    plot_df["pos_bin"] = plot_df["pos_bin"].astype(int)
    if len(plot_df) > 120_000:
        plot_df = plot_df.sample(120_000, random_state=42)

    atg_df = atg_meta.copy()
    if not atg_df.empty:
        atg_df["atg_bin"] = atg_df["atg_rel_mean"] * n_bins - 0.5
        atg_df["tss_bin"] = TSS_REL_POS * n_bins - 0.5

    if contrast is None:
        contrast = scope_bin_significance(bin_sim)
    if not contrast.empty:
        contrast = contrast.copy()
        contrast["meta_cluster"] = contrast["meta_cluster"].astype(int)
        contrast["pos_bin"] = contrast["pos_bin"].astype(int)
        contrast.to_csv(outdir / "table_scope_bin_contrast.tsv", sep="\t", index=False)
        written.append(outdir / "table_scope_bin_contrast.tsv")

    border_colors = {"paralogs": "#E69F00", "orthologs": "#CC3311", "none": "#B0B0B0"}
    bins = sorted(int(b) for b in plot_df["pos_bin"].unique())
    if len(bins) > 25:
        stride = int(np.ceil(len(bins) / 25))
        facet_bins = bins[::stride]
    else:
        facet_bins = bins
    metas = sorted(int(x) for x in plot_df["meta_cluster"].unique())
    winner_lookup = {
        (int(r.meta_cluster), int(r.pos_bin)): str(r.winner)
        for _, r in contrast.iterrows()
    } if not contrast.empty else {}

    # Altair HTML: ordered bins, colored titles for significant cells
    row_charts: list[Any] = []
    for pos_bin in facet_bins:
        cols: list[Any] = []
        for mc in metas:
            cell = plot_df[(plot_df["pos_bin"] == pos_bin) & (plot_df["meta_cluster"] == mc)]
            if cell.empty:
                continue
            winner = winner_lookup.get((mc, pos_bin), "none")
            stroke = border_colors.get(winner, "#B0B0B0")
            box = (
                alt.Chart(cell)
                .mark_boxplot(extent="min-max", size=8)
                .encode(
                    x=alt.X("scope:N", title=None, sort=["full", "orthologs", "paralogs"]),
                    y=alt.Y("similarity:Q", title="sim", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("scope:N", legend=None),
                )
            )
            pts = (
                alt.Chart(cell)
                .mark_circle(size=10, opacity=0.2)
                .encode(
                    x=alt.X("scope:N", sort=["full", "orthologs", "paralogs"]),
                    y="similarity:Q",
                    color=alt.Color("scope:N", legend=None),
                    tooltip=["cluster", "scope", "similarity"],
                )
            )
            title_txt = f"b{pos_bin}|m{mc}"
            if winner == "paralogs":
                title_txt += " P↑"
            elif winner == "orthologs":
                title_txt += " O↑"
            cols.append(
                alt.layer(box, pts).properties(
                    width=70,
                    height=90,
                    title=alt.TitleParams(text=title_txt, fontSize=9, color=stroke),
                )
            )
        if cols:
            row_charts.append(alt.hconcat(*cols))
    if row_charts:
        facet = alt.vconcat(*row_charts).properties(
            title=(
                "Similarity by scope — bin×meta "
                "(orange title=paralogs↑; red=orthologs↑; Wilcoxon+BH q<0.05)"
            )
        )
        written.extend(save_altair_chart(facet, outdir / "Figure_13_facet_bin_by_metacluster_altair"))

    # Publication PNG/PDF with actual colored borders
    fig13 = _annotate_figure13_borders_mpl(
        bin_sim,
        contrast,
        outdir / "Figure_13_facet_bin_by_metacluster_altair.png",
        n_bins=n_bins,
        dpi=dpi,
    )
    written.append(fig13)
    written.append(fig13.with_suffix(".pdf"))

    # Figure_14: per meta-cluster profiles
    panels: list[Any] = []
    for mc in metas:
        sub = plot_df[plot_df["meta_cluster"] == mc]
        box = (
            alt.Chart(sub)
            .mark_boxplot(extent="min-max", outliers=False)
            .encode(
                x=alt.X("pos_bin:O", title="Position bin", sort=sorted(sub["pos_bin"].unique())),
                y=alt.Y("similarity:Q", title="Similarity", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("scope:N", title="Scope"),
            )
        )
        pts = (
            alt.Chart(sub)
            .mark_circle(size=10, opacity=0.2)
            .encode(
                x=alt.X("pos_bin:O", sort=sorted(sub["pos_bin"].unique())),
                y="similarity:Q",
                color="scope:N",
                tooltip=["cluster", "scope", "similarity"],
            )
        )
        layers_mc: list[Any] = [box, pts]
        am = atg_df[atg_df["meta_cluster"] == mc] if not atg_df.empty else atg_df
        tss_df = pd.DataFrame({"x": [TSS_REL_POS * n_bins - 0.5]})
        layers_mc.append(
            alt.Chart(tss_df)
            .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(x="x:Q")
        )
        if not am.empty and np.isfinite(am.iloc[0]["atg_rel_mean"]):
            r = am.iloc[0]
            iqr = pd.DataFrame(
                {
                    "x": [float(r["atg_rel_q25"]) * n_bins - 0.5],
                    "x2": [float(r["atg_rel_q75"]) * n_bins - 0.5],
                }
            )
            atg_pt = pd.DataFrame({"x": [float(r["atg_rel_mean"]) * n_bins - 0.5], "y": [0.0]})
            layers_mc.append(
                alt.Chart(iqr)
                .mark_rule(color="#D55E00", strokeWidth=6, opacity=0.3)
                .encode(x="x:Q", x2="x2:Q")
            )
            layers_mc.append(
                alt.Chart(atg_pt)
                .mark_point(color="#D55E00", size=60, shape="diamond")
                .encode(x="x:Q", y="y:Q")
            )
        panels.append(
            alt.layer(*layers_mc).properties(width=280, height=150, title=f"meta {int(mc)}")
        )
    if panels:
        panel = alt.concat(*panels, columns=4).properties(
            title="Similarity vs bin by meta-cluster (ATG diamond at y=0; TSS dashed)"
        )
        written.extend(save_altair_chart(panel, outdir / "Figure_14_metacluster_bin_profiles_altair"))

    means = (
        plot_df.groupby(["meta_cluster", "pos_bin", "scope"], as_index=False)["similarity"].mean()
    )
    traj_panels: list[Any] = []
    for mc in metas:
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
    if traj_panels:
        layered = alt.concat(*traj_panels, columns=4).properties(
            title="Point + mean similarity trajectories by meta-cluster"
        )
        written.extend(save_altair_chart(layered, outdir / "Figure_15_metacluster_points_means_altair"))

    # Figure_16 / 17: orthologs − paralogs (raw and per-OPG minmax-scaled)
    delta_prof = ortho_minus_para_profiles(bin_sim)
    if not delta_prof.empty:
        delta_path = outdir / "table_meta_profile_ortho_minus_para.tsv"
        delta_prof.to_csv(delta_path, sep="\t", index=False)
        written.append(delta_path)
        written.extend(
            _altair_meta_delta_profile(
                delta_prof,
                y_col="delta_ortho_minus_para",
                title="Meta-cluster similarity profiles — orthologs − paralogs",
                y_title="Mean (orthologs − paralogs) similarity",
                n_bins=n_bins,
                atg_meta=atg_meta,
                stem=outdir / "Figure_16_meta_profile_orthologs_minus_paralogs",
            )
        )
    delta_mm = ortho_minus_para_minmax_profiles(bin_sim)
    if not delta_mm.empty:
        mm_path = outdir / "table_meta_profile_ortho_minus_para_minmax.tsv"
        delta_mm.to_csv(mm_path, sep="\t", index=False)
        written.append(mm_path)
        written.extend(
            _altair_meta_delta_profile(
                delta_mm,
                y_col="delta_minmax_ortho_minus_para",
                title=(
                    "Meta-cluster similarity profiles — "
                    "minmax(orthologs) − minmax(paralogs)"
                ),
                y_title="Mean paired (minmax ortho − minmax para)",
                n_bins=n_bins,
                atg_meta=atg_meta,
                stem=outdir / "Figure_17_meta_profile_ortho_minus_para_minmax",
            )
        )

    return written


def redraw_meta_figures_from_tables(
    outdir: Path | str,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Re-plot Figure_11–15 from existing TSVs (no MAFFT/ATG recompute)."""
    outdir = Path(outdir)
    bin_sim = pd.read_csv(outdir / "table_bin_similarity_meta.tsv", sep="\t")
    atg_meta = pd.read_csv(outdir / "table_atg_by_metacluster.tsv", sep="\t")
    n_bins = int(bin_sim["pos_bin"].max()) + 1
    contrast = scope_bin_significance(bin_sim)
    written: list[Path] = []
    written.extend(plot_meta_cnsplots(bin_sim, atg_meta, outdir, n_bins=n_bins, dpi=dpi))
    written.extend(
        plot_meta_altair(bin_sim, atg_meta, outdir, n_bins=n_bins, contrast=contrast, dpi=dpi)
    )
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
    min_k: int = MIN_META_CLUSTERS,
    fixed_k: int | None = None,
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
        min_k=min_k,
        fixed_k=fixed_k,
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
    contrast = scope_bin_significance(tables["bin_sim"])
    contrast.to_csv(outdir / "table_scope_bin_contrast.tsv", sep="\t", index=False)
    written.append(outdir / "table_scope_bin_contrast.tsv")
    written.extend(
        plot_meta_altair(
            tables["bin_sim"],
            tables["atg_meta"],
            outdir,
            n_bins=n_bins,
            contrast=contrast,
            dpi=dpi,
        )
    )
    return written
