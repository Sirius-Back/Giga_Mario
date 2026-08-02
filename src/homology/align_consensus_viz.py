"""cnsplots + Altair figures for orthoparagroups alignment consensus metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.homology.visualize import _apply_cns_style

SCOPES = ("full", "orthologs", "paralogs")
RAW_COLS = {
    "full": "overall_consensus_rate",
    "orthologs": "orthologs_consensus_rate",
    "paralogs": "paralogs_consensus_rate",
}
VARIANT_SUFFIXES = (
    ("raw", ""),
    ("norm_residual", "_norm_residual"),
    ("norm_ratio", "_norm_ratio"),
    ("norm_z", "_norm_z"),
)
PAIR_SPECS = (
    ("full", "orthologs"),
    ("full", "paralogs"),
    ("orthologs", "paralogs"),
)
DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
N_POS_BINS = 20
# Finer bins + meta-clustering (Figure_10+)
DEFAULT_META_POS_BINS = 50
MAX_META_CLUSTERS = 20


def _col(scope: str, variant: str) -> str:
    base = RAW_COLS[scope]
    for name, suf in VARIANT_SUFFIXES:
        if name == variant:
            return base if not suf else f"{base}{suf}"
    raise KeyError(variant)


def discover_metric_files(metrics_dir: Path | str) -> list[Path]:
    metrics_dir = Path(metrics_dir)
    paths = sorted(metrics_dir.glob("cluster_*.pos.tsv.gz"))
    if not paths:
        raise FileNotFoundError(f"No cluster_*.pos.tsv.gz under {metrics_dir}")
    return paths


def load_metrics_table(
    metrics_dir: Path | str,
    *,
    limit: int = 0,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate per-cluster position metrics."""
    paths = discover_metric_files(metrics_dir)
    if limit > 0:
        paths = paths[:limit]
    usecols = list(columns) if columns is not None else None
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, sep="\t", usecols=usecols)
        if df.empty:
            raise ValueError(f"Empty metrics file: {path}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError(f"No rows loaded from {metrics_dir}")
    return out


def add_relative_position(df: pd.DataFrame, *, n_bins: int = N_POS_BINS) -> pd.DataFrame:
    """Add ``rel_pos`` in [0,1] and integer ``pos_bin`` 0..n_bins-1."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    out = df.copy()
    if "aln_length" not in out.columns:
        lengths = out.groupby("cluster")["position"].transform("max")
        out["aln_length"] = lengths
    denom = (out["aln_length"] - 1).clip(lower=1)
    out["rel_pos"] = (out["position"] - 1) / denom
    out["pos_bin"] = np.minimum(
        (out["rel_pos"] * n_bins).astype(int),
        n_bins - 1,
    )
    out["pos_bin_label"] = (
        (out["pos_bin"] / n_bins).round(2).astype(str)
        + "–"
        + ((out["pos_bin"] + 1) / n_bins).round(2).astype(str)
    )
    return out


def pairwise_correlation_table(
    df: pd.DataFrame,
    *,
    variants: tuple[str, ...] = ("raw", "norm_residual", "norm_ratio", "norm_z"),
    min_points: int = 10,
) -> pd.DataFrame:
    """Per-cluster Pearson r for each scope pair × rate variant."""
    rows: list[dict[str, object]] = []
    for cluster, g in df.groupby("cluster", sort=False):
        for variant in variants:
            for a, b in PAIR_SPECS:
                ca, cb = _col(a, variant), _col(b, variant)
                if ca not in g.columns or cb not in g.columns:
                    raise KeyError(f"Missing columns {ca}/{cb}")
                sub = g[[ca, cb]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) < min_points:
                    r = float("nan")
                elif sub[ca].nunique() < 2 or sub[cb].nunique() < 2:
                    r = float("nan")
                else:
                    r = float(sub[ca].corr(sub[cb], method="pearson"))
                rows.append(
                    {
                        "cluster": cluster,
                        "variant": variant,
                        "pair": f"{a}_vs_{b}",
                        "scope_a": a,
                        "scope_b": b,
                        "pearson_r": r,
                        "n_positions": int(len(sub)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No pairwise correlations computed")
    return out


def similar_length_table(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    variant: str = "raw",
) -> pd.DataFrame:
    """Per cluster × scope × threshold: total and longest contiguous high-rate length."""
    rows: list[dict[str, object]] = []
    for cluster, g in df.groupby("cluster", sort=False):
        g = g.sort_values("position")
        for scope in SCOPES:
            col = _col(scope, variant)
            vals = g[col].to_numpy(dtype=float)
            for thr in thresholds:
                mask = np.isfinite(vals) & (vals >= thr)
                total = int(mask.sum())
                longest = _longest_true_run(mask)
                rows.append(
                    {
                        "cluster": cluster,
                        "scope": scope,
                        "variant": variant,
                        "threshold": float(thr),
                        "similar_length_total": total,
                        "similar_length_longest_run": longest,
                        "aln_length": int(len(vals)),
                        "similar_fraction": total / max(len(vals), 1),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No similar-length rows")
    return out


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def position_long_rates(
    df: pd.DataFrame,
    *,
    variant: str = "raw",
    max_rows_per_scope: int = 80_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Long-form rates with relative position for violin / median profiles."""
    parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    base = add_relative_position(df)
    for scope in SCOPES:
        col = _col(scope, variant)
        sub = base[["cluster", "position", "rel_pos", "pos_bin", "pos_bin_label", col]].copy()
        sub = sub.rename(columns={col: "rate"})
        sub["scope"] = scope
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["rate"])
        if len(sub) > max_rows_per_scope:
            idx = rng.choice(len(sub), size=max_rows_per_scope, replace=False)
            sub = sub.iloc[idx]
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise ValueError("No position rates for long table")
    return out


def median_per_position_bin(
    df: pd.DataFrame,
    *,
    variant: str = "raw",
    n_bins: int = N_POS_BINS,
) -> pd.DataFrame:
    """Median (and quartiles) of rates across clusters within relative-position bins."""
    base = add_relative_position(df, n_bins=n_bins)
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        col = _col(scope, variant)
        for pos_bin, g in base.groupby("pos_bin", sort=True):
            vals = g[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "pos_bin": int(pos_bin),
                    "rel_pos_mid": (int(pos_bin) + 0.5) / n_bins,
                    "n": int(vals.size),
                    "median_rate": float(np.median(vals)),
                    "q25": float(np.quantile(vals, 0.25)),
                    "q75": float(np.quantile(vals, 0.75)),
                    "mean_rate": float(np.mean(vals)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No median-per-position rows")
    return out


def plot_consensus_cnsplots(
    corr_df: pd.DataFrame,
    length_df: pd.DataFrame,
    long_rates: pd.DataFrame,
    median_df: pd.DataFrame,
    outdir: Path | str,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Publication static figures via cnsplots."""
    import matplotlib.pyplot as plt
    import cnsplots as cns

    from src.train_viz.plotting import save_cns_figure

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _apply_cns_style(dpi)
    written: list[Path] = []

    # 1) Pairwise correlation distributions
    for variant in corr_df["variant"].drop_duplicates().tolist():
        sub = corr_df[corr_df["variant"] == variant].dropna(subset=["pearson_r"])
        if sub.empty:
            continue
        cns.figure(width=420, height=280)
        ax = cns.kdeplot(data=sub, x="pearson_r", hue="pair")
        ax.set_xlabel(f"Pearson r across positions ({variant})")
        ax.set_ylabel("Density")
        ax.set_title(f"Pairwise consensus correlation ({variant})")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_01_pair_corr_hist_{variant}", dpi))

        cns.figure(width=420, height=300)
        ax = cns.violinplot(data=sub, x="pair", y="pearson_r", add_box=True)
        ax.set_xlabel("Scope pair")
        ax.set_ylabel(f"Pearson r ({variant})")
        ax.set_title(f"Pairwise consensus correlation violins ({variant})")
        ax.tick_params(axis="x", rotation=20)
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_02_pair_corr_violin_{variant}", dpi))

    # 2) Similar-length distributions
    for metric, stem_tag, xlabel in (
        ("similar_length_total", "total", "Positions with rate ≥ threshold"),
        ("similar_length_longest_run", "longest_run", "Longest contiguous run (positions)"),
    ):
        cns.figure(width=480, height=320)
        plot_df = length_df.copy()
        plot_df["threshold_label"] = plot_df["threshold"].map(lambda x: f"≥{x:g}")
        ax = cns.violinplot(
            data=plot_df,
            x="threshold_label",
            y=metric,
            hue="scope",
            add_box=False,
        )
        ax.set_xlabel("Similarity rate threshold")
        ax.set_ylabel(xlabel)
        ax.set_title(f"Similar-sequence length ({stem_tag}) by threshold × scope")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_03_similar_length_{stem_tag}", dpi))

        # log1p hist per scope at selected thresholds
        for thr in sorted(plot_df["threshold"].unique()):
            sub = plot_df[plot_df["threshold"] == thr].copy()
            sub["log1p_length"] = np.log1p(sub[metric])
            cns.figure(width=400, height=280)
            ax = cns.kdeplot(data=sub, x="log1p_length", hue="scope")
            ax.set_xlabel(f"log1p({stem_tag} length)")
            ax.set_ylabel("Density")
            ax.set_title(f"Similar length ({stem_tag}) at rate ≥ {thr:g}")
            cns.setup_ax(ax)
            thr_tag = str(thr).replace(".", "p")
            written.extend(
                save_cns_figure(outdir / f"Figure_04_similar_length_{stem_tag}_thr{thr_tag}", dpi)
            )

    # 3) Violins of similarity per relative-position bin × scope
    lr_plot = long_rates.copy()
    lr_plot["pos_bin"] = lr_plot["pos_bin"].astype(str)
    cns.figure(width=560, height=320)
    ax = cns.violinplot(
        data=lr_plot,
        x="pos_bin",
        y="rate",
        hue="scope",
        add_box=False,
        order=[str(i) for i in range(N_POS_BINS)],
    )
    ax.set_xlabel(f"Relative-position bin (0–{N_POS_BINS - 1})")
    ax.set_ylabel("Consensus / similarity rate")
    ax.set_title("Similarity distribution per relative position (violins)")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_05_rate_violin_by_posbin", dpi))

    for scope in SCOPES:
        sub = lr_plot[lr_plot["scope"] == scope]
        if sub.empty:
            continue
        cns.figure(width=480, height=280)
        ax = cns.violinplot(
            data=sub,
            x="pos_bin",
            y="rate",
            add_box=True,
            order=[str(i) for i in range(N_POS_BINS)],
        )
        ax.set_xlabel(f"Relative-position bin (0–{N_POS_BINS - 1})")
        ax.set_ylabel("Consensus / similarity rate")
        ax.set_title(f"Similarity vs position — {scope}")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / f"Figure_06_rate_violin_by_posbin_{scope}", dpi))

    # 4) Median points per position
    cns.figure(width=420, height=300)
    ax = cns.scatterplot(
        data=median_df,
        x="rel_pos_mid",
        y="median_rate",
        hue="scope",
        s=28,
        alpha=0.9,
    )
    ax.set_xlabel("Relative position (bin mid)")
    ax.set_ylabel("Median consensus rate")
    ax.set_title("Median similarity per relative position")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_07_median_rate_by_position", dpi))

    cns.figure(width=400, height=280)
    ax = cns.kdeplot(data=median_df, x="median_rate", hue="scope")
    ax.set_xlabel("Median rate (across clusters, per position bin)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of per-position median similarity")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_08_median_rate_distribution", dpi))

    cns.figure(width=400, height=280)
    ax = cns.violinplot(data=median_df, x="scope", y="median_rate", add_box=True)
    ax.set_xlabel("Scope")
    ax.set_ylabel("Median rate per position bin")
    ax.set_title("Per-position medians by scope")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_09_median_rate_violin_by_scope", dpi))

    plt.close("all")
    return written


def plot_consensus_altair(
    corr_df: pd.DataFrame,
    length_df: pd.DataFrame,
    long_rates: pd.DataFrame,
    median_df: pd.DataFrame,
    outdir: Path | str,
) -> list[Path]:
    """Interactive Altair charts (HTML + VL + PNG)."""
    import altair as alt

    from src.train_viz.plotting import save_altair_chart

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alt.data_transformers.disable_max_rows()

    for variant in corr_df["variant"].drop_duplicates().tolist():
        sub = corr_df[corr_df["variant"] == variant].dropna(subset=["pearson_r"])
        if sub.empty:
            continue
        chart = (
            alt.Chart(sub)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("pearson_r:Q", bin=alt.Bin(maxbins=30), title=f"Pearson r ({variant})"),
                y=alt.Y("count()", title="Clusters"),
                color=alt.Color("pair:N", title="Pair"),
                tooltip=["pair", "pearson_r", "cluster"],
            )
            .properties(title=f"Pairwise consensus correlation ({variant})", width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / f"Figure_01_pair_corr_hist_{variant}_altair"))

        chart = (
            alt.Chart(sub)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("pair:N", title="Scope pair"),
                y=alt.Y("pearson_r:Q", title=f"Pearson r ({variant})"),
                color="pair:N",
            )
            .properties(title=f"Pairwise correlation by pair ({variant})", width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / f"Figure_02_pair_corr_box_{variant}_altair"))

    for metric, stem_tag, ylabel in (
        ("similar_length_total", "total", "Positions with rate ≥ threshold"),
        ("similar_length_longest_run", "longest_run", "Longest contiguous run"),
    ):
        plot_df = length_df.copy()
        plot_df["threshold_label"] = plot_df["threshold"].map(lambda x: f"≥{x:g}")
        chart = (
            alt.Chart(plot_df)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("threshold_label:N", title="Similarity threshold"),
                y=alt.Y(f"{metric}:Q", title=ylabel),
                color=alt.Color("scope:N", title="Scope"),
                column=alt.Column("scope:N", title=None),
            )
            .properties(title=f"Similar-sequence length ({stem_tag})", width=120, height=260)
        )
        written.extend(save_altair_chart(chart, outdir / f"Figure_03_similar_length_{stem_tag}_altair"))

        chart = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=30), title=ylabel),
                y=alt.Y("count()", title="Clusters"),
                color="scope:N",
                row=alt.Row("threshold_label:N", title="Threshold"),
            )
            .properties(title=f"Similar length histograms ({stem_tag})", width=360, height=80)
        )
        written.extend(
            save_altair_chart(chart, outdir / f"Figure_04_similar_length_{stem_tag}_hist_altair")
        )

    # Violins approximated as density strips / box+points by pos_bin
    # Subsample for Altair size
    lr = long_rates
    if len(lr) > 60_000:
        lr = lr.sample(60_000, random_state=42)
    chart = (
        alt.Chart(lr)
        .transform_density("rate", as_=["rate", "density"], groupby=["pos_bin", "scope"], extent=[0, 1])
        .mark_area(orient="horizontal", opacity=0.4)
        .encode(
            y=alt.Y("rate:Q", title="Consensus / similarity rate"),
            x=alt.X("density:Q", stack="center", impute=None, title=None, axis=None),
            color=alt.Color("scope:N"),
            column=alt.Column("pos_bin:O", title="Relative-position bin"),
        )
        .properties(title="Similarity density by position bin (violin-like)", width=40, height=220)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_05_rate_violin_by_posbin_altair"))

    for scope in SCOPES:
        sub = lr[lr["scope"] == scope]
        if sub.empty:
            continue
        chart = (
            alt.Chart(sub)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("pos_bin:O", title="Relative-position bin"),
                y=alt.Y("rate:Q", title="Consensus / similarity rate"),
            )
            .properties(title=f"Similarity vs position — {scope}", width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / f"Figure_06_rate_by_posbin_{scope}_altair"))

    chart = (
        alt.Chart(median_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("rel_pos_mid:Q", title="Relative position (bin mid)"),
            y=alt.Y("median_rate:Q", title="Median consensus rate"),
            color=alt.Color("scope:N", title="Scope"),
            tooltip=["scope", "rel_pos_mid", "median_rate", "q25", "q75", "n"],
        )
        .properties(title="Median similarity per relative position", width=420, height=280)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_07_median_rate_by_position_altair"))

    chart = (
        alt.Chart(median_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("median_rate:Q", bin=alt.Bin(maxbins=20), title="Median rate per position bin"),
            y=alt.Y("count()", title="Position bins"),
            color="scope:N",
        )
        .properties(title="Distribution of per-position median similarity", width=420, height=280)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_08_median_rate_distribution_altair"))

    chart = (
        alt.Chart(median_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("scope:N", title="Scope"),
            y=alt.Y("median_rate:Q", title="Median rate per position bin"),
            color="scope:N",
        )
        .properties(title="Per-position medians by scope", width=320, height=280)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_09_median_rate_by_scope_altair"))

    return written


def build_consensus_viz_tables(
    metrics_dir: Path | str,
    *,
    limit: int = 0,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    length_variant: str = "raw",
    rate_variant: str = "raw",
    max_rows_per_scope: int = 80_000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Load metrics and build all analysis tables for plotting."""
    needed = [
        "cluster",
        "position",
        "overall_consensus_rate",
        "orthologs_consensus_rate",
        "paralogs_consensus_rate",
        "overall_consensus_rate_norm_residual",
        "orthologs_consensus_rate_norm_residual",
        "paralogs_consensus_rate_norm_residual",
        "overall_consensus_rate_norm_ratio",
        "orthologs_consensus_rate_norm_ratio",
        "paralogs_consensus_rate_norm_ratio",
        "overall_consensus_rate_norm_z",
        "orthologs_consensus_rate_norm_z",
        "paralogs_consensus_rate_norm_z",
    ]
    df = load_metrics_table(metrics_dir, limit=limit, columns=needed)
    return {
        "corr": pairwise_correlation_table(df),
        "length": similar_length_table(df, thresholds=thresholds, variant=length_variant),
        "long_rates": position_long_rates(
            df, variant=rate_variant, max_rows_per_scope=max_rows_per_scope, seed=seed
        ),
        "median": median_per_position_bin(df, variant=rate_variant),
        "raw": df,
    }


def plot_consensus_metrics(
    metrics_dir: Path | str,
    outdir: Path | str,
    *,
    limit: int = 0,
    dpi: int = 300,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[Path]:
    """End-to-end: tables → cnsplots + Altair under ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables = build_consensus_viz_tables(metrics_dir, limit=limit, thresholds=thresholds)
    # persist tables for reuse / audits
    tables["corr"].to_csv(outdir / "table_pair_correlations.tsv", sep="\t", index=False)
    tables["length"].to_csv(outdir / "table_similar_lengths.tsv", sep="\t", index=False)
    tables["median"].to_csv(outdir / "table_median_by_posbin.tsv", sep="\t", index=False)
    written: list[Path] = [
        outdir / "table_pair_correlations.tsv",
        outdir / "table_similar_lengths.tsv",
        outdir / "table_median_by_posbin.tsv",
    ]
    written.extend(
        plot_consensus_cnsplots(
            tables["corr"], tables["length"], tables["long_rates"], tables["median"], outdir, dpi=dpi
        )
    )
    written.extend(
        plot_consensus_altair(
            tables["corr"], tables["length"], tables["long_rates"], tables["median"], outdir
        )
    )
    return written


# Presentation half-violins: orthologs | paralogs; up=thr_hi, down=thr_lo (no labels).
SCOPE_COLORS = {
    "orthologs": "#0072B2",  # Okabe–Ito blue
    "paralogs": "#D55E00",  # Okabe–Ito vermillion
}
HALFVIOLIN_SCOPES = ("orthologs", "paralogs")
HALFVIOLIN_THR_UP = 0.8
HALFVIOLIN_THR_DOWN = 0.5


def _kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE on ``grid``; delta fallback when variance is ~0."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(grid, dtype=float)
    if vals.size == 1 or float(np.std(vals)) < 1e-12:
        # Narrow bump around the constant value.
        center = float(vals.mean())
        bw = max(abs(center) * 0.02, 1.0)
        return np.exp(-0.5 * ((grid - center) / bw) ** 2)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(vals)
    dens = np.asarray(kde(grid), dtype=float)
    dens[~np.isfinite(dens)] = 0.0
    return dens


def _violin_support(
    values: np.ndarray,
    *,
    n_grid: int = 256,
    cut: float = 2.0,
    trim: bool = False,
) -> np.ndarray:
    """Evaluation grid for a violin KDE.

    ``trim=False`` (default): extend past data extremes by ``cut`` bandwidths
    (seaborn-style). ``trim=True``: clip exactly to [min, max].
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n_grid)
    lo = float(vals.min())
    hi = float(vals.max())
    if trim or vals.size == 1 or float(np.std(vals)) < 1e-12:
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1.0)
            return np.linspace(lo - pad, hi + pad, n_grid)
        return np.linspace(lo, hi, n_grid)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(vals)
    bw = float(np.sqrt(kde.covariance.flat[0]))
    return np.linspace(lo - cut * bw, hi + cut * bw, n_grid)


def _draw_half_violin(
    ax,
    values: np.ndarray,
    *,
    x_offset: float,
    side: str,
    color: str,
    height: float = 0.42,
    n_grid: int = 256,
    alpha: float = 0.85,
    cut: float = 2.0,
    trim: bool = False,
    y_gap: float = 0.0,
) -> None:
    """Fill a half-violin: metric on x; ``side='up'`` dens +y, ``side='down'`` dens −y.

    ``y_gap=0`` (nudge 0): upper and lower halves meet at y=0.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    grid = _violin_support(vals, n_grid=n_grid, cut=cut, trim=trim)
    dens = _kde_density(vals, grid)
    peak = float(dens.max()) if dens.size else 0.0
    if peak <= 0.0:
        return
    dens = dens / peak * height
    x = grid + float(x_offset)
    gap = float(y_gap)
    if side == "up":
        y0, y1 = gap, dens + gap
    else:
        y0, y1 = -gap, -(dens + gap)
    ax.fill_between(x, y0, y1, color=color, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(x, y1, color=color, lw=0.9, alpha=min(1.0, alpha + 0.1), zorder=3)


def plot_halfviolin_similar_lengths(
    length_df: pd.DataFrame,
    outdir: Path | str,
    *,
    metric: str = "similar_length_total",
    thr_up: float = HALFVIOLIN_THR_UP,
    thr_down: float = HALFVIOLIN_THR_DOWN,
    scopes: tuple[str, ...] = HALFVIOLIN_SCOPES,
    aspect: tuple[float, float] = (9.0, 5.0),
    dpi: int = 300,
    stem: str = "Figure_11_halfviolin_ortho_para_thr0p8_up_0p5_down",
    trim: bool = False,
    cut: float = 2.0,
    y_gap: float = 0.0,
) -> list[Path]:
    """Orthologs (left) / paralogs (right); thr_up half-violin up, thr_down down.

    Metric runs horizontally; density opens vertically (up=thr_up, down=thr_down).
    Presentation figure: no titles, axis labels, tick labels, legend, midline, or median.
    Aspect ratio defaults to 9:5. Uses the same similar-length table as Figure_03/04.
    """
    import matplotlib.pyplot as plt

    if metric not in length_df.columns:
        raise KeyError(f"metric {metric!r} not in columns {list(length_df.columns)}")
    needed = {"scope", "threshold", metric}
    missing = needed - set(length_df.columns)
    if missing:
        raise ValueError(f"length_df missing columns: {sorted(missing)}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Presentation figure: plain Agg canvas (avoid cnsplots DPI/size side-effects).
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": int(dpi),
            "savefig.dpi": int(dpi),
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig_w, fig_h = float(aspect[0]), float(aspect[1])
    if fig_w <= 0 or fig_h <= 0:
        raise ValueError(f"aspect must be positive, got {aspect}")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=int(dpi))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.98)

    # Shared metric scale; scopes side-by-side with a gap.
    panel: dict[str, dict[str, np.ndarray]] = {}
    global_max = 0.0
    global_min = 0.0
    for scope in scopes:
        if scope not in SCOPE_COLORS:
            raise KeyError(f"No color for scope {scope!r}; known={sorted(SCOPE_COLORS)}")
        panel[scope] = {}
        for thr in (thr_up, thr_down):
            sub = length_df[
                (length_df["scope"] == scope)
                & (np.isclose(length_df["threshold"].astype(float), thr))
            ]
            vals = sub[metric].to_numpy(dtype=float)
            if vals.size == 0:
                raise ValueError(f"No rows for scope={scope!r} threshold={thr}")
            panel[scope][str(thr)] = vals
            finite = vals[np.isfinite(vals)]
            if finite.size:
                global_max = max(global_max, float(finite.max()))
                global_min = min(global_min, float(finite.min()))

    # Layout span: data range plus untrimmed KDE tails (approx cut·bw ≤ span).
    span = max(global_max - global_min, 1.0)
    layout_span = span * (1.0 + 0.15 if not trim else 1.0)
    gap = max(layout_span * 0.18, 20.0)
    x_offsets = {scope: float(i) * (layout_span + gap) for i, scope in enumerate(scopes)}
    for scope in scopes:
        color = SCOPE_COLORS[scope]
        x0 = x_offsets[scope]
        _draw_half_violin(
            ax,
            panel[scope][str(thr_up)],
            x_offset=x0,
            side="up",
            color=color,
            cut=cut,
            trim=trim,
            y_gap=y_gap,
        )
        _draw_half_violin(
            ax,
            panel[scope][str(thr_down)],
            x_offset=x0,
            side="down",
            color=color,
            cut=cut,
            trim=trim,
            y_gap=y_gap,
        )

    x_right = x_offsets[scopes[-1]] + layout_span + gap * 0.15
    ax.set_xlim(global_min - gap * 0.1, x_right)
    ax.set_ylim(-0.55, 0.55)

    # Strip all chrome for slide overlays (no midline, no median marks).
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    out_stem = outdir / stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = out_stem.with_suffix(f".{ext}")
        save_kw: dict = {
            "facecolor": "white",
            "edgecolor": "none",
            "bbox_inches": None,
            "pad_inches": 0,
        }
        if ext == "png":
            save_kw["dpi"] = int(dpi)
        fig.savefig(path, **save_kw)
        written.append(path)
    plt.close(fig)
    return written
