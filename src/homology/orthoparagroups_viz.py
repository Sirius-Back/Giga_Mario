"""Distribution figures for orthoparagroups ``clusters.tsv`` (cnsplots + Altair)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.homology.visualize import _apply_cns_style


NUMERIC_COLS = (
    "n_nodes",
    "n_distinct_orthology_groups",
    "n_ortholog_edges",
    "n_paralog_edges",
    "n_written_orthologs",
    "n_written_paralogs",
)


def load_clusters(path: Path | str) -> pd.DataFrame:
    """Load ``clusters.tsv`` and derive log / species-span columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"clusters.tsv missing: {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"fna_name", *NUMERIC_COLS, "nodes_per_species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clusters.tsv missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"clusters.tsv is empty: {path}")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(NUMERIC_COLS)].isna().any().any():
        bad = df[list(NUMERIC_COLS)].isna().any(axis=1).sum()
        raise ValueError(f"Non-numeric values in {bad} cluster rows")

    df["log10_n_nodes"] = np.log10(df["n_nodes"].clip(lower=1))
    df["log10_n_ortholog_edges"] = np.log10(df["n_ortholog_edges"].clip(lower=1))
    df["log10_n_paralog_edges"] = np.log10(df["n_paralog_edges"].clip(lower=1))
    df["log10_n_ogs"] = np.log10(df["n_distinct_orthology_groups"].clip(lower=1))
    df["log1p_n_written_paralogs"] = np.log1p(df["n_written_paralogs"])
    df["n_written_total"] = df["n_written_orthologs"] + df["n_written_paralogs"]
    df["n_species"] = df["nodes_per_species"].map(_n_species)
    return df


def expand_nodes_per_species(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (cluster, species) gene count."""
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        raw = str(row["nodes_per_species"] or "")
        if not raw or raw == "nan":
            continue
        for part in raw.split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            species, count_s = part.rsplit(":", 1)
            try:
                count = int(count_s)
            except ValueError as exc:
                raise ValueError(f"Bad nodes_per_species token {part!r}") from exc
            rows.append(
                {
                    "fna_name": row["fna_name"],
                    "species": species,
                    "n_genes": count,
                    "log10_n_genes": float(np.log10(max(count, 1))),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No nodes_per_species entries parsed")
    return out


def _n_species(cell: object) -> int:
    raw = str(cell or "")
    if not raw or raw == "nan":
        return 0
    return sum(1 for p in raw.split(";") if p.strip() and ":" in p)


def plot_clusters_cnsplots(
    df: pd.DataFrame,
    species_long: pd.DataFrame,
    outdir: Path | str,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Publication static histograms / scatter via cnsplots."""
    import matplotlib.pyplot as plt
    import cnsplots as cns

    from src.train_viz.plotting import save_cns_figure

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _apply_cns_style(dpi)
    written: list[Path] = []

    specs: list[tuple[str, str, str, str, str]] = [
        ("log10_n_nodes", "log10(genes per cluster)", "Clusters", "Cluster size", "Figure_01_n_nodes_hist"),
        (
            "log10_n_ogs",
            "log10(distinct orthology groups)",
            "Clusters",
            "Orthology-group count",
            "Figure_02_n_ogs_hist",
        ),
        (
            "log10_n_ortholog_edges",
            "log10(ortholog edges)",
            "Clusters",
            "Ortholog-edge count",
            "Figure_03_n_ortholog_edges_hist",
        ),
        (
            "log10_n_paralog_edges",
            "log10(paralog edges)",
            "Clusters",
            "Paralog-edge count",
            "Figure_04_n_paralog_edges_hist",
        ),
        (
            "n_written_orthologs",
            "Written ortholog sequences",
            "Clusters",
            "Written orthologs",
            "Figure_05_n_written_orthologs_hist",
        ),
        (
            "log1p_n_written_paralogs",
            "log1p(written paralogs)",
            "Clusters",
            "Written paralogs",
            "Figure_06_n_written_paralogs_hist",
        ),
        (
            "n_species",
            "Species represented in cluster",
            "Clusters",
            "Species span",
            "Figure_07_n_species_hist",
        ),
    ]
    for col, xlabel, ylabel, title, stem in specs:
        cns.figure(width=360, height=260)
        ax = cns.histplot(data=df, x=col, bins=40)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / stem, dpi))

    cns.figure(width=360, height=300)
    ax = cns.scatterplot(
        data=df,
        x="n_nodes",
        y="n_paralog_edges",
        alpha=0.35,
        s=12,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Genes per cluster")
    ax.set_ylabel("Paralog edges")
    ax.set_title("Cluster size vs paralog edges")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_08_nodes_vs_paralog_edges", dpi))

    cns.figure(width=360, height=300)
    ax = cns.scatterplot(
        data=df,
        x="n_written_orthologs",
        y="n_written_paralogs",
        alpha=0.35,
        s=12,
    )
    ax.set_xlabel("Written orthologs")
    ax.set_ylabel("Written paralogs")
    ax.set_title("Written ortholog vs paralog counts")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_09_written_ortho_vs_para", dpi))

    cns.figure(width=420, height=280)
    ax = cns.histplot(data=species_long, x="log10_n_genes", bins=40)
    ax.set_xlabel("log10(genes of one species in a cluster)")
    ax.set_ylabel("Cluster×species pairs")
    ax.set_title("Per-species gene counts within clusters")
    cns.setup_ax(ax)
    written.extend(save_cns_figure(outdir / "Figure_10_species_gene_counts_hist", dpi))

    plt.close("all")
    return written


def plot_clusters_altair(
    df: pd.DataFrame,
    species_long: pd.DataFrame,
    outdir: Path | str,
) -> list[Path]:
    """Interactive Altair histograms / scatters (HTML + VL + PNG)."""
    import altair as alt

    from src.train_viz.plotting import save_altair_chart

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alt.data_transformers.disable_max_rows()

    hist_specs: list[tuple[str, str, str, str]] = [
        ("log10_n_nodes", "log10(genes per cluster)", "Cluster size", "Figure_01_n_nodes_hist_altair"),
        (
            "log10_n_ogs",
            "log10(distinct orthology groups)",
            "Orthology-group count",
            "Figure_02_n_ogs_hist_altair",
        ),
        (
            "log10_n_ortholog_edges",
            "log10(ortholog edges)",
            "Ortholog-edge count",
            "Figure_03_n_ortholog_edges_hist_altair",
        ),
        (
            "log10_n_paralog_edges",
            "log10(paralog edges)",
            "Paralog-edge count",
            "Figure_04_n_paralog_edges_hist_altair",
        ),
        (
            "n_written_orthologs",
            "Written ortholog sequences",
            "Written orthologs",
            "Figure_05_n_written_orthologs_hist_altair",
        ),
        (
            "log1p_n_written_paralogs",
            "log1p(written paralogs)",
            "Written paralogs",
            "Figure_06_n_written_paralogs_hist_altair",
        ),
        (
            "n_species",
            "Species represented in cluster",
            "Species span",
            "Figure_07_n_species_hist_altair",
        ),
    ]
    for col, xlabel, title, stem in hist_specs:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=xlabel),
                y=alt.Y("count()", title="Clusters"),
                tooltip=[col, "fna_name", "n_nodes"],
            )
            .properties(title=title, width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / stem))

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.35, size=28)
        .encode(
            x=alt.X("n_nodes:Q", scale=alt.Scale(type="log"), title="Genes per cluster"),
            y=alt.Y("n_paralog_edges:Q", scale=alt.Scale(type="log"), title="Paralog edges"),
            tooltip=[
                "fna_name",
                "n_nodes",
                "n_paralog_edges",
                "n_ortholog_edges",
                "n_distinct_orthology_groups",
            ],
        )
        .properties(title="Cluster size vs paralog edges", width=420, height=320)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_08_nodes_vs_paralog_edges_altair"))

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.35, size=28)
        .encode(
            x=alt.X("n_written_orthologs:Q", title="Written orthologs"),
            y=alt.Y("n_written_paralogs:Q", title="Written paralogs"),
            tooltip=["fna_name", "n_written_orthologs", "n_written_paralogs", "n_nodes"],
        )
        .properties(title="Written ortholog vs paralog counts", width=420, height=320)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_09_written_ortho_vs_para_altair"))

    chart = (
        alt.Chart(species_long)
        .mark_bar()
        .encode(
            x=alt.X("log10_n_genes:Q", bin=alt.Bin(maxbins=40), title="log10(genes / species / cluster)"),
            y=alt.Y("count()", title="Cluster×species pairs"),
            color=alt.Color("species:N", legend=alt.Legend(title="Species")),
            tooltip=["species", "n_genes"],
        )
        .properties(title="Per-species gene counts within clusters", width=420, height=280)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_10_species_gene_counts_hist_altair"))

    med = (
        species_long.groupby("species", as_index=False)["n_genes"]
        .median()
        .sort_values("n_genes", ascending=False)
    )
    chart = (
        alt.Chart(med)
        .mark_bar()
        .encode(
            x=alt.X("n_genes:Q", title="Median genes per cluster (when present)"),
            y=alt.Y("species:N", sort="-x", title="Species"),
            tooltip=["species", "n_genes"],
        )
        .properties(title="Median per-species occupancy in extracted clusters", width=420, height=320)
    )
    written.extend(save_altair_chart(chart, outdir / "Figure_11_species_median_occupancy_altair"))

    return written


def plot_clusters_tsv(
    clusters_tsv: Path | str,
    outdir: Path | str,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Load ``clusters.tsv`` and write cnsplots + Altair distribution figures."""
    df = load_clusters(clusters_tsv)
    species_long = expand_nodes_per_species(df)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(plot_clusters_cnsplots(df, species_long, outdir, dpi=dpi))
    written.extend(plot_clusters_altair(df, species_long, outdir))
    return written
