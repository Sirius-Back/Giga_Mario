"""SBS diagnostics: PCA of feature tables (cnsplots + Altair)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.splits.sbs.features import FeatureTable

# Okabe–Ito-ish categorical palette
_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
    "#999999",
]


def _genome_map_from_id_csv(id_csv: Path | None) -> dict[str, str]:
    if id_csv is None:
        return {}
    from src.pipeline.common import read_csv

    rows = read_csv(Path(id_csv))
    out: dict[str, str] = {}
    for row in rows:
        rid = row.get("ID", "").strip()
        genome = row.get("genome", row.get("Genome", "")).strip()
        if rid:
            out[rid] = genome or "unknown"
    return out


def _labels_from_strat_csv(
    strat_csv: Path, column: str, ids: Sequence[str]
) -> dict[str, str]:
    from src.pipeline.common import read_csv

    rows = read_csv(Path(strat_csv))
    if not rows:
        raise ValueError(f"empty stratification/label file: {strat_csv}")
    if "ID" not in rows[0]:
        raise ValueError(f"{strat_csv} missing ID column")
    if column not in rows[0]:
        raise ValueError(
            f"{strat_csv} missing label column {column!r}; have {list(rows[0])}"
        )
    table = {r["ID"].strip(): str(r.get(column, "")) for r in rows}
    out: dict[str, str] = {}
    missing = []
    for rid in ids:
        if rid not in table:
            missing.append(rid)
        else:
            out[rid] = table[rid] if table[rid] != "" else "NA"
    if missing:
        raise ValueError(
            f"label file missing {len(missing)} id(s); example={missing[0]!r}"
        )
    return out


def _pca_coords(features: FeatureTable, *, seed: int = 42) -> tuple[np.ndarray, dict]:
    from sklearn.decomposition import PCA

    x = features.scaled_matrix()
    n_comp = min(2, features.n, features.n_features)
    if n_comp < 1:
        raise ValueError("cannot run PCA on empty feature table")
    pca = PCA(n_components=n_comp, random_state=seed)
    coords = pca.fit_transform(x)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    meta = {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "n_components": int(n_comp),
    }
    return coords, meta


def _color_map(labels: Sequence[str]) -> dict[str, str]:
    uniq = sorted(set(labels), key=lambda s: (s == "zsv", s))
    return {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(uniq)}


def _save_cns_pca(
    coords: np.ndarray,
    labels: Sequence[str],
    *,
    out_stem: Path,
    title: str,
    dpi: int = 300,
) -> list[str]:
    import matplotlib.pyplot as plt

    used_cns = False
    try:
        import cnsplots as cns

        cns.set_style("nature")
        used_cns = True
    except Exception:  # noqa: BLE001
        pass

    cmap = _color_map(labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab in sorted(set(labels), key=lambda s: (s == "zsv", s)):
        mask = [i for i, L in enumerate(labels) if L == lab]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            c=cmap[lab],
            label=str(lab),
            alpha=0.85,
            edgecolors="none",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(
        loc="best",
        fontsize=7,
        frameon=False,
        markerscale=1.2,
        ncol=1 if len(set(labels)) <= 8 else 2,
    )
    written: list[str] = []
    for ext in ("pdf", "svg", "png"):
        path = out_stem.with_suffix(f".{ext}")
        if used_cns:
            try:
                import cnsplots as cns

                cns.savefig(path)
            except Exception:  # noqa: BLE001
                fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        else:
            fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def _save_altair_pca(
    coords: np.ndarray,
    regions: Sequence[str],
    labels: Sequence[str],
    *,
    out_stem: Path,
    title: str,
    max_points: int = 5000,
) -> list[str]:
    import pandas as pd

    n = len(regions)
    idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
    df = pd.DataFrame(
        {
            "region": [regions[i] for i in idx],
            "PC1": coords[idx, 0],
            "PC2": coords[idx, 1],
            "label": [labels[i] for i in idx],
        }
    )
    written: list[str] = []
    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_circle(size=40, opacity=0.75)
            .encode(
                x=alt.X("PC1:Q"),
                y=alt.Y("PC2:Q"),
                color=alt.Color("label:N", title="label"),
                tooltip=["region", "label", "PC1", "PC2"],
            )
            .properties(width=420, height=360, title=title)
            .interactive()
        )
        html = out_stem.with_name(out_stem.name + "_altair.html")
        vl = out_stem.with_name(out_stem.name + "_altair.vl.json")
        chart.save(str(html))
        vl.write_text(json.dumps(chart.to_dict(), indent=2) + "\n", encoding="utf-8")
        written.extend([str(html), str(vl)])
    except Exception as exc:  # noqa: BLE001
        err = out_stem.with_name(out_stem.name + "_altair_error.txt")
        err.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        written.append(str(err))
    return written


def plot_feature_pca(
    features: FeatureTable,
    *,
    labels_by_region: Mapping[str, str],
    outdir: Path,
    stem: str,
    title: str,
    seed: int = 42,
    dpi: int = 300,
) -> dict[str, Any]:
    """Single PCA scatter colored by ``labels_by_region`` (cnsplots + Altair)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coords, pca_meta = _pca_coords(features, seed=seed)
    labels = [str(labels_by_region.get(rid, "NA")) for rid in features.ids]
    stem_path = outdir / stem
    written = _save_cns_pca(coords, labels, out_stem=stem_path, title=title, dpi=dpi)
    written.extend(
        _save_altair_pca(
            coords,
            list(features.ids),
            labels,
            out_stem=stem_path,
            title=title,
        )
    )
    return {"stem": stem, "title": title, "written": written, "pca": pca_meta}


def plot_sbs_pca_diagnostics(
    features: FeatureTable,
    assignment_rows: Sequence[Mapping[str, str]],
    *,
    outdir: Path,
    id_csv: Path | None = None,
    custom_label_csv: Path | None = None,
    custom_label_column: str | None = None,
    seed: int = 42,
    dpi: int = 300,
) -> dict[str, Any]:
    """PCA feature diagnostics with four standard labelings.

    1. cluster (= fold)
    2. train_test (train/test/val/zsv)
    3. genome (from ID.csv)
    4. custom column from a stratification-like CSV (optional)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_region = {r["region"]: r for r in assignment_rows}

    cluster_labels = {
        rid: str(by_region[rid]["cluster"]) if rid in by_region else "NA"
        for rid in features.ids
    }
    tt_labels = {
        rid: str(by_region[rid]["train_test"]) if rid in by_region else "NA"
        for rid in features.ids
    }
    genomes = _genome_map_from_id_csv(id_csv)
    genome_labels = {rid: genomes.get(rid, "unknown") for rid in features.ids}

    panels = [
        plot_feature_pca(
            features,
            labels_by_region=cluster_labels,
            outdir=outdir,
            stem="pca_by_cluster",
            title="PCA features · cluster (=fold)",
            seed=seed,
            dpi=dpi,
        ),
        plot_feature_pca(
            features,
            labels_by_region=tt_labels,
            outdir=outdir,
            stem="pca_by_train_test",
            title="PCA features · train/test/val",
            seed=seed,
            dpi=dpi,
        ),
        plot_feature_pca(
            features,
            labels_by_region=genome_labels,
            outdir=outdir,
            stem="pca_by_genome",
            title="PCA features · genome",
            seed=seed,
            dpi=dpi,
        ),
    ]

    if custom_label_csv is not None:
        if not custom_label_column:
            raise ValueError(
                "custom_label_column is required when custom_label_csv is set"
            )
        custom = _labels_from_strat_csv(
            Path(custom_label_csv), custom_label_column, list(features.ids)
        )
        panels.append(
            plot_feature_pca(
                features,
                labels_by_region=custom,
                outdir=outdir,
                stem=f"pca_by_custom_{custom_label_column}",
                title=f"PCA features · {custom_label_column}",
                seed=seed,
                dpi=dpi,
            )
        )

    # Combined Altair faceted overview (subsample for browser)
    try:
        import altair as alt
        import pandas as pd

        coords, _ = _pca_coords(features, seed=seed)
        n = features.n
        idx = np.arange(n)
        if n > 3000:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n, size=3000, replace=False))
        frames = []
        for kind, labmap in (
            ("cluster", cluster_labels),
            ("train_test", tt_labels),
            ("genome", genome_labels),
        ):
            for i in idx:
                rid = features.ids[i]
                frames.append(
                    {
                        "panel": kind,
                        "region": rid,
                        "PC1": float(coords[i, 0]),
                        "PC2": float(coords[i, 1]),
                        "label": labmap[rid],
                    }
                )
        df = pd.DataFrame(frames)
        chart = (
            alt.Chart(df)
            .mark_circle(size=30, opacity=0.7)
            .encode(
                x="PC1:Q",
                y="PC2:Q",
                color=alt.Color("label:N", legend=alt.Legend(title="label")),
                tooltip=["region", "label", "panel"],
                facet=alt.Facet("panel:N", columns=3, title=None),
            )
            .properties(width=260, height=240, title="SBS PCA diagnostics")
            .resolve_scale(color="independent")
        )
        html = outdir / "pca_diagnostics_combined_altair.html"
        vl = outdir / "pca_diagnostics_combined_altair.vl.json"
        chart.save(str(html))
        vl.write_text(json.dumps(chart.to_dict(), indent=2) + "\n", encoding="utf-8")
        combined = [str(html), str(vl)]
    except Exception as exc:  # noqa: BLE001
        err = outdir / "pca_diagnostics_combined_altair_error.txt"
        err.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        combined = [str(err)]

    meta = {"panels": panels, "combined": combined}
    (outdir / "pca_diagnostics_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


# Legacy name kept as a thin redirect message for old imports
def plot_distance_heatmap(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise TypeError(
        "plot_distance_heatmap is retired; use plot_sbs_pca_diagnostics / "
        "plot_feature_pca on FeatureTable instead."
    )
