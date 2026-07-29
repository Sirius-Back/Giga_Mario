"""SBS diagnostics: PCA of feature tables (cnsplots + Altair).

Default plot policy:
- random subsample of **10 000** sequences (seeded)
- points sorted by ``PC1 * PC2`` (draw order)
- numeric fold/cluster labels → continuous (viridis) gradient
- categorical labels (train/test/val, genome, …) → discrete Okabe–Ito palette
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.splits.sbs.features import FeatureTable

DEFAULT_PLOT_N = 10_000

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


def _labels_are_numeric(labels: Sequence[str], *, allow_zsv: bool = True) -> bool:
    """True when labels are numeric fold/cluster ids (optional ``zsv`` holdouts)."""
    parsed = 0
    for lab in labels:
        s = str(lab).strip()
        if s == "" or s.upper() == "NA":
            continue
        if allow_zsv and s.lower() in {"zsv", "zeroshotvalidation", "zero-shot-validation"}:
            continue
        try:
            float(s)
        except ValueError:
            return False
        parsed += 1
    return parsed > 0


def _parse_numeric_label(lab: str) -> float | None:
    s = str(lab).strip()
    if s == "" or s.upper() == "NA":
        return None
    if s.lower() in {"zsv", "zeroshotvalidation", "zero-shot-validation"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _subsample_indices(
    n: int, *, max_points: int, seed: int
) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=int(max_points), replace=False)


def _sort_by_pc1_pc2(coords: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Order indices by PC1*PC2 (ascending) for stable gradient draw order."""
    prod = coords[idx, 0] * coords[idx, 1]
    return idx[np.argsort(prod, kind="mergesort")]


def prepare_plot_indices(
    coords: np.ndarray,
    *,
    max_points: int = DEFAULT_PLOT_N,
    seed: int = 42,
) -> np.ndarray:
    """Random subsample (default 10k) then sort by PC1*PC2."""
    n = int(coords.shape[0])
    idx = _subsample_indices(n, max_points=max_points, seed=seed)
    return _sort_by_pc1_pc2(coords, idx)


def _color_map_categorical(labels: Sequence[str]) -> dict[str, str]:
    uniq = sorted(set(labels), key=lambda s: (s == "zsv", s))
    return {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(uniq)}


def _save_cns_pca(
    coords: np.ndarray,
    labels: Sequence[str],
    *,
    out_stem: Path,
    title: str,
    dpi: int = 300,
    max_points: int = DEFAULT_PLOT_N,
    seed: int = 42,
    color_title: str = "label",
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    used_cns = False
    try:
        import cnsplots as cns

        cns.set_style("nature")
        used_cns = True
    except Exception:  # noqa: BLE001
        pass

    idx = prepare_plot_indices(coords, max_points=max_points, seed=seed)
    coords_p = coords[idx]
    labels_p = [str(labels[i]) for i in idx]
    numeric = _labels_are_numeric(labels_p)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if numeric:
        vals = np.array([_parse_numeric_label(v) for v in labels_p], dtype=object)
        num_mask = np.array([v is not None for v in vals], dtype=bool)
        zsv_mask = ~num_mask
        num_vals = np.asarray([float(v) for v in vals[num_mask]], dtype=float)
        if len(num_vals):
            norm = Normalize(vmin=float(np.min(num_vals)), vmax=float(np.max(num_vals)))
            try:
                from matplotlib import colormaps

                cmap = colormaps["viridis"]
            except Exception:  # noqa: BLE001
                cmap = cm.get_cmap("viridis")
            # Draw in PC1*PC2 order (idx already sorted); keep that order within mask
            order = np.where(num_mask)[0]
            sc = ax.scatter(
                coords_p[order, 0],
                coords_p[order, 1],
                s=18,
                c=num_vals,
                cmap=cmap,
                norm=norm,
                alpha=0.85,
                edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(color_title)
            cbar.ax.tick_params(labelsize=7)
        if np.any(zsv_mask):
            ax.scatter(
                coords_p[zsv_mask, 0],
                coords_p[zsv_mask, 1],
                s=22,
                c="#000000",
                marker="x",
                linewidths=0.6,
                alpha=0.7,
                label="zsv",
            )
            ax.legend(loc="best", fontsize=7, frameon=False)
    else:
        cmap = _color_map_categorical(labels_p)
        for lab in sorted(set(labels_p), key=lambda s: (s == "zsv", s)):
            mask = [i for i, L in enumerate(labels_p) if L == lab]
            ax.scatter(
                coords_p[mask, 0],
                coords_p[mask, 1],
                s=18,
                c=cmap[lab],
                label=str(lab),
                alpha=0.85,
                edgecolors="none",
            )
        ax.legend(
            loc="best",
            fontsize=7,
            frameon=False,
            markerscale=1.2,
            ncol=1 if len(set(labels_p)) <= 8 else 2,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
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
    max_points: int = DEFAULT_PLOT_N,
    seed: int = 42,
    color_title: str = "label",
) -> list[str]:
    import pandas as pd

    idx = prepare_plot_indices(coords, max_points=max_points, seed=seed)
    labels_p = [str(labels[i]) for i in idx]
    numeric = _labels_are_numeric(labels_p)
    rows = []
    for j, i in enumerate(idx):
        lab = labels_p[j]
        num = _parse_numeric_label(lab)
        rows.append(
            {
                "region": regions[i],
                "PC1": float(coords[i, 0]),
                "PC2": float(coords[i, 1]),
                "PC1xPC2": float(coords[i, 0] * coords[i, 1]),
                "label": lab,
                "label_num": num if num is not None else float("nan"),
                "is_zsv": num is None and lab.lower().startswith("zsv"),
            }
        )
    df = pd.DataFrame(rows)
    written: list[str] = []
    try:
        import altair as alt

        # DEFAULT_PLOT_N is 10k; Altair's default row cap is 5k.
        alt.data_transformers.disable_max_rows()

        if numeric:
            base = alt.Chart(df).encode(
                x=alt.X("PC1:Q"),
                y=alt.Y("PC2:Q"),
                order=alt.Order("PC1xPC2:Q"),
                tooltip=["region", "label", "PC1", "PC2", "PC1xPC2"],
            )
            grad = (
                base.transform_filter("isValid(datum.label_num)")
                .mark_circle(size=40, opacity=0.75)
                .encode(
                    color=alt.Color(
                        "label_num:Q",
                        title=color_title,
                        scale=alt.Scale(scheme="viridis"),
                    )
                )
            )
            zsv = (
                base.transform_filter("datum.is_zsv")
                .mark_point(shape="cross", size=50, opacity=0.8, color="black")
            )
            chart = (grad + zsv).properties(width=420, height=360, title=title).interactive()
        else:
            chart = (
                alt.Chart(df)
                .mark_circle(size=40, opacity=0.75)
                .encode(
                    x=alt.X("PC1:Q"),
                    y=alt.Y("PC2:Q"),
                    color=alt.Color("label:N", title=color_title),
                    order=alt.Order("PC1xPC2:Q"),
                    tooltip=["region", "label", "PC1", "PC2", "PC1xPC2"],
                )
                .properties(width=420, height=360, title=title)
                .interactive()
            )
        html = out_stem.with_name(out_stem.name + "_altair.html")
        vl = out_stem.with_name(out_stem.name + "_altair.vl.json")
        chart.save(str(html))
        try:
            vl.write_text(json.dumps(chart.to_dict(), indent=2) + "\n", encoding="utf-8")
            written.extend([str(html), str(vl)])
        except TypeError:
            # Layered charts may not JSON-serialize cleanly on older Altair.
            written.append(str(html))
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
    max_points: int = DEFAULT_PLOT_N,
    color_title: str | None = None,
) -> dict[str, Any]:
    """Single PCA scatter colored by ``labels_by_region`` (cnsplots + Altair)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coords, pca_meta = _pca_coords(features, seed=seed)
    labels = [str(labels_by_region.get(rid, "NA")) for rid in features.ids]
    stem_path = outdir / stem
    ctitle = color_title or "label"
    written = _save_cns_pca(
        coords,
        labels,
        out_stem=stem_path,
        title=title,
        dpi=dpi,
        seed=seed,
        max_points=max_points,
        color_title=ctitle,
    )
    written.extend(
        _save_altair_pca(
            coords,
            list(features.ids),
            labels,
            out_stem=stem_path,
            title=title,
            max_points=max_points,
            seed=seed,
            color_title=ctitle,
        )
    )
    return {
        "stem": stem,
        "title": title,
        "written": written,
        "pca": pca_meta,
        "max_points": max_points,
        "numeric_color": _labels_are_numeric(labels),
    }


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
    max_points: int = DEFAULT_PLOT_N,
) -> dict[str, Any]:
    """PCA feature diagnostics with standard labelings.

    1. cluster / fold (numeric → viridis gradient)
    2. train_test (train/test/val/zsv)
    3. genome (from ID.csv)
    4. custom column from a stratification-like CSV (optional)

    Points: random ``max_points`` (default 10 000), sorted by PC1×PC2.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_region = {r["region"]: r for r in assignment_rows}

    # Prefer numeric fold when present; else cluster id (also numeric under SBS).
    cluster_labels: dict[str, str] = {}
    for rid in features.ids:
        row = by_region.get(rid)
        if row is None:
            cluster_labels[rid] = "NA"
            continue
        fold_v = str(row.get("fold", "")).strip()
        cluster_v = str(row.get("cluster", "")).strip()
        cluster_labels[rid] = fold_v if fold_v != "" else cluster_v
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
            title="PCA features · fold (cluster)",
            seed=seed,
            dpi=dpi,
            max_points=max_points,
            color_title="fold",
        ),
        plot_feature_pca(
            features,
            labels_by_region=tt_labels,
            outdir=outdir,
            stem="pca_by_train_test",
            title="PCA features · train/test/val",
            seed=seed,
            dpi=dpi,
            max_points=max_points,
            color_title="split",
        ),
        plot_feature_pca(
            features,
            labels_by_region=genome_labels,
            outdir=outdir,
            stem="pca_by_genome",
            title="PCA features · genome",
            seed=seed,
            dpi=dpi,
            max_points=max_points,
            color_title="genome",
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
                max_points=max_points,
                color_title=custom_label_column,
            )
        )

    # Combined Altair faceted overview (same 10k / PC1×PC2 policy)
    try:
        import altair as alt
        import pandas as pd

        alt.data_transformers.disable_max_rows()
        coords, _ = _pca_coords(features, seed=seed)
        idx = prepare_plot_indices(coords, max_points=max_points, seed=seed)
        frames = []
        for kind, labmap in (
            ("fold", cluster_labels),
            ("train_test", tt_labels),
            ("genome", genome_labels),
        ):
            for i in idx:
                rid = features.ids[i]
                lab = labmap[rid]
                num = _parse_numeric_label(lab)
                frames.append(
                    {
                        "panel": kind,
                        "region": rid,
                        "PC1": float(coords[i, 0]),
                        "PC2": float(coords[i, 1]),
                        "PC1xPC2": float(coords[i, 0] * coords[i, 1]),
                        "label": lab,
                        "label_num": num if num is not None else float("nan"),
                        "is_zsv": bool(
                            num is None and str(lab).lower().startswith("zsv")
                        ),
                    }
                )
        df = pd.DataFrame(frames)
        fold_df = df[df["panel"] == "fold"].copy()
        other_df = df[df["panel"] != "fold"].copy()
        charts = []
        if len(fold_df) and _labels_are_numeric(fold_df["label"].tolist()):
            grad = (
                alt.Chart(fold_df)
                .transform_filter("isValid(datum.label_num)")
                .mark_circle(size=30, opacity=0.7)
                .encode(
                    x="PC1:Q",
                    y="PC2:Q",
                    color=alt.Color(
                        "label_num:Q",
                        title="fold",
                        scale=alt.Scale(scheme="viridis"),
                    ),
                    order="PC1xPC2:Q",
                    tooltip=["region", "label", "panel"],
                )
                .properties(width=260, height=240, title="fold")
            )
            zsv = (
                alt.Chart(fold_df)
                .transform_filter("datum.is_zsv")
                .mark_point(shape="cross", size=40, opacity=0.8, color="black")
                .encode(x="PC1:Q", y="PC2:Q", order="PC1xPC2:Q")
            )
            charts.append(grad + zsv)
        elif len(fold_df):
            other_df = pd.concat([other_df, fold_df], ignore_index=True)
        if len(other_df):
            charts.append(
                alt.Chart(other_df)
                .mark_circle(size=30, opacity=0.7)
                .encode(
                    x="PC1:Q",
                    y="PC2:Q",
                    color=alt.Color("label:N", legend=alt.Legend(title="label")),
                    order="PC1xPC2:Q",
                    tooltip=["region", "label", "panel"],
                    facet=alt.Facet("panel:N", columns=2, title=None),
                )
                .properties(width=260, height=240)
                .resolve_scale(color="independent")
            )
        if len(charts) == 1:
            chart = charts[0].properties(title="SBS PCA diagnostics")
        else:
            chart = alt.hconcat(*charts).properties(title="SBS PCA diagnostics")
        html = outdir / "pca_diagnostics_combined_altair.html"
        vl = outdir / "pca_diagnostics_combined_altair.vl.json"
        chart.save(str(html))
        try:
            vl.write_text(json.dumps(chart.to_dict(), indent=2) + "\n", encoding="utf-8")
            combined = [str(html), str(vl)]
        except TypeError:
            combined = [str(html)]
    except Exception as exc:  # noqa: BLE001
        err = outdir / "pca_diagnostics_combined_altair_error.txt"
        err.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        combined = [str(err)]

    meta = {
        "panels": panels,
        "combined": combined,
        "max_points": max_points,
        "sort": "PC1*PC2",
        "numeric_fold_gradient": True,
    }
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
