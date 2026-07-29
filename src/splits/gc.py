"""GC split strategy — SBS with GC% + AAA% feature clustering.

Caption: ``splits/gc.md``. Wired into ``split-predict`` as ``type=gc``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.splits.sbs.assign import (
    ClusterMethod,
    assign_from_features,
    assignment_rows_to_split_csv,
    write_assignment_table,
)
from src.splits.sbs.backends.gc import GcAaaFeatureBackend
from src.splits.sbs.features import FeatureTable, compute_feature_table
from src.splits.sbs.fna_io import FastaMode
from src.splits.sbs.visualize import plot_sbs_pca_diagnostics

__all__ = (
    "SPLIT_ID",
    "run_gc_split_assign",
    "compute_gc_feature_table",
)

SPLIT_ID = "gc"


def compute_gc_feature_table(
    fna: Path,
    *,
    mode: FastaMode = "auto",
    ids: list[str] | None = None,
    max_ids: int | None = None,
) -> FeatureTable:
    """C1 helper: FNA → FeatureTable (GC_pct, AAA_pct)."""
    return compute_feature_table(
        fna,
        GcAaaFeatureBackend(),
        mode=mode,
        ids=ids,
        max_ids=max_ids,
    )


def run_gc_split_assign(
    *,
    outdir: Path,
    fna: Path,
    id_csv: Path | None = None,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    seed: int = 42,
    max_ids: int | None = None,
    ids: list[str] | None = None,
    fna_mode: FastaMode = "auto",
    n_clusters: int | Literal["auto"] = "auto",
    cluster_method: ClusterMethod = "dbscan",
    ratios: tuple[float, float, float] | None = None,
    plot: bool = True,
    plot_max_n: int | None = None,
    custom_label_csv: Path | None = None,
    custom_label_column: str | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
) -> dict[str, Any]:
    """FNA → GC%/AAA% features → SBS assignment → ``split.csv`` (+ PCA diagnostics)."""
    _ = plot_max_n  # retained for CLI compat; PCA subsamples internally
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    features = compute_gc_feature_table(
        fna, mode=fna_mode, ids=ids, max_ids=max_ids
    )
    feat_csv = features.write_csv(outdir / "feature_table.csv")

    rows, meta = assign_from_features(
        features,
        fold_csv=fold_csv,
        stratification_csv=stratification_csv,
        seed=seed,
        n_clusters=n_clusters,
        cluster_method=cluster_method,
        ratios=ratios,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    assign_path = write_assignment_table(rows, outdir / "sbs_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, outdir)

    plot_meta: dict[str, Any] | None = None
    if plot:
        # Optional custom panel: reuse stratification file/column when requested
        custom_csv = custom_label_csv
        custom_col = custom_label_column
        if custom_csv is None and stratification_csv is not None and custom_col:
            custom_csv = stratification_csv
        plot_meta = plot_sbs_pca_diagnostics(
            features,
            rows,
            outdir=outdir / "figures",
            id_csv=id_csv,
            custom_label_csv=custom_csv,
            custom_label_column=custom_col,
            seed=seed,
        )

    summary = {
        "split_id": SPLIT_ID,
        "seed": seed,
        "fna": str(fna),
        "n_ids": features.n,
        "feature_names": list(features.feature_names),
        "split_csv": str(split_csv),
        "assignment_csv": str(assign_path),
        "feature_table": str(feat_csv),
        "assign_meta": meta,
        "plot": plot_meta,
        "cluster_method": cluster_method,
    }
    (outdir / "gc_split_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
