"""K-mer split strategy — SBS with DSK k-mer composition features.

Caption: ``splits/kmer.md``. Wired into ``split-predict`` as ``type=kmer``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from src.splits.sbs.assign import (
    ClusterMethod,
    assign_from_features,
    assignment_rows_to_split_csv,
    write_assignment_table,
)
from src.splits.sbs.backends.kmer import KmerFeatureBackend, normalize_k_list
from src.splits.sbs.features import FeatureTable, compute_feature_table
from src.splits.sbs.fna_io import FastaMode
from src.splits.sbs.visualize import plot_sbs_pca_diagnostics

__all__ = (
    "SPLIT_ID",
    "run_kmer_split_assign",
    "compute_kmer_feature_table",
)

SPLIT_ID = "kmer"


def compute_kmer_feature_table(
    fna: Path,
    *,
    k: int | Sequence[int] = 5,
    mode: FastaMode = "auto",
    ids: list[str] | None = None,
    max_ids: int | None = None,
    normalize: str = "relative",
    log_transform: bool = False,
    engine: str = "auto",
    dsk_bin: str | Path | None = None,
    dsk2ascii_bin: str | Path | None = None,
    abundance_min: int = 1,
    threads: int = 1,
) -> FeatureTable:
    """C1 helper: FNA → FeatureTable (observed k-mer composition)."""
    backend = KmerFeatureBackend(
        k=k,
        normalize=normalize,
        log_transform=log_transform,
        engine=engine,  # type: ignore[arg-type]
        dsk_bin=dsk_bin,
        dsk2ascii_bin=dsk2ascii_bin,
        abundance_min=abundance_min,
        threads=threads,
    )
    return compute_feature_table(
        fna,
        backend,
        mode=mode,
        ids=ids,
        max_ids=max_ids,
    )


def run_kmer_split_assign(
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
    k: int | Sequence[int] = 5,
    kmer_length: int | Sequence[int] | None = None,
    n_clusters: int | Literal["auto"] = "auto",
    cluster_method: str | ClusterMethod = "dbscan",
    ratios: tuple[float, float, float] | None = None,
    plot: bool = True,
    plot_max_n: int | None = None,
    custom_label_csv: Path | None = None,
    custom_label_column: str | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    normalize: str = "relative",
    log_transform: bool = False,
    engine: str = "auto",
    dsk_bin: str | Path | None = None,
    dsk2ascii_bin: str | Path | None = None,
    abundance_min: int = 1,
    threads: int = 1,
    features_npz: Path | None = None,
) -> dict[str, Any]:
    """FNA → k-mer features → SBS assignment → ``split.csv`` (+ PCA diagnostics).

    If ``features_npz`` is set (or ``outdir/feature_table.npz`` exists and
    ``features_npz`` is the string/path reuse sentinel via existing file when
    passed explicitly), skip recomputing k-mer counts and assign from the dump.
    """
    from src.splits.sbs.assign import normalize_cluster_method
    from src.splits.sbs.visualize import DEFAULT_PLOT_N

    cluster_method = normalize_cluster_method(cluster_method)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    k_arg = kmer_length if kmer_length is not None else k
    k_list = normalize_k_list(k_arg)

    # Auto-resume: durable feature dump present and split.csv not yet written.
    if features_npz is None:
        auto_npz = outdir / "feature_table.npz"
        if auto_npz.is_file() and not (outdir / "split.csv").is_file():
            features_npz = auto_npz
            print(
                f"kmer: auto-resume features from {auto_npz} "
                "(split.csv missing)",
                flush=True,
            )

    npz_reuse = Path(features_npz) if features_npz is not None else None
    if npz_reuse is not None:
        print(f"kmer: reusing features from {npz_reuse}", flush=True)
        from src.pipeline.mem_guard import ensure_allocation_fits, wait_for_ram_headroom

        wait_for_ram_headroom(0.95, label="kmer_assign_reuse")
        features = FeatureTable.load_npz(npz_reuse, backend="kmer")
        ensure_allocation_fits(
            int(features.n) * int(features.n_features) * 4,
            safety=1.15,
            label="kmer_assign_matrix_resident",
        )
        feat_csv: Path | str = npz_reuse
    else:
        features = compute_kmer_feature_table(
            fna,
            k=k_list,
            mode=fna_mode,
            ids=ids,
            max_ids=max_ids,
            normalize=normalize,
            log_transform=log_transform,
            engine=engine,
            dsk_bin=dsk_bin,
            dsk2ascii_bin=dsk2ascii_bin,
            abundance_min=abundance_min,
            threads=threads,
        )
        # Large panels: skip dense CSV (n×d text OOMs); keep npz for audit + assign in-RAM.
        import numpy as np

        n_cells = int(features.n) * int(features.n_features)
        if n_cells >= 20_000_000:
            npz_path = outdir / "feature_table.npz"
            features.write_npz(npz_path)
            feat_csv = npz_path
            print(
                f"kmer: skipped feature_table.csv (n_cells={n_cells}); wrote {npz_path}",
                flush=True,
            )
        else:
            feat_csv = features.write_csv(outdir / "feature_table.csv")

    print(
        f"kmer: assign cluster_method={cluster_method} n={features.n} "
        f"d={features.n_features}",
        flush=True,
    )
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
        elbow_checkpoint_path=outdir / "elbow_inertias.json",
    )
    assign_path = write_assignment_table(rows, outdir / "sbs_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, outdir)
    (outdir / "stage_assign_done.json").write_text(
        json.dumps(
            {
                "assignment_csv": str(assign_path),
                "split_csv": str(split_csv),
                "cluster_method": cluster_method,
                "n_clusters": n_clusters,
                "assign_meta": {
                    "n_total": (meta or {}).get("n_total"),
                    "n_zsv": (meta or {}).get("n_zsv"),
                    "n_assignable": (meta or {}).get("n_assignable"),
                    "n_features": (meta or {}).get("n_features"),
                    "cluster": {
                        k: (meta.get("cluster") or {}).get(k)
                        for k in (
                            "method_used",
                            "n_clusters",
                            "k_info",
                            "standardize_inplace",
                            "method_requested",
                        )
                    },
                    "train_test_by_fold": (meta or {}).get("train_test_by_fold"),
                    "additional_embeds_features": (meta or {}).get(
                        "additional_embeds_features"
                    ),
                },
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_meta: dict[str, Any] | None = None
    if plot:
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
            max_points=int(plot_max_n) if plot_max_n else DEFAULT_PLOT_N,
        )

    summary = {
        "split_id": SPLIT_ID,
        "seed": seed,
        "fna": str(fna),
        "k": list(k_list),
        "n_ids": features.n,
        "n_features": len(features.feature_names),
        "split_csv": str(split_csv),
        "assignment_csv": str(assign_path),
        "feature_table": str(feat_csv),
        "assign_meta": meta,
        "plot": plot_meta,
        "cluster_method": cluster_method,
        "normalize": normalize,
        "log_transform": log_transform,
        "engine": (features.extras or {}).get("engine", engine),
    }
    if len(features.feature_names) <= 512:
        summary["feature_names"] = list(features.feature_names)
    (outdir / "kmer_split_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
