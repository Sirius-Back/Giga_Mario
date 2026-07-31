"""MMseqs2 homology split — cluster-first SBS strategy.

Caption: ``splits/mmseqs.md``. Wired into ``split-predict`` as ``type=mmseqs``.

Flow:
  MARKED/ → multifasta → ``mmseqs easy-cluster`` → cluster = fold →
  fold-grain train/test/val at Locked ``ratios=(0.6, 0.2, 0.2)`` → ``split.csv``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline.common import read_csv
from src.splits.sbs.assign import (
    assign_from_features,
    assignment_rows_to_split_csv,
    write_assignment_table,
)
from src.splits.sbs.backends.mmseqs import (
    DEFAULT_MIN_SEQ_ID,
    DEFAULT_SENSITIVITY,
    cluster_map_to_dense_ids,
    find_mmseqs,
    parse_cluster_tsv,
    run_mmseqs_easy_cluster,
    write_multifasta,
)
from src.splits.sbs.features import FeatureTable
from src.splits.sbs.fna_io import FastaMode, load_fna_sequences
from src.splits.sbs.visualize import plot_sbs_pca_diagnostics

__all__ = (
    "SPLIT_ID",
    "DEFAULT_RATIOS",
    "DEFAULT_MIN_SEQ_ID",
    "DEFAULT_SENSITIVITY",
    "run_mmseqs_split_assign",
)

SPLIT_ID = "mmseqs"
# Locked: train/val/test = 60:20:20 → API order train:test:val
DEFAULT_RATIOS: tuple[float, float, float] = (0.6, 0.2, 0.2)


def _resolve_ids(
    *,
    fna: Path,
    id_csv: Path | None,
    ids: list[str] | None,
    max_ids: int | None,
    seed: int,
) -> list[str]:
    if ids is not None:
        selected = list(ids)
    elif id_csv is not None:
        rows = read_csv(Path(id_csv))
        if not rows or "ID" not in rows[0]:
            raise ValueError(f"id_csv must have ID column: {id_csv}")
        selected = [r["ID"].strip() for r in rows if r.get("ID", "").strip()]
    else:
        from src.splits.sbs.fna_io import iter_fasta_paths

        selected = [p.stem for p in iter_fasta_paths(Path(fna))]
    if not selected:
        raise ValueError("no IDs to cluster")
    if max_ids is not None and len(selected) > int(max_ids):
        import random

        rng = random.Random(int(seed))
        sampled = list(selected)
        rng.shuffle(sampled)
        selected = sorted(sampled[: int(max_ids)], key=lambda x: (len(x), x))
    return selected


def run_mmseqs_split_assign(
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
    ratios: tuple[float, float, float] | None = None,
    plot: bool = False,
    plot_max_n: int | None = None,
    custom_label_csv: Path | None = None,
    custom_label_column: str | None = None,
    mmseqs_bin: str | Path | None = None,
    threads: int = 8,
    sensitivity: float = DEFAULT_SENSITIVITY,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    force: bool = False,
) -> dict[str, Any]:
    """MARKED/FNA → easy-cluster → SBS assign → ``split.csv``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fna = Path(fna)
    if not fna.exists():
        raise FileNotFoundError(f"FNA / MARKED path missing: {fna}")

    mmseqs_path = find_mmseqs(mmseqs_bin)
    use_ratios = DEFAULT_RATIOS if ratios is None else ratios

    selected = _resolve_ids(
        fna=fna,
        id_csv=id_csv,
        ids=ids,
        max_ids=max_ids,
        seed=seed,
    )
    sequences = load_fna_sequences(
        fna, mode=fna_mode, ids=selected, max_ids=None
    )
    missing = [rid for rid in selected if rid not in sequences]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} IDs absent from FNA/MARKED (e.g. {missing[0]!r})"
        )

    work = outdir / "mmseqs_work"
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "all.fa"
    if force or not fasta.is_file() or fasta.stat().st_size == 0:
        write_multifasta(sequences, fasta)

    cluster_tsv = run_mmseqs_easy_cluster(
        fasta,
        work=work,
        mmseqs_bin=mmseqs_path,
        threads=threads,
        sensitivity=sensitivity,
        min_seq_id=min_seq_id,
        force=force,
    )
    member_to_rep = parse_cluster_tsv(cluster_tsv)
    dense = cluster_map_to_dense_ids(member_to_rep, ids=selected)

    # Minimal FeatureTable for SBS C2 + optional PCA (cluster id as feature).
    matrix = np.asarray(
        [[float(dense[rid])] for rid in selected], dtype=np.float32
    )
    features = FeatureTable(
        ids=tuple(selected),
        feature_names=("mmseqs_cluster",),
        matrix=matrix,
        backend=SPLIT_ID,
        extras={
            "mmseqs_bin": str(mmseqs_path),
            "min_seq_id": float(min_seq_id),
            "sensitivity": float(sensitivity),
            "cluster_tsv": str(cluster_tsv),
        },
    )
    feat_csv = features.write_csv(outdir / "feature_table.csv")

    rows, meta = assign_from_features(
        features,
        fold_csv=fold_csv,
        stratification_csv=stratification_csv,
        seed=seed,
        ratios=use_ratios,
        precomputed_clusters=dense,
    )
    assign_path = write_assignment_table(rows, outdir / "sbs_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, outdir)

    plot_meta: dict[str, Any] | None = None
    if plot:
        from src.splits.sbs.visualize import DEFAULT_PLOT_N

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

    summary: dict[str, Any] = {
        "split_id": SPLIT_ID,
        "seed": seed,
        "fna": str(fna),
        "n_ids": len(selected),
        "n_clusters": len(set(dense.values())),
        "ratios": list(use_ratios),
        "min_seq_id": float(min_seq_id),
        "sensitivity": float(sensitivity),
        "threads": int(threads),
        "mmseqs_bin": str(mmseqs_path),
        "cluster_tsv": str(cluster_tsv),
        "split_csv": str(split_csv),
        "assignment_csv": str(assign_path),
        "feature_table": str(feat_csv),
        "assign_meta": meta,
        "plot": plot_meta,
    }
    (outdir / "mmseqs_split_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
