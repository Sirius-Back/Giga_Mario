"""Assign IDs to train/test/val (+ fold) → split.csv.

Random assignment imports Caduceus-aligned fold ratios from
``src.splits.common`` (same helpers used by ``src.splits.random``).

Supported ``type`` values:
  - ``random`` — independent per-ID random / stratified assignment
  - ``gc`` — split-by-similarity (SBS) with GC% + AAA% features
    (requires ``marked_fasta`` or ``fna`` directory of per-ID FASTA)
  - ``kmer`` — SBS with DSK k-mer composition features (same FNA inputs)
  - ``hashfrag`` — homology-aware orthogonal splits via hashFrag+BLAST
    (requires ``marked`` / ``fna`` and an explicit ``threshold``)
  - ``pangenome`` — C++ k-mer contingency / repeat-graph connected components
    (requires ``marked``; filters to PARSED IDs)
  - ``blastp`` — CDS protein BLASTP homology (adapt fna+gtf → filter ∩ PARSED
    → sparse BLASTP connected components)
  - ``mmseqs`` — MMseqs2 ``easy-cluster`` homology folds (MARKED; Locked
    ratios 60:20:20 train/val/test)
  - ``paralogs_only`` — orthogroup 1-rep train; remainder 50/50 test/val
    (Compara graph; unmapped → test/val only)
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

# Import fold assignment from the random split implementation (re-exported helpers).
from src.splits.random import assign_folds_random, assign_folds_stratified

from .common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from .generate_fold import is_zsv_fold, normalize_fold_label

SUPPORTED_SPLIT_TYPES = (
    "random",
    "gc",
    "kmer",
    "hashfrag",
    "pangenome",
    "blastp",
    "mmseqs",
    "paralogs_only",
    "loco",
)


def _load_optional_table(
    path: Path | None, *, min_cols: list[str], label: str
) -> dict[str, dict[str, str]]:
    """Load an optional pipe-delimited table and validate its ID key."""
    if path is None:
        return {}
    rows = read_csv(Path(path))
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = [c for c in min_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"{label} missing columns {missing}; have {list(rows[0])}")
    table: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        identifier = row["ID"].strip()
        if not identifier:
            raise ValueError(f"{label} has blank ID at row {row_number}")
        if identifier in table:
            raise ValueError(f"{label} has duplicate ID {identifier!r}")
        table[identifier] = row
    return table


def _load_ids(path: Path) -> list[str]:
    """Load a non-empty, duplicate-free list of IDs."""
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"id_csv is empty: {path}")
    if "ID" not in rows[0]:
        raise ValueError(f"id_csv missing column ['ID']; have {list(rows[0])}")
    ids: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        identifier = row["ID"].strip()
        if not identifier:
            raise ValueError(f"id_csv has blank ID at row {row_number}")
        if identifier in seen:
            raise ValueError(f"id_csv has duplicate ID {identifier!r}")
        seen.add(identifier)
        ids.append(identifier)
    return ids


def _strat_columns(sample_row: dict[str, str]) -> list[str]:
    """All stratification columns: every column except ID (and optional meta)."""
    skip = {"ID", "id"}
    cols = [c for c in sample_row if c not in skip]
    # Prefer explicit strat* columns when present; otherwise use all non-ID cols
    strat_star = [c for c in cols if c.lower().startswith("strat")]
    return strat_star if strat_star else cols


def _composite_stratum(row: dict[str, str], columns: list[str]) -> str:
    return "||".join(str(row.get(c, "")) for c in columns)


def _assign_random_train_test(
    ids: list[str],
    *,
    seed: int,
    strat_map: dict[str, dict[str, str]],
    ratios: tuple[float, float, float] | None,
) -> dict[str, str]:
    """
    Assign train/val/test using ``assign_folds_random`` / stratified helpers
    from ``src.splits.common`` (same ratios as ``src.splits.random``).
    """
    if not ids:
        return {}
    if len(ids) < 3:
        raise ValueError(f"need >=3 non-ZSV IDs for train/val/test; got {len(ids)}")

    rng = random.Random(seed)

    if strat_map:
        sample_row = next(iter(strat_map.values()))
        strat_cols = _strat_columns(sample_row)
        if not strat_cols:
            raise ValueError("stratification.csv has no stratification columns")
        missing_strat = [i for i in ids if i not in strat_map]
        if missing_strat:
            raise ValueError(
                f"stratification.csv missing ID {missing_strat[0]!r} "
                f"(required when --stratification is set)"
            )
        strata = [_composite_stratum(strat_map[i], strat_cols) for i in ids]
        labels = assign_folds_stratified(ids, strata, rng, ratios=ratios)
        return dict(zip(ids, labels))

    # Unstratified: shuffle index order, then zip with fixed-size fold labels
    # (matches src.splits.random M1 pairing).
    order = list(range(len(ids)))
    rng.shuffle(order)
    folds = assign_folds_random(len(ids), ratios=ratios)
    out: dict[str, str] = {}
    for idx, fold in zip(order, folds):
        out[ids[idx]] = fold
    return out


def _run_gc_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    stratification_csv: Path | None,
    fna: Path | None,
    marked_fasta: Path | None,
    ratios: tuple[float, float, float] | None,
    max_ids: int | None,
    n_clusters: int | Literal["auto"],
    plot: bool,
    cluster_method: str = "dbscan",
    custom_label_column: str | None = None,
) -> Path:
    """SBS GC path: MARKED/FNA dir → GC%/AAA% features → assignment → split.csv."""
    from src.splits.gc import run_gc_split_assign

    fna_root = marked_fasta or fna
    if fna_root is None:
        raise ValueError(
            "split-predict type=gc requires --marked (MARKED dir) or --fna"
        )
    ids = _load_ids(Path(id_csv)) if id_csv is not None else None
    summary = run_gc_split_assign(
        outdir=outdir,
        fna=Path(fna_root),
        id_csv=Path(id_csv) if id_csv else None,
        fold_csv=Path(fold_csv) if fold_csv else None,
        stratification_csv=Path(stratification_csv) if stratification_csv else None,
        seed=seed,
        max_ids=max_ids,
        ids=ids,
        ratios=ratios,
        n_clusters=n_clusters,
        cluster_method=cluster_method,  # type: ignore[arg-type]
        plot=plot,
        custom_label_csv=Path(stratification_csv) if (
            stratification_csv is not None and custom_label_column
        ) else None,
        custom_label_column=custom_label_column,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"gc split did not write split.csv: {out}")
    return out


def _normalize_kmer_engine(engine: str) -> str:
    """Map CLI/Hydra aliases onto ``KmerFeatureBackend`` engines."""
    e = str(engine).strip().lower()
    if e in {"cpp", "c++", "cxx"}:
        return "native"
    if e in {"auto", "native", "python", "dsk"}:
        return e
    raise ValueError(
        f"kmer engine must be auto|native|cpp|python|dsk; got {engine!r}"
    )


def _run_kmer_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    stratification_csv: Path | None,
    fna: Path | None,
    marked_fasta: Path | None,
    ratios: tuple[float, float, float] | None,
    max_ids: int | None,
    n_clusters: int | Literal["auto"],
    plot: bool,
    cluster_method: str = "dbscan",
    custom_label_column: str | None = None,
    kmer_size: Sequence[int] | int = 5,
    log_transform: bool = False,
    engine: str = "auto",
    threads: int = 1,
) -> Path:
    """SBS k-mer path: MARKED/FNA → k-mer features → assignment → split.csv."""
    from src.splits.kmer import run_kmer_split_assign

    fna_root = marked_fasta or fna
    if fna_root is None:
        raise ValueError(
            "split-predict type=kmer requires --marked (MARKED dir) or --fna"
        )
    ids = _load_ids(Path(id_csv)) if id_csv is not None else None
    summary = run_kmer_split_assign(
        outdir=outdir,
        fna=Path(fna_root),
        id_csv=Path(id_csv) if id_csv else None,
        fold_csv=Path(fold_csv) if fold_csv else None,
        stratification_csv=Path(stratification_csv) if stratification_csv else None,
        seed=seed,
        max_ids=max_ids,
        ids=ids,
        ratios=ratios,
        n_clusters=n_clusters,
        cluster_method=cluster_method,  # type: ignore[arg-type]
        plot=plot,
        custom_label_csv=Path(stratification_csv) if (
            stratification_csv is not None and custom_label_column
        ) else None,
        custom_label_column=custom_label_column,
        k=kmer_size,
        log_transform=log_transform,
        engine=_normalize_kmer_engine(engine),
        threads=threads,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"kmer split did not write split.csv: {out}")
    return out


def _run_hashfrag_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    fna: Path | None,
    marked_fasta: Path | None,
    threshold: float | None,
    max_ids: int | None,
    p_train: float | None,
    p_test: float | None,
    threads: int,
    force: bool,
) -> Path:
    """hashFrag path: MARKED → orthogonal homology splits → split.csv."""
    from src.splits.hashfrag import run_hashfrag_split_assign

    fna_root = marked_fasta or fna
    if fna_root is None:
        raise ValueError(
            "split-predict type=hashfrag requires --marked (MARKED dir) or --fna"
        )
    if threshold is None:
        raise ValueError(
            "split-predict type=hashfrag requires --threshold (hashFrag -t); "
            "do not invent a default"
        )
    if id_csv is None:
        raise ValueError("split-predict type=hashfrag requires --id-csv")
    summary = run_hashfrag_split_assign(
        outdir=outdir,
        marked=Path(fna_root),
        threshold=float(threshold),
        id_csv=Path(id_csv),
        fold_csv=Path(fold_csv) if fold_csv else None,
        seed=seed,
        max_ids=max_ids,
        p_train=p_train,
        p_test=p_test,
        threads=threads,
        force=force,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"hashfrag split did not write split.csv: {out}")
    return out


def _normalize_pangenome_k(kmer_size: Sequence[int] | int) -> int:
    """Pangenome contingency uses a single k (default 21)."""
    if isinstance(kmer_size, Sequence) and not isinstance(kmer_size, (str, bytes)):
        vals = [int(x) for x in kmer_size]
        if not vals:
            return 21
        if len(vals) > 1:
            raise ValueError(
                f"type=pangenome accepts a single --kmer-size; got {vals}"
            )
        return vals[0]
    return int(kmer_size) if kmer_size is not None else 21


def _run_pangenome_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    stratification_csv: Path | None,
    fna: Path | None,
    marked_fasta: Path | None,
    parsed: Path | None,
    ratios: tuple[float, float, float] | None,
    max_ids: int | None,
    plot: bool,
    kmer_size: Sequence[int] | int,
    genomes: Sequence[str] | None,
    min_shared: int,
    min_df: int = 2,
    cluster_method: str = "hash_majority",
    reuse_panel_marked: bool = False,
    marked_pangenome: Path | None = None,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    modularity_refine: bool = False,
    max_fold_size: int | None = None,
) -> Path:
    """Pangenome path: resolve MARKED_pangenome → MARKED_parsed → C++ CC → split.csv."""
    from src.splits.pangenome import run_pangenome_split_assign

    panel_marked = marked_fasta or fna
    k = _normalize_pangenome_k(kmer_size)
    # Default: if caller only passed --marked (panel), require explicit reuse flag
    # unless they also provided adapt window / marked_pangenome.
    from src.splits.pangenome import DEFAULT_MAX_FOLD_SIZE

    summary = run_pangenome_split_assign(
        outdir=outdir,
        parsed=Path(parsed) if parsed is not None else None,
        id_csv=Path(id_csv) if id_csv else None,
        fold_csv=Path(fold_csv) if fold_csv else None,
        stratification_csv=Path(stratification_csv) if stratification_csv else None,
        seed=seed,
        max_ids=max_ids,
        genomes=genomes,
        k=k,
        min_shared=int(min_shared),
        min_df=int(min_df),
        cluster_method=str(cluster_method),
        ratios=ratios,
        plot=plot,
        modularity_refine=bool(modularity_refine),
        max_fold_size=int(max_fold_size) if max_fold_size is not None else DEFAULT_MAX_FOLD_SIZE,
        marked_pangenome=Path(marked_pangenome) if marked_pangenome else None,
        panel_marked=Path(panel_marked) if panel_marked else None,
        reuse_panel_marked=reuse_panel_marked,
        gtf_dir=Path(gtf_dir) if gtf_dir else None,
        fna_dir=Path(fna_dir) if fna_dir else None,
        environment=environment,
        window=window,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"pangenome split did not write split.csv: {out}")
    return out


def _run_blastp_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    stratification_csv: Path | None,
    fna: Path | None,
    marked_fasta: Path | None,
    parsed: Path | None,
    ratios: tuple[float, float, float] | None,
    max_ids: int | None,
    genomes: Sequence[str] | None,
    reuse_panel_marked: bool = False,
    marked_blastp: Path | None = None,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    genetic_code: str = "universal",
    threads: int = 8,
    force: bool = False,
    evalue: float | None = None,
    max_target_seqs: int | None = None,
    min_bitscore: float | None = None,
) -> Path:
    """BLASTP path: MARKED_blastp → PARSED filter → CDS proteins → sparse BLASTP CC."""
    from src.splits.blastp import (
        DEFAULT_EVALUE,
        DEFAULT_MAX_TARGET_SEQS,
        DEFAULT_MIN_BITSCORE,
        run_blastp_split_assign,
    )

    panel_marked = marked_fasta or fna
    summary = run_blastp_split_assign(
        outdir=outdir,
        parsed=Path(parsed) if parsed is not None else None,
        id_csv=Path(id_csv) if id_csv else None,
        fold_csv=Path(fold_csv) if fold_csv else None,
        stratification_csv=Path(stratification_csv) if stratification_csv else None,
        seed=seed,
        max_ids=max_ids,
        genomes=genomes,
        ratios=ratios,
        genetic_code=genetic_code,
        threads=threads,
        evalue=float(evalue) if evalue is not None else DEFAULT_EVALUE,
        max_target_seqs=(
            int(max_target_seqs)
            if max_target_seqs is not None
            else DEFAULT_MAX_TARGET_SEQS
        ),
        min_bitscore=(
            float(min_bitscore) if min_bitscore is not None else DEFAULT_MIN_BITSCORE
        ),
        force=force,
        marked_blastp=Path(marked_blastp) if marked_blastp else None,
        panel_marked=Path(panel_marked) if panel_marked else None,
        reuse_panel_marked=reuse_panel_marked,
        gtf_dir=Path(gtf_dir) if gtf_dir else None,
        fna_dir=Path(fna_dir) if fna_dir else None,
        environment=environment,
        window=window,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"blastp split did not write split.csv: {out}")
    return out


def _run_mmseqs_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    stratification_csv: Path | None,
    fna: Path | None,
    marked_fasta: Path | None,
    ratios: tuple[float, float, float] | None,
    max_ids: int | None,
    plot: bool,
    custom_label_column: str | None,
    threads: int,
    force: bool,
    min_seq_id: float | None,
    mmseqs_sensitivity: float | None,
    mmseqs_bin: Path | None,
) -> Path:
    """MMseqs2 path: MARKED → easy-cluster → fold-grain 60:20:20 → split.csv."""
    from src.splits.mmseqs import (
        DEFAULT_MIN_SEQ_ID,
        DEFAULT_RATIOS,
        DEFAULT_SENSITIVITY,
        run_mmseqs_split_assign,
    )

    fna_root = marked_fasta or fna
    if fna_root is None:
        raise ValueError(
            "split-predict type=mmseqs requires --marked (MARKED dir) or --fna"
        )
    summary = run_mmseqs_split_assign(
        outdir=outdir,
        fna=Path(fna_root),
        id_csv=Path(id_csv) if id_csv else None,
        fold_csv=Path(fold_csv) if fold_csv else None,
        stratification_csv=Path(stratification_csv) if stratification_csv else None,
        seed=seed,
        max_ids=max_ids,
        ratios=ratios if ratios is not None else DEFAULT_RATIOS,
        plot=plot,
        custom_label_column=custom_label_column,
        mmseqs_bin=mmseqs_bin,
        threads=threads,
        sensitivity=(
            float(mmseqs_sensitivity)
            if mmseqs_sensitivity is not None
            else DEFAULT_SENSITIVITY
        ),
        min_seq_id=(
            float(min_seq_id) if min_seq_id is not None else DEFAULT_MIN_SEQ_ID
        ),
        force=force,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"mmseqs split did not write split.csv: {out}")
    return out



def _run_paralogs_only_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    homology_edges: Path | None,
    homology_nodes: Path | None,
    max_ids: int | None,
) -> Path:
    """Homology orthogroup-rep train / remainder test-val (C++ native)."""
    from src.splits.paralogs_only import run_paralogs_only_split_assign

    if id_csv is None:
        raise ValueError("type=paralogs_only requires --id-csv")
    summary = run_paralogs_only_split_assign(
        outdir=outdir,
        id_csv=Path(id_csv),
        homology_edges=Path(homology_edges) if homology_edges else None,
        homology_nodes=Path(homology_nodes) if homology_nodes else None,
        fold_csv=Path(fold_csv) if fold_csv else None,
        seed=seed,
        max_ids=max_ids,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"paralogs_only split did not write split.csv: {out}")
    return out


def _run_loco_split_predict(
    *,
    outdir: Path,
    seed: int,
    id_csv: Path | None,
    fold_csv: Path | None,
    ratios: tuple[float, float, float] | None,
    max_ids: int | None,
    plot: bool,
) -> Path:
    """Chromosome-grain LOCO split (metadata-only from ID.csv)."""
    from src.splits.loco import run_loco_split_assign

    if id_csv is None:
        raise ValueError("type=loco requires --id-csv")
    summary = run_loco_split_assign(
        outdir=outdir,
        id_csv=Path(id_csv),
        fold_csv=Path(fold_csv) if fold_csv else None,
        seed=seed,
        ratios=ratios,
        max_ids=max_ids,
        plot=plot,
    )
    out = Path(summary["split_csv"])
    if not out.is_file():
        raise FileNotFoundError(f"loco split did not write split.csv: {out}")
    return out


def run_split_predict(
    *,
    outdir: Path,
    type: str = "random",
    seed: int = 42,
    id_csv: Path | None = None,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    stratification_column: str | None = None,
    intersect_csv: Path | None = None,
    fna: Path | None = None,
    gtf: Path | None = None,
    marked_fasta: Path | None = None,
    ratios: tuple[float, float, float] | None = None,
    max_ids: int | None = None,
    n_clusters: int | Literal["auto"] = "auto",
    plot: bool = False,
    cluster_method: str = "dbscan",
    custom_label_column: str | None = None,
    threshold: float | None = None,
    p_train: float | None = None,
    p_test: float | None = None,
    threads: int = 2,
    force: bool = False,
    kmer_size: Sequence[int] | int = 5,
    log_transform: bool = False,
    engine: str = "auto",
    genomes: Sequence[str] | None = None,
    min_shared: int = 1,
    min_df: int = 2,
    pangenome_cluster_method: str = "hash_majority",
    parsed: Path | None = None,
    reuse_panel_marked: bool = False,
    marked_pangenome: Path | None = None,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    modularity_refine: bool = False,
    max_fold_size: int | None = None,
    genetic_code: str = "universal",
    marked_blastp: Path | None = None,
    evalue: float | None = None,
    max_target_seqs: int | None = None,
    min_bitscore: float | None = None,
    min_seq_id: float | None = None,
    mmseqs_sensitivity: float | None = None,
    mmseqs_bin: Path | None = None,
    homology_edges: Path | None = None,
    homology_nodes: Path | None = None,
) -> Path:
    """
    Write `{outdir}/split.csv` with columns ID|train_test|fold.

    ``type=random`` — Caduceus-aligned random / stratified assignment.
    ``type=gc`` — SBS on GC% + AAA% feature table (``src.splits.sbs`` / ``gc``).
    ``type=kmer`` — SBS on DSK k-mer composition features (``src.splits.kmer``).
    ``type=hashfrag`` — hashFrag+BLAST orthogonal homology splits (MARKED).
    ``type=pangenome`` — C++ k-mer contingency connected components (MARKED∩PARSED).
    ``type=blastp`` — CDS protein BLASTP connected components (adapt fna+gtf).
    ``type=mmseqs`` — MMseqs2 easy-cluster folds (MARKED; default ratios 60:20:20).
    ``type=paralogs_only`` — orthogroup 1-rep train; remainder 50/50 test/val
      (Compara homology graph; unmapped IDs → test/val only).
    ``type=loco`` — chromosome-grain split from ID.csv ``(genome, chr)``;
      fold→train/val/test stratified by normalized chromosome number.

    When ``fold.csv`` is present:
      - folds labeled zsv / zeroshotvalidation → train_test=zsv (excluded from
        random / SBS / hashfrag / pangenome / blastp / mmseqs / paralogs_only /
        loco assignment; materialize moves them to
        zero-shot-validation/)
      - other IDs get train/test/val; fold column keeps fold.csv value (random),
        SBS cluster id (gc/kmer), homologous-group id (hashfrag), contingency
        component id (pangenome), BLASTP component id (blastp), MMseqs
        cluster id (mmseqs), or ``genome|chr`` (loco)

    When ``fold.csv`` is omitted, emits:
      ``Warning: folds are not included``
    """
    _ = (gtf, intersect_csv)
    if type not in SUPPORTED_SPLIT_TYPES:
        raise ValueError(
            f"split-predict type={type!r} not implemented; "
            f"supported: {', '.join(SUPPORTED_SPLIT_TYPES)}"
        )
    outdir = ensure_dir(Path(outdir))

    if type == "gc":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        # Deprecated single-column flag still accepted as custom PCA label column
        custom_col = custom_label_column or stratification_column
        return _run_gc_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            stratification_csv=stratification_csv,
            fna=fna,
            marked_fasta=marked_fasta,
            ratios=ratios,
            max_ids=max_ids,
            n_clusters=n_clusters,
            plot=plot,
            cluster_method=cluster_method,
            custom_label_column=custom_col,
        )

    if type == "kmer":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        custom_col = custom_label_column or stratification_column
        return _run_kmer_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            stratification_csv=stratification_csv,
            fna=fna,
            marked_fasta=marked_fasta,
            ratios=ratios,
            max_ids=max_ids,
            n_clusters=n_clusters,
            plot=plot,
            cluster_method=cluster_method,
            custom_label_column=custom_col,
            kmer_size=kmer_size,
            log_transform=log_transform,
            engine=engine,
            threads=threads,
        )

    if type == "hashfrag":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        return _run_hashfrag_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            fna=fna,
            marked_fasta=marked_fasta,
            threshold=threshold,
            max_ids=max_ids,
            p_train=p_train,
            p_test=p_test,
            threads=threads,
            force=force,
        )

    if type == "pangenome":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        return _run_pangenome_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            stratification_csv=stratification_csv,
            fna=fna,
            marked_fasta=marked_fasta,
            parsed=parsed,
            ratios=ratios,
            max_ids=max_ids,
            plot=plot,
            kmer_size=kmer_size,
            genomes=genomes,
            min_shared=min_shared,
            min_df=min_df,
            cluster_method=pangenome_cluster_method,
            reuse_panel_marked=reuse_panel_marked,
            marked_pangenome=marked_pangenome,
            gtf_dir=gtf_dir,
            fna_dir=fna_dir,
            environment=environment,
            window=window,
            modularity_refine=modularity_refine,
            max_fold_size=max_fold_size,
        )

    if type == "blastp":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        return _run_blastp_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            stratification_csv=stratification_csv,
            fna=fna,
            marked_fasta=marked_fasta,
            parsed=parsed,
            ratios=ratios,
            max_ids=max_ids,
            genomes=genomes,
            reuse_panel_marked=reuse_panel_marked,
            marked_blastp=marked_blastp,
            gtf_dir=gtf_dir,
            fna_dir=fna_dir,
            environment=environment,
            window=window,
            genetic_code=genetic_code,
            threads=threads,
            force=force,
            evalue=evalue,
            max_target_seqs=max_target_seqs,
            min_bitscore=min_bitscore,
        )

    if type == "mmseqs":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        custom_col = custom_label_column or stratification_column
        return _run_mmseqs_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            stratification_csv=stratification_csv,
            fna=fna,
            marked_fasta=marked_fasta,
            ratios=ratios,
            max_ids=max_ids,
            plot=plot,
            custom_label_column=custom_col,
            threads=threads,
            force=force,
            min_seq_id=min_seq_id,
            mmseqs_sensitivity=mmseqs_sensitivity,
            mmseqs_bin=mmseqs_bin,
        )

    if type == "paralogs_only":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        return _run_paralogs_only_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            homology_edges=homology_edges,
            homology_nodes=homology_nodes,
            max_ids=max_ids,
        )

    if type == "loco":
        if fold_csv is None:
            warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)
        return _run_loco_split_predict(
            outdir=outdir,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            ratios=ratios,
            max_ids=max_ids,
            plot=plot,
        )

    fold_map = _load_optional_table(fold_csv, min_cols=["ID", "fold"], label="fold.csv")
    strat_map = _load_optional_table(
        stratification_csv, min_cols=["ID"], label="strat.csv"
    )
    if stratification_csv is not None and strat_map:
        # Ensure at least one strat column exists
        sample = next(iter(strat_map.values()))
        if not _strat_columns(sample):
            raise ValueError(
                "stratification.csv must include at least one non-ID column"
            )

    if fold_csv is None:
        warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)

    if id_csv is not None:
        ids = _load_ids(Path(id_csv))
    elif fold_map:
        ids = list(fold_map.keys())
    elif strat_map:
        ids = list(strat_map.keys())
    else:
        raise ValueError("Provide --id-csv, --fold, or --stratification")

    id_set = set(ids)
    for label, table in (("fold.csv", fold_map), ("strat.csv", strat_map)):
        unknown_ids = set(table) - id_set
        if unknown_ids:
            example = sorted(unknown_ids)[0]
            raise ValueError(f"{label} contains ID absent from split IDs: {example!r}")

    zsv_ids: list[str] = []
    assignable: list[str] = []
    fold_values: dict[str, str] = {}
    for i in ids:
        raw_fold = fold_map[i]["fold"] if i in fold_map else "0"
        fold_values[i] = normalize_fold_label(raw_fold)
        if is_zsv_fold(fold_values[i]):
            zsv_ids.append(i)
            fold_values[i] = "zsv"
        else:
            assignable.append(i)

    train_test_map = _assign_random_train_test(
        assignable, seed=seed, strat_map=strat_map, ratios=ratios
    )

    zsv_set = set(zsv_ids)
    rows: list[dict[str, Any]] = []
    for i in ids:
        if i in zsv_set:
            rows.append({"ID": i, "train_test": "zsv", "fold": "zsv"})
        else:
            rows.append(
                {
                    "ID": i,
                    "train_test": train_test_map[i],
                    "fold": fold_values[i],
                }
            )

    out = outdir / "split.csv"
    write_csv(out, rows, SPLIT_CSV_COLUMNS)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="split-predict → split.csv")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument(
        "--type",
        default="random",
        choices=list(SUPPORTED_SPLIT_TYPES),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id-csv", type=Path, default=None)
    p.add_argument("--fold", "--fold-csv", dest="fold", type=Path, default=None)
    p.add_argument(
        "--stratification",
        "--stratification-csv",
        dest="stratification",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--stratification-column",
        default=None,
        help="Deprecated: all stratification columns are used",
    )
    p.add_argument("--intersect", type=Path, default=None)
    p.add_argument("--fna", type=Path, default=None)
    p.add_argument("--gtf", type=Path, default=None)
    p.add_argument("--marked", type=Path, default=None)
    p.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "TEST", "VAL"),
        default=None,
        help="Optional train:test:val split weights; default preserves Caduceus ratios",
    )
    p.add_argument(
        "--max-ids",
        type=int,
        default=None,
        help="Optional cap on sequences for SBS / hashfrag strategies",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="hashFrag homology score threshold (-t); required for type=hashfrag",
    )
    p.add_argument(
        "--p-train",
        type=float,
        default=None,
        help="hashFrag train-pool proportion (with --p-test; must sum to 1)",
    )
    p.add_argument(
        "--p-test",
        type=float,
        default=None,
        help="hashFrag test proportion (with --p-train; must sum to 1)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=2,
        help="BLAST threads for type=hashfrag (default: 2)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to hashFrag (overwrite existing BLAST outputs)",
    )
    p.add_argument(
        "--n-clusters",
        default="auto",
        help="SBS cluster count or 'auto' (gc; ignored by dbscan)",
    )
    p.add_argument(
        "--cluster-method",
        default="dbscan",
        choices=[
            "dbscan",
            "kmeans",
            "kmeans_elbow",
            "hierarchical",
            "pca_kmeans",
            "auto",
        ],
        help="SBS clustering on feature table (default: dbscan)",
    )
    p.add_argument(
        "--custom-label-column",
        default=None,
        help="Optional column in --stratification for custom PCA coloring (gc/kmer)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write SBS PCA (gc/kmer) or contingency graph (pangenome) figures",
    )
    p.add_argument(
        "--kmer-size",
        "--kmer-length",
        dest="kmer_size",
        type=int,
        nargs="+",
        default=None,
        help="K-mer size(s) for type=kmer (default: 5), or single k for "
        "type=pangenome (default: 21)",
    )
    p.add_argument(
        "--log-transform",
        action="store_true",
        help="Apply log1p to k-mer features after normalization (type=kmer)",
    )
    p.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "native", "cpp", "python", "dsk"],
        help="K-mer counting engine for type=kmer (cpp ≡ native C++)",
    )
    p.add_argument(
        "--parsed",
        type=Path,
        default=None,
        help="PARSED dir for type=pangenome|blastp filter (default: <marked>/../PARSED)",
    )
    p.add_argument(
        "--genome",
        dest="genomes",
        action="append",
        default=None,
        help="Restrict type=pangenome/blastp to genome id(s) from ID.csv (repeatable)",
    )
    p.add_argument(
        "--min-shared",
        type=int,
        default=1,
        help="Min shared k-mers to emit a contingency edge (type=pangenome; legacy)",
    )
    p.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Min document frequency for repeat k-mer hashes (type=pangenome hash-majority)",
    )
    p.add_argument(
        "--pangenome-cluster-method",
        type=str,
        default="hash_majority",
        choices=["hash_majority", "region_contingency"],
        help="Pangenome clustering: hash_majority (default) or legacy region_contingency",
    )
    p.add_argument(
        "--marked-pangenome",
        type=Path,
        default=None,
        help="Existing MARKED_pangenome dir (pangenome window; preferred over panel MARKED)",
    )
    p.add_argument(
        "--marked-blastp",
        type=Path,
        default=None,
        help="Existing MARKED_blastp dir (blastp adapt window; preferred over panel MARKED)",
    )
    p.add_argument(
        "--reuse-panel-marked",
        action="store_true",
        help="Opt-in: reuse --marked as MARKED_pangenome/MARKED_blastp only when windows match",
    )
    p.add_argument(
        "--gtf-dir",
        type=Path,
        default=None,
        help="Raw GTF dir for A2A adapt (type=pangenome|blastp)",
    )
    p.add_argument(
        "--fna-dir",
        type=Path,
        default=None,
        help="Raw FNA dir for A2A adapt / CDS translation (type=pangenome|blastp)",
    )
    p.add_argument(
        "--environment",
        default=None,
        choices=["gene", "random"],
        help="adapt environment for A2A MARKED (type=pangenome|blastp)",
    )
    p.add_argument(
        "--window",
        default=None,
        help='adapt window JSON for A2A MARKED, e.g. \'{"pos1":0,"pos2":0}\'',
    )
    p.add_argument(
        "--genetic-code",
        default="universal",
        help="Translation table for type=blastp (default: universal / NCBI 1)",
    )
    p.add_argument(
        "--evalue",
        type=float,
        default=None,
        help="BLASTP e-value cutoff for type=blastp (default: 1e-5)",
    )
    p.add_argument(
        "--max-target-seqs",
        type=int,
        default=None,
        help="BLASTP -max_target_seqs for type=blastp (default: 50)",
    )
    p.add_argument(
        "--min-bitscore",
        type=float,
        default=None,
        help="Min BLASTP bitscore to keep an edge (type=blastp; default: 50)",
    )
    p.add_argument(
        "--min-seq-id",
        type=float,
        default=None,
        help="MMseqs --min-seq-id for type=mmseqs (default: 0.8)",
    )
    p.add_argument(
        "--mmseqs-sensitivity",
        type=float,
        default=None,
        help="MMseqs -s sensitivity for type=mmseqs (default: 7.5)",
    )
    p.add_argument(
        "--mmseqs-bin",
        type=Path,
        default=None,
        help="Explicit path to mmseqs binary (type=mmseqs)",
    )
    p.add_argument(
        "--homology-edges",
        type=Path,
        default=None,
        help="Compara ortholog/paralog edge TSV[.gz] (type=paralogs_only)",
    )
    p.add_argument(
        "--homology-nodes",
        type=Path,
        default=None,
        help="Ensembl→marked_id node map TSV (type=paralogs_only)",
    )
    args = p.parse_args(argv)
    n_clusters: int | Literal["auto"]
    if str(args.n_clusters).lower() == "auto":
        n_clusters = "auto"
    else:
        n_clusters = int(args.n_clusters)
    kmer_size: Sequence[int] | int
    if args.type == "pangenome":
        kmer_size = 21
    else:
        kmer_size = 5
    if args.kmer_size is not None:
        kmer_size = tuple(int(x) for x in args.kmer_size)
        if len(kmer_size) == 1:
            kmer_size = kmer_size[0]
    window: dict[str, int] | None = None
    if args.window is not None:
        import json as _json

        decoded = _json.loads(args.window)
        if not isinstance(decoded, dict):
            p.error("--window must decode to a JSON object")
        window = {str(k): int(v) for k, v in decoded.items()}
    path = run_split_predict(
        outdir=args.outdir,
        type=args.type,
        seed=args.seed,
        id_csv=args.id_csv,
        fold_csv=args.fold,
        stratification_csv=args.stratification,
        stratification_column=args.stratification_column,
        intersect_csv=args.intersect,
        fna=args.fna,
        gtf=args.gtf,
        marked_fasta=args.marked,
        ratios=tuple(args.ratios) if args.ratios is not None else None,
        max_ids=args.max_ids,
        n_clusters=n_clusters,
        plot=bool(args.plot),
        cluster_method=str(args.cluster_method),
        custom_label_column=args.custom_label_column,
        threshold=args.threshold,
        p_train=args.p_train,
        p_test=args.p_test,
        threads=int(args.threads),
        force=bool(args.force),
        kmer_size=kmer_size,
        log_transform=bool(args.log_transform),
        engine=str(args.engine),
        genomes=list(args.genomes) if args.genomes else None,
        min_shared=int(args.min_shared),
        min_df=int(args.min_df),
        pangenome_cluster_method=str(args.pangenome_cluster_method),
        parsed=args.parsed,
        reuse_panel_marked=bool(args.reuse_panel_marked),
        marked_pangenome=args.marked_pangenome,
        gtf_dir=args.gtf_dir,
        fna_dir=args.fna_dir,
        environment=args.environment,
        window=window,
        genetic_code=str(args.genetic_code),
        marked_blastp=args.marked_blastp,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        min_bitscore=args.min_bitscore,
        min_seq_id=args.min_seq_id,
        mmseqs_sensitivity=args.mmseqs_sensitivity,
        mmseqs_bin=args.mmseqs_bin,
        homology_edges=args.homology_edges,
        homology_nodes=args.homology_nodes,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
