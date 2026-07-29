"""Assign IDs to train/test/val (+ fold) → split.csv.

Random assignment imports Caduceus-aligned fold ratios from
``src.splits.common`` (same helpers used by ``src.splits.random``).

Supported ``type`` values:
  - ``random`` — independent per-ID random / stratified assignment
  - ``gc`` — split-by-similarity (SBS) with delta-GC distance
    (requires ``marked_fasta`` or ``fna`` directory of per-ID FASTA)
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Any, Literal

# Import fold assignment from the random split implementation (re-exported helpers).
from src.splits.random import assign_folds_random, assign_folds_stratified

from .common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from .generate_fold import is_zsv_fold, normalize_fold_label

SUPPORTED_SPLIT_TYPES = ("random", "gc")


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
) -> Path:
    """
    Write `{outdir}/split.csv` with columns ID|train_test|fold.

    ``type=random`` — Caduceus-aligned random / stratified assignment.
    ``type=gc`` — SBS on GC% + AAA% feature table (``src.splits.sbs`` / ``gc``).

    When ``fold.csv`` is present:
      - folds labeled zsv / zeroshotvalidation → train_test=zsv (excluded from
        random / SBS assignment; materialize moves them to zero-shot-validation/)
      - other IDs get train/test/val; fold column keeps fold.csv value (random)
        or SBS cluster id (gc)

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
    p.add_argument("--type", default="random")
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
        help="Optional cap on sequences for SBS strategies (gc)",
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
        help="Optional column in --stratification for custom PCA coloring (gc)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write SBS PCA diagnostics (gc)",
    )
    args = p.parse_args(argv)
    n_clusters: int | Literal["auto"]
    if str(args.n_clusters).lower() == "auto":
        n_clusters = "auto"
    else:
        n_clusters = int(args.n_clusters)
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
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
