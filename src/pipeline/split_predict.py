"""Assign IDs to train/test/val (+ fold) → split.csv.

Random assignment imports Caduceus-aligned fold ratios from
``src.splits.common`` (same helpers used by ``src.splits.random``).
Only ``type=random`` is implemented.
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Any

# Import fold assignment from the random split implementation (re-exported helpers).
from src.splits.random import assign_folds_random, assign_folds_stratified

from .common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from .generate_fold import is_zsv_fold, normalize_fold_label


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
) -> Path:
    """
    Write `{outdir}/split.csv` with columns ID|train_test|fold.

    Only ``type=random`` is implemented (imports assignment from
    ``src.splits.common``, shared with ``src.splits.random``).

    When ``fold.csv`` is present:
      - folds labeled zsv / zeroshotvalidation → train_test=zsv (excluded from
        random assignment; materialize moves them to zero-shot-validation/)
      - other IDs get train/test/val via random/stratified import; fold column
        keeps the fold.csv value (or ``0``)

    When ``fold.csv`` is omitted, emits:
      ``Warning: folds are not included``
    """
    _ = (fna, gtf, marked_fasta, intersect_csv, stratification_column)
    if type != "random":
        raise ValueError(
            f"split-predict type={type!r} not implemented yet "
            "(only random is available; other strategies removed for now)"
        )
    outdir = ensure_dir(Path(outdir))

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

    rows: list[dict[str, Any]] = []
    for i in ids:
        if i in zsv_ids:
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
    args = p.parse_args(argv)
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
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
