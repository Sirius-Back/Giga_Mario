"""Assign IDs to train/test/val (+ fold) → split.csv."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from .common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv


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


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1; got {value}")


def run_split_predict(
    *,
    outdir: Path,
    type: str = "random",
    seed: int = 42,
    id_csv: Path | None = None,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    stratification_column: str = "strat1",
    intersect_csv: Path | None = None,
    fna: Path | None = None,
    gtf: Path | None = None,
    marked_fasta: Path | None = None,
    test_fraction: float = 0.10,
    val_fraction_of_rest: float = 0.10,
) -> Path:
    """
    Write `{outdir}/split.csv` with columns ID|train_test|fold.

    For type=random, FNA/GTF/marked_FASTA may be omitted.
    """
    _ = (fna, gtf, marked_fasta, intersect_csv)  # reserved for non-random strategies
    if type != "random":
        raise ValueError(f"split-predict type={type!r} not implemented yet (use random)")
    _validate_fraction("test_fraction", test_fraction)
    _validate_fraction("val_fraction_of_rest", val_fraction_of_rest)
    outdir = ensure_dir(Path(outdir))

    fold_map = _load_optional_table(fold_csv, min_cols=["ID", "fold"], label="fold.csv")
    strat_map = _load_optional_table(
        stratification_csv, min_cols=["ID", stratification_column], label="strat.csv"
    )

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

    rng = random.Random(seed)
    ids_shuffled = ids[:]
    rng.shuffle(ids_shuffled)

    n = len(ids_shuffled)
    n_test = max(1, int(round(n * test_fraction))) if n >= 3 else max(0, n // 3)
    rest = ids_shuffled[n_test:]
    n_val = max(1, int(round(len(rest) * val_fraction_of_rest))) if len(rest) >= 2 else 0
    test_ids = set(ids_shuffled[:n_test])
    val_ids = set(rest[:n_val])
    # If fold.csv provided, prefer its fold; else fold=0
    rows: list[dict[str, Any]] = []
    for i in ids_shuffled:
        if i in test_ids:
            split = "test"
        elif i in val_ids:
            split = "val"
        else:
            split = "train"
        fold = fold_map[i]["fold"] if i in fold_map else "0"
        rows.append({"ID": i, "train_test": split, "fold": fold})

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
    p.add_argument("--stratification-column", default="strat1")
    p.add_argument("--intersect", type=Path, default=None)
    p.add_argument("--fna", type=Path, default=None)
    p.add_argument("--gtf", type=Path, default=None)
    p.add_argument("--marked", type=Path, default=None)
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
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
