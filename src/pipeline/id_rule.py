"""Filter / remap ID lists using ID.csv column rules."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from .common import read_csv


def _id_csv_columns(id_csv: Path) -> list[str]:
    """Return the pipe-delimited ID.csv header, including for header-only files."""
    with id_csv.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh, delimiter="|"), None)
    if header is None:
        raise ValueError(f"ID.csv is empty: {id_csv}")
    return header


def run_id_rule(
    id_list: Sequence[str | int],
    id_csv: Path,
    *,
    id_col_1: str,
    id_col_2: str,
) -> list[str]:
    """
    Map values from `id_col_1` → `id_col_2` for IDs present in `id_list`.

    `id_list` is matched against column `id_col_1` (string compare).
    Output is the corresponding `id_col_2` values (stable order of input hits).
    If an input key occurs in multiple ID.csv rows, every corresponding value is
    returned in ID.csv order.
    """
    id_csv = Path(id_csv)
    columns = _id_csv_columns(id_csv)
    missing = [col for col in (id_col_1, id_col_2) if col not in columns]
    if missing:
        raise ValueError(
            f"ID.csv missing columns {missing}; have {columns}"
        )

    index: dict[str, list[str]] = {}
    for row in read_csv(id_csv):
        index.setdefault(str(row[id_col_1]), []).append(str(row[id_col_2]))

    out: list[str] = []
    for item in id_list:
        out.extend(index.get(str(item), []))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Apply ID.csv column mapping to an ID list")
    ids_group = p.add_mutually_exclusive_group(required=True)
    ids_group.add_argument("--ids", help="Comma-separated ID list matched to id_col_1")
    ids_group.add_argument(
        "--ids-file",
        type=Path,
        help="One ID per line, matched to id_col_1",
    )
    p.add_argument("--id-csv", required=True, type=Path)
    p.add_argument("--id-col-1", required=True)
    p.add_argument("--id-col-2", required=True)
    args = p.parse_args(argv)
    if args.ids is not None:
        ids = [x for x in args.ids.split(",") if x != ""]
    else:
        assert args.ids_file is not None
        with args.ids_file.open(encoding="utf-8") as fh:
            ids = [line.strip() for line in fh if line.strip()]
    mapped = run_id_rule(ids, args.id_csv, id_col_1=args.id_col_1, id_col_2=args.id_col_2)
    print("\n".join(mapped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
