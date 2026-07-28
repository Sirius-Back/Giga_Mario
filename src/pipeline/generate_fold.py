"""Generate fold.csv from ID.csv + prepare_fold.csv rules via ``id_rule``.

prepare_fold.csv columns (pipe or semicolon):
  identificator|column|fold

Example row:
  GCF_000005845.2|genome|zsv

Meaning: resolve matching ``ID`` values with
``run_id_rule([identificator], id_csv, id_col_1=column, id_col_2="ID")``
and assign fold ``zsv`` to those IDs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .common import ID_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from .id_rule import run_id_rule

FOLD_OUT_COLUMNS = ID_CSV_COLUMNS + ["fold"]
ZSV_LABELS = frozenset(
    {
        "zsv",
        "zeroshotvalidation",
        "zero_shot",
        "zero-shot",
        "zero-shot-validation",
        "zeroshot",
        "zero_shot_validation",
    }
)


def normalize_fold_label(raw: str) -> str:
    s = raw.strip()
    key = s.lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in {"zsv", "zeroshotvalidation", "zeroshot"} or s.lower() in ZSV_LABELS:
        return "zsv"
    if s.lower() in {"validation", "val"}:
        return "val"
    if s.lower() in {"train", "test"}:
        return s.lower()
    return s


def is_zsv_fold(raw: str) -> bool:
    return normalize_fold_label(raw) == "zsv"


def _read_prepare_fold(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text.strip() else ""
    delim = ";" if first.count(";") >= first.count("|") else "|"
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        required = {"identificator", "column", "fold"}
        # allow slight header variants
        fieldmap = {((f or "").strip().lower()): (f or "").strip() for f in (reader.fieldnames or [])}
        aliases = {
            "identificator": ["identificator", "identifier", "id_value", "value"],
            "column": ["column", "col", "id_col"],
            "fold": ["fold", "split", "label"],
        }
        resolved: dict[str, str] = {}
        for canon, opts in aliases.items():
            for opt in opts:
                if opt in fieldmap:
                    resolved[canon] = fieldmap[opt]
                    break
        missing = [c for c in required if c not in resolved]
        if missing:
            raise ValueError(
                f"prepare_fold.csv missing columns {missing}; have {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            rows.append(
                {
                    "identificator": (row.get(resolved["identificator"]) or "").strip(),
                    "column": (row.get(resolved["column"]) or "").strip(),
                    "fold": (row.get(resolved["fold"]) or "").strip(),
                }
            )
    if not rows:
        raise ValueError(f"prepare_fold.csv has no rows: {path}")
    return rows


def run_generate_fold(
    id_csv: Path,
    prepare_fold_csv: Path,
    *,
    outdir: Path,
    default_fold: str = "0",
) -> Path:
    """Write ``{outdir}/fold.csv`` with ID.csv columns plus ``fold``.

    Fold assignment uses ``run_id_rule`` per prepare_fold rule
    (``id_col_1`` = rule column, ``id_col_2`` = ``ID``). Later rules override
    earlier ones; unmatched IDs keep ``default_fold``.
    """
    id_csv = Path(id_csv)
    id_rows = read_csv(id_csv)
    if not id_rows:
        raise ValueError(f"Empty ID.csv: {id_csv}")
    missing = [c for c in ID_CSV_COLUMNS if c not in id_rows[0]]
    if missing:
        raise ValueError(f"ID.csv missing {missing}")

    rules = _read_prepare_fold(Path(prepare_fold_csv))
    for rule in rules:
        col = rule["column"]
        if col not in id_rows[0]:
            raise ValueError(
                f"prepare_fold rule column {col!r} not in ID.csv columns {list(id_rows[0])}"
            )
        if not rule["identificator"]:
            raise ValueError("prepare_fold identificator must not be blank")
        if not rule["fold"]:
            raise ValueError("prepare_fold fold must not be blank")

    fold_by_id: dict[str, str] = {str(row["ID"]): default_fold for row in id_rows}
    # later rules override earlier ones
    for rule in rules:
        ids = run_id_rule(
            [rule["identificator"]],
            id_csv,
            id_col_1=rule["column"],
            id_col_2="ID",
        )
        label = normalize_fold_label(rule["fold"])
        for rid in ids:
            fold_by_id[str(rid)] = label

    out_rows: list[dict[str, Any]] = [
        {**row, "fold": fold_by_id[str(row["ID"])]} for row in id_rows
    ]

    outdir = ensure_dir(Path(outdir))
    out_path = outdir / "fold.csv"
    write_csv(out_path, out_rows, FOLD_OUT_COLUMNS)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="generate_fold: ID.csv + prepare_fold.csv → fold.csv")
    p.add_argument("--id-csv", required=True, type=Path)
    p.add_argument("--prepare-fold", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--default-fold", default="0")
    args = p.parse_args(argv)
    print(run_generate_fold(args.id_csv, args.prepare_fold, outdir=args.outdir, default_fold=args.default_fold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
