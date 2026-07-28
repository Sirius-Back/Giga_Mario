"""Materialize SPLIT/FASTA and SPLIT/PREDICT trees from split.csv."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import ensure_dir, read_csv, sanitize_filename, write_csv
from src.splits.common import link_or_copy


def _bucket(train_test: str, fold: str, strategy: str) -> list[str]:
    tt = train_test.strip().lower()
    if tt == "validation":
        tt = "val"
    if tt not in {"train", "test", "val"}:
        raise ValueError(
            "split.csv train_test must be train, test, val, or validation; "
            f"got {train_test!r}"
        )
    if strategy == "traintest":
        if tt == "val":
            tt = "train"  # merge val into train for train/test-only
        return [tt.upper()]
    if strategy == "traintestval":
        return [tt.upper()]
    if strategy == "fold":
        if not fold.strip():
            raise ValueError("split.csv fold must not be blank for strategy='fold'")
        return [f"FOLD{sanitize_filename(fold)}"]
    raise ValueError(f"Unknown strategy {strategy!r}")


def _resolve_artifact_dir(path: Path, name: str) -> Path:
    """Accept either an artifact directory or its parent stage output."""
    path = Path(path)
    resolved = path / name if (path / name).is_dir() else path
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Expected {name}/ directory at {path / name} or {path}; "
            f"neither exists"
        )
    return resolved


def _load_split_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"split.csv is empty: {path}")
    required = ("ID", "train_test", "fold")
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(f"split.csv missing columns {missing}; have {list(rows[0])}")

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        rid = row["ID"].strip()
        if not rid:
            raise ValueError(f"split.csv has blank ID at row {row_number}")
        if rid in seen:
            raise ValueError(f"split.csv has duplicate ID {rid!r}")
        seen.add(rid)
        row["ID"] = rid
    return rows


def _load_predict_rows(predict_root: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    path = predict_root / "predict.csv"
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"predict.csv is empty: {path}")
    if "id" not in rows[0]:
        raise ValueError(f"predict.csv missing column 'id'; have {list(rows[0])}")

    by_id: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        rid = row["id"].strip()
        if not rid:
            raise ValueError(f"predict.csv has blank id at row {row_number}")
        if rid in by_id:
            raise ValueError(f"predict.csv has duplicate id {rid!r}")
        row["id"] = rid
        by_id[rid] = row
    return by_id, list(rows[0])


def run_split(
    split_csv: Path,
    parsed_target: Path,
    parsed_data: Path,
    *,
    outdir: Path,
    strategy: str = "traintestval",
) -> Path:
    """
    Copy PARSED + PREDICT files into:
      SPLIT/PREDICT/{TRAIN|TEST|VAL|FOLD{X}}/
      SPLIT/FASTA/{TRAIN|TEST|VAL|FOLD{X}}/
    """
    if strategy not in {"traintest", "traintestval", "fold"}:
        raise ValueError("strategy must be traintest|traintestval|fold")

    rows = _load_split_rows(Path(split_csv))
    predict_root = _resolve_artifact_dir(Path(parsed_target), "PREDICT")
    data_root = _resolve_artifact_dir(Path(parsed_data), "PARSED")
    predict_table, predict_fields = _load_predict_rows(predict_root)

    bucket_rows: dict[str, list[dict[str, str]]] = {}
    planned_files: list[tuple[Path, Path, Path, Path]] = []
    sanitized_ids: dict[str, str] = {}
    for row in rows:
        rid = row["ID"]
        safe_id = sanitize_filename(rid)
        other_id = sanitized_ids.setdefault(safe_id, rid)
        if other_id != rid:
            raise ValueError(
                f"split.csv IDs {other_id!r} and {rid!r} map to the same filename "
                f"{safe_id!r}"
            )
        src_ext = predict_root / f"{safe_id}.ext"
        src_fa = data_root / f"{safe_id}.ext"
        missing: list[str] = []
        if rid not in predict_table:
            missing.append("predict.csv")
        if not src_ext.is_file():
            missing.append(f"PREDICT/{safe_id}.ext")
        if not src_fa.is_file():
            missing.append(f"PARSED/{safe_id}.ext")
        if missing:
            raise FileNotFoundError(
                f"ID {rid!r} from split.csv is missing required artifacts: "
                + ", ".join(missing)
            )
        for bucket in _bucket(row["train_test"], str(row["fold"]), strategy):
            bucket_rows.setdefault(bucket, []).append(predict_table[rid])
            planned_files.append(
                (
                    src_ext,
                    src_fa,
                    Path("PREDICT") / bucket / f"{safe_id}.ext",
                    Path("FASTA") / bucket / f"{safe_id}.ext",
                )
            )

    outdir = ensure_dir(Path(outdir))
    split_root = outdir / "SPLIT"
    if split_root.exists() or split_root.is_symlink():
        if split_root.is_symlink() or split_root.is_file():
            split_root.unlink()
        else:
            shutil.rmtree(split_root)
    split_root.mkdir()
    for src_ext, src_fa, pred_relative, fasta_relative in planned_files:
        link_or_copy(src_ext, split_root / pred_relative)
        link_or_copy(src_fa, split_root / fasta_relative)
    for bucket, brows in bucket_rows.items():
        write_csv(split_root / "PREDICT" / bucket / "predict.csv", brows, predict_fields)

    return split_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="split materialize → SPLIT/")
    p.add_argument("--split-csv", required=True, type=Path)
    p.add_argument("--parsed-target", required=True, type=Path)
    p.add_argument("--parsed-data", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--strategy", default="traintestval", choices=["traintest", "traintestval", "fold"])
    args = p.parse_args(argv)
    print(run_split(args.split_csv, args.parsed_target, args.parsed_data, outdir=args.outdir, strategy=args.strategy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
