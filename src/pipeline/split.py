"""Materialize SPLIT/FASTA and SPLIT/PREDICT trees from split.csv.

Supports merged ``PREDICT/{ID}.ext`` and mapped ``PREDICT/{sample_id}/{ID}.ext``
layouts. Mapped panels are flattened to composite unique ``id`` values
(``{sample}__{region}``) so SPLIT/PREDICT, SPLIT/FASTA, and ``predict.csv`` share
the same unique ID column. Optional ``intersect_allow`` skips IDs missing
artifacts (default True).
"""
from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

from src.splits.common import link_or_copy

from .common import (
    checkout_ids_before_split,
    ensure_dir,
    index_unique_predict_rows,
    make_mapped_unique_id,
    sanitize_filename,
    write_csv,
)
from .generate_fold import is_zsv_fold


def _bucket(train_test: str, fold: str, strategy: str) -> list[str]:
    tt = train_test.strip().lower()
    if tt in {"validation", "val"}:
        tt = "val"
    if is_zsv_fold(tt) or is_zsv_fold(fold):
        return ["zero-shot-validation"]
    if tt not in {"train", "test", "val"}:
        raise ValueError(
            "split.csv train_test must be train, test, val, validation, or zsv; "
            f"got {train_test!r}"
        )
    if strategy == "traintest":
        if tt == "val":
            tt = "train"
        return [tt.upper()]
    if strategy == "traintestval":
        return [tt.upper()]
    if strategy == "fold":
        if not fold.strip():
            raise ValueError("split.csv fold must not be blank for strategy='fold'")
        return [f"FOLD{sanitize_filename(fold)}"]
    raise ValueError(f"Unknown strategy {strategy!r}")


def _resolve_artifact_dir(path: Path, name: str) -> Path:
    path = Path(path)
    resolved = path / name if (path / name).is_dir() else path
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Expected {name}/ directory at {path / name} or {path}; "
            f"neither exists"
        )
    return resolved


def _load_split_rows(path: Path) -> list[dict[str, str]]:
    from .common import read_csv

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


def _predict_src_path(
    predict_root: Path, safe_region: str, *, mapped: bool, sample_id: str
) -> Path:
    if mapped:
        if not sample_id:
            raise ValueError(f"Mapped PREDICT row for {safe_region!r} has blank sample_id")
        return predict_root / sanitize_filename(sample_id) / f"{safe_region}.ext"
    return predict_root / f"{safe_region}.ext"


def _out_predict_row(
    prow: dict[str, str], *, mapped: bool, unique_id: str, region_id: str
) -> dict[str, str]:
    """Build a predict.csv row whose ``id`` is unique for training."""
    out = dict(prow)
    out["id"] = unique_id
    if mapped:
        out["sample_id"] = prow["sample_id"]
        out["region_id"] = region_id
    return out


def run_split(
    split_csv: Path,
    parsed_target: Path,
    parsed_data: Path,
    *,
    outdir: Path,
    strategy: str = "traintestval",
    intersect_allow: bool = True,
    sample_id: str | None = None,
    id_csv: Path | None = None,
) -> Path:
    """
    Copy PARSED + PREDICT files into:
      SPLIT/PREDICT/{TRAIN|TEST|VAL|FOLD{X}}/{unique_id}.ext
      SPLIT/FASTA/{TRAIN|TEST|VAL|FOLD{X}}/{unique_id}.ext

    For merged panels ``unique_id`` is the region ID from ID.csv / split.csv.
    For mapped panels ``unique_id`` is ``{sample_id}__{region_id}`` so the
    ``predict.csv`` ``id`` column is unique and matches both artifact trees.

    Zero-shot-validation rows go to:
      {outdir}/PREDICT/zero-shot-validation/{unique_id}.ext
      {outdir}/PARSED/zero-shot-validation/{unique_id}.ext
    """
    if strategy not in {"traintest", "traintestval", "fold"}:
        raise ValueError("strategy must be traintest|traintestval|fold")

    rows = _load_split_rows(Path(split_csv))
    predict_root = _resolve_artifact_dir(Path(parsed_target), "PREDICT")
    data_root = _resolve_artifact_dir(Path(parsed_data), "PARSED")
    split_ids = [r["ID"] for r in rows]

    mapped, predict_rows = checkout_ids_before_split(
        predict_root=predict_root,
        parsed_root=data_root,
        split_ids=split_ids,
        sample_id=sample_id,
        id_csv=id_csv,
        intersect_allow=intersect_allow,
    )

    by_region: dict[str, list[dict[str, str]]] = {}
    for prow in predict_rows:
        by_region.setdefault(prow["id"], []).append(prow)

    # Output predict.csv fields: unique id (+ sample/region provenance when mapped)
    if mapped:
        predict_fields = ["id", "sample_id", "region_id"]
        for key in predict_rows[0]:
            if key not in predict_fields and key != "id":
                predict_fields.append(key)
    else:
        predict_fields = list(predict_rows[0].keys())

    outdir = ensure_dir(Path(outdir))
    split_root = outdir / "SPLIT"
    if split_root.exists() or split_root.is_symlink():
        if split_root.is_symlink() or split_root.is_file():
            split_root.unlink()
        else:
            shutil.rmtree(split_root)
    split_root.mkdir()

    for zsv_name in ("PREDICT", "PARSED"):
        zsv_dir = outdir / zsv_name / "zero-shot-validation"
        if zsv_dir.exists():
            shutil.rmtree(zsv_dir)

    bucket_rows: dict[str, list[dict[str, str]]] = {}
    n_skipped = 0
    skipped_examples: list[str] = []
    used_unique: dict[str, str] = {}

    for row in rows:
        rid = row["ID"]
        safe_region = sanitize_filename(rid)
        is_zsv = is_zsv_fold(row["train_test"]) or is_zsv_fold(row["fold"])
        src_fa = data_root / f"{safe_region}.ext"
        pred_list = by_region.get(rid, [])

        missing: list[str] = []
        if not pred_list:
            missing.append("predict.csv")
        if not src_fa.is_file():
            missing.append(f"PARSED/{safe_region}.ext")

        pred_sources: list[tuple[dict[str, str], Path, str]] = []
        for prow in pred_list:
            sid = prow.get("sample_id", "") if mapped else ""
            src_ext = _predict_src_path(
                predict_root, safe_region, mapped=mapped, sample_id=sid
            )
            if mapped:
                unique_id = make_mapped_unique_id(sid, rid)
            else:
                unique_id = safe_region
            other = used_unique.setdefault(unique_id, rid if not mapped else f"{sid}:{rid}")
            expect = rid if not mapped else f"{sid}:{rid}"
            if other != expect:
                raise ValueError(
                    f"Composite/unique id collision {unique_id!r}: {other!r} vs {expect!r}"
                )
            if not src_ext.is_file():
                missing.append(
                    f"PREDICT/{sanitize_filename(sid) + '/' if sid else ''}{safe_region}.ext"
                )
            else:
                pred_sources.append((prow, src_ext, unique_id))

        if not pred_sources or not src_fa.is_file():
            if intersect_allow:
                n_skipped += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append(
                        f"{rid}: {', '.join(missing) if missing else 'incomplete'}"
                    )
                continue
            raise FileNotFoundError(
                f"ID {rid!r} from split.csv is missing required artifacts: "
                + ", ".join(missing or ["incomplete artifacts"])
            )
        if missing and not intersect_allow:
            raise FileNotFoundError(
                f"ID {rid!r} from split.csv is missing required artifacts: "
                + ", ".join(missing)
            )

        if is_zsv:
            for prow, src_ext, unique_id in pred_sources:
                link_or_copy(
                    src_ext,
                    outdir / "PREDICT" / "zero-shot-validation" / f"{unique_id}.ext",
                )
                link_or_copy(
                    src_fa,
                    outdir / "PARSED" / "zero-shot-validation" / f"{unique_id}.ext",
                )
            continue

        buckets = _bucket(row["train_test"], str(row["fold"]), strategy)
        for bucket in buckets:
            for prow, src_ext, unique_id in pred_sources:
                out_row = _out_predict_row(
                    prow, mapped=mapped, unique_id=unique_id, region_id=rid
                )
                bucket_rows.setdefault(bucket, []).append(out_row)
                pred_rel = Path("PREDICT") / bucket / f"{unique_id}.ext"
                fasta_rel = Path("FASTA") / bucket / f"{unique_id}.ext"
                link_or_copy(src_ext, split_root / pred_rel)
                link_or_copy(src_fa, split_root / fasta_rel)

    for bucket, brows in bucket_rows.items():
        # Enforce unique id column before write
        index_unique_predict_rows(brows, label=f"SPLIT/PREDICT/{bucket}/predict.csv")
        write_csv(split_root / "PREDICT" / bucket / "predict.csv", brows, predict_fields)

    if n_skipped and intersect_allow:
        warnings.warn(
            f"intersect_allow=True: skipped {n_skipped} ID(s) missing PARSED/PREDICT "
            f"(examples: {skipped_examples})",
            UserWarning,
            stacklevel=2,
        )

    return split_root


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"t", "true", "1", "yes", "y"}:
        return True
    if v in {"f", "false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected T/F (bool), got {value!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="split materialize → SPLIT/")
    p.add_argument("--split-csv", required=True, type=Path)
    p.add_argument("--parsed-target", required=True, type=Path)
    p.add_argument("--parsed-data", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument(
        "--strategy", default="traintestval", choices=["traintest", "traintestval", "fold"]
    )
    p.add_argument(
        "--intersect-allow",
        type=_parse_bool,
        default=True,
        help="T (default): skip IDs missing PARSED/PREDICT; F: raise",
    )
    p.add_argument(
        "--sample-id",
        default=None,
        help="Optional filter for mapped PREDICT rows (sample_id column)",
    )
    p.add_argument(
        "--id-csv",
        type=Path,
        default=None,
        help="Optional ID.csv for pre-split unique-ID checkout",
    )
    args = p.parse_args(argv)
    print(
        run_split(
            args.split_csv,
            args.parsed_target,
            args.parsed_data,
            outdir=args.outdir,
            strategy=args.strategy,
            intersect_allow=args.intersect_allow,
            sample_id=args.sample_id,
            id_csv=args.id_csv,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
