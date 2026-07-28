"""Create a human_legnet TSV from a universal materialized SPLIT tree."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import read_csv

FOLD_TO_CV = {"TEST": 1, "VAL": 2, "TRAIN": 3}


def build_legnet_tsv(*, split_root: Path, out_tsv: Path) -> Path:
    """Join real 230 bp SPLIT DNA and scalar PREDICT labels into LegNet input.

    Human LegNet reserves CV folds 1 and 2 for test and validation respectively;
    the materialized TRAIN split is assigned fold 3.  The source split is never
    modified, and strict ID/length checks prevent accidental label mismatches.
    """
    split_root = Path(split_root)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for bucket, cv_fold in FOLD_TO_CV.items():
        fasta_dir = split_root / "FASTA" / bucket
        predict_csv = split_root / "PREDICT" / bucket / "predict.csv"
        if not fasta_dir.is_dir() or not predict_csv.is_file():
            raise FileNotFoundError(
                f"Need {fasta_dir}/ and {predict_csv} to build LegNet input"
            )
        predictions = read_csv(predict_csv)
        by_id = {row["id"]: row for row in predictions}
        if len(by_id) != len(predictions):
            raise ValueError(f"Duplicate id values in {predict_csv}")
        sequence_ids = {path.stem for path in fasta_dir.glob("*.ext")}
        if sequence_ids != set(by_id):
            raise ValueError(
                f"ID mismatch in {bucket}: FASTA={len(sequence_ids)}, "
                f"PREDICT={len(by_id)}"
            )
        for identifier in sorted(sequence_ids):
            if identifier in seen:
                raise ValueError(f"Duplicate SPLIT sequence id: {identifier}")
            seen.add(identifier)
            sequence = (fasta_dir / f"{identifier}.ext").read_text(
                encoding="utf-8"
            ).strip().upper()
            if len(sequence) != 230:
                raise ValueError(
                    f"LegNet requires 230 bp sequences; {identifier} has {len(sequence)} bp"
                )
            try:
                value = float(by_id[identifier]["predict_var1"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric predict_var1 for {identifier} in {predict_csv}"
                ) from exc
            rows.append(
                {
                    "seq_id": identifier,
                    "seq": sequence,
                    "mean_value": str(value),
                    "fold": str(cv_fold),
                    "rev": "0",
                }
            )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seq_id", "seq", "mean_value", "fold", "rev"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_tsv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_legnet_tsv(split_root=args.split_root, out_tsv=args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
