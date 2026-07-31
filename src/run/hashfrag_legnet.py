#!/usr/bin/env python3
"""Convert ready_legnet MARKED → multi-FASTA files for hashFrag.

Reads MARKED/{ID}.fa files and split.csv (ID|train_test|fold), then writes
  output/hashfrag/legnet_run0/fasta/train.fa
  output/hashfrag/legnet_run0/fasta/test.fa

Headers: >{ID}  (filename stem = join key for split.csv and ID.csv).
val / zsv labels are excluded by default (skill contract).

Usage:
    python -m src.run.hashfrag_legnet [--panel ready_legnet] [--run-id legnet_run0]
                                       [--max-ids N] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path


def _load_split_csv(split_path: Path) -> dict[str, str]:
    """Return {id: train_test} from split.csv; skip val/zsv."""
    result: dict[str, str] = {}
    with open(split_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            label = row["train_test"].strip()
            if label in ("train", "test"):
                result[row["ID"].strip()] = label
    return result


def _read_one_fasta(path: Path) -> str:
    """Return sequence from a single-record FASTA (MARKED contract)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if ">" not in text:
        return "".join(text.split()).upper()
    seq_chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        seq_chunks.append(line)
    return "".join(seq_chunks).upper()


def _write_fasta(records: dict[str, str], out_path: Path) -> int:
    """Write multi-FASTA with 80-col wrapping; return record count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rid, seq in records.items():
            fh.write(f">{rid}\n")
            for chunk in textwrap.wrap(seq, 80):
                fh.write(chunk + "\n")
    return len(records)


def validate_inputs(panel_dir: Path, split_path: Path) -> None:
    if not panel_dir.is_dir():
        sys.exit(f"ERROR: panel dir not found: {panel_dir}")
    marked = panel_dir / "MARKED"
    if not marked.is_dir():
        sys.exit(f"ERROR: MARKED dir not found: {marked}")
    fa_files = list(marked.glob("*.fa")) + list(marked.glob("*.fasta"))
    if not fa_files:
        sys.exit(f"ERROR: no .fa/.fasta files under {marked}")
    if not split_path.is_file():
        sys.exit(f"ERROR: split.csv not found: {split_path}")
    print(f"[preflight] MARKED files: {len(fa_files):,}")
    print(f"[preflight] split.csv: {split_path}")


def run(
    panel: str = "ready_legnet",
    run_id: str = "legnet_run0",
    max_ids: int | None = None,
) -> None:
    panel_dir = Path(panel)
    marked_dir = panel_dir / "MARKED"
    split_path = panel_dir / "split.csv"

    validate_inputs(panel_dir, split_path)

    print("[step 1] Loading split.csv (train/test only; val+zsv excluded)...")
    split_map = _load_split_csv(split_path)
    train_ids = sorted(k for k, v in split_map.items() if v == "train")
    test_ids = sorted(k for k, v in split_map.items() if v == "test")
    print(f"[step 1] train={len(train_ids):,}  test={len(test_ids):,}  (val/zsv excluded)")

    if max_ids:
        train_ids = train_ids[:max_ids]
        test_ids = test_ids[:max_ids]
        print(f"[step 1] Subsampling to max_ids={max_ids}: "
              f"train={len(train_ids):,}  test={len(test_ids):,}")

    overlap = set(train_ids) & set(test_ids)
    if overlap:
        sys.exit(f"ERROR: train/test overlap in split.csv: {len(overlap)} IDs. "
                 f"Example: {sorted(overlap)[0]}")

    out_fasta = Path("output/hashfrag") / run_id / "fasta"
    out_fasta.mkdir(parents=True, exist_ok=True)

    print("[step 2] Building train.fa...")
    train_records: dict[str, str] = {}
    missing_train: list[str] = []
    for rid in train_ids:
        fa = marked_dir / f"{rid}.fa"
        if not fa.exists():
            fa = marked_dir / f"{rid}.fasta"
        if not fa.exists():
            missing_train.append(rid)
            continue
        seq = _read_one_fasta(fa)
        if not seq:
            sys.exit(f"ERROR: empty sequence for {rid} in {fa}")
        train_records[rid] = seq
    if missing_train:
        sys.exit(f"ERROR: {len(missing_train)} MARKED files missing for train IDs. "
                 f"Example: {missing_train[0]}")
    n_train = _write_fasta(train_records, out_fasta / "train.fa")
    print(f"[step 2] train.fa written: {n_train:,} records → {out_fasta/'train.fa'}")

    print("[step 2] Building test.fa...")
    test_records: dict[str, str] = {}
    missing_test: list[str] = []
    for rid in test_ids:
        fa = marked_dir / f"{rid}.fa"
        if not fa.exists():
            fa = marked_dir / f"{rid}.fasta"
        if not fa.exists():
            missing_test.append(rid)
            continue
        seq = _read_one_fasta(fa)
        if not seq:
            sys.exit(f"ERROR: empty sequence for {rid} in {fa}")
        test_records[rid] = seq
    if missing_test:
        sys.exit(f"ERROR: {len(missing_test)} MARKED files missing for test IDs. "
                 f"Example: {missing_test[0]}")
    n_test = _write_fasta(test_records, out_fasta / "test.fa")
    print(f"[step 2] test.fa written:  {n_test:,} records → {out_fasta/'test.fa'}")

    print("[preflight-out] Validating output FASTA headers match split.csv IDs...")
    assert n_train == len(train_ids), "train record count mismatch"
    assert n_test == len(test_ids), "test record count mismatch"
    assert set(train_records) & set(test_records) == set(), "train/test seq ID overlap"
    print("[preflight-out] OK — IDs consistent, no overlap")
    print(f"\nOutput dir: {out_fasta.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="ready_legnet",
                        help="Panel dir containing MARKED/ and split.csv (default: ready_legnet)")
    parser.add_argument("--run-id", default="legnet_run0",
                        help="Run label (output/hashfrag/<run-id>/) (default: legnet_run0)")
    parser.add_argument("--max-ids", type=int, default=None,
                        help="Limit to first N train+test IDs (smoke test)")
    args = parser.parse_args()
    run(panel=args.panel, run_id=args.run_id, max_ids=args.max_ids)


if __name__ == "__main__":
    main()
