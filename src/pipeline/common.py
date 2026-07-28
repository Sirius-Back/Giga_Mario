"""Shared CSV / ID helpers for the universal pipeline."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

ID_CSV_COLUMNS = [
    "genome",
    "chr",
    "pos1",
    "pos2",
    "gene_nameORnon_coding_ID",
    "raw_target_ID",
    "ID",
]

FOLD_CSV_MIN = ["ID", "fold"]
STRAT_CSV_MIN = ["ID", "strat1"]
SPLIT_CSV_COLUMNS = ["ID", "train_test", "fold"]
PREDICT_CSV_ID_COL = "id"
INTERSECT_COLUMNS = ["ID1", "ID2", "intersection_size"]

MARKED_HEADER_FIELDS = [
    "genome",
    "chr",
    "pos1",
    "pos2",
    "gene_nameORnon_coding_ID",
    "raw_target_ID",
    "ID",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str], *, delimiter: str = "|") -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: Path, *, delimiter: str = "|") -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def require_columns(rows: list[dict[str, str]], required: Iterable[str], *, label: str) -> None:
    if not rows:
        # header-only files still need validation of fieldnames via separate check
        return
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{label} missing columns {missing}; have {list(rows[0])}")


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "sample"


def parse_gtf_attrs(attr_field: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\s+"([^"]*)"', attr_field):
        out[m.group(1)] = m.group(2)
    return out


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks).upper()
    return seqs


def write_fasta_record(path: Path, header_fields: Sequence[str], sequence: str) -> None:
    ensure_dir(path.parent)
    header = "|" + "|".join(str(x) for x in header_fields)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            fh.write(sequence[i : i + 80] + "\n")


def parse_marked_header(header: str) -> dict[str, str]:
    h = header[1:] if header.startswith(">") else header
    if h.startswith("|"):
        h = h[1:]
    parts = h.split("|")
    if len(parts) < 7:
        raise ValueError(f"Marked FASTA header needs 7 fields, got {parts!r}")
    return {
        "genome": parts[0],
        "chr": parts[1],
        "pos1": parts[2],
        "pos2": parts[3],
        "gene_nameORnon_coding_ID": parts[4],
        "raw_target_ID": parts[5],
        "ID": parts[6],
    }
