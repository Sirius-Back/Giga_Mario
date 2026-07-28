"""Parse MARKED FASTA records into model-ready ``PARSED/<ID>.ext`` files.

Caduceus: raw DNA text (non-ACGTN → N, matching tokenizer UNK→N normalization).
LegNet: only exact CRS_BP inserts are stitched via ``src.legnet_preprocess.stitch_adapters``;
incomplete edge windows are skipped (same policy as ``legnet_preprocess`` CRS OOB skip).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.legnet_preprocess import (
    ADAPTER_3,
    ADAPTER_5,
    CRS_BP,
    STITCHED_LEN,
    stitch_adapters,
)

from .common import ensure_dir, parse_marked_header, sanitize_filename

DNA_CORE = frozenset("ACGTN")
_MARKED_SUFFIXES = (".fa", ".fasta")


def _marked_paths(marked: Path) -> list[Path]:
    if marked.is_file():
        return [marked]
    if not marked.is_dir():
        raise FileNotFoundError(f"MARKED input does not exist: {marked}")
    paths = sorted(
        path
        for path in marked.iterdir()
        if path.is_file() and path.suffix.lower() in _MARKED_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No marked FASTA (*.fa, *.fasta) under {marked}")
    return paths


def _normalize_dna(sequence: str) -> str:
    """Uppercase; map any non-ACGTN IUPAC/ambiguity base to N."""
    seq = sequence.upper()
    return "".join(ch if ch in DNA_CORE else "N" for ch in seq)


def read_marked_fasta(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"MARKED FASTA is missing or empty: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid MARKED FASTA header: {path}")
    if any(line.startswith(">") for line in lines[1:]):
        raise ValueError(f"MARKED FASTA must contain exactly one record: {path}")
    meta = parse_marked_header(lines[0])
    if not meta["ID"]:
        raise ValueError(f"MARKED FASTA has an empty ID: {path}")
    sequence = _normalize_dna("".join(lines[1:]))
    if not sequence:
        raise ValueError(f"MARKED FASTA has no sequence: {path}")
    return meta, sequence


def to_legnet_sequence(sequence: str) -> str:
    if len(sequence) != CRS_BP:
        raise ValueError(
            f"LegNet parsing requires a {CRS_BP} bp MARKED CRS; got {len(sequence)} bp"
        )
    stitched = stitch_adapters(sequence)
    if len(stitched) != STITCHED_LEN:
        raise ValueError(f"LegNet stitch length {len(stitched)} != {STITCHED_LEN}")
    if not stitched.startswith(ADAPTER_5) or not stitched.endswith(ADAPTER_3):
        raise ValueError("LegNet adapter stitch did not preserve adapters")
    return stitched


def run_parse_data(
    marked: Path,
    *,
    outdir: Path,
    to_type: str = "caduceus",
    skip_incomplete_legnet: bool = True,
) -> Path:
    """Write ``outdir/PARSED/<ID>.ext``. Also writes ``outdir/parse_data_stats.json``."""
    if to_type not in {"caduceus", "legnet"}:
        raise ValueError(f"to_type must be caduceus|legnet, got {to_type!r}")

    parsed = ensure_dir(Path(outdir) / "PARSED")
    seen: set[str] = set()
    n_ok = 0
    n_skip = 0
    skip_reasons: dict[str, int] = {}
    for path in _marked_paths(Path(marked)):
        meta, sequence = read_marked_fasta(path)
        rid = str(meta["ID"])
        if rid in seen:
            raise ValueError(f"Duplicate MARKED ID would overwrite output: {rid!r}")
        seen.add(rid)
        if to_type == "caduceus":
            body = sequence
        else:
            if len(sequence) != CRS_BP:
                if not skip_incomplete_legnet:
                    raise ValueError(
                        f"LegNet requires {CRS_BP} bp; got {len(sequence)} in {path}"
                    )
                n_skip += 1
                skip_reasons["bad_crs_len"] = skip_reasons.get("bad_crs_len", 0) + 1
                continue
            body = to_legnet_sequence(sequence)
        (parsed / f"{sanitize_filename(rid)}.ext").write_text(body + "\n", encoding="utf-8")
        n_ok += 1

    stats = {
        "parsed": str(parsed),
        "to_type": to_type,
        "n_written": n_ok,
        "n_skipped": n_skip,
        "skip_reasons": skip_reasons,
    }
    (Path(outdir) / "parse_data_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="parse_data: MARKED FASTA → PARSED/")
    parser.add_argument("--marked", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--to-type", required=True, choices=["caduceus", "legnet"])
    parser.add_argument(
        "--strict-legnet",
        action="store_true",
        help="Fail on non-CRS_BP MARKED records instead of skipping",
    )
    args = parser.parse_args(argv)
    parsed = run_parse_data(
        args.marked,
        outdir=args.outdir,
        to_type=args.to_type,
        skip_incomplete_legnet=not args.strict_legnet,
    )
    stats_path = Path(args.outdir) / "parse_data_stats.json"
    print(parsed)
    if stats_path.is_file():
        print(stats_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
