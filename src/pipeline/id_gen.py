"""Generate stable integer IDs from GTF features.

``ID.csv`` is pipe-delimited and assigns IDs in input file order.  CDS records
are the exception: their rows are aggregated to one min--max CDS span per
``(chrom, gene_id)`` in first-seen CDS order.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

from src.preprocessing import genome_prefix, open_text, parse_attrs

from .common import ID_CSV_COLUMNS, ensure_dir, write_csv


def _feature_identity(attrs: dict[str, str]) -> tuple[str, str]:
    """Return display name and the stable ID used for later target joins."""
    raw_target_id = (
        attrs.get("gene_id")
        or attrs.get("gene")
        or attrs.get("transcript_id")
        or attrs.get("gene_name")
    )
    if not raw_target_id:
        raise ValueError("GTF feature has no gene_id, gene, transcript_id, or gene_name")
    display_name = attrs.get("gene") or attrs.get("gene_name") or raw_target_id
    return display_name, raw_target_id


def _iter_feature_rows(gtf_path: Path, feature: str) -> Iterator[dict[str, Any]]:
    """Yield matching GTF feature rows in source order using shared parsing."""
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != feature:
                continue
            try:
                pos1, pos2 = int(parts[3]), int(parts[4])
            except ValueError as exc:
                raise ValueError(f"Invalid coordinates in {gtf_path}: {line.rstrip()}") from exc
            if pos1 > pos2:
                raise ValueError(f"Start exceeds end in {gtf_path}: {line.rstrip()}")
            name, raw_target_id = _feature_identity(parse_attrs(parts[8]))
            yield {
                "chr": parts[0],
                "pos1": pos1,
                "pos2": pos2,
                "gene_nameORnon_coding_ID": name,
                "raw_target_ID": raw_target_id,
            }


def _iter_cds_rows(gtf_path: Path) -> Iterator[dict[str, Any]]:
    """Aggregate CDS spans like ``preprocessing.iter_cds_genes``."""
    spans: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_feature_rows(gtf_path, "CDS"):
        key = (str(row["chr"]), str(row["raw_target_ID"]))
        if key not in spans:
            spans[key] = row
            continue
        spans[key]["pos1"] = min(int(spans[key]["pos1"]), int(row["pos1"]))
        spans[key]["pos2"] = max(int(spans[key]["pos2"]), int(row["pos2"]))
    yield from spans.values()


def iter_gtf_features(gtf_path: Path, feature: str) -> Iterator[dict[str, Any]]:
    """Yield ID rows for a GTF feature type, aggregating CDS features."""
    if feature == "CDS":
        yield from _iter_cds_rows(gtf_path)
    else:
        yield from _iter_feature_rows(gtf_path, feature)


def _discover_gtfs(gtf: Path) -> list[Path]:
    if not gtf.exists():
        raise FileNotFoundError(f"GTF input does not exist: {gtf}")
    if gtf.is_file():
        if not gtf.name.endswith((".gtf", ".gtf.gz")):
            raise ValueError(f"Expected .gtf or .gtf.gz input, got: {gtf}")
        return [gtf]
    paths = sorted(
        path
        for path in gtf.iterdir()
        if path.is_file() and path.name.endswith((".gtf", ".gtf.gz"))
    )
    if not paths:
        raise FileNotFoundError(f"No .gtf or .gtf.gz files under {gtf}")
    return paths


def run_id_gen(
    gtf: Path,
    *,
    gtf_column: str = "transcript",
    outdir: Path,
    genome: str | None = None,
) -> Path:
    """
    Build `{outdir}/ID.csv` from a GTF file or directory of GTFs.

    `gtf_column` is the GTF feature type (column 3), e.g. transcript, gene, CDS.
    """
    gtf = Path(gtf)
    if not gtf_column:
        raise ValueError("gtf_column must be a non-empty GTF feature type")
    outdir = ensure_dir(Path(outdir))
    paths = _discover_gtfs(gtf)

    out_rows: list[dict[str, Any]] = []
    next_id = 1
    for path in paths:
        gname = genome or genome_prefix(path.name)
        for feat in iter_gtf_features(path, gtf_column):
            out_rows.append(
                {
                    "genome": gname,
                    "chr": feat["chr"],
                    "pos1": feat["pos1"],
                    "pos2": feat["pos2"],
                    "gene_nameORnon_coding_ID": feat["gene_nameORnon_coding_ID"],
                    "raw_target_ID": feat["raw_target_ID"],
                    "ID": next_id,
                }
            )
            next_id += 1

    if not out_rows:
        raise ValueError(f"No features of type {gtf_column!r} in {gtf}")

    ids = [int(r["ID"]) for r in out_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ID values must be unique")

    out_path = outdir / "ID.csv"
    write_csv(out_path, out_rows, ID_CSV_COLUMNS)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate ID.csv from GTF")
    p.add_argument("--gtf", required=True, type=Path)
    p.add_argument("--gtf-column", default="transcript")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--genome", default=None)
    args = p.parse_args(argv)
    path = run_id_gen(args.gtf, gtf_column=args.gtf_column, outdir=args.outdir, genome=args.genome)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
