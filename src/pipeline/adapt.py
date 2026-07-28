"""Universal adapt stage: FNA/GTF + ID.csv → MARKED FASTA + intersect.csv.

Gene windows are expressed as signed offsets from the strand-aware gene anchor.
When a window crosses zero, coordinate zero (the anchor base) is deliberately
excluded: ``{"pos1": -100, "pos2": 100}`` yields 200 bases.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.preprocessing import (
    Fasta,
    chrom_length,
    extract_forward,
    free_intervals,
    genome_prefix,
    open_text,
    parse_attrs,
)

from .common import (
    ID_CSV_COLUMNS,
    INTERSECT_COLUMNS,
    ensure_dir,
    read_csv,
    require_columns,
    sanitize_filename,
    write_csv,
    write_fasta_record,
)

SEED = 42


def _index_folder(folder: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    out: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.endswith(s) for s in suffixes):
            continue
        # Pass the filename rather than a manually stripped stem: accession
        # versions (for example ``GCF_TEST000001.1``) must remain intact.
        g = genome_prefix(path.name)
        if g in out:
            raise ValueError(f"Multiple inputs resolve to genome {g}: {out[g]}, {path}")
        out[g] = path
    return out


def _parse_gtf(gtf_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    """Index gene/CDS spans by stable ``gene_id`` and occupied gene intervals."""
    features: dict[tuple[str, str], dict[str, Any]] = {}
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in {"gene", "CDS"}:
                continue
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError as exc:
                raise ValueError(f"Invalid GTF coordinates in {gtf_path}: {line.rstrip()}") from exc
            attrs = parse_attrs(parts[8])
            raw = attrs.get("gene_id") or attrs.get("gene") or attrs.get("transcript_id")
            if not raw:
                continue
            chrom, strand = parts[0], parts[6] if parts[6] in "+-" else "+"
            key = (chrom, raw)
            item = features.setdefault(
                key,
                {
                    "chr": chrom,
                    "raw_target_ID": raw,
                    "gene_nameORnon_coding_ID": attrs.get("gene") or attrs.get("gene_name") or raw,
                    "strand": strand,
                    "start": start,
                    "end": end,
                },
            )
            item["start"] = min(int(item["start"]), start)
            item["end"] = max(int(item["end"]), end)
            if parts[2] == "gene":
                occupied[chrom].append((start, end))
    for chrom in occupied:
        occupied[chrom].sort()
    return features, occupied


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _offset_segments(anchor: int, strand: str, start_offset: int, end_offset: int) -> list[tuple[int, int]]:
    """Map signed gene-oriented offsets to forward-genomic inclusive segments."""
    offsets = list(range(start_offset, end_offset + 1))
    if start_offset < 0 < end_offset:
        offsets.remove(0)
    positions = sorted(anchor + offset if strand == "+" else anchor - offset for offset in offsets)
    if not positions:
        return []
    segments: list[tuple[int, int]] = []
    seg_start = previous = positions[0]
    for pos in positions[1:]:
        if pos == previous + 1:
            previous = pos
            continue
        segments.append((seg_start, previous))
        seg_start = previous = pos
    segments.append((seg_start, previous))
    return segments


def _center_trim(segments: list[tuple[int, int]], maximum: int) -> list[tuple[int, int]]:
    """Trim a forward-genomic sequence symmetrically from both sequence ends."""
    total = sum(end - start + 1 for start, end in segments)
    if total <= maximum:
        return segments
    drop_left = (total - maximum) // 2
    drop_right = total - maximum - drop_left
    positions: list[int] = []
    for start, end in segments:
        positions.extend(range(start, end + 1))
    retained = positions[drop_left : total - drop_right]
    out: list[tuple[int, int]] = []
    seg_start = previous = retained[0]
    for pos in retained[1:]:
        if pos == previous + 1:
            previous = pos
        else:
            out.append((seg_start, previous))
            seg_start = previous = pos
    out.append((seg_start, previous))
    return out


def _sequence_for_segments(fasta: Any, chrom: str, segments: list[tuple[int, int]]) -> str:
    return "".join(extract_forward(fasta, chrom, start, end) for start, end in segments)


def _overlap_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return sum(max(0, min(a_end, b_end) - max(a_start, b_start) + 1) for a_start, a_end in a for b_start, b_end in b)


def run_adapt(
    gtf_dir: Path,
    fna_dir: Path,
    *,
    outdir: Path,
    id_csv: Path,
    environment: str = "gene",
    window: dict[str, int] | None = None,
    max_window: int | None = None,
    genomes: list[str] | None = None,
    seed: int = SEED,
) -> dict[str, Path]:
    """
    Write `intersect.csv` and `MARKED/{id}.fa`.

    FASTA header: >|genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID
    """
    if Fasta is None:
        raise RuntimeError("pyfaidx is required for adapt; install pyfaidx in the active environment")
    if environment not in {"gene", "random"}:
        raise ValueError("environment must be gene|random")
    if max_window is not None and max_window <= 0:
        raise ValueError("max_window must be a positive integer or None")
    if environment == "gene":
        if not isinstance(window, dict) or set(window) != {"pos1", "pos2"}:
            raise ValueError("gene environment requires window {'pos1': int, 'pos2': int}")
        if not all(isinstance(value, int) for value in window.values()):
            raise ValueError("window offsets must be integers")
        if window["pos1"] > window["pos2"]:
            raise ValueError("window pos1 must be <= pos2")
    id_rows = read_csv(Path(id_csv))
    require_columns(id_rows, ID_CSV_COLUMNS, label="ID.csv")
    if not id_rows:
        raise ValueError("ID.csv has no rows")

    gtf_index = _index_folder(Path(gtf_dir), (".gtf",))
    fna_index = _index_folder(Path(fna_dir), (".fna", ".fa", ".fasta", ".fna.gz", ".fa.gz", ".fasta.gz"))
    available_genomes = sorted(set(gtf_index) & set(fna_index))
    if genomes is not None:
        selected = set(genomes)
        available_genomes = [genome for genome in available_genomes if genome in selected]
    if not available_genomes:
        raise FileNotFoundError("No matching GTF/FNA genome pairs")

    outdir = ensure_dir(Path(outdir))
    marked = ensure_dir(outdir / "MARKED")
    intersect_rows: list[dict[str, Any]] = []
    written = 0

    for genome in available_genomes:
        fasta = Fasta(str(fna_index[genome]), as_raw=True, sequence_always_upper=True)
        features, occupied = _parse_gtf(gtf_index[genome])
        rows = [row for row in id_rows if row["genome"] == genome]
        if not rows:
            fasta.close()
            continue
        resolved: list[dict[str, Any]] = []
        free_by_chrom = {
            chrom: free_intervals(chrom_length(fasta, chrom), _merge_intervals(intervals))
            for chrom, intervals in occupied.items()
            if chrom in fasta
        }
        rng = random.Random(seed)
        for meta in rows:
            chrom = meta["chr"]
            if chrom not in fasta:
                continue
            if environment == "gene":
                if window == {"pos1": 0, "pos2": 0}:
                    segments = [(int(meta["pos1"]), int(meta["pos2"]))]
                else:
                    feat = features.get((chrom, meta["raw_target_ID"]))
                    if feat is None:
                        # fall back only when the display name is uniquely resolvable.
                        candidates = [v for v in features.values() if v["chr"] == chrom and v["gene_nameORnon_coding_ID"] == meta["gene_nameORnon_coding_ID"]]
                        if len(candidates) != 1:
                            continue
                        feat = candidates[0]
                    anchor = int(feat["start"]) if feat["strand"] == "+" else int(feat["end"])
                    segments = _offset_segments(anchor, str(feat["strand"]), window["pos1"], window["pos2"])
                chrom_len = chrom_length(fasta, chrom)
                segments = [(max(1, start), min(chrom_len, end)) for start, end in segments]
                segments = [(start, end) for start, end in segments if start <= end]
                label = meta["gene_nameORnon_coding_ID"]
            else:
                target_length = int(meta["pos2"]) - int(meta["pos1"]) + 1
                if max_window is not None:
                    target_length = min(target_length, max_window)
                candidates = [(start, end) for start, end in free_by_chrom.get(chrom, []) if end - start + 1 >= target_length]
                if not candidates:
                    continue
                start, end = rng.choice(candidates)
                pos1 = rng.randint(start, end - target_length + 1)
                segments = [(pos1, pos1 + target_length - 1)]
                label = f"NC_{genome}_{meta['ID']}"
                updated: list[tuple[int, int]] = []
                for free_start, free_end in free_by_chrom[chrom]:
                    if segments[0][1] < free_start or segments[0][0] > free_end:
                        updated.append((free_start, free_end))
                    else:
                        if free_start < segments[0][0]:
                            updated.append((free_start, segments[0][0] - 1))
                        if segments[0][1] < free_end:
                            updated.append((segments[0][1] + 1, free_end))
                free_by_chrom[chrom] = updated
            if not segments:
                continue
            if max_window is not None:
                segments = _center_trim(segments, max_window)
            sequence = _sequence_for_segments(fasta, chrom, segments)
            if not sequence:
                continue
            p1, p2 = segments[0][0], segments[-1][1]
            rid = meta["ID"]
            header = [
                genome,
                chrom,
                str(p1),
                str(p2),
                label,
                meta["raw_target_ID"],
                str(rid),
            ]
            fa_path = marked / f"{sanitize_filename(str(rid))}.fa"
            write_fasta_record(fa_path, header, sequence)
            resolved.append({"ID": rid, "chr": chrom, "segments": segments})
            written += 1
        fasta.close()

        # Record only true overlap lengths across potentially discontinuous segments.
        by_chrom: dict[str, list[dict[str, Any]]] = {}
        for w in resolved:
            by_chrom.setdefault(w["chr"], []).append(w)
        for chrom, items in by_chrom.items():
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    size_i = _overlap_length(a["segments"], b["segments"])
                    if size_i > 0:
                        intersect_rows.append(
                            {
                                "ID1": a["ID"],
                                "ID2": b["ID"],
                                "intersection_size": size_i,
                            }
                        )
    if written == 0:
        raise ValueError("adapt wrote 0 MARKED files — check ID.csv join / GTF feature")

    intersect_path = outdir / "intersect.csv"
    if not intersect_rows:
        # still emit header-only valid table
        write_csv(intersect_path, [], INTERSECT_COLUMNS)
    else:
        write_csv(intersect_path, intersect_rows, INTERSECT_COLUMNS)

    return {"marked_dir": marked, "intersect_csv": intersect_path}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="adapt → MARKED + intersect.csv")
    p.add_argument("--gtf", required=True, type=Path)
    p.add_argument("--fna", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--id-csv", required=True, type=Path)
    p.add_argument("--environment", required=True, choices=["gene", "random"])
    p.add_argument("--window", default=None, help="JSON offsets, e.g. '{\"pos1\":-100,\"pos2\":100}'")
    p.add_argument("--max-window", default="null", help="Integer cap, or null/none")
    p.add_argument("--genomes", nargs="*", default=None, help="Optional GCF accession filter")
    args = p.parse_args(argv)
    window: dict[str, int] | None = None
    if args.window is not None:
        try:
            decoded = json.loads(args.window)
        except json.JSONDecodeError as exc:
            p.error(f"--window must be valid JSON: {exc.msg}")
        if not isinstance(decoded, dict):
            p.error("--window must decode to an object")
        window = decoded
    raw_max = str(args.max_window).strip().lower()
    if raw_max in {"null", "none", ""}:
        max_window = None
    else:
        try:
            max_window = int(raw_max)
        except ValueError:
            p.error("--max-window must be an integer, null, or none")
    out = run_adapt(
        args.gtf,
        args.fna,
        outdir=args.outdir,
        id_csv=args.id_csv,
        environment=args.environment,
        window=window,
        max_window=max_window,
        genomes=args.genomes,
    )
    print(out["intersect_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
