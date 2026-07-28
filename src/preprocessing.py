#!/usr/bin/env python3
"""Raw FNA/GTF/TPM → data_ready Caduceus-prep windows.

Pipeline (LOCKED 2026-07-27):
  1. Pair genomes via raw/{fna,gtf,tpm} + random_borzoi_expr_file_mappings.csv
  2. CDS span ± flank (default 10_000 bp); neighbour-trim; large-gene crop
  3. Match intergenic (non-coding) windows to gene length & GC distributions
  4. Write ready.fna / ready.csv / side tables / caduceus_ready/

Does NOT assign train/val/test folds.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

try:
    from pyfaidx import Fasta
except ImportError:  # pragma: no cover
    Fasta = None  # type: ignore[assignment,misc]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Locked defaults
# ---------------------------------------------------------------------------
FLANK_BP = 10_000
LARGE_GENE_BP = 130_000
LARGE_BODY_BP = 120_000  # 10k flank + 120k CDS body = 130k window
SEED = 42
GC_TOL = 0.03  # absolute GC fraction tolerance for non-coding match
STRIDE_FRAC = 0.25  # search stride as fraction of window length
SEP = "|"
ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')


def stable_hash(s: str) -> int:
    """Deterministic 32-bit hash (avoid PYTHONHASHSEED nondeterminism)."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GenomeBundle:
    genome_id: str  # GCF accession prefix, e.g. GCF_000001405.40
    fasta: Path
    gtf: Path
    tpm: Path | None
    tpm_id: str | None
    notes: list[str] = field(default_factory=list)


@dataclass
class GeneCDS:
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    cds_start: int  # 1-based inclusive
    cds_end: int  # 1-based inclusive

    @property
    def cds_length(self) -> int:
        return self.cds_end - self.cds_start + 1


@dataclass
class Window:
    genome: str
    gene_or_id: str
    chrom: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    strand: str
    kind: str  # gene | non_coding
    tpm: float
    sequence: str = ""
    length: int = 0
    gc: float = 0.0
    large_gene: bool = False
    neighbour_trimmed: bool = False

    def finalize(self, seq: str) -> None:
        self.sequence = seq.upper()
        self.length = len(self.sequence)
        if self.length == 0:
            self.gc = 0.0
            return
        gc = sum(1 for b in self.sequence if b in "GC")
        atgc = sum(1 for b in self.sequence if b in "ACGT")
        self.gc = (gc / atgc) if atgc else 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def open_text(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def parse_attrs(attr_field: str) -> dict[str, str]:
    return {k: v for k, v in ATTR_RE.findall(attr_field)}


def sanitize_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "sample"


def genome_prefix(name: str) -> str:
    """GCF_000001405.40_GRCh38... → GCF_000001405.40"""
    m = re.match(r"(GCF_\d+\.\d+)", name)
    return m.group(1) if m else Path(name).stem.split("_genomic")[0]


def load_tpm_row(path: Path) -> dict[str, float]:
    """Wide TPM CSV: header = gene symbols, single data row. Duplicate cols → first kept."""
    with open_text(path) as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
            values = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty TPM file: {path}") from exc
    out: dict[str, float] = {}
    for name, raw in zip(header, values):
        key = name.strip()
        if not key or key in out:
            continue
        try:
            out[key] = float(raw)
        except ValueError:
            out[key] = 0.0
    return out


def discover_raw(
    raw_dir: Path,
    *,
    tpm_merged_only: bool = False,
) -> tuple[list[GenomeBundle], list[str]]:
    """Pair fna/gtf/tpm under raw/ using mapping CSV when present.

    If ``tpm_merged_only``, only ``*_merged.csv`` TPM files are eligible (and
    mapping ids must resolve to those stems).
    """
    notes: list[str] = []
    fna_dir = raw_dir / "fna"
    gtf_dir = raw_dir / "gtf"
    tpm_dir = raw_dir / "tpm"
    for d, label in ((fna_dir, "fna"), (gtf_dir, "gtf"), (tpm_dir, "tpm")):
        if not d.is_dir():
            raise FileNotFoundError(f"Critical: missing raw/{label}/ at {d}")

    fnas = {
        genome_prefix(p.name): p
        for p in sorted(fna_dir.iterdir())
        if p.suffix in {".fna", ".fa", ".fasta"} or p.name.endswith((".fna.gz", ".fa.gz", ".fasta.gz"))
    }
    gtfs = {
        genome_prefix(p.name): p
        for p in sorted(gtf_dir.iterdir())
        if ".gtf" in p.name
    }
    tpm_paths = sorted(tpm_dir.glob("*.csv"))
    if tpm_merged_only:
        tpm_paths = [p for p in tpm_paths if p.stem.endswith("_merged")]
        notes.append(f"tpm_merged_only: {len(tpm_paths)} *_merged.csv eligible")
    tpms_by_id = {p.stem: p for p in tpm_paths}

    mapping_path = raw_dir / "random_borzoi_expr_file_mappings.csv"
    genome_to_tpm: dict[str, tuple[str, Path | None]] = {}
    if mapping_path.exists():
        with mapping_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                gid = row["genome"].strip()
                tid = row["id"].strip()
                if tpm_merged_only and not tid.endswith("_merged"):
                    notes.append(f"Skip mapping id {tid} for {gid}: not *_merged")
                    continue
                local = tpms_by_id.get(tid)
                genome_to_tpm[gid] = (tid, local)
                if local is None:
                    notes.append(f"TPM id {tid} mapped to {gid} but file missing under raw/tpm/")
    elif tpm_merged_only:
        # Pair GCF accession → {stem}_merged.csv when stem starts with that GCF
        for stem, path in tpms_by_id.items():
            gid = genome_prefix(stem)
            if gid in genome_to_tpm:
                notes.append(f"Duplicate merged TPM for {gid}: keeping {genome_to_tpm[gid][0]}, skip {stem}")
                continue
            genome_to_tpm[gid] = (stem, path)
        notes.append("No mapping CSV — paired genomes to *_merged.csv by GCF prefix")
    else:
        notes.append("No random_borzoi_expr_file_mappings.csv — cannot auto-pair TPM")

    bundles: list[GenomeBundle] = []
    for gid in sorted(set(fnas) | set(gtfs)):
        fa = fnas.get(gid)
        gt = gtfs.get(gid)
        miss = []
        if fa is None:
            miss.append("fna")
        if gt is None:
            miss.append("gtf")
        if miss:
            notes.append(f"Skip {gid}: missing {','.join(miss)}")
            continue
        tid, tpm_path = genome_to_tpm.get(gid, (None, None))
        if tpm_path is None:
            notes.append(f"Skip {gid}: no usable TPM (id={tid})")
            continue
        assert fa is not None and gt is not None
        bundles.append(
            GenomeBundle(genome_id=gid, fasta=fa, gtf=gt, tpm=tpm_path, tpm_id=tid)
        )
    if not bundles:
        raise FileNotFoundError("No complete genome bundles (fna+gtf+tpm) found under raw/")
    return bundles, notes

# ---------------------------------------------------------------------------
# GTF → CDS genes
# ---------------------------------------------------------------------------
def iter_cds_genes(gtf_path: Path, max_genes: int | None = None) -> Iterator[GeneCDS]:
    """Aggregate CDS features per (chrom, gene_id) into a single CDS span."""
    spans: dict[tuple[str, str], dict[str, Any]] = {}
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            chrom, start_s, end_s, strand, attrs_s = (
                parts[0],
                parts[3],
                parts[4],
                parts[6],
                parts[8],
            )
            attrs = parse_attrs(attrs_s)
            gene_id = attrs.get("gene_id") or attrs.get("gene")
            if not gene_id:
                continue
            gene_name = attrs.get("gene") or gene_id
            start, end = int(start_s), int(end_s)
            key = (chrom, gene_id)
            if key not in spans:
                spans[key] = {
                    "gene_name": gene_name,
                    "strand": strand if strand in "+-" else "+",
                    "start": start,
                    "end": end,
                }
                if max_genes is not None and len(spans) >= max_genes:
                    # Keep reading a bit more lines for this gene's remaining CDS?
                    # Stop accepting new genes; continue until file end would be slow.
                    # Break early: partial last gene OK for smoke tests.
                    break
            else:
                spans[key]["start"] = min(spans[key]["start"], start)
                spans[key]["end"] = max(spans[key]["end"], end)
                if not spans[key]["gene_name"]:
                    spans[key]["gene_name"] = gene_name

    for (chrom, gene_id), info in spans.items():
        yield GeneCDS(
            gene_id=gene_id,
            gene_name=info["gene_name"] or gene_id,
            chrom=chrom,
            strand=info["strand"],
            cds_start=int(info["start"]),
            cds_end=int(info["end"]),
        )


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------
def ideal_window(
    gene: GeneCDS, chrom_len: int, flank: int = FLANK_BP
) -> tuple[int, int, bool]:
    """Return (start, end, is_large) 1-based inclusive, clipped to chromosome."""
    cds_len = gene.cds_length
    if cds_len > LARGE_GENE_BP:
        if gene.strand == "-":
            # 120k of gene toward 5' (higher genomic coord) + 10k downstream of TSS-end
            end = min(chrom_len, gene.cds_end + flank)
            start = max(1, gene.cds_end - LARGE_BODY_BP + 1)
        else:
            start = max(1, gene.cds_start - flank)
            end = min(chrom_len, gene.cds_start + LARGE_BODY_BP - 1)
        return start, end, True

    start = max(1, gene.cds_start - flank)
    end = min(chrom_len, gene.cds_end + flank)
    return start, end, False


def neighbour_trim(
    gene: GeneCDS,
    start: int,
    end: int,
    by_chrom: dict[str, list[GeneCDS]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Trim window at nearest neighbouring CDS corners; record overlaps."""
    records: list[dict[str, Any]] = []
    others = by_chrom.get(gene.chrom, [])
    new_start, new_end = start, end

    # lists are sorted by cds_start — binary-ish linear scan for nearest neighbours
    prev: GeneCDS | None = None
    nxt: GeneCDS | None = None
    for other in others:
        if other.gene_id == gene.gene_id and other.cds_start == gene.cds_start:
            continue
        if other.cds_end < gene.cds_start:
            if prev is None or other.cds_end > prev.cds_end:
                prev = other
        elif other.cds_start > gene.cds_end:
            if nxt is None or other.cds_start < nxt.cds_start:
                nxt = other
                break  # sorted → first downstream is nearest start
        else:
            # overlapping CDS bodies
            if other.cds_start < gene.cds_start:
                new_start = max(new_start, other.cds_end + 1)
            if other.cds_end > gene.cds_end:
                new_end = min(new_end, other.cds_start - 1)
            records.append(
                {
                    "Gene": gene.gene_name,
                    "Neighbour": other.gene_name,
                    "Chr": gene.chrom,
                    "Side": "overlap",
                    "Original_start": start,
                    "Original_end": end,
                    "Trimmed_start": new_start,
                    "Trimmed_end": new_end,
                    "Neighbour_start": other.cds_start,
                    "Neighbour_end": other.cds_end,
                    "Cut_from": "",
                }
            )

    if prev is not None and prev.cds_end >= new_start:
        trimmed_from = new_start
        new_start = prev.cds_end + 1
        records.append(
            {
                "Gene": gene.gene_name,
                "Neighbour": prev.gene_name,
                "Chr": gene.chrom,
                "Side": "upstream",
                "Original_start": start,
                "Original_end": end,
                "Trimmed_start": new_start,
                "Trimmed_end": new_end,
                "Neighbour_start": prev.cds_start,
                "Neighbour_end": prev.cds_end,
                "Cut_from": trimmed_from,
            }
        )
    if nxt is not None and nxt.cds_start <= new_end:
        trimmed_from = new_end
        new_end = nxt.cds_start - 1
        records.append(
            {
                "Gene": gene.gene_name,
                "Neighbour": nxt.gene_name,
                "Chr": gene.chrom,
                "Side": "downstream",
                "Original_start": start,
                "Original_end": end,
                "Trimmed_start": new_start,
                "Trimmed_end": new_end,
                "Neighbour_start": nxt.cds_start,
                "Neighbour_end": nxt.cds_end,
                "Cut_from": trimmed_from,
            }
        )

    if new_end < new_start:
        return start, end, records
    return new_start, new_end, records


def extract_forward(fasta: Fasta, chrom: str, start: int, end: int) -> str:
    """1-based inclusive → forward genomic sequence."""
    if chrom not in fasta:
        # try hard workarounds: strip version?
        raise KeyError(chrom)
    # pyfaidx is 1-based inclusive when using slice [start:end]
    return str(fasta[chrom][start - 1 : end])


def chrom_length(fasta: Fasta, chrom: str) -> int:
    return len(fasta[chrom])


# ---------------------------------------------------------------------------
# Non-coding matching
# ---------------------------------------------------------------------------
def occupied_intervals(windows: list[Window]) -> dict[str, list[tuple[int, int]]]:
    """Merge gene windows (+ original ±flank occupancy) per chrom as 1-based inclusive."""
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for w in windows:
        raw[w.chrom].append((w.start, w.end))
    merged: dict[str, list[tuple[int, int]]] = {}
    for chrom, ivals in raw.items():
        ivals.sort()
        out: list[tuple[int, int]] = []
        for s, e in ivals:
            if not out or s > out[-1][1] + 1:
                out.append((s, e))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], e))
        merged[chrom] = out
    return merged


def free_intervals(
    chrom_len: int, occupied: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Complement of occupied on [1, chrom_len]."""
    free: list[tuple[int, int]] = []
    cursor = 1
    for s, e in occupied:
        if s > cursor:
            free.append((cursor, s - 1))
        cursor = max(cursor, e + 1)
    if cursor <= chrom_len:
        free.append((cursor, chrom_len))
    return free


def gc_of_slice(prefix_gc: np.ndarray, prefix_atgc: np.ndarray, start: int, end: int) -> float:
    """prefix_* are cumulative over sequence[0..i). start/end 1-based inclusive."""
    a, b = start - 1, end
    gc = int(prefix_gc[b] - prefix_gc[a])
    atgc = int(prefix_atgc[b] - prefix_atgc[a])
    return (gc / atgc) if atgc else 0.0


def build_prefix(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """Compact uint32 prefix sums (avoid Python int lists on mammalian chroms)."""
    n = len(seq)
    arr = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    if arr.size != n:
        # Fallback if non-ascii slipped in
        arr = np.fromiter((ord(c) for c in seq), dtype=np.uint8, count=n)
    # A=65 C=67 G=71 T=84 N=78
    is_gc = (arr == ord("G")) | (arr == ord("C")) | (arr == ord("g")) | (arr == ord("c"))
    is_atgc = is_gc | (arr == ord("A")) | (arr == ord("T")) | (arr == ord("a")) | (arr == ord("t"))
    pg = np.zeros(n + 1, dtype=np.uint32)
    pa = np.zeros(n + 1, dtype=np.uint32)
    pg[1:] = np.cumsum(is_gc, dtype=np.uint32)
    pa[1:] = np.cumsum(is_atgc, dtype=np.uint32)
    return pg, pa


def match_noncoding(
    fasta: Fasta,
    gene_windows: list[Window],
    genome_id: str,
    seed: int = SEED,
    gc_tol: float = GC_TOL,
) -> list[Window]:
    """Sample intergenic windows matching gene length & GC distribution (1:1 greedy)."""
    if not gene_windows:
        return []

    occupied = occupied_intervals(gene_windows)
    by_chrom_targets: dict[str, list[Window]] = defaultdict(list)
    for w in gene_windows:
        by_chrom_targets[w.chrom].append(w)

    placed: list[Window] = []
    nc_idx = 0
    rng_step = max(1, seed % 97)

    for chrom in sorted(by_chrom_targets):
        if chrom not in fasta:
            continue
        targets = sorted(
            by_chrom_targets[chrom], key=lambda w: (-w.length, w.gene_or_id)
        )
        clen = len(fasta[chrom])
        free = free_intervals(clen, occupied.get(chrom, []))
        # One chrom at a time — keep memory bounded
        seq = str(fasta[chrom]).upper()
        pg, pa = build_prefix(seq)
        del seq

        def take_from_free(start: int, end: int) -> None:
            nonlocal free
            updated: list[tuple[int, int]] = []
            for fs, fe in free:
                if end < fs or start > fe:
                    updated.append((fs, fe))
                    continue
                if start > fs:
                    updated.append((fs, start - 1))
                if end < fe:
                    updated.append((end + 1, fe))
            free = updated

        for tw in targets:
            L = tw.length
            if L <= 0:
                continue
            best: tuple[float, int, int] | None = None
            stride = max(1, int(L * STRIDE_FRAC), rng_step)
            for fs, fe in list(free):
                if fe - fs + 1 < L:
                    continue
                span = fe - fs + 1 - L
                offset = (
                    seed * 31 + nc_idx * 17 + stable_hash(tw.gene_or_id) % 10007
                ) % (span + 1)
                starts = list(range(fs + offset, fe - L + 2, stride))
                if fs not in starts:
                    starts.insert(0, fs)
                for s in starts:
                    e = s + L - 1
                    if e > fe:
                        break
                    g = gc_of_slice(pg, pa, s, e)
                    diff = abs(g - tw.gc)
                    if diff <= gc_tol:
                        best = (diff, s, e)
                        break
                    if best is None or diff < best[0]:
                        best = (diff, s, e)
                if best is not None and best[0] <= gc_tol:
                    break
            if best is None:
                continue
            _, s, e = best
            dna = str(fasta[chrom][s - 1 : e]).upper()
            nc_idx += 1
            wid = f"NC_{genome_id}_{chrom}_{nc_idx:06d}"
            w = Window(
                genome=genome_id,
                gene_or_id=wid,
                chrom=chrom,
                start=s,
                end=e,
                strand="+",
                kind="non_coding",
                tpm=0.0,
            )
            w.finalize(dna)
            placed.append(w)
            take_from_free(s, e)

        del pg, pa

    return placed


# ---------------------------------------------------------------------------
# Per-genome processing
# ---------------------------------------------------------------------------
def process_genome(
    bundle: GenomeBundle,
    flank: int = FLANK_BP,
    max_genes: int | None = None,
    seed: int = SEED,
) -> tuple[list[Window], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assert bundle.tpm is not None
    if Fasta is None:  # pragma: no cover - environment-dependent dependency
        raise SystemExit("pyfaidx required: conda install -n caduceus_env pyfaidx")
    if np is None:  # pragma: no cover - environment-dependent dependency
        raise SystemExit("numpy required")
    tpm_map = load_tpm_row(bundle.tpm)
    fasta = Fasta(str(bundle.fasta), as_raw=True, sequence_always_upper=True)

    genes = list(iter_cds_genes(bundle.gtf, max_genes=max_genes))
    genes.sort(key=lambda g: (g.chrom, g.cds_start, g.gene_id))
    if max_genes is not None:
        genes = genes[:max_genes]
    print(f"  parsed_cds_genes={len(genes)}", flush=True)

    by_chrom: dict[str, list[GeneCDS]] = defaultdict(list)
    for g in genes:
        by_chrom[g.chrom].append(g)

    windows: list[Window] = []
    neighbours: list[dict[str, Any]] = []
    large_genes: list[dict[str, Any]] = []
    skipped = 0
    missing_chrom = 0
    empty_after_trim = 0

    for g in genes:
        if g.chrom not in fasta:
            missing_chrom += 1
            continue
        clen = len(fasta[g.chrom])
        start, end, is_large = ideal_window(g, clen, flank=flank)
        start2, end2, neigh = neighbour_trim(g, start, end, by_chrom)
        if neigh:
            neighbours.extend({**r, "Genome": bundle.genome_id} for r in neigh)
        if end2 < start2:
            empty_after_trim += 1
            continue
        try:
            seq = extract_forward(fasta, g.chrom, start2, end2)
        except Exception:
            skipped += 1
            continue
        if not seq or set(seq.upper()) <= {"N"}:
            skipped += 1
            continue

        tpm = tpm_map.get(g.gene_name, tpm_map.get(g.gene_id, 0.0))
        w = Window(
            genome=bundle.genome_id,
            gene_or_id=g.gene_name,
            chrom=g.chrom,
            start=start2,
            end=end2,
            strand=g.strand,
            kind="gene",
            tpm=float(tpm),
            large_gene=is_large,
            neighbour_trimmed=bool(neigh),
        )
        w.finalize(seq)
        windows.append(w)

        if is_large:
            large_genes.append(
                {
                    "Genome": bundle.genome_id,
                    "Gene": g.gene_name,
                    "Chr": g.chrom,
                    "CDS_start": g.cds_start,
                    "CDS_end": g.cds_end,
                    "CDS_length": g.cds_length,
                    "Window_start": start2,
                    "Window_end": end2,
                    "Strand": g.strand,
                }
            )

    nc_windows = match_noncoding(fasta, windows, bundle.genome_id, seed=seed)
    fasta.close()

    stats = {
        "genome": bundle.genome_id,
        "tpm_id": bundle.tpm_id,
        "n_cds_genes": len(genes),
        "n_gene_windows": len(windows),
        "n_noncoding_windows": len(nc_windows),
        "n_large_genes": len(large_genes),
        "n_neighbour_events": len(neighbours),
        "skipped": skipped,
        "missing_chrom": missing_chrom,
        "empty_after_trim": empty_after_trim,
        "gene_length_mean": _mean([w.length for w in windows]),
        "gene_gc_mean": _mean([w.gc for w in windows]),
        "nc_length_mean": _mean([w.length for w in nc_windows]),
        "nc_gc_mean": _mean([w.gc for w in nc_windows]),
    }
    return windows + nc_windows, neighbours, large_genes, stats


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def histogram(values: Iterable[float], bins: int = 20) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"bins": [], "counts": [], "n": 0}
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return {"bins": [lo, hi], "counts": [len(vals)], "n": len(vals), "min": lo, "max": hi}
    width = (hi - lo) / bins
    counts = [0] * bins
    edges = [lo + i * width for i in range(bins + 1)]
    for v in vals:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1
    return {
        "edges": edges,
        "counts": counts,
        "n": len(vals),
        "min": lo,
        "max": hi,
        "mean": _mean(vals),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_pipe_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(SEP.join(fields) + "\n")
        for r in rows:
            fh.write(SEP.join(str(r.get(c, "")) for c in fields) + "\n")


def write_ready_fna(path: Path, windows: list[Window]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for w in windows:
            hdr = SEP.join(
                [w.genome, w.gene_or_id, w.chrom, str(w.start), str(w.end)]
            )
            fh.write(f">{hdr}\n")
            seq = w.sequence
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def write_ready_csv(path: Path, windows: list[Window]) -> None:
    fields = ["Genome", "GeneOrID", "Chr", "Position_start", "Position_end", "TPM"]
    rows = [
        {
            "Genome": w.genome,
            "GeneOrID": w.gene_or_id,
            "Chr": w.chrom,
            "Position_start": w.start,
            "Position_end": w.end,
            "TPM": w.tpm,
        }
        for w in windows
    ]
    write_pipe_table(path, rows, fields)


def write_caduceus_ready(out: Path, windows: list[Window]) -> None:
    root = out / "caduceus_ready" / "all"
    seq_dir = root / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, Any]] = []
    for w in windows:
        sid = sanitize_id(
            f"{w.genome}_{w.gene_or_id}_{w.chrom}_{w.start}_{w.end}"
        )
        fn = seq_dir / f"{sid}.txt"
        fn.write_text(w.sequence + "\n", encoding="utf-8")
        labels.append(
            {
                "sample_id": sid,
                "path": f"all/sequences/{sid}.txt",
                "TPM": w.tpm,
                "genome": w.genome,
                "gene_id": w.gene_or_id,
                "window_length": w.length,
                "strand": w.strand,
                "kind": w.kind,
            }
        )
    # TSV for trainer
    with (root / "labels.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "path",
                "TPM",
                "genome",
                "gene_id",
                "window_length",
                "strand",
                "kind",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(labels)
    (out / "caduceus_ready" / "README.md").write_text(
        "\n".join(
            [
                "# Caduceus-ready export (`data_ready`)",
                "",
                "Continuous TPM labels; DNA windows from `src/preprocessing.py`.",
                "",
                "```",
                "caduceus_ready/all/sequences/<sample_id>.txt",
                "caduceus_ready/all/labels.tsv",
                "```",
                "",
                "Non-coding samples have TPM=0. Folds are not assigned here (`@split`).",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    raw_dir: Path,
    out_dir: Path,
    flank: int = FLANK_BP,
    seed: int = SEED,
    genomes: list[str] | None = None,
    max_genes: int | None = None,
    tpm_merged_only: bool = False,
) -> int:
    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles, audit_notes = discover_raw(raw_dir, tpm_merged_only=tpm_merged_only)
    for note in audit_notes:
        print(f"AUDIT: {note}", flush=True)
    if genomes:
        want = set(genomes)
        bundles = [b for b in bundles if b.genome_id in want]
        if not bundles:
            print(f"ERROR: no bundles match --genomes {genomes}", file=sys.stderr)
            return 2

    if tpm_merged_only:
        bad = [b for b in bundles if not (b.tpm_id or "").endswith("_merged")]
        if bad:
            print(
                "ERROR: --tpm-merged-only but non-merged TPM selected: "
                + ", ".join(f"{b.genome_id}:{b.tpm_id}" for b in bad),
                file=sys.stderr,
            )
            return 2

    all_windows: list[Window] = []
    all_neighbours: list[dict[str, Any]] = []
    all_large: list[dict[str, Any]] = []
    per_genome: list[dict[str, Any]] = []

    for i, bundle in enumerate(bundles, 1):
        print(
            f"[{i}/{len(bundles)}] {bundle.genome_id} "
            f"fasta={bundle.fasta.name} gtf={bundle.gtf.name} tpm={bundle.tpm_id}",
            flush=True,
        )
        wins, neigh, large, stats = process_genome(
            bundle, flank=flank, max_genes=max_genes, seed=seed
        )
        all_windows.extend(wins)
        all_neighbours.extend(neigh)
        all_large.extend(large)
        per_genome.append(stats)
        print(
            f"  gene_windows={stats['n_gene_windows']} "
            f"noncoding={stats['n_noncoding_windows']} "
            f"large={stats['n_large_genes']} neighbours={stats['n_neighbour_events']}",
            flush=True,
        )

    if not all_windows:
        print("ERROR: zero windows produced — aborting.", file=sys.stderr)
        return 2

    gene_wins = [w for w in all_windows if w.kind == "gene"]
    nc_wins = [w for w in all_windows if w.kind == "non_coding"]

    # non_coding.csv: gene properties THEN non-coding samples (unified columns)
    nc_table_rows = [
        {
            "GeneOrID": w.gene_or_id,
            "Chr": w.chrom,
            "Position_start": w.start,
            "Position_end": w.end,
            "Length": w.length,
            "GC": f"{w.gc:.6f}",
            "kind": w.kind,
            "Genome": w.genome,
        }
        for w in gene_wins + nc_wins
    ]
    write_pipe_table(
        out_dir / "non_coding.csv",
        nc_table_rows,
        ["GeneOrID", "Chr", "Position_start", "Position_end", "Length", "GC", "kind", "Genome"],
    )
    write_pipe_table(
        out_dir / "neighbours.csv",
        all_neighbours,
        [
            "Genome",
            "Gene",
            "Neighbour",
            "Chr",
            "Side",
            "Original_start",
            "Original_end",
            "Trimmed_start",
            "Trimmed_end",
            "Neighbour_start",
            "Neighbour_end",
            "Cut_from",
        ],
    )
    write_pipe_table(
        out_dir / "large_genes.csv",
        all_large,
        [
            "Genome",
            "Gene",
            "Chr",
            "CDS_start",
            "CDS_end",
            "CDS_length",
            "Window_start",
            "Window_end",
            "Strand",
        ],
    )

    write_ready_fna(out_dir / "ready.fna", all_windows)
    write_ready_csv(out_dir / "ready.csv", all_windows)
    write_caduceus_ready(out_dir, all_windows)

    stats_doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "flank_bp": flank,
        "large_gene_bp": LARGE_GENE_BP,
        "large_body_bp": LARGE_BODY_BP,
        "seed": seed,
        "n_genomes": len(bundles),
        "n_gene_windows": len(gene_wins),
        "n_noncoding_windows": len(nc_wins),
        "n_total_windows": len(all_windows),
        "n_neighbour_events": len(all_neighbours),
        "n_large_genes": len(all_large),
        "audit_notes": audit_notes,
        "per_genome": per_genome,
        "gene_length_distribution": histogram(w.length for w in gene_wins),
        "gene_gc_distribution": histogram(w.gc for w in gene_wins),
        "noncoding_length_distribution": histogram(w.length for w in nc_wins),
        "noncoding_gc_distribution": histogram(w.gc for w in nc_wins),
        "software": {
            "python": sys.version.split()[0],
            "script": str(Path(__file__).resolve()),
        },
    }
    (out_dir / "statistics.json").write_text(
        json.dumps(stats_doc, indent=2), encoding="utf-8"
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tool": "src/preprocessing.py",
                "created_at": stats_doc["created_at"],
                "bundles": [
                    {
                        "genome": b.genome_id,
                        "fasta": str(b.fasta),
                        "gtf": str(b.gtf),
                        "tpm": str(b.tpm),
                        "tpm_id": b.tpm_id,
                    }
                    for b in bundles
                ],
                "assumptions": [
                    "Windows centered on CDS span (min–max CDS), not full gene/transcript",
                    f"Default flank ±{flank} bp; neighbour-trimmed at adjacent CDS corners",
                    f"CDS length >{LARGE_GENE_BP}: keep 10 kb before strand-aware start + {LARGE_BODY_BP} bp of CDS",
                    "Forward genomic sequence only (no RC export)",
                    "Non-coding TPM = 0; matched to gene length & GC via greedy intergenic placement",
                    "Missing TPM file for a mapped genome → genome skipped (not invented)",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# data_ready",
                "",
                "Produced by `src/preprocessing.py` from `raw/`.",
                "",
                "| File | Content |",
                "|------|---------|",
                "| `ready.fna` | DNA windows; header `>Genome\\|GeneOrID\\|Chr\\|start\\|end` |",
                "| `ready.csv` | `Genome\\|GeneOrID\\|Chr\\|start\\|end\\|TPM` |",
                "| `non_coding.csv` | Gene + non-coding Length/GC table |",
                "| `neighbours.csv` | Neighbour-trim events |",
                "| `large_genes.csv` | CDS >130 kb crops |",
                "| `caduceus_ready/` | Per-sample `.txt` + `labels.tsv` |",
                "| `statistics.json` | Length/GC distributions + per-genome counts |",
                "",
                "See `wiki/conversion.md`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {out_dir}: genes={len(gene_wins)} noncoding={len(nc_wins)} "
        f"total={len(all_windows)}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("raw"), help="Raw input root")
    ap.add_argument("--out", type=Path, default=Path("data_ready"), help="Output directory")
    ap.add_argument("--flank", type=int, default=FLANK_BP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--genomes",
        nargs="*",
        default=None,
        help="Optional GCF accession filter(s), e.g. GCF_000001405.40",
    )
    ap.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Cap CDS genes per genome (smoke tests)",
    )
    ap.add_argument(
        "--tpm-merged-only",
        action="store_true",
        help="Use only prokaryotes/tpm/{assembly}_merged.csv (ignore per-sample GEO CSVs)",
    )
    args = ap.parse_args(argv)
    return run(
        raw_dir=args.raw,
        out_dir=args.out,
        flank=args.flank,
        seed=args.seed,
        genomes=args.genomes,
        max_genes=args.max_genes,
        tpm_merged_only=args.tpm_merged_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
