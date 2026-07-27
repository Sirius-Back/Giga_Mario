#!/usr/bin/env python3
"""raw/{fna,gtf,tpm} or BED+FASTA → human_legnet-ready 230 bp promoters.

Pipeline (Tentative 2026-07-27):
  1. Pair genomes via raw/{fna,gtf,tpm} + mapping CSV (reuse @adapt discovery)
  2. Strand-aware TSS from GTF gene features → 200 bp CRS centered on TSS
  3. Write promoters.bed; extract gene-oriented DNA; stitch lentiMPRA adapters
  4. Join continuous TPM; write human_legnet TSV (seq_id, seq, mean_value, fold, rev)

Does NOT assign project train/val/test folds (@split). fold column =
(hash % 10) + 1 → 1..10 for human_legnet CV compatibility only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from pyfaidx import Fasta
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyfaidx required: conda install -n legnet pyfaidx") from exc

# Reuse @adapt pairing / TPM loaders
from src.preprocessing import (
    discover_raw,
    genome_prefix,
    load_tpm_row,
    open_text,
    parse_attrs,
    sanitize_id,
    stable_hash,
)

# ---------------------------------------------------------------------------
# Locked defaults (human_legnet / lentiMPRA)
# ---------------------------------------------------------------------------
CRS_BP = 200
ADAPTER_5 = "AGGACCGGATCAACT"  # 15 bp
ADAPTER_3 = "CATTGCGTGAACCGA"  # 15 bp
STITCHED_LEN = len(ADAPTER_5) + CRS_BP + len(ADAPTER_3)  # 230
SEED = 42
COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


@dataclass
class GeneTSS:
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    tss: int  # 1-based genomic TSS coordinate


@dataclass
class PromoterRecord:
    genome: str
    gene_or_id: str
    chrom: str
    start0: int  # 0-based half-open genomic CRS start
    end0: int  # 0-based half-open genomic CRS end
    strand: str
    tpm: float
    crs: str = ""
    stitched: str = ""
    seq_id: str = ""
    fold: int = 0
    notes: list[str] = field(default_factory=list)


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def fold_for_seq(seq: str, seed: int = SEED) -> int:
    """Deterministic 1–10 fold for human_legnet TSV compatibility (not @split)."""
    return (stable_hash(f"{seed}:{seq}") % 10) + 1


def iter_gene_tss(gtf_path: Path, max_genes: int | None = None) -> Iterator[GeneTSS]:
    """Yield one GeneTSS per GTF gene feature (strand-aware TSS)."""
    seen: set[tuple[str, str]] = set()
    n = 0
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
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
            strand = strand if strand in "+-" else "+"
            start_1, end_1 = int(start_s), int(end_s)
            tss = start_1 if strand == "+" else end_1
            key = (chrom, gene_id)
            if key in seen:
                continue
            seen.add(key)
            yield GeneTSS(
                gene_id=gene_id,
                gene_name=gene_name,
                chrom=chrom,
                strand=strand,
                tss=tss,
            )
            n += 1
            if max_genes is not None and n >= max_genes:
                return


def tss_crs_window(
    tss_1based: int, chrom_len: int, crs_bp: int = CRS_BP
) -> tuple[int, int] | None:
    """Return 0-based half-open [start, end) of length crs_bp centered on TSS, or None."""
    if crs_bp % 2 != 0:
        raise ValueError(f"crs_bp must be even (got {crs_bp})")
    half = crs_bp // 2
    # TSS is 1-based; center so genomic interval length == crs_bp
    # Inclusive 1-based center span: [tss-half+1, tss+half] → 0-based [tss-half, tss+half)
    start0 = tss_1based - half
    end0 = tss_1based + half
    if start0 < 0 or end0 > chrom_len:
        return None
    if end0 - start0 != crs_bp:
        return None
    return start0, end0


def extract_crs(
    fasta: Fasta, chrom: str, start0: int, end0: int, strand: str
) -> str:
    """Extract CRS in gene orientation (RC if strand '-')."""
    if chrom not in fasta:
        raise KeyError(f"Chromosome {chrom!r} not in FASTA")
    seq = str(fasta[chrom][start0:end0]).upper()
    if strand == "-":
        seq = revcomp(seq)
    return seq


def stitch_adapters(crs: str, adapter5: str = ADAPTER_5, adapter3: str = ADAPTER_3) -> str:
    if len(crs) != CRS_BP:
        raise ValueError(f"CRS must be {CRS_BP} bp for adapter stitch (got {len(crs)})")
    out = adapter5 + crs + adapter3
    expected = len(adapter5) + CRS_BP + len(adapter3)
    if len(out) != expected:
        raise ValueError(f"adapter stitch length {len(out)} != {expected}")
    return out


def parse_bed_rows(bed_path: Path) -> list[dict[str, Any]]:
    """Parse BED6+ (chrom start end name score strand). start/end 0-based half-open."""
    rows: list[dict[str, Any]] = []
    opener = gzip.open if str(bed_path).endswith(".gz") else open
    with opener(bed_path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[arg-type]
        for lineno, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{bed_path}:{lineno}: need ≥3 BED columns")
            chrom = parts[0]
            start0 = int(parts[1])
            end0 = int(parts[2])
            name = parts[3] if len(parts) > 3 and parts[3] not in (".", "") else f"{chrom}:{start0}-{end0}"
            score_raw = parts[4] if len(parts) > 4 else "0"
            try:
                score = float(score_raw) if score_raw not in (".", "") else 0.0
            except ValueError:
                score = 0.0
            strand = parts[5] if len(parts) > 5 and parts[5] in "+-" else "+"
            if end0 <= start0:
                raise ValueError(f"{bed_path}:{lineno}: end <= start")
            rows.append(
                {
                    "chrom": chrom,
                    "start0": start0,
                    "end0": end0,
                    "name": name,
                    "score": score,
                    "strand": strand,
                }
            )
    if not rows:
        raise ValueError(f"No BED intervals in {bed_path}")
    return rows


def write_bed(path: Path, records: list[PromoterRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        for r in records:
            # BED score is typically 0–1000; store TPM as float string in score col
            fh.write(
                f"{r.chrom}\t{r.start0}\t{r.end0}\t{r.gene_or_id}\t{r.tpm}\t{r.strand}\n"
            )


def write_tsv(path: Path, records: list[PromoterRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["seq_id", "seq", "mean_value", "fold", "rev"])
        for r in records:
            w.writerow([r.seq_id, r.stitched, f"{r.tpm}", r.fold, 0])


def write_fasta(path: Path, records: list[PromoterRecord]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(f">{r.seq_id}\n{r.stitched}\n")


def build_from_raw(
    raw_dir: Path,
    out_dir: Path,
    *,
    crs_bp: int = CRS_BP,
    seed: int = SEED,
    genomes: list[str] | None = None,
    max_genes: int | None = None,
    stitch: bool = True,
    write_fa: bool = True,
) -> dict[str, Any]:
    if crs_bp != CRS_BP:
        raise ValueError(
            f"human_legnet CRS is locked at {CRS_BP} bp (got {crs_bp}); "
            "adapter stitch expects 200 bp insert"
        )
    if len(ADAPTER_5) + crs_bp + len(ADAPTER_3) != STITCHED_LEN:
        raise ValueError("adapter constants inconsistent with STITCHED_LEN")

    bundles, notes = discover_raw(raw_dir)
    if genomes:
        want = set(genomes)
        bundles = [b for b in bundles if b.genome_id in want]
        if not bundles:
            raise FileNotFoundError(f"No bundles match --genomes {genomes}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[PromoterRecord] = []
    per_genome: dict[str, list[PromoterRecord]] = {}
    skip_counts: dict[str, int] = {
        "no_tpm_symbol": 0,
        "crs_out_of_bounds": 0,
        "bad_crs_len": 0,
        "chrom_missing": 0,
        "bad_stitch_len": 0,
    }

    for bundle in bundles:
        assert bundle.tpm is not None
        tpm_map = load_tpm_row(bundle.tpm)
        fasta = Fasta(str(bundle.fasta), as_raw=True, sequence_always_upper=True)
        records: list[PromoterRecord] = []
        for gene in iter_gene_tss(bundle.gtf, max_genes=max_genes):
            symbol = gene.gene_name
            if symbol not in tpm_map and gene.gene_id not in tpm_map:
                skip_counts["no_tpm_symbol"] += 1
                continue
            tpm = tpm_map.get(symbol, tpm_map.get(gene.gene_id, 0.0))
            if gene.chrom not in fasta:
                skip_counts["chrom_missing"] += 1
                continue
            chrom_len = len(fasta[gene.chrom])
            win = tss_crs_window(gene.tss, chrom_len, crs_bp=crs_bp)
            if win is None:
                skip_counts["crs_out_of_bounds"] += 1
                continue
            start0, end0 = win
            try:
                crs = extract_crs(fasta, gene.chrom, start0, end0, gene.strand)
            except KeyError:
                skip_counts["chrom_missing"] += 1
                continue
            if len(crs) != crs_bp:
                skip_counts["bad_crs_len"] += 1
                continue
            if stitch:
                stitched = stitch_adapters(crs)
                if len(stitched) != STITCHED_LEN:
                    skip_counts["bad_stitch_len"] += 1
                    continue
            else:
                stitched = crs
            gene_label = sanitize_id(symbol)
            seq_id = f"{bundle.genome_id}|{gene_label}|{gene.chrom}|{start0}|{end0}"
            rec = PromoterRecord(
                genome=bundle.genome_id,
                gene_or_id=symbol,
                chrom=gene.chrom,
                start0=start0,
                end0=end0,
                strand=gene.strand,
                tpm=float(tpm),
                crs=crs,
                stitched=stitched,
                seq_id=seq_id,
                fold=fold_for_seq(stitched, seed=seed),
            )
            records.append(rec)
        fasta.close()
        per_genome[bundle.genome_id] = records
        all_records.extend(records)
        notes.append(f"{bundle.genome_id}: kept={len(records)} tpm={bundle.tpm_id}")

    if not all_records:
        raise RuntimeError(
            "No promoter records produced. "
            f"skips={skip_counts}; notes={notes[:20]}"
        )

    bed_path = out_dir / "promoters.bed"
    write_bed(bed_path, all_records)
    write_tsv(out_dir / "all.tsv", all_records)
    for gid, recs in per_genome.items():
        if recs:
            write_tsv(out_dir / f"{gid}.tsv", recs)
    if write_fa:
        write_fasta(out_dir / "sequences.fa", all_records)

    stats = {
        "n_records": len(all_records),
        "n_genomes": len(per_genome),
        "per_genome": {g: len(r) for g, r in per_genome.items()},
        "skips": skip_counts,
        "crs_bp": crs_bp,
        "stitched_len": STITCHED_LEN if stitch else crs_bp,
        "adapter_5": ADAPTER_5,
        "adapter_3": ADAPTER_3,
        "seed": seed,
        "notes": notes,
    }
    meta = {
        "producer": "src/legnet_preprocess.py",
        "skill": "legnet-adapt",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw": str(raw_dir),
        "out": str(out_dir),
        "human_legnet_format": True,
        "columns": ["seq_id", "seq", "mean_value", "fold", "rev"],
        "fold_note": "(hash%10)+1 → 1..10 for human_legnet CV only; not project @split",
    }
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return stats


def build_from_bed(
    bed_path: Path,
    fasta_path: Path,
    out_dir: Path,
    *,
    genome_id: str | None = None,
    crs_bp: int = CRS_BP,
    seed: int = SEED,
    stitch: bool = True,
    write_fa: bool = True,
) -> dict[str, Any]:
    if not bed_path.is_file():
        raise FileNotFoundError(f"BED missing: {bed_path}")
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA missing: {fasta_path}")
    rows = parse_bed_rows(bed_path)
    gid = genome_id or genome_prefix(fasta_path.name)
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[PromoterRecord] = []
    skip_counts = {"bad_len": 0, "chrom_missing": 0, "bad_stitch_len": 0}
    for row in rows:
        chrom = row["chrom"]
        start0, end0 = int(row["start0"]), int(row["end0"])
        span = end0 - start0
        if chrom not in fasta:
            skip_counts["chrom_missing"] += 1
            continue
        if stitch and span != crs_bp:
            skip_counts["bad_len"] += 1
            continue
        try:
            crs = extract_crs(fasta, chrom, start0, end0, row["strand"])
        except KeyError:
            skip_counts["chrom_missing"] += 1
            continue
        if stitch:
            if len(crs) != crs_bp:
                skip_counts["bad_len"] += 1
                continue
            stitched = stitch_adapters(crs)
            if len(stitched) != STITCHED_LEN:
                skip_counts["bad_stitch_len"] += 1
                continue
        else:
            stitched = crs
        name = sanitize_id(str(row["name"]))
        seq_id = f"{gid}|{name}|{chrom}|{start0}|{end0}"
        records.append(
            PromoterRecord(
                genome=gid,
                gene_or_id=str(row["name"]),
                chrom=chrom,
                start0=start0,
                end0=end0,
                strand=row["strand"],
                tpm=float(row["score"]),
                crs=crs if stitch else stitched,
                stitched=stitched,
                seq_id=seq_id,
                fold=fold_for_seq(stitched, seed=seed),
            )
        )
    fasta.close()

    if not records:
        raise RuntimeError(f"No records from BED {bed_path}; skips={skip_counts}")

    write_bed(out_dir / "promoters.bed", records)
    write_tsv(out_dir / "all.tsv", records)
    write_tsv(out_dir / f"{gid}.tsv", records)
    if write_fa:
        write_fasta(out_dir / "sequences.fa", records)

    stats = {
        "n_records": len(records),
        "genome": gid,
        "skips": skip_counts,
        "crs_bp": crs_bp,
        "stitched_len": STITCHED_LEN if stitch else None,
        "adapter_5": ADAPTER_5,
        "adapter_3": ADAPTER_3,
        "seed": seed,
        "mode": "bed",
        "bed": str(bed_path),
        "fasta": str(fasta_path),
    }
    meta = {
        "producer": "src/legnet_preprocess.py",
        "skill": "legnet-adapt",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "bed",
        "human_legnet_format": True,
    }
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=None, help="Raw root with fna/gtf/tpm")
    ap.add_argument("--out", type=Path, default=Path("legnet_ready"), help="Output directory")
    ap.add_argument("--crs-bp", type=int, default=CRS_BP, help="CRS length (locked 200)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--genomes", nargs="*", default=None, help="Optional GCF filter(s)")
    ap.add_argument("--max-genes", type=int, default=None, help="Smoke-test gene cap per genome")
    ap.add_argument("--bed", type=Path, default=None, help="Parse existing BED (with --fasta)")
    ap.add_argument("--fasta", type=Path, default=None, help="Reference FASTA for --bed mode")
    ap.add_argument("--genome-id", type=str, default=None, help="Optional genome id for --bed mode")
    ap.add_argument(
        "--stitch-adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap CRS with lentiMPRA adapters to 230 bp (default: on)",
    )
    ap.add_argument("--no-fasta-out", action="store_true", help="Skip sequences.fa")
    args = ap.parse_args(argv)

    if args.bed is not None:
        if args.fasta is None:
            raise SystemExit("--bed requires --fasta")
        stats = build_from_bed(
            args.bed,
            args.fasta,
            args.out,
            genome_id=args.genome_id,
            crs_bp=args.crs_bp,
            seed=args.seed,
            stitch=args.stitch_adapters,
            write_fa=not args.no_fasta_out,
        )
    else:
        raw = args.raw or Path("raw")
        if not raw.is_dir():
            raise SystemExit(f"Raw directory missing: {raw}")
        stats = build_from_raw(
            raw,
            args.out,
            crs_bp=args.crs_bp,
            seed=args.seed,
            genomes=args.genomes,
            max_genes=args.max_genes,
            stitch=args.stitch_adapters,
            write_fa=not args.no_fasta_out,
        )

    print(
        f"Wrote {args.out}: n_records={stats.get('n_records')} "
        f"stitched_len={stats.get('stitched_len')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
