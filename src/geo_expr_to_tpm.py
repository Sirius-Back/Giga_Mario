#!/usr/bin/env python3
"""Convert GEO htseq counts / FPKM tables to wide TPM CSVs for the eukaryotic panel.

Reuses length-normalization helpers from ``src.acquire_prokaryotes_tpm``.
Gene keys written for Caduceus-prep join are RefSeq GTF ``gene`` symbols
(or unique GeneID→symbol via ``db_xref``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

from src.acquire_prokaryotes_tpm import counts_to_tpm, renormalize_tpm, write_wide_csv
from src.summarize_geo import write_wide_tpm

_GENE_ATTR = re.compile(r'(\w+) "([^"]*)"')
_GENEID_XREF = re.compile(r'GeneID:(\d+)')


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def parse_ensembl_gene_table(gtf: Path) -> dict[str, dict]:
    """ENSECAG… → {gene_name, length} from Ensembl gene features."""
    out: dict[str, dict] = {}
    with _open_text(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = dict(_GENE_ATTR.findall(parts[8]))
            gid = attrs.get("gene_id")
            if not gid:
                continue
            start, end = int(parts[3]), int(parts[4])
            out[gid] = {
                "gene_name": attrs.get("gene_name") or attrs.get("gene") or gid,
                "length": end - start + 1,
            }
    if not out:
        raise ValueError(f"No Ensembl genes parsed from {gtf}")
    return out


def parse_refseq_gene_tables(gtf: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Return (GeneID→unique gene symbol, gene_symbol→length) from RefSeq GTF."""
    geneid_to_symbols: dict[str, set[str]] = {}
    symbol_lengths: dict[str, int] = {}
    with _open_text(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = dict(_GENE_ATTR.findall(parts[8]))
            symbol = attrs.get("gene") or attrs.get("gene_name") or attrs.get("gene_id")
            if not symbol:
                continue
            start, end = int(parts[3]), int(parts[4])
            length = end - start + 1
            # Prefer longer span if duplicate symbols (rare)
            if symbol not in symbol_lengths or length > symbol_lengths[symbol]:
                symbol_lengths[symbol] = length
            for m in _GENEID_XREF.finditer(parts[8]):
                geneid_to_symbols.setdefault(m.group(1), set()).add(symbol)
    geneid_to_symbol = {
        gid: next(iter(syms))
        for gid, syms in geneid_to_symbols.items()
        if len(syms) == 1
    }
    if not symbol_lengths:
        raise ValueError(f"No RefSeq genes parsed from {gtf}")
    return geneid_to_symbol, symbol_lengths


def read_htseq_counts(path: Path) -> dict[str, float]:
    counts: dict[str, float] = {}
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("__"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            gene, val = parts[0], parts[1]
            if gene.startswith("__"):
                continue
            counts[gene] = float(val)
    if not counts:
        raise ValueError(f"No counts in {path}")
    return counts


def horse_counts_to_symbol_tpm(
    counts_path: Path,
    ensembl_gtf: Path,
    refseq_gtf: Path,
) -> dict[str, float]:
    """htseq ENSECAG counts → TPM keyed by RefSeq gene symbols (name intersection)."""
    ens = parse_ensembl_gene_table(ensembl_gtf)
    counts = read_htseq_counts(counts_path)
    lengths = {gid: info["length"] for gid, info in ens.items()}
    tpm_by_ens = counts_to_tpm(counts, lengths)

    # Collapse Ensembl gene_id TPM onto unique gene_name
    by_name: dict[str, float] = {}
    name_counts: dict[str, int] = {}
    for gid, tpm in tpm_by_ens.items():
        name = ens[gid]["gene_name"]
        if not name:
            continue
        by_name[name] = by_name.get(name, 0.0) + tpm
        name_counts[name] = name_counts.get(name, 0) + 1

    _, ref_lengths = parse_refseq_gene_tables(refseq_gtf)
    joined = {
        name: val
        for name, val in by_name.items()
        if name in ref_lengths and name_counts.get(name, 0) == 1
    }
    if len(joined) < 1000:
        raise ValueError(
            f"Horse symbol join too small ({len(joined)}); check Ensembl↔RefSeq overlap"
        )
    return renormalize_tpm(joined)


def fpkm_column_to_symbol_tpm(
    fpkm_path: Path,
    sample_column: str,
    refseq_gtf: Path,
    *,
    id_column: str = "gene_id",
) -> dict[str, float]:
    """GEO FPKM matrix column (NCBI GeneID rows) → TPM by RefSeq gene symbol."""
    geneid_to_symbol, _ = parse_refseq_gene_tables(refseq_gtf)
    with _open_text(fpkm_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in {fpkm_path}")
        if sample_column not in reader.fieldnames:
            raise ValueError(
                f"Column {sample_column!r} missing; have {reader.fieldnames[:12]}"
            )
        if id_column not in reader.fieldnames:
            raise ValueError(f"ID column {id_column!r} missing in {fpkm_path}")
        fpkm_by_symbol: dict[str, float] = {}
        n_rows = 0
        n_mapped = 0
        for row in reader:
            n_rows += 1
            gid = (row.get(id_column) or "").strip()
            raw = (row.get(sample_column) or "").strip()
            if not gid or raw in {"", "NA", "na", "None"}:
                continue
            symbol = geneid_to_symbol.get(gid)
            if not symbol:
                continue
            n_mapped += 1
            fpkm_by_symbol[symbol] = fpkm_by_symbol.get(symbol, 0.0) + float(raw)
    if n_mapped < 1000:
        raise ValueError(
            f"Goat GeneID→symbol join too small ({n_mapped}/{n_rows}); check GTF GeneID xrefs"
        )
    # FPKM and TPM are proportional given fixed lengths within a sample
    return renormalize_tpm(fpkm_by_symbol)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("horse", help="GSM7084192 htseq counts → SRX19584896.csv")
    h.add_argument("--counts", type=Path, required=True)
    h.add_argument("--ensembl-gtf", type=Path, required=True)
    h.add_argument("--refseq-gtf", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)

    g = sub.add_parser("goat", help="GSE135692 FPKM column → TPM CSV")
    g.add_argument("--fpkm", type=Path, required=True)
    g.add_argument("--sample-column", required=True)
    g.add_argument("--refseq-gtf", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)

    args = p.parse_args(argv)
    if args.cmd == "horse":
        for path, label in [
            (args.counts, "counts"),
            (args.ensembl_gtf, "ensembl-gtf"),
            (args.refseq_gtf, "refseq-gtf"),
        ]:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing {label}: {path}")
        tpm = horse_counts_to_symbol_tpm(args.counts, args.ensembl_gtf, args.refseq_gtf)
        write_wide_tpm(args.out, tpm)
        print(f"Wrote {args.out} genes={len(tpm)} sum={sum(tpm.values()):.3f}")
        return 0

    if args.cmd == "goat":
        for path, label in [(args.fpkm, "fpkm"), (args.refseq_gtf, "refseq-gtf")]:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing {label}: {path}")
        tpm = fpkm_column_to_symbol_tpm(
            args.fpkm, args.sample_column, args.refseq_gtf
        )
        write_wide_csv(args.out, tpm)
        print(f"Wrote {args.out} genes={len(tpm)} sum={sum(tpm.values()):.3f}")
        return 0

    raise SystemExit(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
