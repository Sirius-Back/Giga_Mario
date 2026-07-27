#!/usr/bin/env python3
"""Summarize GEO-aligned wide TPM CSVs (mean across samples) and GTF gene↔transcript maps.

Used by `@summarize_GEO`. Wide TPM convention: header = gene_id, one numeric row.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROK = ROOT / "prokaryotes"
DEFAULT_MAPPINGS = DEFAULT_PROK / "expr_file_mappings.csv"


def read_wide_tpm(path: Path) -> dict[str, float]:
    """Read a wide TPM CSV (header=gene_id, one data row) → {gene_id: tpm}."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Empty or missing TPM CSV: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"TPM CSV needs header + data row: {path}")
    header, values = rows[0], rows[1]
    if len(header) != len(values):
        raise ValueError(
            f"Header/value length mismatch in {path}: {len(header)} vs {len(values)}"
        )
    out: dict[str, float] = {}
    for g, v in zip(header, values):
        if not g:
            continue
        out[g] = float(v)
    if not out:
        raise ValueError(f"No genes in {path}")
    return out


def write_wide_tpm(path: Path, tpm: dict[str, float]) -> None:
    """Write wide TPM CSV with genes sorted by id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(tpm.keys())
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        w.writerow([f"{tpm[k]:.6f}" for k in keys])


def renormalize_tpm(tpm: dict[str, float]) -> dict[str, float]:
    s = sum(tpm.values())
    if s <= 0:
        return dict(tpm)
    return {k: v / s * 1e6 for k, v in tpm.items()}


def mean_merge_tpm_dicts(
    matrices: Iterable[dict[str, float]],
    *,
    how: str = "intersection",
    renormalize: bool = True,
) -> dict[str, float]:
    """Element-wise mean of wide TPM dicts.

    how:
      - intersection: genes present in every matrix (default; safe for matched panels)
      - union: genes in any matrix; missing treated as 0 before mean
    """
    mats = list(matrices)
    if not mats:
        raise ValueError("No TPM matrices to merge")
    if how not in {"intersection", "union"}:
        raise ValueError(f"Unknown how={how!r}")

    if how == "intersection":
        genes = set(mats[0])
        for m in mats[1:]:
            genes &= set(m)
        if not genes:
            raise ValueError("Empty gene intersection across TPM matrices")
        ordered = sorted(genes)
        means = {
            g: sum(m[g] for m in mats) / len(mats)
            for g in ordered
        }
    else:
        genes = set()
        for m in mats:
            genes |= set(m)
        ordered = sorted(genes)
        means = {
            g: sum(m.get(g, 0.0) for m in mats) / len(mats)
            for g in ordered
        }

    return renormalize_tpm(means) if renormalize else means


def mean_merge_tpm_csvs(
    paths: Iterable[Path | str],
    out_path: Path | str,
    *,
    how: str = "intersection",
    renormalize: bool = True,
) -> dict[str, float]:
    """Load wide TPM CSVs, mean-merge, write `out_path`, return merged dict."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No input TPM paths")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    mats = [read_wide_tpm(p) for p in paths]
    merged = mean_merge_tpm_dicts(mats, how=how, renormalize=renormalize)
    write_wide_tpm(Path(out_path), merged)
    return merged


def load_assembly_tpm_groups(
    mappings_csv: Path | str,
    *,
    root: Path | str | None = None,
) -> dict[str, list[Path]]:
    """Group per-sample TPM paths by `genome_stem` from expr_file_mappings.csv."""
    mappings_csv = Path(mappings_csv)
    root = Path(root) if root else mappings_csv.parent
    if not mappings_csv.is_file():
        raise FileNotFoundError(mappings_csv)
    groups: dict[str, list[Path]] = defaultdict(list)
    with mappings_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("genome_stem") or row.get("genome")
            tpm = row.get("tpm")
            if not stem or not tpm:
                raise ValueError(f"Mappings row missing genome_stem/tpm: {row}")
            p = Path(tpm)
            if not p.is_absolute():
                # Prefer project-relative path, then tpm-dir basename
                candidates = [
                    ROOT / p,
                    root / p,
                    root / "tpm" / p.name,
                    root / p.name,
                ]
                p = next((c for c in candidates if c.is_file()), ROOT / p)
            groups[stem].append(p)
    if not groups:
        raise ValueError(f"No assemblies in {mappings_csv}")
    return dict(groups)


def summarize_assemblies(
    mappings_csv: Path | str,
    out_dir: Path | str,
    *,
    root: Path | str | None = None,
    how: str = "intersection",
    renormalize: bool = True,
    suffix: str = "_merged.csv",
) -> dict[str, Path]:
    """Write `{assembly}_merged.csv` per genome_stem; return {stem: out_path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_assembly_tpm_groups(mappings_csv, root=root)
    written: dict[str, Path] = {}
    suf = suffix if suffix.startswith("_") else f"_{suffix}"
    if not suf.endswith(".csv"):
        suf = f"{suf}.csv"
    for stem, paths in sorted(groups.items()):
        out = out_dir / f"{stem}{suf}"
        mean_merge_tpm_csvs(paths, out, how=how, renormalize=renormalize)
        written[stem] = out
    return written


def parse_gtf_gene_transcript_map(gtf: Path | str) -> dict[str, dict]:
    """Map gene_id → transcript linkage from a RefSeq prokaryotic GTF.

    Returns:
      gene_id -> {
        chrom, start, end, strand, gene, locus_tag,
        transcript_ids: sorted list,
        conversion: "identity" | "single_transcript" | "multi_transcript"
      }
    """
    gtf = Path(gtf)
    if not gtf.is_file():
        raise FileNotFoundError(gtf)

    genes: dict[str, dict] = {}
    gene_txs: dict[str, set[str]] = defaultdict(set)

    with gtf.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2]
            attrs = dict(re.findall(r'(\w+) "([^"]*)"', parts[8]))
            gid = attrs.get("gene_id")
            if not gid:
                continue
            if ftype == "gene":
                genes[gid] = {
                    "chrom": parts[0],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "gene": attrs.get("gene") or attrs.get("gene_name") or "",
                    "locus_tag": attrs.get("locus_tag") or "",
                }
            if ftype in {"CDS", "exon", "transcript", "mRNA"}:
                tid = attrs.get("transcript_id")
                if tid:
                    gene_txs[gid].add(tid)

    out: dict[str, dict] = {}
    for gid, info in genes.items():
        txs = sorted(gene_txs.get(gid, []))
        # Drop empty placeholder
        txs = [t for t in txs if t]
        if not txs:
            conversion = "identity"  # use gene_id as the expression key
        elif len(txs) == 1:
            conversion = "single_transcript"
        else:
            conversion = "multi_transcript"
        out[gid] = {
            **info,
            "transcript_ids": txs,
            "conversion": conversion,
        }
    return out


def gene_to_transcript_easy(gtf: Path | str) -> tuple[dict[str, str], dict]:
    """Build a direct gene_id → transcript_id map when conversion is easy.

    Easy = every gene has 0 or 1 transcript_id (prokaryotic RefSeq typical).
    Genes with no transcript_id map to themselves (identity).

    Returns (gene_to_tx, stats).
    Raises ValueError if any gene has multiple transcript_ids (not easy).
    """
    table = parse_gtf_gene_transcript_map(gtf)
    multi = {g: v["transcript_ids"] for g, v in table.items() if v["conversion"] == "multi_transcript"}
    stats = {
        "n_genes": len(table),
        "identity": sum(1 for v in table.values() if v["conversion"] == "identity"),
        "single_transcript": sum(1 for v in table.values() if v["conversion"] == "single_transcript"),
        "multi_transcript": len(multi),
        "easy": len(multi) == 0,
    }
    if multi:
        examples = list(multi.items())[:5]
        raise ValueError(
            f"Gene→transcript is not 1:1 for {len(multi)} genes (examples={examples})"
        )
    mapping = {
        g: (v["transcript_ids"][0] if v["transcript_ids"] else g)
        for g, v in table.items()
    }
    return mapping, stats


def assess_panel_gene_transcript(
    gtf_dir: Path | str,
) -> list[dict]:
    """Assess gene→transcript ease for all `*_genomic.gtf` under gtf_dir."""
    gtf_dir = Path(gtf_dir)
    rows = []
    for gtf in sorted(gtf_dir.glob("*_genomic.gtf")):
        table = parse_gtf_gene_transcript_map(gtf)
        multi = sum(1 for v in table.values() if v["conversion"] == "multi_transcript")
        single = sum(1 for v in table.values() if v["conversion"] == "single_transcript")
        ident = sum(1 for v in table.values() if v["conversion"] == "identity")
        # For prokaryotes, multi with unassigned_transcript_* is rare annotation noise;
        # still report honest counts.
        rows.append(
            {
                "gtf": gtf.name,
                "assembly": gtf.name.replace("_genomic.gtf", ""),
                "n_genes": len(table),
                "identity": ident,
                "single_transcript": single,
                "multi_transcript": multi,
                "easy": multi == 0,
                "note": (
                    "1:1 gene↔transcript (or gene-only)"
                    if multi == 0
                    else f"{multi} genes with >1 transcript_id"
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Mean-merge GEO-aligned wide TPM CSVs per assembly (@summarize_GEO)."
    )
    p.add_argument(
        "--mappings",
        type=Path,
        default=DEFAULT_MAPPINGS,
        help="expr_file_mappings.csv (default: prokaryotes/expr_file_mappings.csv)",
    )
    p.add_argument(
        "--tpm-dir",
        type=Path,
        default=DEFAULT_PROK / "tpm",
        help="Directory containing per-sample TPM CSVs and merged outputs",
    )
    p.add_argument(
        "--prok-root",
        type=Path,
        default=DEFAULT_PROK,
        help="prokaryotes/ root (for resolving relative mapping paths)",
    )
    p.add_argument(
        "--how",
        choices=["intersection", "union"],
        default="intersection",
        help="Gene set for mean (default: intersection)",
    )
    p.add_argument(
        "--no-renormalize",
        action="store_true",
        help="Do not rescale merged TPM to sum 1e6",
    )
    p.add_argument(
        "--assess-transcripts",
        action="store_true",
        help="Print gene→transcript ease for prokaryotes/gtf and exit",
    )
    args = p.parse_args(argv)

    if args.assess_transcripts:
        gtf_dir = args.prok_root / "gtf"
        for row in assess_panel_gene_transcript(gtf_dir):
            flag = "EASY" if row["easy"] else "NOT_EASY"
            print(
                f"{row['assembly']}\t{flag}\tgenes={row['n_genes']}\t"
                f"single={row['single_transcript']}\tmulti={row['multi_transcript']}\t{row['note']}"
            )
        return 0

    written = summarize_assemblies(
        args.mappings,
        args.tpm_dir,
        root=args.prok_root,
        how=args.how,
        renormalize=not args.no_renormalize,
    )
    for stem, path in written.items():
        tpm = read_wide_tpm(path)
        print(f"{stem}\t{path}\tgenes={len(tpm)}\tsum={sum(tpm.values()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
