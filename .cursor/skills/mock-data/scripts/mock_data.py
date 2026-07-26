#!/usr/bin/env python3
"""Generate reproducible mock fixtures for 16S (QIIME2-style) and WGS (Bracken) hooks.

Outputs land under ./test/ by default (gitignored). Plain files are always written;
optional .qza archives wrap the same payloads in a minimal QIIME 2 ZIP layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Canonical mock content
# ---------------------------------------------------------------------------

DEFAULT_16S_SEQS = {
    "seq1": "ATGATG",
    "seq2": "ATTTTT",
    "seq3": "AAAAA",
}

# SILVA-like semicolon taxonomy (QIIME2 FeatureData[Taxonomy] Taxon column)
DEFAULT_16S_TAXONOMY = {
    "seq1": (
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;"
        "o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli"
    ),
    "seq2": (
        "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;"
        "f__Lactobacillaceae;g__Lactobacillus;s__"
    ),
    "seq3": (
        "d__Eukaryota;p__Ascomycota;c__Saccharomycetes;o__Saccharomycetales;"
        "f__Saccharomycetaceae;g__Saccharomyces;s__cerevisiae"
    ),
}

# Tip order matches sequences; root children for a tiny rooted tree
DEFAULT_NEWICK = "((seq1:0.01,seq2:0.02):0.03,seq3:0.04);"

# taxon → NCBI taxid (None = unresolved / invalid for mock purposes)
MISC_TAXON_TAXIDS: list[tuple[str, Optional[int]]] = [
    ("Homo sapiens", 9606),
    ("Escherichia coli", 562),
    ("invalid species 1", None),
    ("Enterobacteriaceae", 543),
    ("Saccharomyces sp.", 4930),  # genus Saccharomyces
    ("Saccharomyces cerevisiae", 4932),
]

SPELLING_NOTES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Resolve repo root by walking up from this file or CWD."""
    here = Path(__file__).resolve()
    for cur in [here.parent, *here.parents]:
        if (cur / "artifact-registry.md").is_file() or (cur / ".cursor").is_dir():
            # Prefer the directory that contains both .cursor and artifact-registry when possible
            if (cur / ".cursor").is_dir():
                return cur
    return Path.cwd()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def pack_qza(
    out_qza: Path,
    *,
    semantic_type: str,
    directory_format: str,
    data_files: dict[str, bytes],
) -> Path:
    """Write a minimal QIIME 2–compatible ZIP (.qza) without requiring qiime2.

    Layout:
      <uuid>/VERSION
      <uuid>/metadata.yaml
      <uuid>/checksums.md5
      <uuid>/data/<payload files>
    """
    ensure_dir(out_qza.parent)
    archive_uuid = str(uuid.uuid4())
    metadata = (
        f"uuid: {archive_uuid}\n"
        f"type: {semantic_type}\n"
        f"format: {directory_format}\n"
    )
    version = "QIIME 2\narchive: 5\nframework: 2024.10.0\n"

    checksum_lines = []
    for rel, payload in sorted(data_files.items()):
        checksum_lines.append(f"{md5_bytes(payload)}  data/{rel}")
    checksum_lines.append(f"{md5_bytes(metadata.encode())}  metadata.yaml")
    checksum_lines.append(f"{md5_bytes(version.encode())}  VERSION")
    checksums = "\n".join(checksum_lines) + "\n"

    with zipfile.ZipFile(out_qza, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        root = archive_uuid
        zf.writestr(f"{root}/VERSION", version)
        zf.writestr(f"{root}/metadata.yaml", metadata)
        zf.writestr(f"{root}/checksums.md5", checksums)
        for rel, payload in data_files.items():
            zf.writestr(f"{root}/data/{rel}", payload)
    return out_qza


# ---------------------------------------------------------------------------
# 16S mocks
# ---------------------------------------------------------------------------

def write_fasta(path: Path, sequences: dict[str, str]) -> Path:
    lines = []
    for sid, seq in sequences.items():
        lines.append(f">{sid}")
        lines.append(seq)
    return write_text(path, "\n".join(lines))


def write_taxonomy_tsv(path: Path, taxonomy: dict[str, str]) -> Path:
    """QIIME2-exported taxonomy.tsv style: Feature ID, Taxon, Confidence."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["Feature ID", "Taxon", "Confidence"])
        for fid, taxon in taxonomy.items():
            w.writerow([fid, taxon, "0.99"])
    return path


def write_newick(path: Path, newick: str) -> Path:
    return write_text(path, newick.strip())


def write_sample_metadata_16s(
    path: Path,
    sample_ids: Sequence[str],
    *,
    id_column: str = "sampleID",
) -> Path:
    """QIIME2-like metadata TSV. Only obligatory field: sampleID (also aliased)."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        # Include SampleID / sample-id aliases for qiime2R / qiime tools
        fieldnames = [id_column, "SampleID", "sample-id", "group", "replicate"]
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for i, sid in enumerate(sample_ids, start=1):
            w.writerow(
                {
                    id_column: sid,
                    "SampleID": sid,
                    "sample-id": sid,
                    "group": "control" if i % 2 else "treatment",
                    "replicate": str(((i - 1) % 2) + 1),
                }
            )
    return path


def write_qiime_manifest(
    path: Path,
    sample_ids: Sequence[str],
    *,
    seq_dir: Path,
) -> Path:
    """Paired-end QIIME2 manifest (paths may be placeholders for mocks)."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(
            ["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"]
        )
        for sid in sample_ids:
            r1 = seq_dir / f"{sid}_R1.fastq.gz"
            r2 = seq_dir / f"{sid}_R2.fastq.gz"
            w.writerow([sid, str(r1.resolve()), str(r2.resolve())])
    return path


def mock_16s(
    out_dir: Path,
    *,
    include_taxonomy: bool = True,
    include_tree: bool = True,
    write_qza: bool = True,
    sample_ids: Optional[Sequence[str]] = None,
) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    sample_ids = list(sample_ids or ("mock16s_S1", "mock16s_S2", "mock16s_S3"))
    written: dict[str, Path] = {}

    # Clear optional artifacts when omitted so re-runs stay honest
    if not include_taxonomy:
        for p in (out / "taxonomy.tsv", out / "taxonomy.qza"):
            if p.exists():
                p.unlink()
    if not include_tree:
        for p in (out / "tree.nwk", out / "tree.qza"):
            if p.exists():
                p.unlink()
    if not write_qza:
        for p in (out / "sequences.qza", out / "taxonomy.qza", out / "tree.qza"):
            if p.exists():
                p.unlink()

    fasta = write_fasta(out / "sequences.fasta", DEFAULT_16S_SEQS)
    written["sequences.fasta"] = fasta

    if write_qza:
        written["sequences.qza"] = pack_qza(
            out / "sequences.qza",
            semantic_type="FeatureData[Sequence]",
            directory_format="DNASequencesDirectoryFormat",
            data_files={"dna-sequences.fasta": fasta.read_bytes()},
        )

    if include_taxonomy:
        tax = write_taxonomy_tsv(out / "taxonomy.tsv", DEFAULT_16S_TAXONOMY)
        written["taxonomy.tsv"] = tax
        if write_qza:
            written["taxonomy.qza"] = pack_qza(
                out / "taxonomy.qza",
                semantic_type="FeatureData[Taxonomy]",
                directory_format="TSVTaxonomyDirectoryFormat",
                data_files={"taxonomy.tsv": tax.read_bytes()},
            )

    if include_tree:
        tree = write_newick(out / "tree.nwk", DEFAULT_NEWICK)
        written["tree.nwk"] = tree
        if write_qza:
            written["tree.qza"] = pack_qza(
                out / "tree.qza",
                semantic_type="Phylogeny[Rooted]",
                directory_format="NewickDirectoryFormat",
                data_files={"tree.nwk": tree.read_bytes()},
            )

    written["sample-metadata.tsv"] = write_sample_metadata_16s(
        out / "sample-metadata.tsv", sample_ids
    )
    written["manifest.tsv"] = write_qiime_manifest(
        out / "manifest.tsv", sample_ids, seq_dir=out / "fastq"
    )

    # Tiny feature-table count matrix (plain TSV) for non-QIIME consumers
    ft = out / "feature-table.tsv"
    ensure_dir(ft.parent)
    with ft.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["#OTU ID", *sample_ids])
        for i, sid in enumerate(DEFAULT_16S_SEQS):
            counts = [max(1, (i + 1) * (j + 1) * 10) for j in range(len(sample_ids))]
            w.writerow([sid, *counts])
    written["feature-table.tsv"] = ft

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(out))
        except ValueError:
            return str(p)

    manifest = {
        "kind": "16S",
        "required": ["sequences", "sample-metadata (sampleID)"],
        "optional": ["taxonomy", "tree"],
        "include_taxonomy": include_taxonomy,
        "include_tree": include_tree,
        "files": {k: _rel(v) for k, v in written.items()},
    }
    written["manifest.json"] = write_text(
        out / "manifest.json", json.dumps(manifest, indent=2)
    )
    return written


# ---------------------------------------------------------------------------
# WGS / Bracken mocks
# ---------------------------------------------------------------------------

BRACKEN_GENERA = [
    # name, taxonomy_id, taxonomy_lvl, kraken_assigned, added, new_est, fraction
    ("Escherichia", 561, "G", 800, 50, 850, 0.425),
    ("Homo", 9605, "G", 400, 0, 400, 0.200),
    ("Saccharomyces", 4930, "G", 200, 20, 220, 0.110),
    ("Lactobacillus", 1578, "G", 150, 0, 150, 0.075),
    ("root", 1, "R", 0, 0, 2000, 1.0),  # not used in G file body typically
]

BRACKEN_SPECIES_REPORT_ROWS = [
    # pct, reads_clade, reads_direct, rank, taxid, name (indented like Kraken)
    (100.00, 2000, 0, "R", 1, "root"),
    (42.50, 850, 850, "S", 562, "                                                            Escherichia coli"),
    (20.00, 400, 400, "S", 9606, "                                                            Homo sapiens"),
    (11.00, 220, 220, "S", 4932, "                                                            Saccharomyces cerevisiae"),
    (7.50, 150, 150, "S", 1578, "                                                            Lactobacillus"),
    (5.00, 100, 0, "F", 543, "                                                      Enterobacteriaceae"),
]


def write_bracken_genus_table(path: Path, sample_scale: float = 1.0) -> Path:
    """Honey-style `*.nt.G.bracken` with header."""
    ensure_dir(path.parent)
    rows = [
        r
        for r in BRACKEN_GENERA
        if r[2] == "G"
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "name",
                "taxonomy_id",
                "taxonomy_lvl",
                "kraken_assigned_reads",
                "added_reads",
                "new_est_reads",
                "fraction_total_reads",
            ]
        )
        for name, tid, lvl, k_asg, added, new_est, frac in rows:
            w.writerow(
                [
                    name,
                    tid,
                    lvl,
                    int(k_asg * sample_scale),
                    int(added * sample_scale),
                    int(new_est * sample_scale),
                    f"{frac:.5f}",
                ]
            )
    return path


def write_bracken_species_report(path: Path, sample_scale: float = 1.0) -> Path:
    """Kristina-style `*.nt.bracken.S.report` — 6 columns, no header."""
    ensure_dir(path.parent)
    lines = []
    for pct, reads, reads_d, rank, taxid, name in BRACKEN_SPECIES_REPORT_ROWS:
        lines.append(
            f"{pct:.2f}\t{int(reads * sample_scale)}\t{int(reads_d * sample_scale)}"
            f"\t{rank}\t{taxid}\t{name}"
        )
    return write_text(path, "\n".join(lines))


def write_sample_metadata_wgs(
    path: Path,
    sample_ids: Sequence[str],
    *,
    id_column: str = "sampleID",
) -> Path:
    """WGS sample metadata. Only obligatory field: sampleID (plus Run alias)."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [id_column, "Run", "BioProject", "group"]
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for i, sid in enumerate(sample_ids, start=1):
            w.writerow(
                {
                    id_column: sid,
                    "Run": sid,
                    "BioProject": "PRJMOCK001",
                    "group": "A" if i % 2 else "B",
                }
            )
    return path


def mock_wgs(
    out_dir: Path,
    *,
    sample_ids: Optional[Sequence[str]] = None,
) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    sample_ids = list(sample_ids or ("mockwgs_S1", "mockwgs_S2", "mockwgs_S3"))
    written: dict[str, Path] = {}

    for i, sid in enumerate(sample_ids):
        scale = 1.0 + 0.1 * i
        g_path = write_bracken_genus_table(out / f"{sid}.nt.G.bracken", scale)
        s_path = write_bracken_species_report(out / f"{sid}.nt.bracken.S.report", scale)
        written[f"{sid}.nt.G.bracken"] = g_path
        written[f"{sid}.nt.bracken.S.report"] = s_path

    written["sample-metadata.csv"] = write_sample_metadata_wgs(
        out / "sample-metadata.csv", sample_ids
    )

    # Sample map (Kristina-style bridge)
    map_path = out / "bracken_sample_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["bracken_id", "sample_id"])
        for sid in sample_ids:
            w.writerow([sid, sid])
    written["bracken_sample_map.csv"] = map_path

    manifest = {
        "kind": "WGS",
        "required": ["bracken report", "sample-metadata (sampleID)"],
        "formats": {
            "genus_table": "*.nt.G.bracken (header; name, taxonomy_id, new_est_reads)",
            "species_report": "*.nt.bracken.S.report (no header; 6 Kraken-style cols)",
        },
        "files": {k: str(v.name) for k, v in written.items()},
    }
    written["manifest.json"] = write_text(
        out / "manifest.json", json.dumps(manifest, indent=2)
    )
    return written


# ---------------------------------------------------------------------------
# Misc: taxon + taxid lists
# ---------------------------------------------------------------------------

def mock_misc(out_dir: Path) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    written: dict[str, Path] = {}

    taxons = [t for t, _ in MISC_TAXON_TAXIDS]
    written["taxons.json"] = write_text(
        out / "taxons.json", json.dumps(taxons, indent=2, ensure_ascii=False)
    )
    written["taxons.txt"] = write_text(out / "taxons.txt", "\n".join(taxons))

    taxids_path = out / "taxids.tsv"
    with taxids_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["taxon", "taxid", "status", "note"])
        for taxon, taxid in MISC_TAXON_TAXIDS:
            if taxid is None:
                status = "unresolved"
                note = SPELLING_NOTES.get(taxon, "invalid or ambiguous name")
                taxid_out = ""
            else:
                status = "ok"
                note = ""
                taxid_out = str(taxid)
            w.writerow([taxon, taxid_out, status, note])
    written["taxids.tsv"] = taxids_path

    # Compact JSON map (null for unresolved)
    taxid_map = {t: tid for t, tid in MISC_TAXON_TAXIDS}
    written["taxids.json"] = write_text(
        out / "taxids.json", json.dumps(taxid_map, indent=2, ensure_ascii=False)
    )

    written["manifest.json"] = write_text(
        out / "manifest.json",
        json.dumps(
            {
                "kind": "misc",
                "taxons": taxons,
                "files": {k: v.name for k, v in written.items() if k != "manifest.json"},
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def mock_all(
    test_root: Path,
    *,
    include_taxonomy: bool = True,
    include_tree: bool = True,
    write_qza: bool = True,
) -> dict[str, dict[str, Path]]:
    ensure_dir(test_root)
    return {
        "16s": mock_16s(
            test_root / "16s",
            include_taxonomy=include_taxonomy,
            include_tree=include_tree,
            write_qza=write_qza,
        ),
        "wgs": mock_wgs(test_root / "wgs"),
        "misc": mock_misc(test_root / "misc"),
    }


def _print_written(label: str, written: dict[str, Path]) -> None:
    print(f"[{label}] wrote {len(written)} files:")
    for name, path in sorted(written.items()):
        print(f"  - {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: <repo>/test)",
    )
    p.add_argument(
        "--target",
        choices=("all", "16s", "wgs", "misc"),
        default="all",
        help="Which mock set to generate",
    )
    p.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Omit 16S taxonomy (plain + qza)",
    )
    p.add_argument(
        "--no-tree",
        action="store_true",
        help="Omit 16S tree (plain + qza)",
    )
    p.add_argument(
        "--no-qza",
        action="store_true",
        help="Do not write .qza archives",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Generate then validate required files exist",
    )
    return p


def validate(test_root: Path, *, expect_taxonomy: bool, expect_tree: bool) -> list[str]:
    errors: list[str] = []

    def need(path: Path) -> None:
        if not path.is_file():
            errors.append(f"missing: {path}")

    need(test_root / "16s" / "sequences.fasta")
    need(test_root / "16s" / "sample-metadata.tsv")
    need(test_root / "16s" / "manifest.tsv")
    if expect_taxonomy:
        need(test_root / "16s" / "taxonomy.tsv")
    if expect_tree:
        need(test_root / "16s" / "tree.nwk")

    # sampleID column present
    meta = test_root / "16s" / "sample-metadata.tsv"
    if meta.is_file():
        header = meta.read_text(encoding="utf-8").splitlines()[0].split("\t")
        if "sampleID" not in header:
            errors.append("16s sample-metadata.tsv lacks sampleID column")

    fasta = (test_root / "16s" / "sequences.fasta").read_text(encoding="utf-8") if (test_root / "16s" / "sequences.fasta").is_file() else ""
    for sid in ("seq1", "seq2", "seq3"):
        if f">{sid}" not in fasta:
            errors.append(f"sequences.fasta missing {sid}")

    need(test_root / "wgs" / "sample-metadata.csv")
    wgs_meta = test_root / "wgs" / "sample-metadata.csv"
    if wgs_meta.is_file():
        header = wgs_meta.read_text(encoding="utf-8").splitlines()[0].split(",")
        if "sampleID" not in header:
            errors.append("wgs sample-metadata.csv lacks sampleID column")

    bracken = list((test_root / "wgs").glob("*.nt.G.bracken"))
    if not bracken:
        errors.append("no WGS *.nt.G.bracken files")
    else:
        cols = bracken[0].read_text(encoding="utf-8").splitlines()[0].split("\t")
        for c in ("name", "taxonomy_id", "new_est_reads"):
            if c not in cols:
                errors.append(f"bracken missing column {c}")

    need(test_root / "misc" / "taxons.json")
    need(test_root / "misc" / "taxids.tsv")
    if (test_root / "misc" / "taxons.json").is_file():
        taxons = json.loads((test_root / "misc" / "taxons.json").read_text(encoding="utf-8"))
        expected = [t for t, _ in MISC_TAXON_TAXIDS]
        if taxons != expected:
            errors.append(f"taxons.json mismatch: {taxons!r} != {expected!r}")

    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    test_root = (args.out or (root / "test")).resolve()

    include_taxonomy = not args.no_taxonomy
    include_tree = not args.no_tree
    write_qza = not args.no_qza

    if args.target == "all":
        results = mock_all(
            test_root,
            include_taxonomy=include_taxonomy,
            include_tree=include_tree,
            write_qza=write_qza,
        )
        for label, written in results.items():
            _print_written(label, written)
    elif args.target == "16s":
        _print_written(
            "16s",
            mock_16s(
                test_root / "16s",
                include_taxonomy=include_taxonomy,
                include_tree=include_tree,
                write_qza=write_qza,
            ),
        )
    elif args.target == "wgs":
        _print_written("wgs", mock_wgs(test_root / "wgs"))
    else:
        _print_written("misc", mock_misc(test_root / "misc"))

    if args.self_test or args.target == "all":
        # Validate whatever we intended to write for this run
        if args.target == "16s":
            # only check 16s subset lightly via full validator may fail — run full if all
            pass
        errs = validate(
            test_root,
            expect_taxonomy=include_taxonomy if args.target in ("all", "16s") else True,
            expect_tree=include_tree if args.target in ("all", "16s") else True,
        )
        # When targeting a subset, filter errors to that subtree
        if args.target != "all":
            prefix = str(test_root / args.target)
            errs = [e for e in errs if prefix in e or args.target in e.lower() or (args.target == "misc" and "taxon" in e)]
            # For subset runs, ignore missing other kinds
            if args.target == "16s":
                errs = [e for e in errs if "/wgs/" not in e and "WGS" not in e and "/misc/" not in e and "taxons" not in e]
            elif args.target == "wgs":
                errs = [e for e in errs if "/16s/" not in e and "/misc/" not in e and "taxons" not in e and "sequences" not in e]
            elif args.target == "misc":
                errs = [e for e in errs if "/16s/" not in e and "/wgs/" not in e and "bracken" not in e and "sample-metadata" not in e]

        if errs:
            print("VALIDATION FAILED:", file=sys.stderr)
            for e in errs:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print("VALIDATION OK")

    print(f"Mock root: {test_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
