#!/usr/bin/env python3
"""Build coherent genome+TPM sample manifest for Caduceus-oriented folds.

Follows @genome-tpm-caduceus-reformat. Does NOT assign train/val/test folds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_EXCLUDE = ("GCF_041296265.1",)
COMMIT_DEFAULT = "0060a6d8079b6a040fc55d505e15972a327b70a6"

# Fallback organism names if genome_download_status.tsv unavailable
SPECIES_FALLBACK = {
    "GCF_000001405.40": "Homo sapiens",
    "GCF_000001635.27": "Mus musculus",
    "GCF_036323735.1": "Rattus norvegicus",
    "GCF_000003025.6": "Sus scrofa",
    "GCF_002263795.3": "Bos taurus",
    "GCF_011100685.1": "Canis lupus familiaris",
    "GCF_049350105.2": "Macaca mulatta",
    "GCF_016772045.2": "Ovis aries",
    "GCF_964237555.1": "Oryctolagus cuniculus",
    "GCF_041296265.1": "Equus caballus",
}


def load_species_map(root: Path) -> dict[str, str]:
    status = root / "data" / "manifests" / "genome_download_status.tsv"
    out = dict(SPECIES_FALLBACK)
    if status.is_file():
        with status.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                acc = (row.get("accession") or row.get("genome") or "").strip()
                org = (
                    row.get("organism")
                    or row.get("species")
                    or row.get("notes")
                    or ""
                ).strip()
                if acc and org and not org.startswith("data/"):
                    out[acc] = org
    return out


def first_match(dirpath: Path, patterns: list[str]) -> Path | None:
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(dirpath.rglob(pat))
    hits = [p for p in hits if p.is_file() and p.stat().st_size > 0]
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def resolve_paths(genomes_root: Path, gcf: str) -> tuple[Path | None, Path | None]:
    gdir = genomes_root / gcf
    if not gdir.is_dir():
        return None, None
    fna = first_match(gdir, ["*_genomic.fna", "*.fna"])
    gtf = first_match(gdir, ["*genomic.gtf.gz", "*genomic.gtf", "*.gtf.gz", "*.gtf"])
    return fna, gtf


def assert_fasta_header(fna: Path) -> None:
    with fna.open("rb") as fh:
        head = fh.read(64).lstrip()
    if not head.startswith(b">"):
        raise ValueError(f"FNA does not start with '>': {fna}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--mappings", type=Path, default=None)
    ap.add_argument("--genomes-root", type=Path, default=None)
    ap.add_argument("--expr-root", type=Path, default=None)
    ap.add_argument("--genes-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require-tpm", action="store_true", default=True)
    ap.add_argument("--no-require-tpm", action="store_false", dest="require_tpm")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDE),
        help="GCF accessions to force-exclude",
    )
    ap.add_argument("--code-commit", default=COMMIT_DEFAULT)
    args = ap.parse_args()

    root = args.root.resolve()
    mappings = args.mappings or (root / "random" / "random_borzoi_expr_file_mappings.csv")
    genomes_root = args.genomes_root or (root / "data" / "raw" / "genomes")
    expr_root = args.expr_root or (root / "random" / "expression_data")
    genes_root = args.genes_root or (root / "random" / "genes")
    out = args.out or (root / "data" / "reformat" / "random_full")

    for label, path in [
        ("mappings", mappings),
        ("genomes_root", genomes_root),
        ("expr_root", expr_root),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    out.mkdir(parents=True, exist_ok=True)
    species_map = load_species_map(root)
    exclude = set(args.exclude)

    included: list[dict] = []
    exclusions: list[dict] = []

    with mappings.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty mappings: {mappings}")

    for row in rows:
        assay = row["id"].strip()
        gcf = row["genome"].strip()
        species = species_map.get(gcf, gcf)
        sample_id = gcf

        fna, gtf = resolve_paths(genomes_root, gcf)
        tpm = expr_root / assay / "tpm" / f"{assay}.csv"
        genes = genes_root / gcf / f"{gcf}_genes.tsv"

        reason = None
        if gcf in exclude:
            reason = "excluded_by_policy"
        elif args.require_tpm and (not tpm.is_file() or tpm.stat().st_size == 0):
            reason = "no_tpm"
        elif fna is None:
            reason = "missing_fna"
        elif gtf is None:
            reason = "missing_gtf"

        if reason:
            exclusions.append(
                {
                    "sample_id": sample_id,
                    "species": species,
                    "genome_accession": gcf,
                    "assay_id": assay,
                    "reason": reason,
                    "tpm_exists": str(tpm.is_file()).lower(),
                }
            )
            continue

        assert fna is not None and gtf is not None
        assert_fasta_header(fna)
        if fna.stat().st_size == 0 or gtf.stat().st_size == 0:
            raise ValueError(f"empty FNA/GTF for {gcf}")

        included.append(
            {
                "sample_id": sample_id,
                "species": species,
                "genome_accession": gcf,
                "assay_id": assay,
                "fna_path": str(fna.relative_to(root)),
                "gtf_path": str(gtf.relative_to(root)),
                "tpm_path": str(tpm.relative_to(root)),
                "genes_path": str(genes.relative_to(root)) if genes.is_file() else "",
            }
        )

    if len(included) != 9:
        raise SystemExit(
            f"expected 9 included samples, got {len(included)}; "
            f"exclusions={len(exclusions)}"
        )

    horse = [e for e in exclusions if e["genome_accession"] == "GCF_041296265.1"]
    if not horse:
        raise SystemExit("horse GCF_041296265.1 missing from exclusions.tsv")
    if horse[0]["reason"] not in {"no_tpm", "excluded_by_policy"}:
        raise SystemExit(f"unexpected horse exclusion reason: {horse[0]['reason']}")
    # Normalize documented reason for AC: no_tpm
    for e in exclusions:
        if e["genome_accession"] == "GCF_041296265.1":
            e["reason"] = "no_tpm"

    man_cols = [
        "sample_id",
        "species",
        "genome_accession",
        "assay_id",
        "fna_path",
        "gtf_path",
        "tpm_path",
        "genes_path",
    ]
    man_path = out / "manifest.tsv"
    with man_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=man_cols, delimiter="\t")
        w.writeheader()
        for r in sorted(included, key=lambda x: x["sample_id"]):
            w.writerow(r)

    excl_cols = [
        "sample_id",
        "species",
        "genome_accession",
        "assay_id",
        "reason",
        "tpm_exists",
    ]
    excl_path = out / "exclusions.tsv"
    with excl_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=excl_cols, delimiter="\t")
        w.writeheader()
        for r in exclusions:
            w.writerow(r)

    selection = {
        "seed_policy": "folds assigned later by @split with seed 42; this step does not shuffle",
        "n_included": len(included),
        "n_excluded": len(exclusions),
        "require_tpm": args.require_tpm,
        "exclude_accessions": sorted(exclude),
        "code_commit": args.code_commit,
        "out": str(out.relative_to(root)),
        "included_sample_ids": sorted(r["sample_id"] for r in included),
    }
    sel_path = out / "selection.json"
    sel_path.write_text(json.dumps(selection, indent=2) + "\n")

    notes = out / "caduceus_notes.md"
    notes.write_text(
        f"""# Caduceus-oriented layout notes

**Date:** 2026-07-26
**Reformat out:** `{out.relative_to(root)}`
**Caduceus pin:** `{args.code_commit}`

## Upstream GenomicBenchmark expectation

```
{{dest}}/{{dataset_name}}/{{split}}/{{label}}/*.txt   # raw DNA, no FASTA header
```

Whole mammalian assemblies are **not** single GenomicBenchmark examples.

## Project fold layout (after @split)

```
data_splits/full/{{train|val|test}}/{{sample_id}}/
  genome.fna
  annotation.gtf   # or .gtf.gz hardlink
  expression_tpm.csv
  genes.tsv        # optional
```

Native GB `.txt` sequence trees are **off** by default for this panel (too large).

## Manifest

- Included: **{len(included)}** TPM-complete species
- Excluded: **{len(exclusions)}** (horse documented as `no_tpm`)
"""
    )

    print(f"wrote {man_path.relative_to(root)} ({len(included)} rows)")
    print(f"wrote {excl_path.relative_to(root)} ({len(exclusions)} rows)")
    print(f"wrote {sel_path.relative_to(root)}")
    print(f"wrote {notes.relative_to(root)}")


if __name__ == "__main__":
    main()
