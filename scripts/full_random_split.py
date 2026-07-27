#!/usr/bin/env python3
"""Seeded species-level random split → data_splits/full/{train,val,test}/.

Reads data/reformat/random_full/manifest.tsv; hardlinks FNA+GTF+TPM (+genes).
Seed=42; ratios train=5, val=2, test=2; no zero-shot.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

SEED_DEFAULT = 42
COMMIT_DEFAULT = "0060a6d8079b6a040fc55d505e15972a327b70a6"
RATIOS = ("train", "val", "test")  # assigned after shuffle: 5/2/2
COUNTS = {"train": 5, "val": 2, "test": 2}


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def annotation_name(gtf: Path) -> str:
    if gtf.name.endswith(".gtf.gz"):
        return "annotation.gtf.gz"
    return "annotation.gtf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--code-commit", default=COMMIT_DEFAULT)
    ap.add_argument("--split-id", default="random")
    args = ap.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest or (
        root / "data" / "reformat" / "random_full" / "manifest.tsv"
    )
    out = args.out or (root / "data_splits" / "full")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    with manifest_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(rows) != 9:
        raise SystemExit(f"expected 9 manifest rows, got {len(rows)}")

    # Deterministic order then shuffle
    rows = sorted(rows, key=lambda r: r["sample_id"])
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_train, n_val, n_test = COUNTS["train"], COUNTS["val"], COUNTS["test"]
    if n_train + n_val + n_test != len(rows):
        raise SystemExit("ratio sum != N")

    assignment: list[tuple[str, dict]] = []
    for i, row in enumerate(rows):
        if i < n_train:
            fold = "train"
        elif i < n_train + n_val:
            fold = "val"
        else:
            fold = "test"
        assignment.append((fold, row))

    # Clean prior fold trees (keep out root)
    if out.exists():
        for fold in RATIOS:
            fdir = out / fold
            if fdir.exists():
                shutil.rmtree(fdir)
        zs = out / "zero-shot"
        if zs.exists():
            shutil.rmtree(zs)
        fm = out / "fold_manifest.tsv"
        if fm.exists():
            fm.unlink()
    out.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    for fold, row in assignment:
        sample_id = row["sample_id"]
        sample_dir = out / fold / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        fna = root / row["fna_path"]
        gtf = root / row["gtf_path"]
        tpm = root / row["tpm_path"]
        for p, label in [(fna, "fna"), (gtf, "gtf"), (tpm, "tpm")]:
            if not p.is_file() or p.stat().st_size == 0:
                raise FileNotFoundError(f"missing/empty {label}: {p}")

        ann_name = annotation_name(gtf)
        link_fna = hardlink_or_copy(fna, sample_dir / "genome.fna")
        link_gtf = hardlink_or_copy(gtf, sample_dir / ann_name)
        link_tpm = hardlink_or_copy(tpm, sample_dir / "expression_tpm.csv")

        genes_rel = (row.get("genes_path") or "").strip()
        genes_dst = ""
        link_genes = ""
        if genes_rel:
            genes = root / genes_rel
            if genes.is_file() and genes.stat().st_size > 0:
                link_genes = hardlink_or_copy(genes, sample_dir / "genes.tsv")
                genes_dst = str((sample_dir / "genes.tsv").relative_to(root))

        fold_rows.append(
            {
                "sample_id": sample_id,
                "species": row["species"],
                "genome_accession": row["genome_accession"],
                "assay_id": row["assay_id"],
                "fold": fold,
                "seed": str(args.seed),
                "split_id": args.split_id,
                "code_commit": args.code_commit,
                "fna_src": row["fna_path"],
                "gtf_src": row["gtf_path"],
                "tpm_src": row["tpm_path"],
                "genes_src": genes_rel,
                "fold_dir": str(sample_dir.relative_to(root)),
                "genome_fna": str((sample_dir / "genome.fna").relative_to(root)),
                "annotation": str((sample_dir / ann_name).relative_to(root)),
                "expression_tpm": str(
                    (sample_dir / "expression_tpm.csv").relative_to(root)
                ),
                "genes_tsv": genes_dst,
                "link_mode": f"fna={link_fna};gtf={link_gtf};tpm={link_tpm}"
                + (f";genes={link_genes}" if link_genes else ""),
            }
        )

    # Sanity counts
    counts = {f: sum(1 for fr in fold_rows if fr["fold"] == f) for f in RATIOS}
    if counts != COUNTS:
        raise SystemExit(f"fold counts {counts} != {COUNTS}")
    if (out / "zero-shot").exists():
        raise SystemExit("zero-shot/ must not exist")

    cols = [
        "sample_id",
        "species",
        "genome_accession",
        "assay_id",
        "fold",
        "seed",
        "split_id",
        "code_commit",
        "fna_src",
        "gtf_src",
        "tpm_src",
        "genes_src",
        "fold_dir",
        "genome_fna",
        "annotation",
        "expression_tpm",
        "genes_tsv",
        "link_mode",
    ]
    man_out = out / "fold_manifest.tsv"
    with man_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for fr in sorted(fold_rows, key=lambda x: (x["fold"], x["sample_id"])):
            w.writerow(fr)

    print(f"wrote folds under {out.relative_to(root)}: {counts}")
    print(f"wrote {man_out.relative_to(root)}")
    for fr in sorted(fold_rows, key=lambda x: (x["fold"], x["sample_id"])):
        print(f"  {fr['fold']}: {fr['sample_id']} ({fr['species']})")


if __name__ == "__main__":
    main()
