#!/usr/bin/env python3
"""Reproducible smoketest: reformat + N=3 distinct-species select + random folds.

Seed=42. Species grain: Escherichia_coli (K12+O157H7), Shigella_sonnei,
Pseudomonas_aeruginosa. One genome per train/val/test after seeded shuffle.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path

SEED_DEFAULT = 42
N_DEFAULT = 3
COMMIT_DEFAULT = "0060a6d8079b6a040fc55d505e15972a327b70a6"


def parse_species(stem: str) -> str:
    """Locked species grain for genomes_smoketest filenames."""
    if stem.startswith("Escherichia_coli_"):
        return "Escherichia_coli"
    if stem.startswith("Shigella_sonnei_"):
        return "Shigella_sonnei"
    if stem.startswith("Pseudomonas_aeruginosa_"):
        return "Pseudomonas_aeruginosa"
    m = re.match(r"^(.*)_(?:ASM|NCTC)\S*$", stem)
    return m.group(1) if m else stem


def hardlink_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build_manifest(root: Path, src: Path) -> list[dict]:
    pairs: list[dict] = []
    for fna in sorted(src.glob("*.fna")):
        gtf = fna.with_suffix(".gtf")
        if not gtf.exists():
            raise FileNotFoundError(f"unpaired FNA missing GTF: {fna.name}")
        if fna.stat().st_size == 0 or gtf.stat().st_size == 0:
            raise ValueError(f"empty pair: {fna.stem}")
        with fna.open("rb") as fh:
            if not fh.read(8).lstrip().startswith(b">"):
                raise ValueError(f"not FASTA: {fna.name}")
        stem = fna.stem
        pairs.append(
            {
                "sample_id": stem,
                "species": parse_species(stem),
                "fna_path": str(fna.relative_to(root)),
                "gtf_path": str(gtf.relative_to(root)),
            }
        )
    if not pairs:
        raise FileNotFoundError(f"no *.fna under {src}")
    return pairs


def select_distinct(
    pairs: list[dict], n: int, seed: int
) -> list[dict]:
    by_sp: dict[str, list[dict]] = {}
    for p in pairs:
        by_sp.setdefault(p["species"], []).append(p)
    species_list = sorted(by_sp.keys())
    if len(species_list) < n:
        raise ValueError(
            f"only {len(species_list)} unique species < N={n}: {species_list}"
        )
    rng = random.Random(seed)
    one_per = [rng.choice(by_sp[sp]) for sp in species_list]
    if len(one_per) > n:
        one_per = rng.sample(one_per, n)
    return sorted(one_per, key=lambda x: (x["species"], x["sample_id"]))


def assign_folds(selected: list[dict], seed: int) -> dict[str, str]:
    ids = [p["sample_id"] for p in selected]
    if len(ids) != 3:
        raise ValueError(f"N=3 fold policy requires exactly 3 samples; got {len(ids)}")
    order = ids[:]
    random.Random(seed).shuffle(order)
    return {"train": order[0], "val": order[1], "test": order[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--src", type=Path, default=Path("genomes_smoketest"))
    ap.add_argument("--reformat-out", type=Path, default=Path("data/reformat/smoketest_small"))
    ap.add_argument("--folds-out", type=Path, default=Path("data_splits/small"))
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--code-commit", default=COMMIT_DEFAULT)
    ap.add_argument("--reformat-only", action="store_true")
    ap.add_argument("--folds-only", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    src = (root / args.src).resolve() if not args.src.is_absolute() else args.src
    reformat_out = root / args.reformat_out
    folds_out = root / args.folds_out

    if not args.folds_only:
        pairs = build_manifest(root, src)
        reformat_out.mkdir(parents=True, exist_ok=True)
        with (reformat_out / "manifest.tsv").open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["sample_id", "species", "fna_path", "gtf_path"],
                delimiter="\t",
            )
            w.writeheader()
            w.writerows(pairs)
        selected = select_distinct(pairs, args.n, args.seed)
        with (reformat_out / "selected.tsv").open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["sample_id", "species", "fna_path", "gtf_path", "seed"],
                delimiter="\t",
            )
            w.writeheader()
            for p in selected:
                row = dict(p)
                row["seed"] = args.seed
                w.writerow(row)
        payload = {
            "seed": args.seed,
            "n": args.n,
            "distinct_species": True,
            "species_grain": (
                "Escherichia_coli (K12+O157H7), Shigella_sonnei, "
                "Pseudomonas_aeruginosa"
            ),
            "selected_ids": [p["sample_id"] for p in selected],
            "species": [p["species"] for p in selected],
            "rng": (
                "random.Random(seed); sorted species; choice one genome per "
                "species; if >N sample without replacement"
            ),
        }
        (reformat_out / "selection.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    else:
        selected_path = reformat_out / "selected.tsv"
        if not selected_path.exists():
            raise FileNotFoundError(selected_path)
        selected = []
        with selected_path.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                selected.append(row)

    if args.reformat_only:
        return

    fold_map = assign_folds(selected, args.seed)
    id_to = {p["sample_id"]: p for p in selected}
    for fold in ("train", "val", "test"):
        d = folds_out / fold
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()

    rows = []
    for fold, sid in fold_map.items():
        p = id_to[sid]
        src_fna = root / p["fna_path"]
        src_gtf = root / p["gtf_path"]
        dst_fna = folds_out / fold / src_fna.name
        dst_gtf = folds_out / fold / src_gtf.name
        hardlink_or_copy(src_fna, dst_fna)
        hardlink_or_copy(src_gtf, dst_gtf)
        rows.append(
            {
                "sample_id": sid,
                "species": p["species"],
                "fold": fold,
                "seed": args.seed,
                "split_id": "random",
                "fna_path": str(dst_fna.relative_to(root)),
                "gtf_path": str(dst_gtf.relative_to(root)),
                "code_commit": args.code_commit,
            }
        )

    with (folds_out / "fold_manifest.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "species",
                "fold",
                "seed",
                "split_id",
                "fna_path",
                "gtf_path",
                "code_commit",
            ],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(rows)

    zs = folds_out / "zero-shot"
    if zs.exists():
        raise RuntimeError("zero-shot/ must not exist for this run")


if __name__ == "__main__":
    main()
