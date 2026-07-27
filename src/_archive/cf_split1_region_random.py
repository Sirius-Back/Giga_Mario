#!/usr/bin/env python3
"""Caduceus-full Split #1: region-level random split with linked TPM.

Atomic unit = GTF gene interval + TPM from panel expression_tpm.csv (wide).
Missing gene↔TPM join → prediction 0 (documented). Seeded shuffle; Caduceus-style
hold-out test fraction then 90/10 train/val on the remainder.
Never invents TPM values. Relative project paths only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

SEED_DEFAULT = 42
SPLIT_ID = "random"
ZS_FORBIDDEN = "Escherichia_coli_K12_ASM2564343v1"
PANEL_DEFAULT = Path(
    "data/reformat/caduceus_full/cf_random_smoketest_20260726/panel"
)
OUT_DEFAULT = Path(
    "data_splits/caduceus_full/cf_random_smoketest_20260726/split1"
)
TEST_FRACTION = 0.10
VAL_FRACTION_OF_TRAINPOOL = 0.10  # Caduceus train_val_split_seed 90/10


def parse_gene_id(attrs: str) -> str | None:
    m = re.search(r'gene_id\s+"([^"]+)"', attrs)
    return m.group(1) if m else None


def load_wide_tpm(path: Path) -> dict[str, float]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        values = next(reader)
    if len(header) != len(values):
        raise ValueError(f"TPM width mismatch in {path}: {len(header)} vs {len(values)}")
    out: dict[str, float] = {}
    for g, v in zip(header, values):
        g = g.strip()
        if not g:
            continue
        try:
            out[g] = float(v)
        except ValueError as exc:
            raise ValueError(f"non-numeric TPM for {g} in {path}: {v!r}") from exc
    return out


def parse_genes(gtf: Path, genome: str) -> list[dict]:
    rows: list[dict] = []
    with gtf.open() as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom, start_s, end_s, strand, attrs = (
                parts[0],
                parts[3],
                parts[4],
                parts[6],
                parts[8],
            )
            gene_id = parse_gene_id(attrs)
            if not gene_id:
                continue
            start, end = int(start_s), int(end_s)
            if end < start:
                start, end = end, start
            region_id = f"{genome}__{gene_id}"
            rows.append(
                {
                    "region_id": region_id,
                    "genome": genome,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand if strand in ("+", "-") else ".",
                    "gene_id": gene_id,
                }
            )
    if not rows:
        raise ValueError(f"no gene features in {gtf}")
    return rows


def assign_folds(n: int, seed: int) -> list[str]:
    """Return fold label per index after shuffle order (caller shuffles rows)."""
    if n < 3:
        raise ValueError(f"need >=3 regions for train/val/test; got {n}")
    n_test = max(1, int(round(n * TEST_FRACTION)))
    n_remain = n - n_test
    n_val = max(1, int(round(n_remain * VAL_FRACTION_OF_TRAINPOOL)))
    n_train = n_remain - n_val
    if n_train < 1:
        raise ValueError(f"train empty after ratios for n={n}")
    labels = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    if len(labels) != n:
        raise RuntimeError("internal fold label length mismatch")
    return labels


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--split-id", default=SPLIT_ID)
    ap.add_argument(
        "--allowed-genomes",
        nargs="+",
        default=[
            "Escherichia_coli_K12_ASM584v2",
            "Shigella_sonnei_ASM295039v1",
            "Pseudomonas_aeruginosa_ASM676v1",
        ],
    )
    args = ap.parse_args()

    root = args.root.resolve()
    panel = (root / args.panel).resolve() if not args.panel.is_absolute() else args.panel
    out = root / args.out if not args.out.is_absolute() else args.out

    if not panel.is_dir():
        raise FileNotFoundError(f"panel missing: {panel}")

    allowed = list(args.allowed_genomes)
    if ZS_FORBIDDEN in allowed:
        raise SystemExit(f"ZS genome must not be in split pool: {ZS_FORBIDDEN}")

    regions: list[dict] = []
    tpm_stats = {"with_tpm": 0, "pred_zero_no_gene_tpm": 0, "per_genome": {}}

    for genome in sorted(allowed):
        gdir = panel / genome
        fna = gdir / "genome.fna"
        gtf = gdir / "annotation.gtf"
        tpm_path = gdir / "expression_tpm.csv"
        for p, label in ((fna, "fna"), (gtf, "gtf"), (tpm_path, "tpm")):
            if not p.is_file() or p.stat().st_size == 0:
                raise FileNotFoundError(f"missing/empty {label}: {p}")
        tpm = load_wide_tpm(tpm_path)
        genes = parse_genes(gtf, genome)
        n_hit = 0
        n_zero = 0
        for g in genes:
            if g["gene_id"] in tpm:
                g["TPM"] = tpm[g["gene_id"]]
                g["tpm_source"] = "expression_tpm.csv"
                n_hit += 1
            else:
                g["TPM"] = 0.0
                g["tpm_source"] = "no_gene_tpm→0"
                n_zero += 1
            g["fna_src"] = str((gdir / "genome.fna").relative_to(root))
            g["gtf_src"] = str((gdir / "annotation.gtf").relative_to(root))
            g["tpm_src"] = str((gdir / "expression_tpm.csv").relative_to(root))
            regions.append(g)
        tpm_stats["with_tpm"] += n_hit
        tpm_stats["pred_zero_no_gene_tpm"] += n_zero
        tpm_stats["per_genome"][genome] = {
            "n_regions": len(genes),
            "with_tpm": n_hit,
            "pred_zero": n_zero,
            "n_tpm_columns": len(tpm),
        }

    # Deterministic order then seeded shuffle (reproducibility)
    regions = sorted(regions, key=lambda r: r["region_id"])
    rng = random.Random(args.seed)
    rng.shuffle(regions)
    fold_labels = assign_folds(len(regions), args.seed)
    for row, fold in zip(regions, fold_labels):
        row["fold"] = fold
        row["seed"] = str(args.seed)
        row["split_id"] = args.split_id
        row["sequence_path"] = ""  # windows belong to @adapt
        row["pred_path"] = ""  # filled after write

    # Reset OUT fold trees
    if out.exists():
        for fold in ("train", "val", "test"):
            fdir = out / fold
            if fdir.exists():
                shutil.rmtree(fdir)
        zs = out / "zero-shot"
        if zs.exists():
            shutil.rmtree(zs)
        for name in ("fold_manifest.tsv", "split_meta.json", "predictions"):
            p = out / name
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    out.mkdir(parents=True, exist_ok=True)

    by_fold: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in regions:
        by_fold[row["fold"]].append(row)

    region_cols = [
        "region_id",
        "genome",
        "chrom",
        "start",
        "end",
        "strand",
        "gene_id",
        "TPM",
        "tpm_source",
        "fna_src",
        "gtf_src",
        "tpm_src",
        "seed",
        "split_id",
    ]
    label_cols = ["region_id", "TPM", "gene_id", "genome"]
    manifest_cols = [
        "region_id",
        "fold",
        "genome",
        "chrom",
        "start",
        "end",
        "strand",
        "gene_id",
        "TPM",
        "tpm_source",
        "sequence_path",
        "pred_path",
        "regions_path",
        "labels_path",
        "fna_src",
        "gtf_src",
        "tpm_src",
        "seed",
        "split_id",
    ]

    manifest_rows: list[dict] = []
    pred_root = out / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    for fold, rows in by_fold.items():
        rows = sorted(rows, key=lambda r: r["region_id"])
        fold_dir = out / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        regions_rel = str((fold_dir / "regions.tsv").relative_to(root))
        labels_rel = str((fold_dir / "labels.tsv").relative_to(root))
        pred_rel = str((pred_root / f"{fold}.tsv").relative_to(root))

        write_tsv(fold_dir / "regions.tsv", rows, region_cols)
        write_tsv(fold_dir / "labels.tsv", rows, label_cols)
        write_tsv(pred_root / f"{fold}.tsv", rows, label_cols)

        for r in rows:
            if r["genome"] == ZS_FORBIDDEN:
                raise SystemExit(f"ZS genome leaked into fold {fold}: {r['region_id']}")
            man = dict(r)
            man["regions_path"] = regions_rel
            man["labels_path"] = labels_rel
            man["pred_path"] = pred_rel
            man["sequence_path"] = ""
            manifest_rows.append(man)

    write_tsv(out / "fold_manifest.tsv", sorted(manifest_rows, key=lambda r: (r["fold"], r["region_id"])), manifest_cols)

    counts = {f: len(by_fold[f]) for f in ("train", "val", "test")}
    genomes_in_folds = sorted({r["genome"] for r in regions})
    if ZS_FORBIDDEN in genomes_in_folds:
        raise SystemExit("ZS genome present in folds")
    if set(genomes_in_folds) != set(allowed):
        raise SystemExit(f"genome set mismatch: {genomes_in_folds} vs {allowed}")

    # Align check: every region has a label TPM field
    for fold, rows in by_fold.items():
        labels = {r["region_id"]: r["TPM"] for r in rows}
        for r in rows:
            if r["region_id"] not in labels:
                raise SystemExit(f"label missing for {r['region_id']}")

    meta = {
        "run": "cf_random_smoketest_20260726",
        "split_stage": "split1",
        "goal": "predict_TPM",
        "split_id": args.split_id,
        "seed": args.seed,
        "panel": str(args.panel),
        "out": str(args.out),
        "atomic_unit": "GTF gene region + linked TPM",
        "ratios": {
            "test_fraction_of_all": TEST_FRACTION,
            "val_fraction_of_trainpool": VAL_FRACTION_OF_TRAINPOOL,
            "scheme": "shuffle→test holdout→90/10 train/val on remainder (Caduceus-aligned)",
        },
        "counts": counts,
        "n_regions_total": len(regions),
        "tpm_linkage": tpm_stats,
        "zs_excluded": ZS_FORBIDDEN,
        "genomes": genomes_in_folds,
        "rng": "sorted(region_id); random.Random(seed).shuffle; assign contiguous fold blocks",
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps({"out": str(args.out), "counts": counts, "tpm": tpm_stats}, indent=2))


if __name__ == "__main__":
    main()
