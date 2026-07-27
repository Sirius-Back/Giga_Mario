#!/usr/bin/env python3
"""Caduceus-full Split #2: secondary random split; target = split1 fold class.

Reads Split #1 fold_manifest.tsv. Primary learning target is split-1 fold
membership encoded as class ids 0=train, 1=val, 2=test (NOT TPM).
Re-assigns regions into OUT_SPLIT2 train/val/test via the same Caduceus-aligned
seeded random ratios as Split #1 (seed 42). region_id values are preserved so
rows remain joinable to split1. Never invents TPM.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

SEED_DEFAULT = 42
SPLIT_ID = "random"
RUN_DEFAULT = "cf_random_smoketest_20260726"
IN_DEFAULT = Path("data_splits/caduceus_full/cf_random_smoketest_20260726/split1")
OUT_DEFAULT = Path("data_splits/caduceus_full/cf_random_smoketest_20260726/split2")
TEST_FRACTION = 0.10
VAL_FRACTION_OF_TRAINPOOL = 0.10

# Locked encoding (method-decision.md): split1 fold → class id
SPLIT1_FOLD_TO_CLASS = {"train": 0, "val": 1, "test": 2}
CLASS_TO_SPLIT1_FOLD = {v: k for k, v in SPLIT1_FOLD_TO_CLASS.items()}


def assign_folds(n: int) -> list[str]:
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


def load_split1_manifest(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"split1 fold_manifest missing/empty: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    required = {"region_id", "fold", "genome", "chrom", "start", "end", "strand", "gene_id"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--in-split1", type=Path, default=IN_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="RNG seed for Split-2 fold reassignment. Default: seed+1 so Split-2 "
        "is independent of Split-1 when both use the same algorithm+seed.",
    )
    ap.add_argument("--split-id", default=SPLIT_ID)
    ap.add_argument("--run", default=RUN_DEFAULT)
    args = ap.parse_args()

    root = args.root.resolve()
    in_split1 = root / args.in_split1 if not args.in_split1.is_absolute() else args.in_split1
    out = root / args.out if not args.out.is_absolute() else args.out
    manifest_path = in_split1 / "fold_manifest.tsv"

    split1_rows = load_split1_manifest(manifest_path)

    regions: list[dict] = []
    seen: set[str] = set()
    class_counts = {0: 0, 1: 0, 2: 0}

    for r in split1_rows:
        rid = r["region_id"].strip()
        if not rid:
            raise ValueError("empty region_id in split1 manifest")
        if rid in seen:
            raise ValueError(f"duplicate region_id in split1: {rid}")
        seen.add(rid)
        fold1 = r["fold"].strip()
        if fold1 not in SPLIT1_FOLD_TO_CLASS:
            raise ValueError(f"unknown split1 fold {fold1!r} for {rid}")
        label_class = SPLIT1_FOLD_TO_CLASS[fold1]
        class_counts[label_class] += 1
        # TPM may be present for audit/join but is NOT the primary target
        tpm_raw = r.get("TPM", "")
        regions.append(
            {
                "region_id": rid,
                "genome": r["genome"],
                "chrom": r["chrom"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "strand": r["strand"],
                "gene_id": r["gene_id"],
                "split1_fold": fold1,
                "label": label_class,
                "label_name": fold1,
                "TPM_audit": tpm_raw,  # audit only; not primary prediction target
                "tpm_source": r.get("tpm_source", ""),
                "fna_src": r.get("fna_src", ""),
                "gtf_src": r.get("gtf_src", ""),
                "tpm_src": r.get("tpm_src", ""),
                "split1_regions_path": r.get("regions_path", ""),
                "split1_labels_path": r.get("labels_path", ""),
                "split1_pred_path": r.get("pred_path", ""),
                "seed": str(args.seed),
                "split_id": args.split_id,
                "sequence_path": "",
            }
        )

    # Deterministic order then seeded shuffle (independent of split1 shuffle).
    # Same algorithm+seed as Split-1 yields identical fold membership (verified
    # seed=42 → 100% fold==split1_fold). Default shuffle_seed=seed+1 breaks that.
    regions = sorted(regions, key=lambda x: x["region_id"])
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed + 1
    rng = random.Random(shuffle_seed)
    rng.shuffle(regions)
    fold_labels = assign_folds(len(regions))
    for row, fold2 in zip(regions, fold_labels):
        row["fold"] = fold2  # split2 fold membership

    # Reset OUT
    if out.exists():
        for fold in ("train", "val", "test"):
            fdir = out / fold
            if fdir.exists():
                shutil.rmtree(fdir)
        zs = out / "zero-shot"
        if zs.exists():
            shutil.rmtree(zs)
        for name in ("fold_manifest.tsv", "split_meta.json", "predictions", "label_encoding.json"):
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
        "split1_fold",
        "label",
        "label_name",
        "fna_src",
        "gtf_src",
        "tpm_src",
        "split1_regions_path",
        "split1_labels_path",
        "seed",
        "split_id",
    ]
    # Primary prediction target = split1 fold class (not TPM)
    label_cols = ["region_id", "label", "label_name", "split1_fold", "gene_id", "genome"]
    manifest_cols = [
        "region_id",
        "fold",
        "genome",
        "chrom",
        "start",
        "end",
        "strand",
        "gene_id",
        "split1_fold",
        "label",
        "label_name",
        "sequence_path",
        "pred_path",
        "regions_path",
        "labels_path",
        "fna_src",
        "gtf_src",
        "tpm_src",
        "split1_regions_path",
        "split1_labels_path",
        "split1_pred_path",
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
            man = dict(r)
            man["regions_path"] = regions_rel
            man["labels_path"] = labels_rel
            man["pred_path"] = pred_rel
            man["sequence_path"] = ""
            manifest_rows.append(man)

    write_tsv(
        out / "fold_manifest.tsv",
        sorted(manifest_rows, key=lambda r: (r["fold"], r["region_id"])),
        manifest_cols,
    )

    counts = {f: len(by_fold[f]) for f in ("train", "val", "test")}

    # Joinability: every split2 region_id exists in split1
    split1_ids = {r["region_id"] for r in split1_rows}
    split2_ids = {r["region_id"] for r in regions}
    if split2_ids != split1_ids:
        only1 = sorted(split1_ids - split2_ids)[:5]
        only2 = sorted(split2_ids - split1_ids)[:5]
        raise SystemExit(f"region_id set mismatch vs split1; only1={only1} only2={only2}")

    # Labels must encode split1 fold, not TPM as primary
    for fold, rows in by_fold.items():
        for r in rows:
            if int(r["label"]) != SPLIT1_FOLD_TO_CLASS[r["split1_fold"]]:
                raise SystemExit(f"label mismatch for {r['region_id']}")

    encoding = {
        "primary_target": "split1_fold_class",
        "encoding": {"0": "train", "1": "val", "2": "test"},
        "split1_fold_to_class": SPLIT1_FOLD_TO_CLASS,
        "class_to_split1_fold": CLASS_TO_SPLIT1_FOLD,
        "tpm_is_primary_target": False,
        "status": "Locked",
    }
    (out / "label_encoding.json").write_text(json.dumps(encoding, indent=2) + "\n")

    meta = {
        "run": args.run,
        "split_stage": "split2",
        "goal": "predict_split1_fold_membership",
        "split_id": args.split_id,
        "seed": args.seed,
        "shuffle_seed": shuffle_seed,
        "in_split1": str(args.in_split1),
        "out": str(args.out),
        "atomic_unit": "same region_id as split1; label = split1 fold class",
        "primary_target": "split1_fold_class",
        "label_encoding": encoding,
        "ratios": {
            "test_fraction_of_all": TEST_FRACTION,
            "val_fraction_of_trainpool": VAL_FRACTION_OF_TRAINPOOL,
            "scheme": "shuffle→test holdout→90/10 train/val on remainder (Caduceus-aligned; same as split1)",
        },
        "counts": counts,
        "n_regions_total": len(regions),
        "split1_class_counts": {str(k): v for k, v in class_counts.items()},
        "rng": "sorted(region_id); random.Random(shuffle_seed).shuffle; assign contiguous fold blocks; shuffle_seed defaults to seed+1 for independence from split1",
        "join_key": "region_id",
        "joinable_to_split1": True,
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        json.dumps(
            {
                "out": str(args.out),
                "counts": counts,
                "split1_class_counts": class_counts,
                "n_regions": len(regions),
                "primary_target": "split1_fold_class",
                "label_encoding": SPLIT1_FOLD_TO_CLASS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
