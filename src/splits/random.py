#!/usr/bin/env python3
"""Random split strategy: M1 (TPM) + M2 (predict M1 fold, stratified).

Inputs (read-only; never convert ready files):
  raw/     — available for strategies that need genomic metadata
  ready/   — ready.csv + caduceus_ready/ (or data_ready/)

Outputs:
  splits/random/M1/{train,val,test}/   — prediction target: TPM
  splits/random/M2/{train,val,test}/   — prediction target: M1 fold (stratified)
  splits/random/splits_log.csv         — data_input|M1|M2
"""
from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    assign_folds_random,
    assign_folds_stratified,
    load_ready_table,
    materialize_fold,
    resolve_raw_dir,
    resolve_ready_dir,
    write_pipe_csv,
    write_tsv,
)

# Re-export for pipeline split-predict (only random strategy is wired).
__all__ = (
    "run_random_split",
    "assign_folds_random",
    "assign_folds_stratified",
    "SPLIT_ID",
    "M1_FOLD_TO_CLASS",
)

SPLIT_ID = "random"
M1_FOLD_TO_CLASS = {"train": 0, "val": 1, "test": 2}


def run_random_split(
    root: Path,
    *,
    raw_dir: Path | None = None,
    ready_dir: Path | None = None,
    out_dir: Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    holdout_genomes: list[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    raw = resolve_raw_dir(root, raw_dir)
    ready = resolve_ready_dir(root, ready_dir)
    out = out_dir if out_dir is not None else Path("splits") / "random"
    out = out if out.is_absolute() else root / out

    # raw is validated for presence; random assignment uses ready IDs only
    _ = raw

    all_samples = load_ready_table(ready)
    all_samples = sorted(all_samples, key=lambda r: r["sample_id"])
    holdout = set(holdout_genomes or [])
    zs_samples = [s for s in all_samples if s["Genome"] in holdout]
    samples = [s for s in all_samples if s["Genome"] not in holdout]
    if max_samples is not None:
        samples = samples[:max_samples]
        # keep ZS uncapped unless empty after filter
    if len(samples) < 3:
        raise ValueError(f"need >=3 non-holdout ready samples; got {len(samples)}")

    # --- M1: random fold assignment; target = TPM ---
    rng1 = random.Random(seed)
    order1 = list(range(len(samples)))
    rng1.shuffle(order1)
    folds_m1 = assign_folds_random(len(samples))
    for idx, fold in zip(order1, folds_m1):
        samples[idx]["M1"] = fold
        samples[idx]["label_tpm"] = samples[idx]["TPM"]

    # --- M2: stratified by M1; target = M1 fold class ---
    rng2 = random.Random(seed + 1)
    strata = [s["M1"] for s in samples]
    folds_m2 = assign_folds_stratified(samples, strata, rng2)
    for s, fold in zip(samples, folds_m2):
        s["M2"] = fold
        s["label_m1_fold"] = s["M1"]
        s["label"] = M1_FOLD_TO_CLASS[s["M1"]]
        s["label_name"] = s["M1"]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Materialize M1 / M2 fold trees
    counts: dict[str, dict[str, int]] = {"M1": {}, "M2": {}}
    for model, label_field, label_fields in (
        ("M1", "TPM", ["TPM"]),
        ("M2", "label", ["label", "label_name", "TPM"]),
    ):
        for fold in ("train", "val", "test"):
            subset = [s for s in samples if s[model] == fold]
            counts[model][fold] = len(subset)
            # attach prediction alias expected by materialize
            if model == "M1":
                for s in subset:
                    s["TPM"] = s["label_tpm"]
            materialize_fold(
                out / model / fold,
                subset,
                label_field=label_field,
                label_fields=label_fields,
            )

    # Zero-shot-validation holdout — TPM labels; not used in M1/M2 train/val/test
    if zs_samples:
        for s in zs_samples:
            s["TPM"] = s["TPM"]
            s["M1"] = "zsv"
            s["M2"] = "zsv"
        materialize_fold(
            out / "zero-shot-validation" / "all",
            zs_samples,
            label_field="TPM",
            label_fields=["TPM"],
        )
        write_tsv(
            out / "zero-shot-validation" / "fold_manifest.tsv",
            [
                {
                    "sample_id": s["sample_id"],
                    "fold": "zsv",
                    "genome": s["Genome"],
                    "gene_id": s["GeneOrID"],
                    "chrom": s["Chr"],
                    "start": s["Position_start"],
                    "end": s["Position_end"],
                    "TPM": s["TPM"],
                    "sequence_path": (
                        f"zero-shot-validation/all/sequences/{s['sample_id']}.txt"
                    ),
                    "seed": seed,
                    "split_id": SPLIT_ID,
                }
                for s in zs_samples
            ],
            [
                "sample_id",
                "fold",
                "genome",
                "gene_id",
                "chrom",
                "start",
                "end",
                "TPM",
                "sequence_path",
                "seed",
                "split_id",
            ],
        )

    # splits_log.csv: data_input|M1|M2
    log_rows = [
        {"data_input": s["sample_id"], "M1": s["M1"], "M2": s["M2"]}
        for s in sorted(samples, key=lambda r: r["sample_id"])
    ]
    log_rows.extend(
        {
            "data_input": s["sample_id"],
            "M1": "zsv",
            "M2": "zsv",
        }
        for s in sorted(zs_samples, key=lambda r: r["sample_id"])
    )
    write_pipe_csv(out / "splits_log.csv", log_rows, ["data_input", "M1", "M2"])

    # manifests
    for model in ("M1", "M2"):
        man = []
        for s in samples:
            man.append(
                {
                    "sample_id": s["sample_id"],
                    "fold": s[model],
                    "genome": s["Genome"],
                    "gene_id": s["GeneOrID"],
                    "chrom": s["Chr"],
                    "start": s["Position_start"],
                    "end": s["Position_end"],
                    "TPM": s["TPM"],
                    "M1": s["M1"],
                    "M2": s["M2"],
                    "label": s.get("label", ""),
                    "label_name": s.get("label_name", ""),
                    "sequence_path": (
                        f"{model}/{s[model]}/sequences/{s['sample_id']}.txt"
                    ),
                    "seed": seed,
                    "split_id": SPLIT_ID,
                }
            )
        write_tsv(
            out / model / "fold_manifest.tsv",
            man,
            [
                "sample_id",
                "fold",
                "genome",
                "gene_id",
                "chrom",
                "start",
                "end",
                "TPM",
                "M1",
                "M2",
                "label",
                "label_name",
                "sequence_path",
                "seed",
                "split_id",
            ],
        )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_id": SPLIT_ID,
        "seed": seed,
        "m2_shuffle_seed": seed + 1,
        "raw_dir": str(raw.relative_to(root)) if raw.is_relative_to(root) else str(raw),
        "ready_dir": str(ready.relative_to(root))
        if ready.is_relative_to(root)
        else str(ready),
        "out_dir": str(out.relative_to(root)) if out.is_relative_to(root) else str(out),
        "n_samples": len(samples),
        "n_zero_shot": len(zs_samples),
        "holdout_genomes": sorted(holdout),
        "counts": counts,
        "m1_target": "TPM",
        "m2_target": "M1_fold_class",
        "m2_label_encoding": M1_FOLD_TO_CLASS,
        "m2_stratified_by": "M1",
        "ratios": {
            "test_fraction": 0.10,
            "val_fraction_of_train_pool": 0.10,
        },
        "note": "Ready files were not converted; sequences hardlinked/symlinked.",
    }
    if zs_samples:
        meta["counts"]["zero-shot-validation"] = {"all": len(zs_samples)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zs_line = (
        f"- **zero-shot-validation (zsv)** — holdout genomes {sorted(holdout)} "
        f"(n={len(zs_samples)}); TPM eval only; not used in M1/M2"
        if zs_samples
        else "- **zero-shot-validation** — none"
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                f"# {out.name}",
                "",
                "- **M1** — random train/val/test; prediction = **TPM**",
                "- **M2** — stratified by M1 fold; prediction = **M1 fold class** "
                f"({M1_FOLD_TO_CLASS})",
                zs_line,
                "- **splits_log.csv** — `data_input|M1|M2`",
                "",
                f"Seed: {seed}. Train-pool samples: {len(samples)}.",
                "",
                "Produced by `src/splits/random.py`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return meta
