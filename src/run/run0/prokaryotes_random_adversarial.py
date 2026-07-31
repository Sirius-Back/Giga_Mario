#!/usr/bin/env python3
"""Build and re-split the run0 adversarial panel with fold-class PREDICT."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
from src.pipeline.split import run_split
from src.pipeline.split_predict import run_split_predict


def run(*, panel_root: Path, direct_root: Path, adversarial_root: Path) -> Path:
    """Copy real prepared artifacts, random 1:1:3 split, rewrite class targets."""
    run_adversarial(
        outdir_new=adversarial_root,
        split_csv=direct_root / "split.csv",
        parsed_target=panel_root / "PREDICT",
        parsed_data=panel_root / "PARSED",
        intersect_allow=True,
    )
    split_csv = run_split_predict(
        outdir=adversarial_root,
        type="random",
        seed=43,  # direct uses 42; M2-style remix (seed+1)
        id_csv=panel_root / "ID.csv",
        fold_csv=panel_root / "fold.csv",
        ratios=(1, 1, 3),
    )
    # M2-style targets from **previous (direct) split** train/val/test → 0/1/2.
    # New adversarial split_csv only defines training folds (must mix classes).
    apply_fold_class_targets(
        predict_root=adversarial_root / "PREDICT",
        label_split_csv=direct_root / "split.csv",
    )
    return run_split(
        split_csv,
        parsed_target=adversarial_root / "PREDICT",
        parsed_data=adversarial_root / "PARSED",
        outdir=adversarial_root,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=panel_root / "ID.csv",
    )
