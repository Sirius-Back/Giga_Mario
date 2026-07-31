#!/usr/bin/env python3
"""Train human_legnet on the run0 adversarial SPLIT (fold-class targets)."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.legnet_input import build_legnet_tsv
from src.pipeline.train import run_train


def run(*, adversarial_root: Path, epochs: int, run_training: bool) -> Path:
    """Build LegNet TSV from fold-class PREDICT and optionally train + ZSV eval."""
    tsv = build_legnet_tsv(
        split_root=adversarial_root / "SPLIT",
        out_tsv=adversarial_root / "legnet_input" / "all.tsv",
    )
    if run_training:
        run_train(
            model="legnet",
            # Integer 0/1/2 targets; human_legnet remains MSE until CE head exists.
            type="classification",
            folders=tsv,
            outdir=adversarial_root / "train",
            strategy="random",
            epochs=epochs,
            batch_size=1024,
            seed=42,
            n_devices=4,
            num_workers=8,
            legnet_demo=True,
            zsv_root=adversarial_root,
            eval_zsv=True,
        )
    return tsv
