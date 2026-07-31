#!/usr/bin/env python3
"""Direct run0 split and human_legnet training stage (+ ZSV final eval)."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.legnet_input import build_legnet_tsv
from src.pipeline.split import run_split
from src.pipeline.split_predict import run_split_predict
from src.pipeline.train import run_train


def run(*, panel_root: Path, out_root: Path, epochs: int, run_training: bool) -> Path:
    """Split the prepared panel, then optionally train + evaluate ZSV."""
    split_csv = run_split_predict(
        outdir=out_root,
        type="random",
        seed=42,
        id_csv=panel_root / "ID.csv",
        fold_csv=panel_root / "fold.csv",
        ratios=(1, 1, 3),
    )
    split_root = run_split(
        split_csv,
        parsed_target=panel_root / "PREDICT",
        parsed_data=panel_root / "PARSED",
        outdir=out_root,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=panel_root / "ID.csv",
    )
    tsv = build_legnet_tsv(
        split_root=split_root, out_tsv=out_root / "legnet_input" / "all.tsv"
    )
    if run_training:
        run_train(
            model="legnet",
            type="regression",
            folders=tsv,
            outdir=out_root / "direct",
            strategy="random",
            epochs=epochs,
            batch_size=1024,
            seed=42,
            n_devices=4,
            num_workers=8,
            legnet_demo=True,
            zsv_root=out_root,
            eval_zsv=True,
        )
    return tsv
