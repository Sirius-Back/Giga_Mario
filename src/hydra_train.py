#!/usr/bin/env python3
"""Hydra entry for ``/train`` (LegNet or Caduceus).

Composes ``configs/train_job.yaml`` + ``configs/train/{legnet,caduceus}.yaml``
and dispatches to ``src.pipeline.train.run_train`` (dual TensorBoard finalized
there). Avoids ``@hydra.main`` (CPython 3.14 × Hydra argparse bug).

Examples::

  python -m src.hydra_train mode=direct train=legnet run_id=run0 epochs=3
  python -m src.hydra_train mode=adversarial train=caduceus run_id=run0 zsv=true
"""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def _infer_folders(cfg) -> Path:
    if cfg.folders not in (None, "", "null"):
        return Path(str(cfg.folders))
    out_root = Path(str(cfg.out_root))
    name = str(cfg.train.name).lower()
    mode = str(cfg.mode).lower()
    if mode == "adversarial":
        if name in {"legnet", "human_legnet"}:
            return out_root / "adversarial" / "legnet_input" / "all.tsv"
        split = out_root / "adversarial" / "SPLIT"
        adapted = out_root / "adversarial" / "train" / "caduceus_input"
        if adapted.is_dir():
            return adapted
        return split
    if name in {"legnet", "human_legnet"}:
        return out_root / "legnet_input" / "all.tsv"
    adapted = out_root / "direct" / "caduceus_input"
    if adapted.is_dir():
        return adapted
    split = out_root / "SPLIT"
    if split.is_dir():
        return split
    raise FileNotFoundError(
        f"Cannot infer Caduceus folders under {out_root}; set folders=…"
    )


def _infer_outdir(cfg) -> Path:
    if cfg.outdir not in (None, "", "null"):
        return Path(str(cfg.outdir))
    out_root = Path(str(cfg.out_root))
    if str(cfg.mode).lower() == "adversarial":
        return out_root / "adversarial" / "train"
    return out_root / "direct"


def _run(cfg) -> int:
    from src.pipeline.train import run_train

    mode = str(cfg.mode).lower()
    if mode not in {"direct", "adversarial"}:
        raise ValueError(f"mode must be direct|adversarial, got {mode!r}")

    train_name = str(cfg.train.name).lower()
    if train_name not in {"legnet", "human_legnet", "caduceus"}:
        raise ValueError(f"Unknown train.name={train_name!r}")

    task = (
        str(cfg.adversarial_task_type)
        if mode == "adversarial"
        else str(cfg.task_type)
    )
    folders = _infer_folders(cfg)
    outdir = _infer_outdir(cfg)
    zsv = bool(cfg.zsv)
    smoke = bool(cfg.smoke)
    panel = Path(str(cfg.panel_root))

    print(
        OmegaConf.to_yaml(
            {
                "mode": mode,
                "train": train_name,
                "task_type": task,
                "folders": str(folders),
                "outdir": str(outdir),
                "epochs": int(cfg.epochs),
                "zsv": zsv,
                "smoke": smoke,
                "direct_cmd": str(cfg.train.direct_cmd),
                "adversarial_cmd": str(cfg.train.adversarial_cmd),
            }
        )
    )

    if not folders.exists():
        raise FileNotFoundError(f"Train folders missing: {folders}")

    run_train(
        model=train_name,
        type=task,
        folders=folders,
        outdir=outdir,
        strategy=str(cfg.split),
        smoke=smoke,
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        seed=int(cfg.seed),
        n_devices=int(cfg.n_devices),
        num_workers=int(cfg.num_workers),
        legnet_demo=bool(cfg.legnet_demo) and train_name in {"legnet", "human_legnet"},
        zsv_root=panel if zsv else None,
        eval_zsv=zsv and not smoke,
        checkpoint_every_n_epochs=int(cfg.get("checkpoint_every_n_epochs", 10)),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 0) or 0),
        min_epochs=int(cfg.get("min_epochs", 0) or 0),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hydra_train_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )
    print(f"Hydra /train complete → {outdir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from hydra import compose, initialize_config_dir

    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="train_job", overrides=overrides)
    try:
        return _run(cfg)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
