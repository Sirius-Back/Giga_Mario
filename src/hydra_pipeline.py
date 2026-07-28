#!/usr/bin/env python3
"""Hydra entry for the universal train pipeline (reproducible configs).

Examples::

  # dry — validate + stage split/adversarial class targets; no full train
  python -m src.hydra_pipeline mode=dry

  # run — full train + ZSV (uses configs/pipeline.yaml + configs/train/*)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.hydra_pipeline mode=run \\
    run_id=run0 epochs=3 n_devices=4

Concrete model CLIs live under ``configs/train/*.yaml`` (``direct_cmd``,
``adversarial_cmd``, ``zsv_cmd``) — not ad-hoc ``model_dir`` flags on the
orchestrator.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _run_stages(cfg: DictConfig) -> int:
    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train

    mode = str(cfg.mode)
    if mode not in {"dry", "run"}:
        raise ValueError(f"mode must be dry|run, got {mode!r}")

    panel_root = _resolve_path(cfg.panel_root)
    out_root = _resolve_path(cfg.out_root)
    for required in ("ID.csv", "fold.csv", "PARSED", "PREDICT"):
        path = panel_root / required
        if not path.exists():
            raise FileNotFoundError(f"Panel missing {required}: {path}")

    ratios = tuple(int(x) for x in cfg.ratios)
    if len(ratios) != 3:
        raise ValueError(f"ratios must be length-3 train:test:val, got {ratios}")

    seed = int(cfg.seed)
    epochs = int(cfg.epochs)
    train_name = str(cfg.train.name).lower()
    run_training = mode == "run"
    eval_zsv = bool(cfg.zsv)

    # --- direct split ---
    split_csv = run_split_predict(
        outdir=out_root,
        type=str(cfg.split),
        seed=seed,
        id_csv=panel_root / "ID.csv",
        fold_csv=panel_root / "fold.csv",
        ratios=ratios,
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

    if train_name in {"legnet", "human_legnet"}:
        direct_tsv = build_legnet_tsv(
            split_root=split_root, out_tsv=out_root / "legnet_input" / "all.tsv"
        )
        folders = direct_tsv
    else:
        folders = split_root

    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(i) for i in range(int(cfg.n_devices))),
    )

    run_train(
        model=train_name,
        type=str(cfg.task_type),
        folders=folders,
        outdir=out_root / "direct",
        strategy=str(cfg.split),
        smoke=not run_training,
        epochs=epochs,
        batch_size=int(cfg.batch_size),
        seed=seed,
        n_devices=int(cfg.n_devices),
        num_workers=int(cfg.num_workers),
        legnet_demo=train_name in {"legnet", "human_legnet"},
        zsv_root=out_root if eval_zsv else None,
        eval_zsv=eval_zsv and run_training,
    )
    # During/after-train monitoring figures (learning curves + split compare)
    try:
        from src.train_viz.train_monitor import refresh_train_monitor

        mon = refresh_train_monitor(
            out_root / "direct",
            model=f"{cfg.run_id}_direct",
            title=f"{cfg.run_id} direct — train monitor",
            include_split_compare=True,
        )
        print(f"train_monitor[direct] status={mon.get('status')} → {mon.get('outdir')}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: train_monitor[direct] skipped: {type(exc).__name__}: {exc}")

    if not bool(cfg.adversarial):
        print(OmegaConf.to_yaml(cfg))
        print(f"Hydra pipeline {mode} complete (direct only) → {out_root}")
        return 0

    # --- adversarial: copy → re-split → fold-class PREDICT → materialize → train ---
    adv_root = out_root / "adversarial"
    if adv_root.exists():
        import shutil

        shutil.rmtree(adv_root)
    run_adversarial(
        outdir_new=adv_root,
        split_csv=out_root / "split.csv",
        parsed_target=panel_root / "PREDICT",
        parsed_data=panel_root / "PARSED",
        intersect_allow=True,
    )
    # New training folds must differ from direct/M1 (M2 uses seed+1).
    adv_seed = int(seed) + 1
    adv_split = run_split_predict(
        outdir=adv_root,
        type="random",
        seed=adv_seed,
        id_csv=panel_root / "ID.csv",
        fold_csv=panel_root / "fold.csv",
        ratios=ratios,
    )
    # Labels = previous direct/M1 train/val/test → 0/1/2 (not the new adv split).
    apply_fold_class_targets(
        predict_root=adv_root / "PREDICT",
        label_split_csv=out_root / "split.csv",
    )
    run_split(
        adv_split,
        parsed_target=adv_root / "PREDICT",
        parsed_data=adv_root / "PARSED",
        outdir=adv_root,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=panel_root / "ID.csv",
    )

    if train_name in {"legnet", "human_legnet"}:
        adv_folders = build_legnet_tsv(
            split_root=adv_root / "SPLIT",
            out_tsv=adv_root / "legnet_input" / "all.tsv",
        )
    else:
        adv_folders = adv_root / "SPLIT"

    run_train(
        model=train_name,
        type=str(cfg.adversarial_task_type),
        folders=adv_folders,
        outdir=adv_root / "train",
        strategy="random",
        smoke=not run_training,
        epochs=epochs,
        batch_size=int(cfg.batch_size),
        seed=seed,
        n_devices=int(cfg.n_devices),
        num_workers=int(cfg.num_workers),
        legnet_demo=train_name in {"legnet", "human_legnet"},
        zsv_root=adv_root if eval_zsv else None,
        eval_zsv=eval_zsv and run_training,
    )
    try:
        from src.train_viz.train_monitor import refresh_train_monitor

        mon = refresh_train_monitor(
            adv_root / "train",
            model=f"{cfg.run_id}_adversarial",
            title=f"{cfg.run_id} adversarial — train monitor",
            include_split_compare=True,
        )
        print(
            f"train_monitor[adversarial] status={mon.get('status')} → {mon.get('outdir')}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: train_monitor[adversarial] skipped: {type(exc).__name__}: {exc}"
        )

    # Final pass: direct + adversarial monitors + TensorBoard export
    try:
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        pipe_mon = refresh_pipeline_monitors(
            out_root,
            run_id=str(cfg.run_id),
            include_split_compare=True,
        )
        print(
            f"pipeline_monitors status={pipe_mon.get('status')} "
            f"direct={((pipe_mon.get('direct') or {}).get('status'))} "
            f"adversarial={((pipe_mon.get('adversarial') or {}).get('status'))}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: pipeline_monitors skipped: {type(exc).__name__}: {exc}")

    # Persist resolved commands for audit
    cmds = {
        "direct_cmd": str(cfg.train.direct_cmd),
        "adversarial_cmd": str(cfg.train.adversarial_cmd),
        "zsv_cmd": str(cfg.train.zsv_cmd),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "hydra_resolved_commands.yaml").write_text(
        OmegaConf.to_yaml(cmds), encoding="utf-8"
    )
    (out_root / "hydra_resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )

    print(f"Hydra pipeline {mode} complete → {out_root}")
    print(f"Model launch templates:\n{OmegaConf.to_yaml(cmds)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Compose ``configs/pipeline.yaml`` and run stages (no ``@hydra.main``).

    Avoids Hydra ``@hydra.main`` + CPython 3.14 argparse ``LazyCompletionHelp`` crash.
    Overrides are plain Hydra-style CLI tokens: ``mode=run epochs=3``.
    """
    from hydra import compose, initialize_config_dir

    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="pipeline", overrides=overrides)
    try:
        return _run_stages(cfg)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
