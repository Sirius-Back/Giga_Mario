#!/usr/bin/env python3
"""Hydra entry for the universal train pipeline (reproducible configs).

Examples::

  # dry — validate + stage split/adversarial class targets; no full train
  python -m src.hydra_pipeline mode=dry

  # run — full train + ZSV (uses configs/pipeline.yaml + configs/train/*)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.hydra_pipeline mode=run \\
    run_id=run0 epochs=3 n_devices=4

  # rerun — fine-tune under runs_aligned/ without overwriting a prior run
  python -m src.hydra_pipeline mode=run rerun=true \\
    source_split=runs/run3 run_id=run3_caduceus_aligned train=caduceus

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


def _run_stages(cfg: DictConfig, *, cli_overrides: set[str] | None = None) -> int:
    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.rerun_aligned import (
        ALIGNED_EVAL_MAX_SAMPLES,
        ALIGNED_RATIOS,
        apply_rerun_schedule,
        assert_fresh_out_root,
        default_aligned_out_root,
        require_aligned_ratios,
        resolve_source_artifacts,
        stage_source_into_out_root,
        write_rerun_manifest,
    )
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train

    mode = str(cfg.mode)
    if mode not in {"dry", "run"}:
        raise ValueError(f"mode must be dry|run, got {mode!r}")

    overridden = set(cli_overrides or ())
    rerun = bool(cfg.get("rerun", False))
    source_split_cfg = cfg.get("source_split", None)
    if source_split_cfg in (None, "", "null"):
        source_hint: str | None = None
    else:
        source_hint = str(source_split_cfg)

    panel_root = _resolve_path(cfg.panel_root)
    if rerun and "out_root" not in overridden:
        out_root = default_aligned_out_root(str(cfg.run_id), project_root=ROOT)
    else:
        out_root = _resolve_path(cfg.out_root)

    for required in ("ID.csv", "fold.csv", "PARSED", "PREDICT"):
        path = panel_root / required
        if not path.exists():
            raise FileNotFoundError(f"Panel missing {required}: {path}")

    ratios_cfg = cfg.get("ratios", None)
    if ratios_cfg is None:
        ratios = None
    else:
        ratios = tuple(float(x) for x in ratios_cfg)
        if len(ratios) != 3 or any(v <= 0 for v in ratios):
            raise ValueError(
                f"ratios must be null or length-3 positive train:test:val, got {ratios_cfg!r}"
            )

    seed = int(cfg.seed)
    epochs = int(cfg.epochs)
    min_epochs = int(cfg.get("min_epochs", 0) or 0)
    early_stop = int(cfg.get("early_stopping_patience", 0) or 0)
    train_name = str(cfg.train.name).lower()
    run_training = mode == "run"
    eval_zsv = bool(cfg.zsv)
    reuse_folds = False
    source_artifacts = None
    rerun_stage_info: dict | None = None

    if rerun:
        # Aligned defaults: max 30 epochs, min 10, early stop allowed; never
        # overwrite prior run trees (fresh out_root under runs_aligned/).
        epochs, min_epochs, early_stop = apply_rerun_schedule(
            epochs=epochs,
            min_epochs=min_epochs,
            early_stopping_patience=early_stop,
            overridden=overridden,
        )
        assert_fresh_out_root(out_root)
        if source_hint:
            source_artifacts = resolve_source_artifacts(source_hint, project_root=ROOT)
            # Never write into the source run root.
            if out_root.resolve() == source_artifacts.root.resolve():
                raise FileExistsError(
                    "rerun out_root must differ from source_split root "
                    f"({source_artifacts.root}). Use runs_aligned/<new_run_id>."
                )
            if source_artifacts.split_csv.resolve().is_relative_to(out_root.resolve()):
                raise FileExistsError(
                    "source_split resolves under out_root; refusing self-overwrite"
                )
            reuse_folds = True
        else:
            # Fresh train/test/val only at ~3:1:1 (no other ratio schedules).
            if "ratios" not in overridden or ratios is None:
                ratios = ALIGNED_RATIOS
            ratios = require_aligned_ratios(ratios)

    # Fresh destination only — never mutate a prior run tree.
    if rerun:
        assert_fresh_out_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- direct split ---
    split_type = str(cfg.split)
    marked_cfg = cfg.get("marked", None)
    if marked_cfg:
        marked_path: Path | None = _resolve_path(marked_cfg)
    elif split_type in {"gc", "kmer", "hashfrag"}:
        marked_path = panel_root / "MARKED"
        if not marked_path.is_dir():
            raise FileNotFoundError(
                f"split={split_type} requires MARKED dir (or marked=… override): "
                f"{marked_path}"
            )
    else:
        marked_path = None
    max_ids_cfg = cfg.get("max_ids", None)
    max_ids = int(max_ids_cfg) if max_ids_cfg not in (None, "", "null") else None
    plot_split = bool(cfg.get("plot_split", True))
    cluster_method = str(cfg.get("cluster_method", "auto"))
    threshold_cfg = cfg.get("threshold", None)
    threshold = (
        float(threshold_cfg)
        if threshold_cfg not in (None, "", "null")
        else None
    )
    p_train_cfg = cfg.get("p_train", None)
    p_test_cfg = cfg.get("p_test", None)
    p_train = float(p_train_cfg) if p_train_cfg not in (None, "", "null") else None
    p_test = float(p_test_cfg) if p_test_cfg not in (None, "", "null") else None
    threads_cfg = cfg.get("threads", None)
    if threads_cfg in (None, "", "null"):
        threads = max(2, int(cfg.n_devices) * 2)
    else:
        threads = int(threads_cfg)
    force_split = bool(cfg.get("force", False))
    if rerun and force_split:
        raise ValueError(
            "rerun=true forbids force=true (would overwrite prior split/BLAST trees)"
        )
    kmer_size_cfg = cfg.get("kmer_size", None)
    if kmer_size_cfg in (None, "", "null"):
        kmer_size: int | tuple[int, ...] = 5
    elif OmegaConf.is_list(kmer_size_cfg) or isinstance(kmer_size_cfg, (list, tuple)):
        ks = tuple(int(x) for x in kmer_size_cfg)
        kmer_size = ks[0] if len(ks) == 1 else ks
    else:
        kmer_size = int(kmer_size_cfg)
    kmer_engine = str(cfg.get("kmer_engine", cfg.get("engine", "auto")) or "auto")
    log_transform = bool(cfg.get("log_transform", False))

    # blastp / pangenome A2A inputs
    gtf_dir_cfg = cfg.get("gtf_dir", None)
    fna_dir_cfg = cfg.get("fna_dir", None)
    gtf_dir = _resolve_path(gtf_dir_cfg) if gtf_dir_cfg not in (None, "", "null") else None
    fna_dir = _resolve_path(fna_dir_cfg) if fna_dir_cfg not in (None, "", "null") else None
    if split_type in {"blastp", "pangenome"}:
        if gtf_dir is None:
            gtf_dir = ROOT / "raw" / "gtf"
        if fna_dir is None:
            fna_dir = ROOT / "raw" / "fna"
    environment = cfg.get("environment", None)
    if environment in ("", "null"):
        environment = None
    if split_type in {"blastp", "pangenome"} and environment is None:
        environment = "gene"
    window_cfg = cfg.get("window", None)
    window: dict[str, int] | None = None
    if window_cfg not in (None, "", "null"):
        if isinstance(window_cfg, str):
            import json as _json

            window = {str(k): int(v) for k, v in _json.loads(window_cfg).items()}
        else:
            window = {str(k): int(v) for k, v in dict(window_cfg).items()}
    elif split_type == "blastp":
        window = {"pos1": 0, "pos2": 0}
    elif split_type == "pangenome":
        # CDS-oriented default when caller omitted window (override via window=…).
        window = {"pos1": 0, "pos2": 100}
    genetic_code = str(cfg.get("genetic_code", "universal") or "universal")
    parsed_path = panel_root / "PARSED" if split_type in {"blastp", "pangenome"} else None

    if reuse_folds and source_artifacts is not None:
        rerun_stage_info = stage_source_into_out_root(
            source_artifacts, out_root, include_split_tree=False
        )
        split_csv = out_root / "split.csv"
        # Rematerialize into the *new* out_root only (source untouched).
        split_root = run_split(
            split_csv,
            parsed_target=panel_root / "PREDICT",
            parsed_data=panel_root / "PARSED",
            outdir=out_root,
            strategy="traintestval",
            intersect_allow=True,
            id_csv=panel_root / "ID.csv",
        )
        print(
            f"rerun: reused folds from {source_artifacts.split_csv} → "
            f"{split_csv}; materialized {split_root}",
            flush=True,
        )
    else:
        split_csv = run_split_predict(
            outdir=out_root,
            type=split_type,
            seed=seed,
            id_csv=panel_root / "ID.csv",
            fold_csv=panel_root / "fold.csv",
            ratios=ratios,
            marked_fasta=marked_path,
            max_ids=max_ids,
            plot=plot_split and split_type in {"gc", "kmer"},
            cluster_method=cluster_method,
            threshold=threshold,
            p_train=p_train,
            p_test=p_test,
            threads=threads,
            force=force_split,
            kmer_size=kmer_size,
            log_transform=log_transform,
            engine=kmer_engine,
            parsed=parsed_path,
            gtf_dir=gtf_dir,
            fna_dir=fna_dir,
            environment=str(environment) if environment is not None else None,
            window=window,
            genetic_code=genetic_code,
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

    if rerun:
        manifest = {
            "rerun": True,
            "run_id": str(cfg.run_id),
            "out_root": str(out_root),
            "panel_root": str(panel_root),
            "reuse_folds": reuse_folds,
            "source_split": source_hint,
            "ratios": list(ratios) if ratios is not None else None,
            "epochs": epochs,
            "min_epochs": min_epochs,
            "early_stopping_patience": early_stop,
            "eval_max_samples": ALIGNED_EVAL_MAX_SAMPLES,
            "train": train_name,
            "split": split_type,
            "stage": rerun_stage_info,
        }
        write_rerun_manifest(out_root, manifest)
        # Snapshot effective aligned settings (not only pre-default YAML).
        with OmegaConf.open_dict(cfg):
            cfg.epochs = epochs
            cfg.min_epochs = min_epochs
            cfg.early_stopping_patience = early_stop
            cfg.out_root = str(out_root)
            cfg.rerun = True
            if ratios is not None:
                cfg.ratios = list(ratios)
            if source_hint is not None:
                cfg.source_split = source_hint
    _write_hydra_resolved(cfg, out_root)

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

    max_length = int(cfg.get("max_length", 512))
    ckpt_every = int(cfg.get("checkpoint_every_n_epochs", 10))
    direct_outdir = out_root / "direct"
    if rerun and direct_outdir.exists():
        raise FileExistsError(
            f"rerun refuses to overwrite existing train dir: {direct_outdir}"
        )
    run_train(
        model=train_name,
        type=str(cfg.task_type),
        folders=folders,
        outdir=direct_outdir,
        strategy=str(cfg.split),
        smoke=not run_training,
        epochs=epochs,
        batch_size=int(cfg.batch_size),
        max_length=max_length,
        seed=seed,
        n_devices=int(cfg.n_devices),
        num_workers=int(cfg.num_workers),
        legnet_demo=train_name in {"legnet", "human_legnet"},
        zsv_root=out_root if eval_zsv else None,
        eval_zsv=eval_zsv and run_training,
        checkpoint_every_n_epochs=ckpt_every,
        early_stopping_patience=early_stop,
        min_epochs=min_epochs,
    )
    # Visualization stage: train monitor (+ split_compare) and SBS PCA diagnostics.
    # Auto-routes to viz_conda_env when the train env lacks matplotlib/cnsplots.
    plot_train = bool(cfg.get("plot_train", True))
    plot_sbs = bool(cfg.get("plot_sbs", plot_split))
    viz_env = str(cfg.get("viz_conda_env", "caduceus_env") or "caduceus_env")
    try:
        from src.pipeline.pipeline_viz import run_pipeline_viz_auto

        viz = run_pipeline_viz_auto(
            out_root=out_root,
            panel_root=panel_root,
            train_dir=out_root / "direct",
            run_id=str(cfg.run_id),
            seed=seed,
            plot_train=plot_train,
            plot_sbs=plot_sbs,
            include_split_compare=True,
            max_ids=max_ids,
            viz_conda_env=viz_env,
        )
        print(
            f"pipeline_viz status={viz.get('status')} "
            f"train={((viz.get('train') or {}).get('status'))} "
            f"sbs={((viz.get('sbs') or {}).get('status'))} "
            f"→ {viz.get('manifest') or out_root / 'pipeline_viz_manifest.json'}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: pipeline_viz skipped: {type(exc).__name__}: {exc}")

    # Persist resolved Hydra config/commands for every path (including direct-only).
    _write_hydra_resolved(cfg, out_root)

    if not bool(cfg.adversarial):
        print(OmegaConf.to_yaml(cfg))
        print(f"Hydra pipeline {mode} complete (direct only) → {out_root}")
        return 0

    # --- adversarial: copy → re-split → fold-class PREDICT → materialize → train ---
    adv_root = out_root / "adversarial"
    if adv_root.exists():
        if rerun:
            raise FileExistsError(
                f"rerun refuses to overwrite existing adversarial tree: {adv_root}"
            )
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
        max_length=int(cfg.get("max_length", 512)),
        seed=seed,
        n_devices=int(cfg.n_devices),
        num_workers=int(cfg.num_workers),
        legnet_demo=train_name in {"legnet", "human_legnet"},
        zsv_root=adv_root if eval_zsv else None,
        eval_zsv=eval_zsv and run_training,
        checkpoint_every_n_epochs=ckpt_every,
        early_stopping_patience=early_stop,
        min_epochs=min_epochs,
    )
    try:
        from src.pipeline.pipeline_viz import run_pipeline_viz_auto

        viz = run_pipeline_viz_auto(
            out_root=out_root,
            panel_root=panel_root,
            train_dir=adv_root / "train",
            run_id=f"{cfg.run_id}_adversarial",
            seed=seed,
            plot_train=plot_train,
            # SBS PCA already done on direct split.csv; skip duplicate for adv unless requested
            plot_sbs=False,
            include_split_compare=True,
            viz_conda_env=viz_env,
        )
        print(
            f"pipeline_viz[adversarial] status={viz.get('status')} "
            f"→ {viz.get('manifest')}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: pipeline_viz[adversarial] skipped: {type(exc).__name__}: {exc}"
        )

    # Final pass: direct + adversarial monitors + TensorBoard export
    try:
        from src.train_viz.train_monitor import refresh_pipeline_monitors
        from src.pipeline.pipeline_viz import has_viz_deps, resolve_viz_python
        import subprocess

        if has_viz_deps():
            pipe_mon = refresh_pipeline_monitors(
                out_root,
                run_id=str(cfg.run_id),
                include_split_compare=True,
            )
        else:
            viz_py = resolve_viz_python(viz_env)
            if viz_py is None:
                raise ModuleNotFoundError("viz python missing")
            # Refresh both dirs via train_monitor CLI
            for sub in (out_root / "direct", adv_root / "train"):
                if sub.is_dir():
                    subprocess.run(
                        [str(viz_py), "-m", "src.train_viz.train_monitor", "--run-dir", str(sub)],
                        check=False,
                    )
            pipe_mon = {"status": "subprocess", "direct": "refreshed"}
        print(
            f"pipeline_monitors status={pipe_mon.get('status')} "
            f"direct={((pipe_mon.get('direct') or {}).get('status'))} "
            f"adversarial={((pipe_mon.get('adversarial') or {}).get('status'))}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: pipeline_monitors skipped: {type(exc).__name__}: {exc}")

    _write_hydra_resolved(cfg, out_root)

    print(f"Hydra pipeline {mode} complete → {out_root}")
    print(
        "Model launch templates:\n"
        + OmegaConf.to_yaml(
            {
                "direct_cmd": str(cfg.train.direct_cmd),
                "adversarial_cmd": str(cfg.train.adversarial_cmd),
                "zsv_cmd": str(cfg.train.zsv_cmd),
            }
        )
    )
    return 0


def _write_hydra_resolved(cfg: DictConfig, out_root: Path) -> None:
    """Write reproducible Hydra snapshots into the run outdir."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cmds = {
        "direct_cmd": str(cfg.train.direct_cmd),
        "adversarial_cmd": str(cfg.train.adversarial_cmd),
        "zsv_cmd": str(cfg.train.zsv_cmd),
    }
    (out_root / "hydra_resolved_commands.yaml").write_text(
        OmegaConf.to_yaml(cmds), encoding="utf-8"
    )
    (out_root / "hydra_resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Compose ``configs/pipeline.yaml`` and run stages (no ``@hydra.main``).

    Avoids Hydra ``@hydra.main`` + CPython 3.14 argparse ``LazyCompletionHelp`` crash.
    Overrides are plain Hydra-style CLI tokens: ``mode=run epochs=3``.
    """
    from hydra import compose, initialize_config_dir

    from src.pipeline.rerun_aligned import parse_override_keys

    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="pipeline", overrides=overrides)
    try:
        return _run_stages(cfg, cli_overrides=parse_override_keys(overrides))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
