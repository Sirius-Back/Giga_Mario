"""Continue run21: 1-GPU LegNet direct + adversarial (after split_done).

Expects ``runs_unif/legnet/run21_legnet_pangenome_k10_wm100_100/{split.csv,SPLIT,legnet_input}``.
Waits for **1** free GPU unless ``skip_wait``.

Direct: min15 / max30 / patience10 + mice ZSV.
Adversarial: random split, max10 / patience5 + ZSV.

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs_unif.run21_legnet_pangenome_k10_wm100_100.continue_from_split
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_I = 21
MODEL = "legnet"
SPLIT = "pangenome"
SPLIT_PARAMS = "k10_wm100_100"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
RATIOS = (3.0, 1.0, 1.0)
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
ADV_EPOCHS = 10
ADV_MIN_EPOCHS = 0
ADV_EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 8192
NUM_WORKERS = 8
PREFERRED_GPUS = (0, 1, 2, 3)
MEM_FREE_MIB = 1500
POLL_SEC = 60
PEAK_RAM_GIB_TRAIN = 24.0


def _require(path: Path, kind: str = "path") -> Path:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing required dir: {path}")
    return path


def _gpu_used_mib(index: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return int(out.split()[0])
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: nvidia-smi failed for GPU {index}: {exc}", flush=True)
        return None


def wait_for_one_gpu(
    *,
    prefer: tuple[int, ...] = PREFERRED_GPUS,
    thresh: int = MEM_FREE_MIB,
    poll_sec: int = POLL_SEC,
) -> int:
    print(
        f"Waiting for any GPU in {prefer} free (memory.used < {thresh} MiB); "
        f"poll every {poll_sec}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in prefer}
        print(f"GPU memory.used MiB: {used}", flush=True)
        free = [g for g in prefer if used.get(g) is not None and used[g] < thresh]
        if free:
            print(f"GPU {free[0]} free — starting train", flush=True)
            return free[0]
        time.sleep(poll_sec)


def _parse_argv(argv: list[str]) -> dict[str, object]:
    cfg: dict[str, object] = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "adv_epochs": ADV_EPOCHS,
        "adv_min_epochs": ADV_MIN_EPOCHS,
        "adv_patience": ADV_EARLY_STOPPING_PATIENCE,
        "skip_wait": False,
        "split_only": False,
    }
    for tok in list(argv):
        if tok.startswith("batch_size="):
            cfg["batch_size"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            cfg["epochs"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("min_epochs="):
            cfg["min_epochs"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("early_stopping_patience="):
            cfg["patience"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("adversarial_epochs="):
            cfg["adv_epochs"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("adversarial_min_epochs="):
            cfg["adv_min_epochs"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("adversarial_early_stopping_patience="):
            cfg["adv_patience"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok in {"skip_wait=true", "--skip-wait"}:
            cfg["skip_wait"] = True
            argv.remove(tok)
        elif tok in {"split_only=true", "--split-only"}:
            cfg["split_only"] = True
            argv.remove(tok)
    return cfg


def run_train_stages(
    *,
    batch: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    adv_epochs: int,
    adv_min_epochs: int,
    adv_patience: int,
    gpu: int,
) -> None:
    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    tsv = _require(OUT_ROOT / "legnet_input" / "all.tsv", "file")
    _require(OUT_ROOT / "SPLIT", "dir")
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    append_queue_entry(
        f"{RUN_NAME}_train",
        job=(
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"python -m src.runs_unif.{RUN_NAME}.continue_from_split n_devices=1"
        ),
        pid=os.getpid(),
        estimated_time="6-20h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        resources=(
            f"batch {batch}; direct {epochs}/{min_epochs}/p{patience}; "
            f"adv {adv_epochs}/p{adv_patience}"
        ),
        log=str(ROOT / "logs" / f"{RUN_NAME}_train.log"),
    )

    direct_out = OUT_ROOT / "direct"
    if direct_out.exists():
        raise FileExistsError(f"refusing overwrite: {direct_out}")

    print(
        f"direct LegNet train gpu={gpu} epochs={epochs} "
        f"min_epochs={min_epochs} patience={patience}",
        flush=True,
    )
    run_train(
        model="legnet",
        type="regression",
        folders=tsv,
        outdir=direct_out,
        strategy=SPLIT,
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        seed=SEED,
        n_devices=1,
        num_workers=NUM_WORKERS,
        legnet_demo=True,
        zsv_root=OUT_ROOT,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
    )

    try:
        from src.pipeline.pipeline_viz import run_pipeline_viz_auto

        run_pipeline_viz_auto(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            train_dir=direct_out,
            run_id=RUN_NAME,
            seed=SEED,
            plot_train=True,
            plot_sbs=False,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: viz[direct] skipped: {type(exc).__name__}: {exc}", flush=True)

    adv_root = OUT_ROOT / "adversarial"
    if adv_root.exists():
        raise FileExistsError(f"refusing overwrite: {adv_root}")

    print("adversarial: copy + random split + fold-class …", flush=True)
    run_adversarial(
        outdir_new=adv_root,
        split_csv=split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        intersect_allow=True,
    )
    adv_split = run_split_predict(
        outdir=adv_root,
        type="random",
        seed=SEED + 1,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
    )
    apply_fold_class_targets(
        predict_root=adv_root / "PREDICT",
        label_split_csv=split_csv,
    )
    run_split(
        adv_split,
        parsed_target=adv_root / "PREDICT",
        parsed_data=adv_root / "PARSED",
        outdir=adv_root,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL_ROOT / "ID.csv",
    )
    adv_tsv = build_legnet_tsv(
        split_root=adv_root / "SPLIT",
        out_tsv=adv_root / "legnet_input" / "all.tsv",
    )
    print(
        f"adversarial LegNet train gpu={gpu} epochs={adv_epochs} "
        f"patience={adv_patience}",
        flush=True,
    )
    run_train(
        model="legnet",
        type="classification",
        folders=adv_tsv,
        outdir=adv_root / "train",
        strategy="random",
        smoke=False,
        epochs=adv_epochs,
        batch_size=batch,
        seed=SEED,
        n_devices=1,
        num_workers=NUM_WORKERS,
        legnet_demo=True,
        zsv_root=adv_root,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=adv_patience,
        min_epochs=adv_min_epochs,
    )

    try:
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        mon = refresh_pipeline_monitors(
            OUT_ROOT, run_id=RUN_NAME, include_split_compare=True
        )
        print(f"pipeline_monitors status={mon.get('status')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: pipeline_monitors skipped: {type(exc).__name__}: {exc}",
            flush=True,
        )

    (OUT_ROOT / "pipeline_done.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "run_name": RUN_NAME,
                "out_root": str(OUT_ROOT),
                "gpu": gpu,
                "adversarial": True,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = _parse_argv(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    print(
        f"{RUN_NAME}: out={OUT_ROOT} panel={PANEL_ROOT} "
        f"split_only={cfg['split_only']} adversarial=true",
        flush=True,
    )

    split_done = OUT_ROOT / "split_done.json"
    if not split_done.is_file():
        from src.runs_unif.run21_legnet_pangenome_k10_wm100_100.run_split_cpu import (
            main as split_main,
        )

        split_main([])
    else:
        print(f"reuse staged split: {split_done}", flush=True)

    if cfg["split_only"]:
        print("split_only=true — skipping train", flush=True)
        return 0

    if cfg["skip_wait"]:
        used = {g: _gpu_used_mib(g) for g in PREFERRED_GPUS}
        print(f"skip_wait=true GPU memory.used MiB: {used}", flush=True)
        free = [
            g
            for g in PREFERRED_GPUS
            if used.get(g) is not None and used[g] < MEM_FREE_MIB
        ]
        gpu = free[0] if free else 0
        print(f"skip_wait=true — using GPU {gpu}", flush=True)
    else:
        gpu = wait_for_one_gpu()

    run_train_stages(
        batch=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        min_epochs=int(cfg["min_epochs"]),
        patience=int(cfg["patience"]),
        adv_epochs=int(cfg["adv_epochs"]),
        adv_min_epochs=int(cfg["adv_min_epochs"]),
        adv_patience=int(cfg["adv_patience"]),
        gpu=gpu,
    )
    print(f"{RUN_NAME} COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
