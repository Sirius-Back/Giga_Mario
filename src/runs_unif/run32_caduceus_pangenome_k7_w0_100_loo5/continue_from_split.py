"""Continue run32: reuse run31 LOO5 tables → Caduceus direct LOO + adversarial.

Source: ``runs_unif/legnet/run31_legnet_pangenome_k7_w0_100_loo5``
Out: ``runs_unif/caduceus/run32_caduceus_pangenome_k7_w0_100_loo5``

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs_unif.run32_caduceus_pangenome_k7_w0_100_loo5.continue_from_split
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

RUN_I = 32
MODEL = "caduceus"
SPLIT = "pangenome"
SPLIT_PARAMS = "k7_w0_100_loo5"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

SOURCE_UNIF = ROOT / "runs_unif" / "legnet" / "run31_legnet_pangenome_k7_w0_100_loo5"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
N_CV = 5
RATIOS = (3.0, 1.0, 1.0)
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
ADV_EPOCHS = 10
ADV_MIN_EPOCHS = 0
ADV_EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 192
MAX_LENGTH = 256
NUM_WORKERS = 4
PREFERRED_GPUS = (0, 1, 2, 3)
MEM_FREE_MIB = 1500
GPU_CONFIRM_SEC = 30
POLL_SEC = 60
SOURCE_POLL_SEC = 120
PEAK_RAM_GIB_STAGE = 28.0
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


def wait_for_source(*, poll_sec: int = SOURCE_POLL_SEC) -> None:
    done = SOURCE_UNIF / "split_done.json"
    print(f"Waiting for source LOO {done}; poll every {poll_sec}s …", flush=True)
    while True:
        ok = done.is_file() and (SOURCE_UNIF / "fold0" / "split.csv").is_file()
        print(f"source ready? {ok}", flush=True)
        if ok:
            return
        time.sleep(poll_sec)


def wait_for_one_gpu(
    *,
    prefer: tuple[int, ...] = PREFERRED_GPUS,
    thresh: int = MEM_FREE_MIB,
    poll_sec: int = POLL_SEC,
    confirm_sec: int = GPU_CONFIRM_SEC,
) -> int:
    print(
        f"Waiting for any GPU in {prefer} free (memory.used < {thresh} MiB); "
        f"poll every {poll_sec}s; re-confirm {confirm_sec}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in prefer}
        print(f"GPU memory.used MiB: {used}", flush=True)
        free = [g for g in prefer if used.get(g) is not None and used[g] < thresh]
        if not free:
            time.sleep(poll_sec)
            continue
        gpu = free[0]
        print(f"GPU {gpu} candidate — confirming idle for {confirm_sec}s …", flush=True)
        time.sleep(confirm_sec)
        used2 = _gpu_used_mib(gpu)
        if used2 is not None and used2 < thresh:
            print(f"GPU {gpu} free (used={used2} MiB) — starting train", flush=True)
            return gpu
        print(
            f"GPU {gpu} no longer free (used={used2} MiB) — keep waiting",
            flush=True,
        )


def _parse_argv(argv: list[str]) -> dict[str, object]:
    cfg: dict[str, object] = {
        "skip_wait": False,
        "split_only": False,
        "skip_direct": False,
        "force_adv": False,
    }
    for tok in list(argv):
        if tok in {"skip_wait=true", "--skip-wait"}:
            cfg["skip_wait"] = True
            argv.remove(tok)
        elif tok in {"split_only=true", "--split-only"}:
            cfg["split_only"] = True
            argv.remove(tok)
        elif tok in {"skip_direct=true", "--skip-direct"}:
            cfg["skip_direct"] = True
            argv.remove(tok)
        elif tok in {"force_adv=true", "--force-adv"}:
            cfg["force_adv"] = True
            argv.remove(tok)
    return cfg


def stage_split() -> dict:
    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.loo_cv import rematerialize_loo_folds_for_panel
    from src.pipeline.rerun_aligned import assert_fresh_out_root

    wait_for_source()
    _require(SOURCE_UNIF / "split_done.json", "file")
    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "PARSED", "dir")
    _require(PANEL_ROOT / "PREDICT", "dir")

    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    assert_fresh_out_root(OUT_ROOT)
    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB_STAGE,
        gpus=(),
        job_class=CLASS_CPU_RAM_HEAVY,
        label=f"{RUN_NAME}_stage",
    )
    append_queue_entry(
        f"{RUN_NAME}_stage",
        job=f"python -m src.runs_unif.{RUN_NAME}.continue_from_split split_only=true",
        pid=os.getpid(),
        estimated_time="2-8h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB_STAGE,
        resources="rematerialize run31 LOO folds on ready_caduceus",
        log=str(ROOT / "logs" / f"{RUN_NAME}_stage.log"),
    )
    return rematerialize_loo_folds_for_panel(
        source_loo_root=SOURCE_UNIF,
        out_root=OUT_ROOT,
        panel_root=PANEL_ROOT,
        n_cv=N_CV,
        build_legnet=False,
    )


def run_train_and_adv(*, gpu: int, skip_direct: bool, force_adv: bool) -> None:
    from src.pipeline.adversarial import setup_adversarial_random_fold_class
    from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry
    from src.pipeline.loo_cv import run_loo_direct_trains
    from src.pipeline.train import run_train

    _require(OUT_ROOT / "split_done.json", "file")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    append_queue_entry(
        f"{RUN_NAME}_train",
        job=f"CUDA_VISIBLE_DEVICES={gpu} python -m src.runs_unif.{RUN_NAME}.continue_from_split",
        pid=os.getpid(),
        estimated_time="30-80h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        resources=f"LOO{N_CV} Caduceus direct + adv",
        log=str(ROOT / "logs" / f"{RUN_NAME}_train.log"),
    )

    if not skip_direct:
        run_loo_direct_trains(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            model="caduceus",
            n_cv=N_CV,
            seed=SEED,
            epochs=EPOCHS,
            min_epochs=MIN_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            n_devices=1,
            num_workers=NUM_WORKERS,
            strategy_name=SPLIT,
        )

    adv_root = OUT_ROOT / "adversarial"
    if adv_root.exists():
        if not force_adv:
            raise FileExistsError(f"refusing overwrite: {adv_root}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        adv_root.rename(OUT_ROOT / f"adversarial_FAILED_{stamp}")

    setup_adversarial_random_fold_class(
        adv_root=adv_root,
        label_split_csv=OUT_ROOT / "split.csv",
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        fold_csv=PANEL_ROOT / "fold.csv",
        seed=SEED + 1,
        ratios=RATIOS,
        intersect_allow=True,
        build_legnet_input=False,
    )
    run_train(
        model="caduceus",
        type="classification",
        folders=adv_root / "SPLIT",
        outdir=adv_root / "train",
        strategy="random",
        smoke=False,
        epochs=ADV_EPOCHS,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        seed=SEED,
        n_devices=1,
        num_workers=NUM_WORKERS,
        zsv_root=adv_root,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=ADV_EARLY_STOPPING_PATIENCE,
        min_epochs=ADV_MIN_EPOCHS,
    )
    try:
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        refresh_pipeline_monitors(OUT_ROOT, run_id=RUN_NAME, include_split_compare=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: monitors skipped: {type(exc).__name__}: {exc}", flush=True)

    (OUT_ROOT / "pipeline_done.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "run_name": RUN_NAME,
                "gpu": gpu,
                "n_cv": N_CV,
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

    print(f"{RUN_NAME}: source={SOURCE_UNIF} → out={OUT_ROOT}", flush=True)
    if not (OUT_ROOT / "split_done.json").is_file():
        stage_split()
    else:
        print(f"reuse staged: {OUT_ROOT / 'split_done.json'}", flush=True)

    if cfg["split_only"]:
        return 0

    if cfg["skip_wait"]:
        used = {g: _gpu_used_mib(g) for g in PREFERRED_GPUS}
        free = [g for g in PREFERRED_GPUS if used.get(g) is not None and used[g] < MEM_FREE_MIB]
        gpu = free[0] if free else 0
    else:
        gpu = wait_for_one_gpu()

    run_train_and_adv(
        gpu=gpu,
        skip_direct=bool(cfg["skip_direct"]),
        force_adv=bool(cfg["force_adv"]),
    )
    print(f"{RUN_NAME} COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
