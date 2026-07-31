"""Continue run17 from pangenome split → LegNet direct + adversarial (4 GPUs).

Expects ``runs/run17_pangenome_CDS_legnet/{split.csv,SPLIT,legnet_input/all.tsv}``
and mice ZSV trees. Waits until physical GPUs 0–3 are free unless ``skip_wait``.

Launch::

  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n legnet --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.continue_from_split
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run17_pangenome_CDS_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
EPOCHS = 50
MIN_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 10
N_DEVICES = 4
BATCH_SIZE = 4096
NUM_WORKERS = 8
SEED = 42
# Caduceus-aligned default (~81% / 10% / 9% train/test/val); see run_split_cpu.
RATIOS = None
WAIT_GPUS = (0, 1, 2, 3)
MEM_FREE_MIB = 500
POLL_SEC = 60
PEAK_RAM_GIB = 16.0


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


def wait_for_gpus(
    gpus: tuple[int, ...] = WAIT_GPUS, thresh: int = MEM_FREE_MIB
) -> None:
    print(
        f"Waiting for GPUs {gpus} free (memory.used < {thresh} MiB); "
        f"poll every {POLL_SEC}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in gpus}
        print(f"GPU memory.used MiB: {used}", flush=True)
        if all(v is not None and v < thresh for v in used.values()):
            print(f"GPUs {gpus} free — starting train", flush=True)
            return
        time.sleep(POLL_SEC)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    min_epochs = MIN_EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    n_devices = N_DEVICES
    skip_wait = False
    for tok in list(argv):
        if tok.startswith("batch_size="):
            batch = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("min_epochs="):
            min_epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("early_stopping_patience="):
            patience = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("n_devices="):
            n_devices = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok in {"skip_wait=true", "--skip-wait"}:
            skip_wait = True
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    _require(OUT_ROOT / "SPLIT", "dir")
    tsv = _require(OUT_ROOT / "legnet_input" / "all.tsv", "file")
    if tsv.stat().st_size == 0:
        raise ValueError(f"legnet TSV is empty: {tsv}")
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")
    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "fold.csv", "file")

    if not skip_wait:
        wait_for_gpus()
    else:
        print("skip_wait=true — not polling GPUs", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.job_queue import (
        CLASS_GPU_TRAIN,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train

    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB,
        gpus=WAIT_GPUS,
        job_class=CLASS_GPU_TRAIN,
        label=f"{RUN_ID}_train",
    )
    append_queue_entry(
        f"{RUN_ID}_train",
        job=f"python -m src.runs.{RUN_ID}.continue_from_split n_devices={n_devices}",
        pid=os.getpid(),
        estimated_time="6-12h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB,
        gpus=WAIT_GPUS,
        resources=f"LegNet direct+adv; epochs={epochs} min={min_epochs} patience={patience}",
        log=f"logs/{RUN_ID}_train.log",
    )

    print(
        f"continue_from_split: tsv={tsv} epochs={epochs} min_epochs={min_epochs} "
        f"patience={patience} n_devices={n_devices} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    direct_out = OUT_ROOT / "direct"
    if direct_out.exists():
        shutil.rmtree(direct_out)

    run_train(
        model="legnet",
        type="regression",
        folders=tsv,
        outdir=direct_out,
        strategy="pangenome",
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        seed=SEED,
        n_devices=n_devices,
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

        viz = run_pipeline_viz_auto(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            train_dir=direct_out,
            run_id=RUN_ID,
            seed=SEED,
            plot_train=True,
            plot_sbs=False,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
        print(f"pipeline_viz[direct] status={viz.get('status')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: pipeline_viz[direct] skipped: {type(exc).__name__}: {exc}",
            flush=True,
        )

    adv_root = OUT_ROOT / "adversarial"
    if adv_root.exists():
        shutil.rmtree(adv_root)
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
    run_train(
        model="legnet",
        type="classification",
        folders=adv_tsv,
        outdir=adv_root / "train",
        strategy="random",
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        seed=SEED,
        n_devices=n_devices,
        num_workers=NUM_WORKERS,
        legnet_demo=True,
        zsv_root=adv_root,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
    )

    try:
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        mon = refresh_pipeline_monitors(
            OUT_ROOT, run_id=RUN_ID, include_split_compare=True
        )
        print(f"pipeline_monitors status={mon.get('status')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: pipeline_monitors skipped: {type(exc).__name__}: {exc}",
            flush=True,
        )

    (OUT_ROOT / "pipeline_done.json").write_text(
        '{"status":"COMPLETED","run_id":"%s"}\n' % RUN_ID, encoding="utf-8"
    )
    print(f"run17_pangenome_CDS_legnet continue COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
