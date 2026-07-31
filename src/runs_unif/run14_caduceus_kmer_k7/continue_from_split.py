"""Continue legacy run14_7mer_caduceus with rebuilt ≈3:1:1 split → Caduceus direct only.

Legacy (read-only):
  ``src/runs/run14_7mer_caduceus``, ``runs/run14_7mer_caduceus``
  (Caduceus / kmer k=7 / ready_caduceus)

New (write):
  ``src/runs_unif/run14_caduceus_kmer_k7``
  ``runs_unif/caduceus/run14_caduceus_kmer_k7``

Flow:
  1. Rewrite ``split.csv`` from the present SBS/kmer assignment to
     train:test:val ≈ 3:1:1 (**whole clusters**; never mutate legacy).
  2. Stage k-mer intermediates + materialize SPLIT.
  3. Wait until **1** GPU is free, then direct train
     (min 15 / max 30 / early-stop patience 10) + mice ZSV.
  4. No adversarial.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs_unif.run14_caduceus_kmer_k7.continue_from_split
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_I = 14
MODEL = "caduceus"
SPLIT = "kmer"
SPLIT_PARAMS = "k7"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

LEGACY_SRC = ROOT / "src" / "runs" / "run14_7mer_caduceus"
LEGACY_OUT = ROOT / "runs" / "run14_7mer_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10

# Aligned with legacy run8 Caduceus train defaults (1 GPU).
BATCH_SIZE = 480
MAX_LENGTH = 208
NUM_WORKERS = 4
PREFERRED_GPUS = (0, 1, 2, 3)
# Require nearly empty device (avoid race where used≈400 MiB looks "free" then OOM).
MEM_FREE_MIB = 200
POLL_SEC = 60
PEAK_RAM_GIB_TRAIN = 24.0
PEAK_RAM_GIB_SPLIT = 16.0
GPU_CONFIRM_SEC = 5

KMER_INTERMEDIATES = (
    "feature_table.csv",
    "feature_table.npz",
    "sbs_assignment.csv",
    "kmer_split_meta.json",
)


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
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
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
        elif tok in {"skip_wait=true", "--skip-wait"}:
            cfg["skip_wait"] = True
            argv.remove(tok)
        elif tok in {"split_only=true", "--split-only"}:
            cfg["split_only"] = True
            argv.remove(tok)
    return cfg


def stage_split(*, seed: int = SEED) -> dict:
    """Rewrite split table + stage k-mer sidecars + materialize SPLIT."""
    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.rerun_aligned import (
        assert_fresh_out_root,
        rewrite_split_table_aligned,
        write_rerun_manifest,
    )
    from src.pipeline.split import run_split

    _require(LEGACY_OUT / "split.csv", "file")
    _require(LEGACY_OUT / "sbs_assignment.csv", "file")
    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "PARSED", "dir")
    _require(PANEL_ROOT / "PREDICT", "dir")
    _require(PANEL_ROOT / "fold.csv", "file")

    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    assert_fresh_out_root(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB_SPLIT,
        gpus=(),
        job_class=CLASS_CPU_RAM_HEAVY,
        label=f"{RUN_NAME}_split",
    )
    append_queue_entry(
        f"{RUN_NAME}_split",
        job=f"python -m src.runs_unif.{RUN_NAME}.continue_from_split split_only=true",
        pid=os.getpid(),
        estimated_time="1-3h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB_SPLIT,
        log=str(ROOT / "logs" / f"{RUN_NAME}_split.log"),
    )

    rewrite_info = rewrite_split_table_aligned(
        LEGACY_OUT / "split.csv",
        OUT_ROOT / "split.csv",
        seed=seed,
        prefer_label_swap=True,
        assignment_csv=LEGACY_OUT / "sbs_assignment.csv",
        allow_id_reassign=False,  # kmer SBS: never random ID sweep
    )
    print(f"split rewrite: {json.dumps(rewrite_info, sort_keys=True)}", flush=True)

    staged_extra: dict[str, str] = {}
    for rel in KMER_INTERMEDIATES:
        src = LEGACY_OUT / rel
        if not src.is_file():
            continue
        dest = OUT_ROOT / rel
        if dest.exists():
            staged_extra[rel] = str(dest)
            print(f"reuse staged {rel}", flush=True)
            continue
        shutil.copy2(src, dest)
        staged_extra[rel] = str(dest)
        print(f"staged {rel}", flush=True)

    split_csv = OUT_ROOT / "split.csv"
    split_root = run_split(
        split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        outdir=OUT_ROOT,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL_ROOT / "ID.csv",
    )
    print(f"SPLIT ready: {split_root}", flush=True)

    manifest = {
        "rerun": True,
        "aligned_run": RUN_I,
        "run_name": RUN_NAME,
        "legacy": {"src": str(LEGACY_SRC), "out": str(LEGACY_OUT)},
        "out_root": str(OUT_ROOT),
        "panel_root": str(PANEL_ROOT),
        "model": MODEL,
        "split": SPLIT,
        "split_params": SPLIT_PARAMS,
        "kmer_size": 7,
        "ratios": [3, 1, 1],
        "rewrite": rewrite_info,
        "staged_extra": staged_extra,
        "zsv": "mice",
        "adversarial": False,
        "direct": {
            "epochs": EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "n_devices": 1,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
        },
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    write_rerun_manifest(OUT_ROOT, manifest)
    (OUT_ROOT / "split_done.json").write_text(
        json.dumps(
            {"status": "ok", "split_csv": str(split_csv), "split_root": str(split_root)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"split_csv": split_csv, "split_root": split_root, "manifest": manifest}


def run_train_direct(
    *,
    batch: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    gpu: int,
) -> None:
    from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry
    from src.pipeline.train import run_train

    split_root = _require(OUT_ROOT / "SPLIT", "dir")
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    append_queue_entry(
        f"{RUN_NAME}_train",
        job=(
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"python -m src.runs_unif.{RUN_NAME}.continue_from_split"
        ),
        pid=os.getpid(),
        estimated_time="6-20h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        resources=(
            f"batch {batch} max_len {MAX_LENGTH}; "
            f"direct {epochs}/{min_epochs}/p{patience}; no adversarial"
        ),
        log=str(ROOT / "logs" / f"{RUN_NAME}_train.log"),
    )

    direct_out = OUT_ROOT / "direct"
    if direct_out.exists():
        raise FileExistsError(f"refusing overwrite: {direct_out}")

    print(
        f"direct Caduceus train gpu={gpu} epochs={epochs} "
        f"min_epochs={min_epochs} patience={patience} batch={batch}",
        flush=True,
    )
    run_train(
        model="caduceus",
        type="regression",
        folders=split_root,
        outdir=direct_out,
        strategy=SPLIT,
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        max_length=MAX_LENGTH,
        seed=SEED,
        n_devices=1,
        num_workers=NUM_WORKERS,
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
            plot_sbs=True,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: viz skipped: {type(exc).__name__}: {exc}", flush=True)

    (OUT_ROOT / "pipeline_done.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "run_name": RUN_NAME,
                "out_root": str(OUT_ROOT),
                "gpu": gpu,
                "adversarial": False,
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
        f"{RUN_NAME}: legacy={LEGACY_OUT} → out={OUT_ROOT} "
        f"panel={PANEL_ROOT} split_only={cfg['split_only']} "
        f"adversarial=false",
        flush=True,
    )

    split_done = OUT_ROOT / "split_done.json"
    if not split_done.is_file():
        stage_split(seed=SEED)
    else:
        print(f"reuse staged split: {split_done}", flush=True)
        _require(LEGACY_OUT / "split.csv", "file")

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

    run_train_direct(
        batch=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        min_epochs=int(cfg["min_epochs"]),
        patience=int(cfg["patience"]),
        gpu=gpu,
    )
    print(f"{RUN_NAME} COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
