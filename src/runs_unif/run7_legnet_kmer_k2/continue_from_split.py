"""Continue legacy run7_2mer_legnet with rebuilt 3:1:1 split → LegNet direct only (no adversarial).

Legacy (read-only):
  ``src/runs/run7_2mer_legnet``, ``runs/run7_2mer_legnet`` (kmer k=2 / cpp)

New (write):
  ``src/runs_unif/run7_legnet_kmer_k2``
  ``runs_unif/legnet/run7_legnet_kmer_k2``

Flow:
  1. Rewrite ``split.csv`` from the present table to train:test:val ≈ 3:1:1
     (prefer train↔val swap when legacy was inverted; never mutate ``runs/run7_2mer_legnet``).
  2. Stage k-mer intermediates (feature_table / sbs_assignment / kmer meta) + materialize.
  3. Wait until **1** GPU is free, then direct train
     (min 15 / max 30 / early-stop patience 10) + mice ZSV.
  4. No adversarial.

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs_unif.run7_legnet_kmer_k2.continue_from_split
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

RUN_I = 7
MODEL = "legnet"
SPLIT = "kmer"
SPLIT_PARAMS = "k2"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

LEGACY_OUT = ROOT / "runs" / "run7_2mer_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 8192
NUM_WORKERS = 8
PREFERRED_GPUS = (0, 1, 2, 3)
MEM_FREE_MIB = 1500
POLL_SEC = 60
PEAK_RAM_GIB_TRAIN = 24.0
PEAK_RAM_GIB_SPLIT = 16.0

# Extra k-mer intermediates to stage from legacy (never mutate source).
KMER_INTERMEDIATES = (
    "feature_table.csv",
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
    """Rewrite split table + stage k-mer sidecars + materialize SPLIT + LegNet TSV."""
    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.rerun_aligned import (
        assert_fresh_out_root,
        rewrite_split_table_aligned,
        write_rerun_manifest,
    )
    from src.pipeline.split import run_split

    _require(LEGACY_OUT / "split.csv", "file")
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
        allow_id_reassign=False,  # SBS: never random ID sweep
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

    tsv = build_legnet_tsv(
        split_root=split_root, out_tsv=OUT_ROOT / "legnet_input" / "all.tsv"
    )
    print(f"legnet TSV: {tsv}", flush=True)

    manifest = {
        "rerun": True,
        "aligned_run": RUN_I,
        "run_name": RUN_NAME,
        "legacy": {
            "src": str(ROOT / "src" / "runs" / "run7_2mer_legnet"),
            "out": str(LEGACY_OUT),
        },
        "out_root": str(OUT_ROOT),
        "panel_root": str(PANEL_ROOT),
        "model": MODEL,
        "split": SPLIT,
        "split_params": SPLIT_PARAMS,
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
        },
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    write_rerun_manifest(OUT_ROOT, manifest)
    (OUT_ROOT / "split_done.json").write_text(
        json.dumps(
            {"status": "ok", "split_csv": str(split_csv), "tsv": str(tsv)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"split_csv": split_csv, "tsv": tsv, "manifest": manifest}


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
        estimated_time="4-12h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        resources=f"batch {batch}; direct {epochs}/{min_epochs}/p{patience}; no adversarial",
        log=str(ROOT / "logs" / f"{RUN_NAME}_train.log"),
    )

    direct_out = OUT_ROOT / "direct"
    if direct_out.exists():
        raise FileExistsError(f"refusing overwrite: {direct_out}")

    print(
        f"direct LegNet train n_devices=1 gpu={gpu} epochs={epochs} "
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
