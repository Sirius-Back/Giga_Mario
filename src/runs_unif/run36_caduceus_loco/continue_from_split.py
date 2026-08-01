"""Run36: reuse run35 LOCO split → Caduceus direct + adversarial.

Source (read-only):
  ``runs_unif/legnet/run35_legnet_loco`` (LOCO ``split.csv``)

New (write):
  ``src/runs_unif/run36_caduceus_loco``
  ``runs_unif/caduceus/run36_caduceus_loco``

Flow:
  1. Reuse run35 ``split.csv`` (+ ``loco_meta.json``). Do **not** re-run LOCO;
     do **not** rewrite to random / forced 3:1:1 for direct.
  2. Materialize ``SPLIT/`` from ``ready_caduceus`` PARSED/PREDICT (CPU).
  3. Wait until **1** GPU is free; direct Caduceus min15/max30/p10 + mice ZSV.
  4. Adversarial: random ⊆ direct IDs + fold-class at ≈3:1:1; max10/p5 + ZSV.
  5. Reports / TensorBoard via pipeline viz + train_monitor.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs_unif.run36_caduceus_loco.continue_from_split
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

RUN_I = 36
MODEL = "caduceus"
SPLIT = "loco"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}"

SOURCE_UNIF = ROOT / "runs_unif" / "legnet" / "run35_legnet_loco"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
ADV_RATIOS = (3.0, 1.0, 1.0)
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
ADV_EPOCHS = 10
ADV_MIN_EPOCHS = 0
ADV_EARLY_STOPPING_PATIENCE = 5

BATCH_SIZE = 192
MAX_LENGTH = 256
NUM_WORKERS = 4
PREFERRED_GPUS = (3, 2, 0, 1)
MEM_FREE_MIB = 200
POLL_SEC = 60
GPU_CONFIRM_SEC = 5
PEAK_RAM_GIB_TRAIN = 24.0
PEAK_RAM_GIB_SPLIT = 16.0
STAGE_FILES = (
    "split.csv",
    "loco_meta.json",
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
    peak_ram_gib: float = PEAK_RAM_GIB_TRAIN,
) -> int:
    """Pick a GPU that is free in nvidia-smi **and** clear in queue.md politics.

    Re-selects on every poll so a race (device free → another job claims it in
    ``queue.md``) does not pin this waiter forever to a single busy index.
    """
    from src.pipeline.job_queue import CLASS_GPU_TRAIN, can_launch_parallel

    print(
        f"Waiting for any GPU in {prefer} free (memory.used < {thresh} MiB) "
        f"+ queue-clear; poll every {poll_sec}s; re-confirm {confirm_sec}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in prefer}
        print(f"GPU memory.used MiB: {used}", flush=True)
        free = [g for g in prefer if used.get(g) is not None and used[g] < thresh]
        if not free:
            time.sleep(poll_sec)
            continue
        for gpu in free:
            ok, reason = can_launch_parallel(
                peak_ram_gib=peak_ram_gib,
                gpus=(gpu,),
                job_class=CLASS_GPU_TRAIN,
            )
            if not ok:
                print(f"GPU {gpu} smi-free but queue blocks: {reason}", flush=True)
                continue
            print(
                f"GPU {gpu} candidate — confirming idle for {confirm_sec}s …",
                flush=True,
            )
            time.sleep(confirm_sec)
            used2 = _gpu_used_mib(gpu)
            if used2 is None or used2 >= thresh:
                print(
                    f"GPU {gpu} no longer free (used={used2} MiB) — keep waiting",
                    flush=True,
                )
                continue
            ok2, reason2 = can_launch_parallel(
                peak_ram_gib=peak_ram_gib,
                gpus=(gpu,),
                job_class=CLASS_GPU_TRAIN,
            )
            if not ok2:
                print(
                    f"GPU {gpu} lost queue clearance after confirm: {reason2}",
                    flush=True,
                )
                continue
            print(f"GPU {gpu} free (used={used2} MiB) — starting train", flush=True)
            return gpu
        time.sleep(poll_sec)


def wait_for_source_split(
    *,
    poll_sec: int = POLL_SEC,
    marker: str = "split_done.json",
) -> Path:
    """Block until run35 has staged LOCO split.csv (CPU reuse contract)."""
    done = SOURCE_UNIF / marker
    split_csv = SOURCE_UNIF / "split.csv"
    print(f"Waiting for source {done} …", flush=True)
    while True:
        if done.is_file() and split_csv.is_file():
            print(f"source ready: {split_csv}", flush=True)
            return split_csv
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
        "gpu": None,
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
        elif tok.startswith("gpu="):
            cfg["gpu"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok in {"skip_wait=true", "--skip-wait"}:
            cfg["skip_wait"] = True
            argv.remove(tok)
        elif tok in {"split_only=true", "--split-only"}:
            cfg["split_only"] = True
            argv.remove(tok)
    return cfg


def _assert_loco_split(split_csv: Path) -> dict:
    """Sanity: folds look like genome|chr; no role leaks within a chrom fold."""
    from collections import Counter, defaultdict

    from src.pipeline.common import read_csv
    from src.pipeline.generate_fold import is_zsv_fold
    from src.pipeline.rerun_aligned import count_train_test_val

    rows = list(read_csv(split_csv))
    counts = count_train_test_val(rows)
    zsv_n = sum(
        1
        for r in rows
        if is_zsv_fold(r["train_test"]) or str(r["train_test"]).lower() == "zsv"
    )
    by_fold: dict[str, set[str]] = defaultdict(set)
    n_loco = 0
    n_assign = 0
    for r in rows:
        if is_zsv_fold(r["train_test"]) or str(r["train_test"]).lower() == "zsv":
            continue
        n_assign += 1
        fold = str(r.get("fold") or "")
        by_fold[fold].add(str(r["train_test"]).strip().lower())
        if "|" in fold:
            n_loco += 1
    leaks = {f: labs for f, labs in by_fold.items() if len(labs) > 1}
    if n_assign == 0 or n_loco < max(1, n_assign // 2):
        raise RuntimeError(
            f"split.csv does not look like LOCO (genome|chr); "
            f"loco-like={n_loco}/{n_assign}"
        )
    if leaks:
        example = sorted(leaks)[0]
        raise RuntimeError(
            f"chromosome fold split across roles: {example!r} → {leaks[example]}"
        )
    if counts["train"] < 1 or counts["test"] < 1 or counts["val"] < 1:
        raise RuntimeError(f"source split missing a role: {counts}")
    return {
        "counts": counts,
        "zsv": zsv_n,
        "n_rows": len(rows),
        "n_folds": len(by_fold),
        "loco_fold_fraction": n_loco / max(1, n_assign),
        "note": "reuse run35 LOCO (Caduceus-aligned chrom-grain)",
    }


def stage_split(*, seed: int = SEED) -> dict:
    """Reuse run35 LOCO split + materialize Caduceus SPLIT (CPU only)."""
    from src.pipeline.common import read_csv, write_csv
    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.rerun_aligned import (
        SPLIT_CSV_COLUMNS,
        assert_fresh_out_root,
        write_rerun_manifest,
    )
    from src.pipeline.split import run_split

    wait_for_source_split()
    src_split = _require(SOURCE_UNIF / "split.csv", "file")
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
        resources="reuse run35 LOCO + materialize ready_caduceus SPLIT",
        log=str(ROOT / "logs" / f"{RUN_NAME}_split.log"),
    )

    audit = _assert_loco_split(src_split)
    print(f"source LOCO audit: {json.dumps(audit, sort_keys=True)}", flush=True)

    staged_extra: dict[str, str] = {}
    for rel in STAGE_FILES:
        src = SOURCE_UNIF / rel
        if not src.is_file():
            if rel == "loco_meta.json":
                print(f"WARNING: optional missing {src}", flush=True)
                continue
            raise FileNotFoundError(f"missing source artifact: {src}")
        dest = OUT_ROOT / rel
        shutil.copy2(src, dest)
        staged_extra[rel] = str(dest)
        print(f"staged {rel}", flush=True)

    split_csv = OUT_ROOT / "split.csv"
    rows = list(read_csv(split_csv))
    write_csv(
        split_csv,
        [
            {
                "ID": str(r["ID"]),
                "train_test": str(r["train_test"]).strip().lower(),
                "fold": str(r.get("fold") or ""),
            }
            for r in rows
        ],
        SPLIT_CSV_COLUMNS,
    )
    _assert_loco_split(split_csv)

    rewrite_info = {
        "method": "reuse_run35_loco_split_csv",
        "source_split_csv": str(src_split),
        "dest_split_csv": str(split_csv),
        "audit": audit,
        "seed": int(seed),
        "note": (
            "Reuse run35 LegNet LOCO labels; rematerialize on ready_caduceus. "
            "Direct keeps chrom-grain Caduceus-aligned ratios. "
            "Adversarial random uses ≈3:1:1."
        ),
    }
    print(f"split reuse: {json.dumps(rewrite_info, sort_keys=True)}", flush=True)

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
        "source_unif": str(SOURCE_UNIF),
        "out_root": str(OUT_ROOT),
        "panel_root": str(PANEL_ROOT),
        "model": MODEL,
        "split": SPLIT,
        "source_run": "run35_legnet_loco",
        "rewrite": rewrite_info,
        "staged_extra": staged_extra,
        "zsv": "mice",
        "adversarial": True,
        "direct": {
            "epochs": EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "n_devices": 1,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
        },
        "adversarial_schedule": {
            "epochs": ADV_EPOCHS,
            "min_epochs": ADV_MIN_EPOCHS,
            "early_stopping_patience": ADV_EARLY_STOPPING_PATIENCE,
            "split": "random",
            "ratios": list(ADV_RATIOS),
        },
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    write_rerun_manifest(OUT_ROOT, manifest)
    (OUT_ROOT / "split_done.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "split_csv": str(split_csv),
                "split_root": str(split_root),
                "source_unif": str(SOURCE_UNIF),
                "audit": audit,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "split_cpu_done.json").write_text(
        json.dumps({**manifest, "status": "COMPLETED"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"split_csv": split_csv, "split_root": split_root, "manifest": manifest}


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
    from src.pipeline.adversarial import setup_adversarial_random_fold_class
    from src.pipeline.job_queue import (
        CLASS_GPU_TRAIN,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.train import run_train

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    split_root = _require(OUT_ROOT / "SPLIT", "dir")
    _assert_loco_split(split_csv)
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    # GPU already cleared by wait_for_one_gpu (smi + queue). Short re-check only.
    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        job_class=CLASS_GPU_TRAIN,
        label=f"{RUN_NAME}_train",
        timeout_sec=120.0,
        poll_sec=10.0,
    )
    append_queue_entry(
        f"{RUN_NAME}_train",
        job=(
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"python -m src.runs_unif.{RUN_NAME}.continue_from_split"
        ),
        pid=os.getpid(),
        estimated_time="8-24h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        resources=(
            f"batch {batch} max_len {MAX_LENGTH}; "
            f"direct {epochs}/{min_epochs}/p{patience}; "
            f"adv {adv_epochs}/p{adv_patience}"
        ),
        log=str(ROOT / "logs" / f"{RUN_NAME}.log"),
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
        print(f"WARNING: viz[direct] skipped: {type(exc).__name__}: {exc}", flush=True)

    adv_root = OUT_ROOT / "adversarial"
    if adv_root.exists():
        raise FileExistsError(f"refusing overwrite: {adv_root}")

    print("adversarial: copy + random(⊆direct IDs) + fold-class …", flush=True)
    setup_adversarial_random_fold_class(
        adv_root=adv_root,
        label_split_csv=split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        fold_csv=PANEL_ROOT / "fold.csv",
        seed=SEED + 1,
        ratios=ADV_RATIOS,
        intersect_allow=True,
        build_legnet_input=False,
    )
    adv_split_root = _require(adv_root / "SPLIT", "dir")
    print(
        f"adversarial Caduceus train gpu={gpu} epochs={adv_epochs} "
        f"patience={adv_patience}",
        flush=True,
    )
    run_train(
        model="caduceus",
        type="classification",
        folders=adv_split_root,
        outdir=adv_root / "train",
        strategy="random",
        smoke=False,
        epochs=adv_epochs,
        batch_size=batch,
        max_length=MAX_LENGTH,
        seed=SEED,
        n_devices=1,
        num_workers=NUM_WORKERS,
        zsv_root=adv_root,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=adv_patience,
        min_epochs=adv_min_epochs,
    )

    try:
        from src.pipeline.pipeline_viz import run_pipeline_viz_auto
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        run_pipeline_viz_auto(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            train_dir=adv_root / "train",
            run_id=RUN_NAME,
            seed=SEED,
            plot_train=True,
            plot_sbs=False,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
        refresh_pipeline_monitors(OUT_ROOT, run_id=RUN_NAME, include_split_compare=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: viz/monitor skipped: {type(exc).__name__}: {exc}", flush=True)

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
        f"{RUN_NAME}: source={SOURCE_UNIF} → out={OUT_ROOT} "
        f"panel={PANEL_ROOT} split_only={cfg['split_only']} "
        f"adversarial=true source=run35_loco",
        flush=True,
    )

    split_done = OUT_ROOT / "split_done.json"
    if not split_done.is_file():
        stage_split(seed=SEED)
    else:
        print(f"reuse staged split: {split_done}", flush=True)
        _require(OUT_ROOT / "split.csv", "file")
        _require(OUT_ROOT / "SPLIT", "dir")
        _assert_loco_split(OUT_ROOT / "split.csv")

    if cfg["split_only"]:
        print("split_only=true — skipping train", flush=True)
        return 0

    if cfg.get("gpu") is not None:
        gpu = int(cfg["gpu"])
        print(f"gpu={gpu} locked by argv", flush=True)
    elif cfg["skip_wait"]:
        used = {g: _gpu_used_mib(g) for g in PREFERRED_GPUS}
        print(f"skip_wait=true GPU memory.used MiB: {used}", flush=True)
        free = [
            g
            for g in PREFERRED_GPUS
            if used.get(g) is not None and used[g] < MEM_FREE_MIB
        ]
        gpu = free[0] if free else PREFERRED_GPUS[0]
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
