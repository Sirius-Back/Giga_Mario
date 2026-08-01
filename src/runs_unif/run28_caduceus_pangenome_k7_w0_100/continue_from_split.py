"""Continue unif run27 pangenome split → Caduceus direct + adversarial.

Source (read-only):
  ``runs_unif/legnet/run27_legnet_pangenome_k7_w0_100``
  (pangenome k=7, window 0…100, modularity-refined CCs, ≈3:1:1)

New (write):
  ``src/runs_unif/run28_caduceus_pangenome_k7_w0_100``
  ``runs_unif/caduceus/run28_caduceus_pangenome_k7_w0_100``

Flow:
  1. Wait until run27 ``split.csv`` (+ sidecars / ``split_done.json``) exist.
  2. Reuse that split table; verify ≈3:1:1 and fold sizes.
  3. Materialize ``SPLIT/`` from ``ready_caduceus`` PARSED/PREDICT (CPU).
  4. Wait until **1** GPU is free, then direct Caduceus
     (min 15 / max 30 / patience 10) + mice ZSV.
  5. Adversarial: random ⊆ direct IDs + fold-class; max 10 / patience 5 + ZSV.
  6. Reports / TensorBoard via pipeline viz + train_monitor.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs_unif.run28_caduceus_pangenome_k7_w0_100.continue_from_split
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_I = 28
MODEL = "caduceus"
SPLIT = "pangenome"
SPLIT_PARAMS = "k7_w0_100"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

SOURCE_UNIF = ROOT / "runs_unif" / "legnet" / "run27_legnet_pangenome_k7_w0_100"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
RATIOS = (3.0, 1.0, 1.0)
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
ADV_EPOCHS = 10
ADV_MIN_EPOCHS = 0
ADV_EARLY_STOPPING_PATIENCE = 5

# Caduceus defaults (1 GPU); conservative vs long ready_caduceus windows.
BATCH_SIZE = 192
MAX_LENGTH = 256
NUM_WORKERS = 4
PREFERRED_GPUS = (0, 1, 2, 3)
MEM_FREE_MIB = 200
POLL_SEC = 60
GPU_CONFIRM_SEC = 5
SOURCE_POLL_SEC = 120
PEAK_RAM_GIB_TRAIN = 24.0
PEAK_RAM_GIB_SPLIT = 24.0
MAX_FOLD_SIZE = 1000  # modularity cap from run27

PANGENOME_INTERMEDIATES = (
    "split.csv",
    "sbs_assignment.csv",
    "pangenome_assignment.csv",
    "pangenome_split_meta.json",
    "intersect_pangenome.csv",
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


def wait_for_source_split(*, poll_sec: int = SOURCE_POLL_SEC) -> Path:
    """Block until run27 has a reusable split table (prefer split_done.json)."""
    split_csv = SOURCE_UNIF / "split.csv"
    split_done = SOURCE_UNIF / "split_done.json"
    sbs = SOURCE_UNIF / "sbs_assignment.csv"
    print(
        f"Waiting for source split from {SOURCE_UNIF} "
        f"(need split.csv + sbs_assignment.csv; prefer split_done.json); "
        f"poll every {poll_sec}s …",
        flush=True,
    )
    while True:
        has_csv = split_csv.is_file()
        has_sbs = sbs.is_file()
        has_done = split_done.is_file()
        print(
            f"source ready? split.csv={has_csv} sbs={has_sbs} "
            f"split_done={has_done}",
            flush=True,
        )
        if has_csv and has_sbs:
            if has_done:
                print(f"source split ready: {split_done}", flush=True)
            else:
                print(
                    f"source split.csv present (split_done pending) — proceeding: "
                    f"{split_csv}",
                    flush=True,
                )
            return split_csv
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
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "adv_epochs": ADV_EPOCHS,
        "adv_min_epochs": ADV_MIN_EPOCHS,
        "adv_patience": ADV_EARLY_STOPPING_PATIENCE,
        "skip_wait": False,
        "split_only": False,
        "skip_direct": False,
        "force_adv": False,
        "max_fold_size": MAX_FOLD_SIZE,
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
        elif tok.startswith("max_fold_size="):
            cfg["max_fold_size"] = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok in {"skip_wait=true", "--skip-wait"}:
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


def _audit_split(split_csv: Path, *, max_fold_size: int) -> dict:
    from src.pipeline.common import read_csv
    from src.pipeline.generate_fold import is_zsv_fold
    from src.pipeline.rerun_aligned import count_train_test_val, is_aligned_ratios

    rows = list(read_csv(split_csv))
    counts = count_train_test_val(rows)
    fold_sizes = Counter(
        str(r.get("fold") or "")
        for r in rows
        if not is_zsv_fold(r["train_test"]) and not is_zsv_fold(r.get("fold", ""))
    )
    max_sz = max(fold_sizes.values()) if fold_sizes else 0
    aligned = is_aligned_ratios(
        (counts["train"], counts["test"], counts["val"]), tol=0.10
    )
    info = {
        "counts": counts,
        "n_folds": len(fold_sizes),
        "max_fold_size": int(max_sz),
        "max_fold_size_cap": int(max_fold_size),
        "aligned_3_1_1": bool(aligned),
        "top_folds": fold_sizes.most_common(5),
    }
    if not aligned:
        raise RuntimeError(
            f"source split not near ≈3:1:1: {counts} (from {split_csv})"
        )
    if max_sz > int(max_fold_size):
        raise RuntimeError(
            f"fold size {max_sz} exceeds max_fold_size={max_fold_size}; "
            f"source should have been modularity-refined (top={fold_sizes.most_common(5)})"
        )
    return info


def stage_split(*, seed: int = SEED, max_fold_size: int = MAX_FOLD_SIZE) -> dict:
    """Reuse run27 split table + materialize Caduceus SPLIT (CPU only)."""
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
    _require(SOURCE_UNIF / "sbs_assignment.csv", "file")
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

    audit = _audit_split(src_split, max_fold_size=max_fold_size)
    print(f"source split audit: {json.dumps(audit, sort_keys=True)}", flush=True)

    staged_extra: dict[str, str] = {}
    for rel in PANGENOME_INTERMEDIATES:
        src = SOURCE_UNIF / rel
        if not src.is_file():
            continue
        dest = OUT_ROOT / rel
        shutil.copy2(src, dest)
        staged_extra[rel] = str(dest)
        print(f"staged {rel}", flush=True)

    graph_src = SOURCE_UNIF / "graph"
    if graph_src.is_dir():
        graph_dest = OUT_ROOT / "graph"
        if graph_dest.exists():
            shutil.rmtree(graph_dest)
        shutil.copytree(graph_src, graph_dest)
        staged_extra["graph/"] = str(graph_dest)
        print(f"staged graph/ ({graph_src})", flush=True)

    # Ensure split.csv columns normalized; keep run27 labels (already ≈3:1:1).
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
    rewrite_info = {
        "method": "reuse_run27_unif_split_csv",
        "source_split_csv": str(src_split),
        "dest_split_csv": str(split_csv),
        "modularity_refine": True,
        "max_fold_size": int(max_fold_size),
        "audit": audit,
        "seed": int(seed),
        "ratios": list(RATIOS),
        "note": (
            "Run27 applied pangenome k=7 + modularity refine for large "
            "connected components; folds reused for Caduceus panel "
            "(ready_caduceus PARSED/PREDICT rematerialized)."
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
        "split_params": SPLIT_PARAMS,
        "kmer_size": 7,
        "window": {"pos1": 0, "pos2": 100},
        "modularity_refine": True,
        "max_fold_size": int(max_fold_size),
        "ratios": list(RATIOS),
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
            },
            indent=2,
        )
        + "\n",
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
    skip_direct: bool = False,
    force_adv: bool = False,
) -> None:
    from src.pipeline.adversarial import setup_adversarial_random_fold_class
    from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry
    from src.pipeline.train import run_train

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    split_root = _require(OUT_ROOT / "SPLIT", "dir")
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    append_queue_entry(
        f"{RUN_NAME}_train",
        job=(
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"python -m src.runs_unif.{RUN_NAME}.continue_from_split "
            f"skip_direct={str(skip_direct).lower()} force_adv={str(force_adv).lower()}"
        ),
        pid=os.getpid(),
        estimated_time="8-24h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=(gpu,),
        resources=(
            f"batch {batch} max_len {MAX_LENGTH}; "
            f"direct {epochs}/{min_epochs}/p{patience}; "
            f"adv {adv_epochs}/p{adv_patience}; skip_direct={skip_direct}"
        ),
        log=str(ROOT / "logs" / f"{RUN_NAME}_train.log"),
    )

    direct_out = OUT_ROOT / "direct"
    if skip_direct:
        if not (direct_out / "best_model" / "best_meta.json").is_file():
            raise FileNotFoundError(
                f"skip_direct=true but missing {direct_out / 'best_model' / 'best_meta.json'}"
            )
        print(f"skip_direct=true — reuse {direct_out}", flush=True)
    else:
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
        if not force_adv:
            raise FileExistsError(
                f"refusing overwrite: {adv_root} (pass force_adv=true to archive)"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = OUT_ROOT / f"adversarial_FAILED_{stamp}"
        print(f"force_adv=true — archive {adv_root} → {archived}", flush=True)
        adv_root.rename(archived)

    print("adversarial: copy + random(⊆direct IDs) + fold-class …", flush=True)
    setup_adversarial_random_fold_class(
        adv_root=adv_root,
        label_split_csv=split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        fold_csv=PANEL_ROOT / "fold.csv",
        seed=SEED + 1,
        ratios=RATIOS,
        intersect_allow=True,
        build_legnet_input=False,  # Caduceus trains on SPLIT, not 230 bp TSV
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
        f"adversarial=true modularity_max_fold={cfg['max_fold_size']}",
        flush=True,
    )

    split_done = OUT_ROOT / "split_done.json"
    if not split_done.is_file():
        stage_split(seed=SEED, max_fold_size=int(cfg["max_fold_size"]))
    else:
        print(f"reuse staged split: {split_done}", flush=True)
        _require(SOURCE_UNIF / "split.csv", "file")

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
        skip_direct=bool(cfg["skip_direct"]),
        force_adv=bool(cfg["force_adv"]),
    )
    print(f"{RUN_NAME} COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
