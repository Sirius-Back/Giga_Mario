"""Continue legacy run2 with a rebuilt 3:1:1 split → LegNet + adversarial.

Legacy (read-only):
  ``src/runs/run2``, ``runs/run2``

New (write):
  ``src/runs_unif/run2_legnet_random``
  ``runs_unif/legnet/run2_legnet_random``

Flow:
  1. Rewrite ``split.csv`` from the present table to train:test:val ≈ 3:1:1
     (legacy inverted 1:1:3 → swap train↔val; never mutate ``runs/run2``).
  2. Materialize ``SPLIT/`` + LegNet TSV (CPU; no GPU wait).
  3. Wait until 4 GPUs are free (fallback 2), then direct train
     (min 15 / max 30 / early-stop patience 10) + mice ZSV.
  4. Adversarial: random split, max 10 epochs, early-stop patience 5.

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs_unif.run2_legnet_random.continue_from_split
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

RUN_I = 2
MODEL = "legnet"
SPLIT = "random"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}"

LEGACY_OUT = ROOT / "runs" / f"run{RUN_I}"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
# Direct train
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
# Adversarial train
ADV_EPOCHS = 10
ADV_MIN_EPOCHS = 0
ADV_EARLY_STOPPING_PATIENCE = 5

BATCH_SIZE = 8192
NUM_WORKERS = 8
PREFERRED_GPUS = (0, 1, 2, 3)
FALLBACK_GPUS = (0, 1)
MEM_FREE_MIB = 1500
POLL_SEC = 60
PEAK_RAM_GIB_TRAIN = 48.0
PEAK_RAM_GIB_SPLIT = 16.0


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


def select_free_gpus(
    *,
    prefer: tuple[int, ...] = PREFERRED_GPUS,
    fallback: tuple[int, ...] = FALLBACK_GPUS,
    thresh: int = MEM_FREE_MIB,
) -> tuple[int, ...]:
    """Return prefer (4) or fallback (2) when all listed GPUs are under thresh."""
    used = {g: _gpu_used_mib(g) for g in prefer}
    print(f"GPU memory.used MiB: {used}", flush=True)
    if all(v is not None and v < thresh for v in (used[g] for g in prefer)):
        return prefer
    if all(v is not None and v < thresh for v in (used[g] for g in fallback)):
        return fallback
    return ()


def wait_for_train_gpus(
    *,
    prefer: tuple[int, ...] = PREFERRED_GPUS,
    fallback: tuple[int, ...] = FALLBACK_GPUS,
    thresh: int = MEM_FREE_MIB,
    poll_sec: int = POLL_SEC,
) -> tuple[int, ...]:
    print(
        f"Waiting for GPUs {prefer} (or fallback {fallback}) "
        f"free (memory.used < {thresh} MiB); poll every {poll_sec}s …",
        flush=True,
    )
    while True:
        gpus = select_free_gpus(prefer=prefer, fallback=fallback, thresh=thresh)
        if gpus:
            print(f"GPUs {gpus} free — starting train", flush=True)
            return gpus
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
        "n_devices": None,
        "skip_direct": False,
        "force_adv": False,
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
        elif tok.startswith("n_devices="):
            cfg["n_devices"] = int(tok.split("=", 1)[1])
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


def stage_split(*, seed: int = SEED) -> dict:
    """Rewrite split table + materialize SPLIT + LegNet TSV (CPU only)."""
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
        log=str(ROOT / "logs" / f"{RUN_NAME}.log"),
    )

    rewrite_info = rewrite_split_table_aligned(
        LEGACY_OUT / "split.csv",
        OUT_ROOT / "split.csv",
        seed=seed,
        prefer_label_swap=True,
    )
    print(f"split rewrite: {json.dumps(rewrite_info, sort_keys=True)}", flush=True)

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

    # Provenance: never touch legacy trees.
    manifest = {
        "rerun": True,
        "aligned_run": RUN_I,
        "run_name": RUN_NAME,
        "legacy": {
            "src": str(ROOT / "src" / "runs" / f"run{RUN_I}"),
            "out": str(LEGACY_OUT),
        },
        "out_root": str(OUT_ROOT),
        "panel_root": str(PANEL_ROOT),
        "model": MODEL,
        "split": SPLIT,
        "ratios": [3, 1, 1],
        "rewrite": rewrite_info,
        "zsv": "mice",
        "direct": {
            "epochs": EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "adversarial": {
            "split": "random",
            "epochs": ADV_EPOCHS,
            "min_epochs": ADV_MIN_EPOCHS,
            "early_stopping_patience": ADV_EARLY_STOPPING_PATIENCE,
        },
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    write_rerun_manifest(OUT_ROOT, manifest)
    (OUT_ROOT / "split_done.json").write_text(
        json.dumps({"status": "ok", "split_csv": str(split_csv), "tsv": str(tsv)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"split_csv": split_csv, "tsv": tsv, "manifest": manifest}


def run_train_stages(
    *,
    batch: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    adv_epochs: int,
    adv_min_epochs: int,
    adv_patience: int,
    gpus: tuple[int, ...],
    skip_wait: bool = False,
    skip_direct: bool = False,
    force_adv: bool = False,
) -> None:
    from src.pipeline.adversarial import setup_adversarial_random_fold_class
    from src.pipeline.job_queue import (
        CLASS_GPU_TRAIN,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.train import run_train

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    tsv = _require(OUT_ROOT / "legnet_input" / "all.tsv", "file")
    _require(OUT_ROOT / "SPLIT", "dir")
    # Mice ZSV trees must exist under out_root after materialize.
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    n_devices = len(gpus)

    if skip_wait:
        print(
            f"skip_wait=true — not blocking on queue GPU politics; "
            f"using verified free GPUs {gpus}",
            flush=True,
        )
    else:
        wait_until_launchable(
            peak_ram_gib=PEAK_RAM_GIB_TRAIN,
            gpus=gpus,
            job_class=CLASS_GPU_TRAIN,
            label=f"{RUN_NAME}_train",
        )
    append_queue_entry(
        f"{RUN_NAME}_train",
        job=(
            f"python -m src.runs_unif.{RUN_NAME}.continue_from_split "
            f"n_devices={n_devices}"
        ),
        pid=os.getpid(),
        estimated_time="6-20h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=gpus,
        resources=f"batch {batch}; direct {epochs}/{min_epochs}/p{patience}; "
        f"adv {adv_epochs}/p{adv_patience}; skip_direct={skip_direct}",
        log=str(ROOT / "logs" / f"{RUN_NAME}.log"),
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
            f"direct LegNet train n_devices={n_devices} epochs={epochs} "
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
            n_devices=n_devices,
            num_workers=NUM_WORKERS,
            legnet_demo=True,
            zsv_root=OUT_ROOT,
            eval_zsv=True,
            checkpoint_every_n_epochs=10,
            early_stopping_patience=patience,
            min_epochs=min_epochs,
        )

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
    _adv_split, adv_tsv = setup_adversarial_random_fold_class(
        adv_root=adv_root,
        label_split_csv=split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        fold_csv=PANEL_ROOT / "fold.csv",
        seed=SEED + 1,
        ratios=(3.0, 1.0, 1.0),
        intersect_allow=True,
    )
    print(
        f"adversarial LegNet train epochs={adv_epochs} "
        f"min_epochs={adv_min_epochs} patience={adv_patience}",
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
        n_devices=n_devices,
        num_workers=NUM_WORKERS,
        legnet_demo=True,
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
            train_dir=direct_out,
            run_id=RUN_NAME,
            seed=SEED,
            plot_train=True,
            plot_sbs=True,
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
                "gpus": list(gpus),
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
        f"{RUN_NAME}: legacy={LEGACY_OUT} → out={OUT_ROOT} "
        f"panel={PANEL_ROOT} split_only={cfg['split_only']}",
        flush=True,
    )

    split_done = OUT_ROOT / "split_done.json"
    if not split_done.is_file():
        stage_split(seed=SEED)
    else:
        print(f"reuse staged split: {split_done}", flush=True)
        # Ensure legacy still untouched / present.
        _require(LEGACY_OUT / "split.csv", "file")

    if cfg["split_only"]:
        print("split_only=true — skipping train", flush=True)
        return 0

    forced_n = cfg.get("n_devices")
    if forced_n is not None:
        n = int(forced_n)
        if n < 1:
            raise ValueError(f"n_devices must be >=1, got {n}")
        if cfg["skip_wait"]:
            # Prefer already-free GPUs; else first n indices.
            free = select_free_gpus(prefer=PREFERRED_GPUS, fallback=FALLBACK_GPUS)
            if n == 1:
                used = {g: _gpu_used_mib(g) for g in PREFERRED_GPUS}
                gpus = tuple(
                    g for g in PREFERRED_GPUS
                    if used.get(g) is not None and used[g] < MEM_FREE_MIB
                )[:1]
                if not gpus:
                    gpus = (0,)
            elif free and len(free) >= n:
                gpus = free[:n]
            else:
                gpus = PREFERRED_GPUS[:n]
            print(f"n_devices={n} skip_wait=true — using GPUs {gpus}", flush=True)
        else:
            if n == 1:
                print("Waiting for any single free GPU …", flush=True)
                while True:
                    used = {g: _gpu_used_mib(g) for g in PREFERRED_GPUS}
                    print(f"GPU memory.used MiB: {used}", flush=True)
                    free1 = [
                        g for g in PREFERRED_GPUS
                        if used.get(g) is not None and used[g] < MEM_FREE_MIB
                    ]
                    if free1:
                        gpus = (free1[0],)
                        break
                    time.sleep(POLL_SEC)
            else:
                gpus = wait_for_train_gpus()
                gpus = gpus[:n]
            print(f"n_devices={n} — using GPUs {gpus}", flush=True)
    elif cfg["skip_wait"]:
        gpus = select_free_gpus() or FALLBACK_GPUS
        print(f"skip_wait=true — using GPUs {gpus}", flush=True)
    else:
        gpus = wait_for_train_gpus()

    run_train_stages(
        batch=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        min_epochs=int(cfg["min_epochs"]),
        patience=int(cfg["patience"]),
        adv_epochs=int(cfg["adv_epochs"]),
        adv_min_epochs=int(cfg["adv_min_epochs"]),
        adv_patience=int(cfg["adv_patience"]),
        gpus=gpus,
        skip_wait=bool(cfg["skip_wait"]),
        skip_direct=bool(cfg["skip_direct"]),
        force_adv=bool(cfg["force_adv"]),
    )
    print(f"{RUN_NAME} COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
