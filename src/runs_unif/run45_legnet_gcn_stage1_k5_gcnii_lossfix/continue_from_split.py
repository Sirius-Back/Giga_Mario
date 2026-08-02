"""Run45: GCNII Stage-1 lossfix → LegNet direct (no adversarial).

New (write):
  ``src/runs_unif/run45_legnet_gcn_stage1_k5_gcnii_lossfix``
  ``runs_unif/legnet/run45_legnet_gcn_stage1_k5_gcnii_lossfix``

Flow:
  1. GCN cascade on pretrained ``VGAE/stage1_region_k5_gcnii_lossfix`` (architecture=gcnii, homology_first, k=5):
     reuse ``split.csv`` if present, else infer from ``checkpoints/best.pt``,
     else train (graph from run37). Overlay mice ZSV from ``fold.csv``.
  2. Materialize ``SPLIT/`` + LegNet TSV (CPU; reusable via ``split_done.json``).
  3. Wait for **1** free GPU; direct train (min 15 / max 30 / patience 10)
     + mice ZSV; best → final_model.
  4. No adversarial (not requested for this run).

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs_unif.run45_legnet_gcn_stage1_k5_gcnii_lossfix.continue_from_split
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

RUN_I = 45
MODEL = "legnet"
SPLIT = "gcn"
SPLIT_PARAMS = "stage1_k5_gcnii_lossfix"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME
VGAE_ROOT = ROOT / "VGAE"
GCN_MODEL = "stage1_region_k5_gcnii_lossfix"
# Contingency graph used when cascade must train (not needed for reuse/infer).
VGAE_GRAPH = (
    ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100" / "graph"
)

SEED = 42
RATIOS = (3.0, 1.0, 1.0)
EPOCHS = 30
MIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
ADV_EPOCHS = 10
ADV_MIN_EPOCHS = 0
ADV_EARLY_STOPPING_PATIENCE = 5
ADV_RATIOS = (3.0, 1.0, 1.0)

BATCH_SIZE = 8192
NUM_WORKERS = 8
PREFERRED_GPUS = (1, 2, 0, 3)
MEM_FREE_MIB = 400  # allow small idle CUDA contexts (~311 MiB)
POLL_SEC = 60
PEAK_RAM_GIB_TRAIN = 24.0
PEAK_RAM_GIB_SPLIT = 24.0


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
    confirm_sec: float = 5.0,
) -> int:
    print(
        f"Waiting for any GPU in {prefer} free (memory.used < {thresh} MiB "
        f"for {confirm_sec}s); poll every {poll_sec}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in prefer}
        print(f"GPU memory.used MiB: {used}", flush=True)
        free = [g for g in prefer if used.get(g) is not None and used[g] < thresh]
        if free:
            gpu = free[0]
            time.sleep(confirm_sec)
            used2 = _gpu_used_mib(gpu)
            if used2 is not None and used2 < thresh:
                print(f"GPU {gpu} free — starting train", flush=True)
                return gpu
            print(f"GPU {gpu} re-occupied ({used2} MiB) — keep waiting", flush=True)
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
        "skip_direct": False,
        "force_adv": False,
        "gpu": None,
        "gcn_model": GCN_MODEL,
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
        elif tok.startswith("gcn_model="):
            cfg["gcn_model"] = tok.split("=", 1)[1].strip()
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


def _count_split(split_csv: Path) -> dict[str, int]:
    from src.pipeline.common import read_csv
    from src.pipeline.generate_fold import is_zsv_fold

    counts = {"train": 0, "test": 0, "val": 0, "zsv": 0, "other": 0}
    for row in read_csv(split_csv):
        lab = str(row["train_test"]).strip().lower()
        if is_zsv_fold(lab) or lab == "zsv":
            counts["zsv"] += 1
        elif lab in counts:
            counts[lab] += 1
        elif lab == "validation":
            counts["val"] += 1
        else:
            counts["other"] += 1
    return counts


def _assert_vgae_split(split_csv: Path) -> None:
    """Refuse random-looking folds; require vgae_/gcn_ fold prefixes or roles."""
    from src.pipeline.common import read_csv
    from src.pipeline.generate_fold import is_zsv_fold

    n_checked = 0
    n_ok = 0
    for row in read_csv(split_csv):
        if is_zsv_fold(row["train_test"]) or str(row.get("fold", "")).lower() == "zsv":
            continue
        fold = str(row.get("fold", ""))
        n_checked += 1
        if fold.startswith("vgae_") or fold.startswith("gcn_"):
            n_ok += 1
        if n_checked >= 500:
            break
    if n_checked == 0 or n_ok < max(1, n_checked // 2):
        raise RuntimeError(
            f"split.csv does not look like VGAE/GCN (vgae_/gcn_ folds); "
            f"ok={n_ok}/{n_checked} in sample — refusing train"
        )


def apply_mice_zsv_overlay(split_csv: Path, fold_csv: Path) -> dict[str, int]:
    """Force fold.csv zsv IDs to train_test=zsv (VGAE assigns mice into TVT)."""
    from src.pipeline.common import SPLIT_CSV_COLUMNS, read_csv, write_csv
    from src.pipeline.generate_fold import is_zsv_fold

    zsv_ids: set[str] = set()
    for row in read_csv(fold_csv):
        if is_zsv_fold(row.get("fold", "")) or str(row.get("fold", "")).lower() == "zsv":
            zsv_ids.add(str(row["ID"]))
    if not zsv_ids:
        raise RuntimeError(f"No zsv IDs in {fold_csv} — mice fold missing")

    rows = read_csv(split_csv)
    seen: set[str] = set()
    n_relabel = 0
    out_rows: list[dict[str, str]] = []
    for row in rows:
        rid = str(row["ID"])
        seen.add(rid)
        if rid in zsv_ids:
            if str(row.get("train_test", "")).lower() != "zsv":
                n_relabel += 1
            out_rows.append({"ID": rid, "train_test": "zsv", "fold": "zsv"})
        else:
            out_rows.append(
                {
                    "ID": rid,
                    "train_test": str(row["train_test"]),
                    "fold": str(row.get("fold") or row["train_test"]),
                }
            )
    n_append = 0
    for rid in sorted(zsv_ids - seen, key=lambda x: int(x) if x.isdigit() else x):
        out_rows.append({"ID": rid, "train_test": "zsv", "fold": "zsv"})
        n_append += 1
    write_csv(split_csv, out_rows, SPLIT_CSV_COLUMNS)
    stats = {
        "n_zsv_fold": len(zsv_ids),
        "n_relabel": n_relabel,
        "n_append": n_append,
        "n_rows": len(out_rows),
    }
    print(f"mice ZSV overlay: {stats}", flush=True)
    return stats


def stage_split(*, seed: int = SEED, gcn_model: str = GCN_MODEL) -> dict:
    """GCN/VGAE cascade + mice ZSV overlay + materialize SPLIT + LegNet TSV."""
    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.rerun_aligned import assert_fresh_out_root, write_rerun_manifest
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.runs_unif.run45_legnet_gcn_stage1_k5_gcnii_lossfix.ensure_mice_fold import (
        main as ensure_fold,
    )

    ensure_fold()

    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "PARSED", "dir")
    _require(PANEL_ROOT / "PREDICT", "dir")
    _require(PANEL_ROOT / "fold.csv", "file")
    _require(PANEL_ROOT / "MARKED", "dir")

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
        resources=f"gcn cascade model={gcn_model} + mice ZSV + materialize",
        log=str(ROOT / "logs" / f"{RUN_NAME}_split.log"),
    )

    graph = VGAE_GRAPH if VGAE_GRAPH.is_dir() else None
    print(
        f"split_predict type=gcn model={gcn_model} "
        f"(reuse→infer→train); graph={'yes' if graph else 'no'}",
        flush=True,
    )
    split_csv = run_split_predict(
        outdir=OUT_ROOT,
        type="gcn",
        seed=seed,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        marked_fasta=PANEL_ROOT / "MARKED",
        ratios=RATIOS,
        kmer_size=5,
        gcn_model=gcn_model,
        gcn_vgae_root=VGAE_ROOT,
        vgae_graph_dir=graph,
    )
    split_csv = Path(split_csv)
    # Provenance: cascade source (reuse / infer / train)
    cascade_note = {
        "gcn_model": gcn_model,
        "vgae_root": str(VGAE_ROOT),
        "pretrained_meta": None,
    }
    meta_path = VGAE_ROOT / gcn_model / "train_meta.json"
    if meta_path.is_file():
        cascade_note["pretrained_meta"] = json.loads(
            meta_path.read_text(encoding="utf-8")
        )

    zsv_stats = apply_mice_zsv_overlay(split_csv, PANEL_ROOT / "fold.csv")
    counts = _count_split(split_csv)
    print(f"vgae(+mice zsv) counts: {counts}", flush=True)
    _assert_vgae_split(split_csv)
    if counts["zsv"] < 1000:
        raise RuntimeError(f"Expected mice ZSV rows; got zsv={counts['zsv']}")

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
        "rerun": False,
        "aligned_run": RUN_I,
        "run_name": RUN_NAME,
        "out_root": str(OUT_ROOT),
        "panel_root": str(PANEL_ROOT),
        "model": MODEL,
        "split": SPLIT,
        "split_params": SPLIT_PARAMS,
        "gcn_model": gcn_model,
        "cascade": cascade_note,
        "zsv_overlay": zsv_stats,
        "split_counts": counts,
        "ratios": list(RATIOS),
        "zsv": "mice",
        "direct": {
            "epochs": EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "n_devices": 1,
        },
        "adversarial": False,
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    write_rerun_manifest(OUT_ROOT, manifest)
    (OUT_ROOT / "split_done.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "split_csv": str(split_csv),
                "tsv": str(tsv),
                "counts": counts,
                "gcn_model": gcn_model,
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
    from src.pipeline.job_queue import (
        CLASS_GPU_TRAIN,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.train import run_train

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    tsv = _require(OUT_ROOT / "legnet_input" / "all.tsv", "file")
    _require(OUT_ROOT / "SPLIT", "dir")
    _assert_vgae_split(split_csv)
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")

    gpus = gpus[:1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    n_devices = 1

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
            f"gpu={gpus[0]}"
        ),
        pid=os.getpid(),
        estimated_time="6-20h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB_TRAIN,
        gpus=gpus,
        resources=f"batch {batch}; direct {epochs}/{min_epochs}/p{patience}; "
        f"no-adv; skip_direct={skip_direct}",
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

    # Adversarial not requested for run45.
    print("adversarial: SKIPPED (not requested)", flush=True)

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
        f"{RUN_NAME}: out={OUT_ROOT} panel={PANEL_ROOT} "
        f"gcn_model={cfg['gcn_model']} split_only={cfg['split_only']}",
        flush=True,
    )

    split_done = OUT_ROOT / "split_done.json"
    if not split_done.is_file():
        stage_split(seed=SEED, gcn_model=str(cfg["gcn_model"]))
    else:
        print(f"reuse staged split: {split_done}", flush=True)
        _require(OUT_ROOT / "split.csv", "file")
        _require(OUT_ROOT / "legnet_input" / "all.tsv", "file")
        _require(OUT_ROOT / "SPLIT", "dir")
        _assert_vgae_split(OUT_ROOT / "split.csv")
        counts = _count_split(OUT_ROOT / "split.csv")
        print(f"reused split counts: {counts}", flush=True)

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
        gpus=(gpu,),
        skip_wait=bool(cfg["skip_wait"]),
        skip_direct=bool(cfg["skip_direct"]),
        force_adv=bool(cfg["force_adv"]),
    )
    print(f"{RUN_NAME} COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
