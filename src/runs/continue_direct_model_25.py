"""Continue early-stopped *direct* trains to 25 epochs under ``model_25_epoch/``.

Does **not** modify existing ``best_model/`` / ``final_model/`` / logs under
``direct/``. Writes all new artifacts to ``{run}/direct/model_25_epoch/``, then
runs zero-shot validation on that folder only (no adversarial).

Eligible: ``direct/`` with both ``best_model`` and ``final_model``, max logged
epoch ``< 25``, excluding 1-epoch probes.

Usage::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.runs.continue_direct_model_25
    python -m src.runs.continue_direct_model_25 --dry-run
    python -m src.runs.continue_direct_model_25 --only run4,run5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.pipeline.job_queue import (
    CLASS_GPU_TRAIN,
    append_queue_entry,
    can_launch_parallel,
    wait_until_launchable,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET_EPOCHS = 25
GPUS = (0, 1, 2, 3)
N_DEVICES = 4
PEAK_RAM_LEGNET_GIB = 24.0
PEAK_RAM_CADUCEUS_GIB = 20.0
GPU_FREE_MIB = 500
GPU_POLL_SEC = 60


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
    except (subprocess.CalledProcessError, ValueError, OSError) as exc:
        print(f"WARNING: nvidia-smi GPU {index}: {exc}", flush=True)
        return None


def wait_for_gpus_free(
    gpus: tuple[int, ...] = GPUS,
    *,
    thresh_mib: int = GPU_FREE_MIB,
    poll_sec: float = GPU_POLL_SEC,
    timeout_sec: float | None = 24 * 3600,
    label: str = "gpus",
) -> None:
    """Block until each GPU reports memory.used below ``thresh_mib``."""
    t0 = time.monotonic()
    print(
        f"[gpu_wait] {label}: waiting for GPUs {gpus} free "
        f"(mem.used < {thresh_mib} MiB); poll {poll_sec}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in gpus}
        print(f"[gpu_wait] {label}: memory.used MiB={used}", flush=True)
        if all(v is not None and v < thresh_mib for v in used.values()):
            print(f"[gpu_wait] {label}: GPUs {gpus} free", flush=True)
            return
        if timeout_sec is not None and (time.monotonic() - t0) >= timeout_sec:
            raise TimeoutError(f"[gpu_wait] {label}: timed out; used={used}")
        time.sleep(poll_sec)


@dataclass(frozen=True)
class Candidate:
    run_id: str
    direct: Path
    model: str  # legnet | caduceus
    max_epoch: int
    batch_size: int
    seed: int
    resume_ckpt: Path | None
    splits_dir: Path | None
    data_path: Path | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _max_logged_epoch(direct: Path) -> int | None:
    jsonl = direct / "logs" / "train_metrics.jsonl"
    if not jsonl.is_file():
        return None
    epochs: list[int] = []
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = obj.get("epoch")
            if isinstance(ep, int):
                epochs.append(ep)
            elif isinstance(ep, float) and ep == int(ep):
                epochs.append(int(ep))
    return max(epochs) if epochs else None


def _detect_model(direct: Path) -> str | None:
    if (direct / "caduceus_input").is_dir():
        return "caduceus"
    cfg = direct / "logs" / "run_config.json"
    if cfg.is_file():
        raw = cfg.read_text(encoding="utf-8")
        if '"skill": "legnet"' in raw or "src/legnet.py" in raw:
            return "legnet"
        if "caduceus" in raw.lower():
            return "caduceus"
    if (direct / "model_2_1").is_dir() or list(direct.glob("**/pearson-*.ckpt")):
        return "legnet"
    rc = direct / "run_config.json"
    if rc.is_file() and "caduceus" in rc.read_text(encoding="utf-8").lower():
        return "caduceus"
    return None


def _legnet_last_ckpt(direct: Path) -> Path | None:
    ckpts = sorted(direct.rglob("last_model-*.ckpt"))
    skip = {"best_model", "final_model", "model_25_epoch"}
    usable = [p for p in ckpts if not any(part in skip for part in p.parts)]
    if not usable:
        return None
    return max(usable, key=lambda p: p.stat().st_mtime)


def _legnet_best_pearson_ckpt(direct: Path) -> Path | None:
    best_dir = direct / "best_model"
    if best_dir.is_dir():
        hits = sorted(best_dir.glob("pearson-*.ckpt"))
        if hits:
            return hits[0]
    hits = sorted(direct.rglob("pearson-*.ckpt"))
    skip = {"final_model", "model_25_epoch", "checkpoints"}
    usable = [p for p in hits if not any(part in skip for part in p.parts)]
    return usable[0] if usable else None


def _is_train_complete(out: Path, model: str) -> bool:
    if not (out / "final_model").exists() or not (out / "train_time.json").is_file():
        return False
    if model == "caduceus":
        timing = _read_json(out / "train_time.json")
        return int(timing.get("epochs_completed") or 0) >= TARGET_EPOCHS
    mx = _max_logged_epoch(out)
    # LegNet logs 0-indexed epochs; epoch 24 ⇒ 25 completed.
    return mx is not None and mx >= TARGET_EPOCHS - 1


def discover_candidates(
    runs_root: Path,
    *,
    only: Iterable[str] | None = None,
) -> list[Candidate]:
    only_set = {x.strip() for x in only} if only else None
    out: list[Candidate] = []
    for direct in sorted(runs_root.glob("*/direct")):
        run_id = direct.parent.name
        if only_set is not None and run_id not in only_set:
            continue
        if not (direct / "best_model").exists() or not (direct / "final_model").exists():
            continue
        max_ep = _max_logged_epoch(direct)
        if max_ep is None or max_ep >= TARGET_EPOCHS:
            continue
        model = _detect_model(direct)
        if model is None:
            continue
        rc_paths = [
            direct / "logs" / "run_config.json",
            direct / "run_config.json",
            direct / "config.json",
        ]
        requested = None
        seed = 42
        batch = 1024
        data_path: Path | None = None
        splits_dir: Path | None = None
        for rp in rc_paths:
            if not rp.is_file():
                continue
            try:
                cfg = _read_json(rp)
            except json.JSONDecodeError:
                continue
            if "epochs" in cfg:
                requested = int(cfg["epochs"])
            if "epoch_num" in cfg:
                requested = int(cfg["epoch_num"])
            if "seed" in cfg:
                seed = int(cfg["seed"])
            if "batch_size" in cfg:
                batch = int(cfg["batch_size"])
            if "train_batch_size" in cfg:
                batch = int(cfg["train_batch_size"])
            if "data_path" in cfg:
                data_path = Path(cfg["data_path"])
            if "splits_dir" in cfg:
                splits_dir = Path(cfg["splits_dir"])
        # Skip intentional 1-epoch probes.
        if requested is not None and requested <= 1 and max_ep <= 0:
            continue
        if model == "caduceus":
            splits_dir = splits_dir or (direct / "caduceus_input")
            if not splits_dir.is_dir():
                continue
            resume_ckpt = None
        else:
            if data_path is None or not data_path.is_file():
                alt = direct.parent / "legnet_input" / "all.tsv"
                if alt.is_file():
                    data_path = alt
                else:
                    continue
            resume_ckpt = _legnet_last_ckpt(direct)
            if resume_ckpt is None:
                continue
            # Keep global batch ≈ original single-GPU batch when scaling to 4 GPUs.
            if batch >= 4096:
                batch = max(512, batch // N_DEVICES)
        out.append(
            Candidate(
                run_id=run_id,
                direct=direct,
                model=model,
                max_epoch=max_ep,
                batch_size=batch,
                seed=seed,
                resume_ckpt=resume_ckpt,
                splits_dir=splits_dir,
                data_path=data_path,
            )
        )
    return out


def _prepare_out_dir(cand: Candidate) -> Path:
    """Create ``model_25_epoch`` without clobbering an existing completed tree."""
    out = cand.direct / "model_25_epoch"
    if out.exists():
        if _is_train_complete(out, cand.model):
            return out
        raise FileExistsError(
            f"{out} exists but is incomplete; remove it manually to retry "
            "(will not override in place)."
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if cand.model == "caduceus":
        src_best = cand.direct / "best_model"
        if not (src_best / "best_meta.json").is_file():
            raise FileNotFoundError(f"Missing Caduceus best_meta under {src_best}")
        shutil.copytree(src_best, out / "best_model")
        meta = {
            "source_direct": str(cand.direct),
            "resume_from": "best_model",
            "target_epochs": TARGET_EPOCHS,
            "parent_max_epoch": cand.max_epoch,
        }
    else:
        assert cand.resume_ckpt is not None
        seed_dir = out / "seed_from_parent"
        seed_dir.mkdir(parents=True, exist_ok=True)
        best_p = _legnet_best_pearson_ckpt(cand.direct)
        if best_p is not None:
            shutil.copy2(best_p, seed_dir / best_p.name)
        meta = {
            "source_direct": str(cand.direct),
            "resume_ckpt": str(cand.resume_ckpt),
            "seed_best": str(best_p) if best_p else None,
            "target_epochs": TARGET_EPOCHS,
            "parent_max_epoch": cand.max_epoch,
        }
    (out / "logs" / "continue_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out


def _conda_run(env: str, *args: str, cwd: Path, log: Path) -> int:
    cmd = ["conda", "run", "-n", env, "--no-capture-output", *args]
    log.parent.mkdir(parents=True, exist_ok=True)
    print("EXEC:", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# {time.strftime('%Y-%m-%dT%H:%M:%S%z')} EXEC {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in GPUS)},
        )
    return int(proc.returncode)


def _train_caduceus(cand: Candidate, out: Path, log: Path) -> int:
    assert cand.splits_dir is not None
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={N_DEVICES}",
        "-m",
        "src.caduceus",
        "--splits-dir",
        str(cand.splits_dir.resolve()),
        "--out",
        str(out.resolve()),
        "--epochs",
        str(TARGET_EPOCHS),
        "--batch-size",
        str(cand.batch_size),
        "--max-length",
        "208",
        "--seed",
        str(cand.seed),
        "--task",
        "regression",
        "--checkpoint-every-n-epochs",
        "10",
        "--early-stopping-patience",
        "0",
        "--min-epochs",
        str(TARGET_EPOCHS),
        "--resume",
    ]
    return _conda_run("caduceus_env", *cmd, cwd=ROOT, log=log)


def _train_legnet(cand: Candidate, out: Path, log: Path) -> int:
    assert cand.data_path is not None and cand.resume_ckpt is not None
    cmd = [
        "python",
        "-m",
        "src.legnet",
        "--data-path",
        str(cand.data_path.resolve()),
        "--out",
        str(out.resolve()),
        "--epochs",
        str(TARGET_EPOCHS),
        "--n-devices",
        str(N_DEVICES),
        "--train-batch-size",
        str(cand.batch_size),
        "--valid-batch-size",
        str(cand.batch_size),
        "--num-workers",
        "8",
        "--seed",
        str(cand.seed),
        "--checkpoint-every-n-epochs",
        "10",
        "--early-stopping-patience",
        "0",
        "--min-epochs",
        str(TARGET_EPOCHS),
        "--resume-ckpt",
        str(cand.resume_ckpt.resolve()),
        "--demo",
    ]
    return _conda_run("legnet", *cmd, cwd=ROOT, log=log)


def _run_zsv(cand: Candidate, out: Path, log: Path) -> int:
    split_root = cand.direct.parent
    env = "legnet" if cand.model == "legnet" else "caduceus_env"
    cmd = [
        "python",
        "-m",
        "src.pipeline.zsv_eval",
        "--model",
        cand.model,
        "--outdir",
        str(out.resolve()),
        "--split-root",
        str(split_root.resolve()),
        "--device",
        "0",
    ]
    return _conda_run(env, *cmd, cwd=ROOT, log=log)


def _mark_queue(name: str, status: str, *, log: str = "") -> None:
    """Flip the latest RUNNING heading for ``name`` to ``status`` (in place)."""
    path = ROOT / "queue.md"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        needle = f"### {name} — RUNNING"
        idx = text.rfind(needle)
        if idx >= 0:
            # Replace heading + trailing **status:** RUNNING in that block.
            end = text.find("\n### ", idx + 1)
            if end < 0:
                end = len(text)
            block = text[idx:end]
            block2 = block.replace(
                f"### {name} — RUNNING", f"### {name} — {status}", 1
            )
            block2 = block2.replace("**status:** RUNNING", f"**status:** {status}")
            path.write_text(text[:idx] + block2 + text[end:], encoding="utf-8")
            return
    append_queue_entry(
        name,
        job=f"status→{status}",
        pid=os.getpid(),
        estimated_time="-",
        status=status,
        resources="continue_direct_model_25",
        log=log,
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=0.0,
        gpus=GPUS,
    )


def process_one(cand: Candidate, *, dry_run: bool = False) -> dict[str, Any]:
    out = cand.direct / "model_25_epoch"
    result: dict[str, Any] = {
        "run_id": cand.run_id,
        "model": cand.model,
        "parent_max_epoch": cand.max_epoch,
        "out": str(out),
    }
    zsv_json = out / "logs" / "zero_shot_metrics.json"
    train_done = out.is_dir() and _is_train_complete(out, cand.model)
    if train_done and zsv_json.is_file():
        result["status"] = "SKIPPED"
        result["reason"] = "model_25_epoch already complete with ZSV"
        return result

    if dry_run:
        result["status"] = "READY"
        result["plan"] = {
            "n_devices": N_DEVICES,
            "gpus": list(GPUS),
            "epochs": TARGET_EPOCHS,
            "batch_size": cand.batch_size,
            "resume_ckpt": str(cand.resume_ckpt) if cand.resume_ckpt else None,
            "splits_dir": str(cand.splits_dir) if cand.splits_dir else None,
            "data_path": str(cand.data_path) if cand.data_path else None,
            "train_needed": not train_done,
            "zsv_needed": not zsv_json.is_file(),
        }
        return result

    peak = PEAK_RAM_LEGNET_GIB if cand.model == "legnet" else PEAK_RAM_CADUCEUS_GIB
    ok, reason = can_launch_parallel(
        peak_ram_gib=peak, gpus=GPUS, job_class=CLASS_GPU_TRAIN
    )
    if not ok:
        print(f"[{cand.run_id}] queue busy — waiting: {reason}", flush=True)
        wait_until_launchable(
            peak_ram_gib=peak,
            gpus=GPUS,
            job_class=CLASS_GPU_TRAIN,
            label=f"continue25_{cand.run_id}",
            poll_sec=60.0,
            timeout_sec=24 * 3600,
        )
    # Also wait on live nvidia-smi (queue.md may miss foreign GPU jobs).
    wait_for_gpus_free(GPUS, label=f"continue25_{cand.run_id}")

    log = ROOT / "logs" / f"continue_model_25_{cand.run_id}.log"
    job_name = f"continue_model_25_{cand.run_id}"
    append_queue_entry(
        job_name,
        job=f"python -m src.runs.continue_direct_model_25 --only {cand.run_id}",
        pid=os.getpid(),
        estimated_time="1–3h",
        status="RUNNING",
        resources=f"{cand.model} epochs→{TARGET_EPOCHS} batch={cand.batch_size}",
        log=str(log),
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=peak,
        gpus=GPUS,
    )
    try:
        if not train_done:
            out = _prepare_out_dir(cand)
            if cand.model == "caduceus":
                rc = _train_caduceus(cand, out, log)
            else:
                rc = _train_legnet(cand, out, log)
            if rc != 0:
                result["status"] = "FAILED"
                result["train_rc"] = rc
                _mark_queue(job_name, "FAILED", log=str(log))
                return result
        if not zsv_json.is_file():
            zrc = _run_zsv(cand, out, log)
            result["zsv_rc"] = zrc
            if zrc != 0:
                result["status"] = "FAILED"
                _mark_queue(job_name, "FAILED", log=str(log))
                return result
        result["status"] = "COMPLETED"
        result["zsv"] = str(out / "logs" / "zero_shot_metrics.json")
        _mark_queue(job_name, "COMPLETED", log=str(log))
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _mark_queue(job_name, "FAILED", log=str(log))
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs",
        help="Root containing run*/direct",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated run_id filter",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidates and planned commands; do not train",
    )
    args = ap.parse_args(argv)
    only = args.only.split(",") if args.only else None
    cands = discover_candidates(args.runs_root, only=only)
    summary: dict[str, Any] = {
        "target_epochs": TARGET_EPOCHS,
        "n_devices": N_DEVICES,
        "gpus": list(GPUS),
        "n_candidates": len(cands),
        "candidates": [
            {
                "run_id": c.run_id,
                "model": c.model,
                "max_epoch": c.max_epoch,
                "batch_size": c.batch_size,
            }
            for c in cands
        ],
        "results": [],
    }
    print(json.dumps({"discover": summary["candidates"]}, indent=2), flush=True)
    for cand in cands:
        print(
            f"=== {cand.run_id} ({cand.model}) max_epoch={cand.max_epoch} ===",
            flush=True,
        )
        summary["results"].append(process_one(cand, dry_run=args.dry_run))
    out_json = ROOT / "logs" / "continue_direct_model_25_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    failed = [r for r in summary["results"] if r.get("status") == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
