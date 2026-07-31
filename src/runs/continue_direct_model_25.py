"""Continue early-stopped *direct* trains to 25 epochs under ``model_25_epoch/``.

Does **not** modify existing ``best_model/`` / ``final_model/`` / logs under
``direct/``. Writes all new artifacts to ``{run}/direct/model_25_epoch/``, then
runs zero-shot validation on that folder only (no adversarial).

Scheduling:

- **LegNet** — one job per free unreserved GPU (`n_devices=1`); parallel when
  multiple free GPUs (fast; does not wait for a full 4-GPU Caduceus slot).
- **Caduceus** — wait until ≥4 free unreserved GPUs, then train on 4.

Eligible: ``direct/`` with both ``best_model`` and ``final_model``, max logged
epoch ``< 25``, excluding 1-epoch probes.

Usage::

    python -m src.runs.continue_direct_model_25
    python -m src.runs.continue_direct_model_25 --dry-run
    python -m src.runs.continue_direct_model_25 --only run4,run5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.pipeline.job_queue import (
    CLASS_GPU_TRAIN,
    append_queue_entry,
    can_launch_parallel,
    running_jobs,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET_EPOCHS = 25
GPUS = (0, 1, 2, 3)
LEGNET_MIN_GPUS = 1  # one LegNet job per free GPU (DDP-2 hangs on this host)
CADUCEUS_MIN_GPUS = 4
LEGNET_N_DEVICES = 1
PEAK_RAM_LEGNET_GIB = 16.0
PEAK_RAM_CADUCEUS_GIB = 20.0
GPU_FREE_MIB = 500
GPU_POLL_SEC = 30


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


def reserved_gpus(*, exclude_pid: int | None = None) -> set[int]:
    """GPU indices claimed by alive RUNNING ``gpu_train`` queue entries."""
    claimed: set[int] = set()
    for j in running_jobs():
        if j.job_class != CLASS_GPU_TRAIN:
            continue
        if exclude_pid is not None and j.pid == exclude_pid:
            continue
        claimed.update(j.gpus)
    return claimed


def available_gpus(
    pool: Sequence[int] = GPUS,
    *,
    thresh_mib: int = GPU_FREE_MIB,
    exclude_pid: int | None = None,
) -> list[int]:
    """Free on nvidia-smi and not reserved by another gpu_train job."""
    reserved = reserved_gpus(exclude_pid=exclude_pid)
    free: list[int] = []
    for g in pool:
        if g in reserved:
            continue
        used = _gpu_used_mib(g)
        if used is not None and used < thresh_mib:
            free.append(int(g))
    return free


def wait_for_n_gpus(
    n: int,
    pool: Sequence[int] = GPUS,
    *,
    thresh_mib: int = GPU_FREE_MIB,
    poll_sec: float = GPU_POLL_SEC,
    timeout_sec: float | None = 24 * 3600,
    label: str = "gpus",
) -> list[int]:
    """Block until ``n`` unreserved free GPUs exist; return their indices."""
    t0 = time.monotonic()
    print(
        f"[gpu_wait] {label}: need {n} free unreserved GPUs from {tuple(pool)} "
        f"(mem.used < {thresh_mib} MiB); poll {poll_sec}s …",
        flush=True,
    )
    while True:
        free = available_gpus(pool, thresh_mib=thresh_mib)
        used = {g: _gpu_used_mib(g) for g in pool}
        reserved = reserved_gpus()
        print(
            f"[gpu_wait] {label}: free={free} reserved={sorted(reserved)} "
            f"used_mib={used}",
            flush=True,
        )
        if len(free) >= n:
            chosen = free[:n]
            print(f"[gpu_wait] {label}: using GPUs {chosen}", flush=True)
            return chosen
        if timeout_sec is not None and (time.monotonic() - t0) >= timeout_sec:
            raise TimeoutError(
                f"[gpu_wait] {label}: timed out needing {n}; free={free}"
            )
        time.sleep(poll_sec)


@dataclass(frozen=True)
class Candidate:
    run_id: str
    direct: Path
    model: str  # legnet | caduceus
    max_epoch: int
    batch_size: int  # original per-config batch (scale at launch for LegNet)
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
    return mx is not None and mx >= TARGET_EPOCHS - 1


def _scaled_legnet_batch(orig: int, n_devices: int) -> int:
    """Keep approximate global batch when spreading across devices."""
    if orig >= 4096 and n_devices > 1:
        return max(512, orig // n_devices)
    return orig


def discover_candidates(
    runs_root: Path,
    *,
    only: Iterable[str] | None = None,
) -> list[Candidate]:
    only_set = {x.strip() for x in only} if only else None
    # Outdirs currently being trained by a live process (skip mid-flight).
    busy_outdirs: set[str] = set()
    try:
        ps = subprocess.check_output(["ps", "-eo", "args="], text=True)
        for line in ps.splitlines():
            if "src.caduceus" not in line and "src.legnet" not in line:
                continue
            if "model_25_epoch" in line:
                continue
            if "--out" in line:
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "--out" and i + 1 < len(parts):
                        busy_outdirs.add(str(Path(parts[i + 1]).resolve()))
    except (subprocess.CalledProcessError, OSError):
        pass

    out: list[Candidate] = []
    for direct in sorted(runs_root.glob("*/direct")):
        run_id = direct.parent.name
        if only_set is not None and run_id not in only_set:
            continue
        if str(direct.resolve()) in busy_outdirs:
            print(f"[discover] skip {run_id}: live train writing {direct}", flush=True)
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
    # Prefer LegNet in discovery order so schedulers see them first.
    out.sort(key=lambda c: (0 if c.model == "legnet" else 1, c.run_id))
    return out


def _prepare_out_dir(cand: Candidate) -> Path:
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


def _conda_run(
    env: str,
    *args: str,
    cwd: Path,
    log: Path,
    gpus: Sequence[int],
) -> int:
    cmd = ["conda", "run", "-n", env, "--no-capture-output", *args]
    log.parent.mkdir(parents=True, exist_ok=True)
    gpu_s = ",".join(str(g) for g in gpus)
    print(f"EXEC (CUDA_VISIBLE_DEVICES={gpu_s}): {' '.join(cmd)}", flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n# {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
            f"CUDA_VISIBLE_DEVICES={gpu_s} EXEC {' '.join(cmd)}\n"
        )
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_s},
        )
    return int(proc.returncode)


def _train_caduceus(
    cand: Candidate, out: Path, log: Path, *, gpus: Sequence[int], n_devices: int
) -> int:
    assert cand.splits_dir is not None
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n_devices}",
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
        # Match parent runs: cap epoch eval. Without this, full val (n≈2.8e5)
        # dominates wall time (~70 min/epoch vs ~90s train).
        "--eval-max-samples",
        "8192",
        "--train-eval-max-samples",
        "8192",
    ]
    return _conda_run("caduceus_env", *cmd, cwd=ROOT, log=log, gpus=gpus)


def _train_legnet(
    cand: Candidate, out: Path, log: Path, *, gpus: Sequence[int], n_devices: int
) -> int:
    assert cand.data_path is not None and cand.resume_ckpt is not None
    batch = _scaled_legnet_batch(cand.batch_size, n_devices)
    # ddp_spawn + num_workers>0 often aborts DataLoader workers on this host.
    num_workers = 0 if n_devices > 1 else 8
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
        str(n_devices),
        "--train-batch-size",
        str(batch),
        "--valid-batch-size",
        str(batch),
        "--num-workers",
        str(num_workers),
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
    return _conda_run("legnet", *cmd, cwd=ROOT, log=log, gpus=gpus)


def _run_zsv(cand: Candidate, out: Path, log: Path, *, gpus: Sequence[int]) -> int:
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
    return _conda_run(env, *cmd, cwd=ROOT, log=log, gpus=gpus[:1] or (0,))


def _mark_queue(name: str, status: str, *, log: str = "") -> None:
    path = ROOT / "queue.md"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        needle = f"### {name} — RUNNING"
        idx = text.rfind(needle)
        if idx >= 0:
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
        gpus=(),
    )


def process_one(
    cand: Candidate,
    *,
    gpus: Sequence[int],
    n_devices: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    out = cand.direct / "model_25_epoch"
    batch = (
        _scaled_legnet_batch(cand.batch_size, n_devices)
        if cand.model == "legnet"
        else cand.batch_size
    )
    result: dict[str, Any] = {
        "run_id": cand.run_id,
        "model": cand.model,
        "parent_max_epoch": cand.max_epoch,
        "out": str(out),
        "gpus": list(gpus),
        "n_devices": n_devices,
        "batch_size": batch,
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
            "n_devices": n_devices,
            "gpus": list(gpus),
            "epochs": TARGET_EPOCHS,
            "batch_size": batch,
            "resume_ckpt": str(cand.resume_ckpt) if cand.resume_ckpt else None,
            "splits_dir": str(cand.splits_dir) if cand.splits_dir else None,
            "data_path": str(cand.data_path) if cand.data_path else None,
            "train_needed": not train_done,
            "zsv_needed": not zsv_json.is_file(),
        }
        return result

    peak = PEAK_RAM_LEGNET_GIB if cand.model == "legnet" else PEAK_RAM_CADUCEUS_GIB
    ok, reason = can_launch_parallel(
        peak_ram_gib=peak, gpus=gpus, job_class=CLASS_GPU_TRAIN
    )
    if not ok:
        print(f"[{cand.run_id}] queue busy — {reason}; re-check shortly", flush=True)
        result["status"] = "BLOCKED"
        result["reason"] = reason
        return result

    # Confirm assigned GPUs still free (race with other agents).
    still = available_gpus()
    if not all(g in still for g in gpus):
        result["status"] = "BLOCKED"
        result["reason"] = f"assigned GPUs {list(gpus)} no longer free; free={still}"
        return result

    log = ROOT / "logs" / f"continue_model_25_{cand.run_id}.log"
    job_name = f"continue_model_25_{cand.run_id}"
    append_queue_entry(
        job_name,
        job=f"python -m src.runs.continue_direct_model_25 --only {cand.run_id}",
        pid=os.getpid(),
        estimated_time="20–90m" if cand.model == "legnet" else "1–3h",
        status="RUNNING",
        resources=(
            f"{cand.model} epochs→{TARGET_EPOCHS} batch={batch} "
            f"n_devices={n_devices}"
        ),
        log=str(log),
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=peak,
        gpus=gpus,
    )
    try:
        if not train_done:
            out = _prepare_out_dir(cand)
            if cand.model == "caduceus":
                rc = _train_caduceus(
                    cand, out, log, gpus=gpus, n_devices=n_devices
                )
            else:
                rc = _train_legnet(cand, out, log, gpus=gpus, n_devices=n_devices)
            if rc != 0:
                result["status"] = "FAILED"
                result["train_rc"] = rc
                _mark_queue(job_name, "FAILED", log=str(log))
                return result
        if not (out / "logs" / "zero_shot_metrics.json").is_file():
            zrc = _run_zsv(cand, out, log, gpus=gpus)
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


def run_schedule(
    cands: list[Candidate], *, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Drain candidates: LegNet on each free GPU (1 device); Caduceus when ≥4 free.

    When multiple LegNet jobs and multiple free GPUs are available, launch them
    in parallel (one GPU each) — LegNet is fast and should not wait for a full
    4-GPU Caduceus slot.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending = list(cands)
    results: list[dict[str, Any]] = []
    if dry_run:
        for cand in pending:
            need = (
                LEGNET_MIN_GPUS if cand.model == "legnet" else CADUCEUS_MIN_GPUS
            )
            n_dev = LEGNET_N_DEVICES if cand.model == "legnet" else CADUCEUS_MIN_GPUS
            free = available_gpus()
            gpus = free[:need] if len(free) >= need else list(GPUS)[:need]
            results.append(
                process_one(cand, gpus=gpus, n_devices=n_dev, dry_run=True)
            )
        return results

    while pending:
        free = available_gpus()
        legs = [c for c in pending if c.model == "legnet"]
        cads = [c for c in pending if c.model == "caduceus"]

        if legs and free:
            # Pair each free GPU with one LegNet candidate (parallel).
            pairs = list(zip(legs, free))
            print(
                f"=== schedule {len(pairs)} LegNet job(s) on GPUs "
                f"{[g for _, g in pairs]} ===",
                flush=True,
            )

            def _one(pair: tuple[Candidate, int]) -> dict[str, Any]:
                cand, gpu = pair
                return process_one(
                    cand,
                    gpus=(gpu,),
                    n_devices=LEGNET_N_DEVICES,
                    dry_run=False,
                )

            with ThreadPoolExecutor(max_workers=max(1, len(pairs))) as pool:
                futs = {pool.submit(_one, p): p[0].run_id for p in pairs}
                blocked = 0
                for fut in as_completed(futs):
                    rid = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = {
                            "run_id": rid,
                            "model": "legnet",
                            "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    if res.get("status") == "BLOCKED":
                        blocked += 1
                        print(
                            f"[{rid}] blocked ({res.get('reason')}); will retry",
                            flush=True,
                        )
                        continue
                    results.append(res)
                    pending = [c for c in pending if c.run_id != rid]
                if blocked and blocked == len(pairs):
                    time.sleep(GPU_POLL_SEC)
            continue

        if cads and len(free) >= CADUCEUS_MIN_GPUS:
            next_cad = cads[0]
            gpus = free[:CADUCEUS_MIN_GPUS]
            print(
                f"=== schedule Caduceus {next_cad.run_id} on GPUs {gpus} ===",
                flush=True,
            )
            res = process_one(
                next_cad,
                gpus=gpus,
                n_devices=CADUCEUS_MIN_GPUS,
                dry_run=False,
            )
            if res.get("status") == "BLOCKED":
                print(
                    f"[{next_cad.run_id}] blocked ({res.get('reason')}); "
                    f"sleep {GPU_POLL_SEC}s",
                    flush=True,
                )
                time.sleep(GPU_POLL_SEC)
                continue
            results.append(res)
            pending = [c for c in pending if c.run_id != next_cad.run_id]
            continue

        need = LEGNET_MIN_GPUS if legs else (CADUCEUS_MIN_GPUS if cads else 0)
        who = legs[0].run_id if legs else (cads[0].run_id if cads else "?")
        print(
            f"[schedule] waiting for {need} free GPU(s) to run {who}; "
            f"free={free}; pending={[c.run_id for c in pending]}",
            flush=True,
        )
        time.sleep(GPU_POLL_SEC)
    return results


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
    ap.add_argument(
        "--skip-caduceus",
        action="store_true",
        default=True,
        help="Skip Caduceus continues (default True; LegNet-only). "
        "Use --no-skip-caduceus to include Caduceus again.",
    )
    ap.add_argument(
        "--no-skip-caduceus",
        action="store_false",
        dest="skip_caduceus",
        help="Include Caduceus continues (requires eval caps; 4 free GPUs).",
    )
    args = ap.parse_args(argv)
    only = args.only.split(",") if args.only else None
    cands = discover_candidates(args.runs_root, only=only)
    if args.skip_caduceus:
        before = len(cands)
        cands = [c for c in cands if c.model != "caduceus"]
        print(
            json.dumps(
                {
                    "skip_caduceus": True,
                    "dropped": before - len(cands),
                    "remaining": [c.run_id for c in cands],
                }
            ),
            flush=True,
        )
    summary: dict[str, Any] = {
        "target_epochs": TARGET_EPOCHS,
        "legnet_min_gpus": LEGNET_MIN_GPUS,
        "caduceus_min_gpus": CADUCEUS_MIN_GPUS,
        "gpus_pool": list(GPUS),
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
        "available_gpus_now": available_gpus(),
        "results": [],
    }
    print(
        json.dumps(
            {
                "discover": summary["candidates"],
                "available_gpus_now": summary["available_gpus_now"],
            },
            indent=2,
        ),
        flush=True,
    )
    summary["results"] = run_schedule(cands, dry_run=args.dry_run)
    out_json = ROOT / "logs" / "continue_direct_model_25_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    failed = [r for r in summary["results"] if r.get("status") == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
