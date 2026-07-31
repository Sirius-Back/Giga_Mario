"""run14 Caduceus full ZSV only (no adversarial) on any 2 free GPUs.

Expects ``direct/final_model`` (or ``best_model``) and panel ZSV trees under
``runs/run14_7mer_caduceus/{PARSED,PREDICT}/zero-shot-validation``.

Polls until two physical GPUs have ``memory.used < MEM_FREE_MIB``, then runs
sharded multi-GPU ZSV (not DataParallel). Overwrites ``logs/zero_shot_metrics.json``.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run14_7mer_caduceus.run_zsv
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run14_7mer_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
TRAIN_OUT = OUT_ROOT / "direct"
N_GPUS = 2
MEM_FREE_MIB = 1500
POLL_SEC = 60
PEAK_RAM_GIB = 12.0


def _gpu_used_mib() -> dict[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx_s, mem_s = [x.strip() for x in line.split(",")]
        used[int(idx_s)] = int(mem_s)
    return used


def wait_for_n_free_gpus(n: int = N_GPUS, thresh: int = MEM_FREE_MIB) -> list[int]:
    print(
        f"Waiting for {n} GPUs with memory.used < {thresh} MiB; poll every {POLL_SEC}s …",
        flush=True,
    )
    while True:
        used = _gpu_used_mib()
        free = sorted(i for i, m in used.items() if m < thresh)
        print(f"GPU memory.used MiB: {used}; free<{thresh}: {free}", flush=True)
        if len(free) >= n:
            chosen = free[:n]
            print(f"Using physical GPUs {chosen}", flush=True)
            return chosen
        time.sleep(POLL_SEC)


def main(argv: list[str] | None = None) -> int:
    del argv
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry
    from src.pipeline.zsv_eval import eval_zsv_from_train_outdir

    final = TRAIN_OUT / "final_model"
    best = TRAIN_OUT / "best_model"
    if not (final / "config.json").is_file() and not (best / "config.json").is_file():
        raise FileNotFoundError(
            f"Need {final} or {best} with config.json before ZSV"
        )
    for req in (
        OUT_ROOT / "PARSED" / "zero-shot-validation",
        OUT_ROOT / "PREDICT" / "zero-shot-validation",
    ):
        if not req.is_dir():
            raise FileNotFoundError(f"ZSV tree missing: {req}")

    physical = wait_for_n_free_gpus(N_GPUS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical)
    # Inside the visible set, shards use logical 0..n-1.
    logical = tuple(range(N_GPUS))

    append_queue_entry(
        f"{RUN_ID}_zsv_full",
        job=f"python -m src.runs.{RUN_ID}.run_zsv",
        pid=os.getpid(),
        estimated_time="30-90m",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=PEAK_RAM_GIB,
        gpus=physical,
        resources="Caduceus full ZSV sharded; no adversarial",
        log=f"logs/{RUN_ID}_zsv.log",
    )

    print(
        f"run14 full ZSV: outdir={TRAIN_OUT} zsv_root={OUT_ROOT} "
        f"physical_gpus={physical} logical_device_ids={logical} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    result = eval_zsv_from_train_outdir(
        model="caduceus",
        outdir=TRAIN_OUT,
        split_root=OUT_ROOT,
        device=0,
        device_ids=logical,
    )
    if result is None:
        raise RuntimeError("ZSV eval produced no metrics")
    metrics = result.get("metrics") or {}
    print(json.dumps(metrics, sort_keys=True), flush=True)
    (OUT_ROOT / "direct_zsv_done.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "run_id": RUN_ID,
                "adversarial": False,
                "zsv": True,
                "full_zsv": True,
                "physical_gpus": physical,
                "device_ids": list(logical),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"run14 full ZSV COMPLETED (no adversarial) → {TRAIN_OUT / 'logs'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
