"""Evaluate run14 best/final Caduceus on ZSV (8192 random) — no adversarial.

Promotes ``best_model`` → ``final_model`` when final lacks weights, then runs
universal ZSV on GPU 0 with a seeded 8k subsample.

Note: multi-GPU ``DataParallel`` fails on Caduceus/Mamba Triton kernels, so
ZSV stays single-GPU.

Launch::

  CUDA_VISIBLE_DEVICES=3 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run14_7mer_caduceus.run_zsv_8k
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run14_7mer_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
DIRECT = OUT_ROOT / "direct"
MAX_SAMPLES = 8192
SEED = 42
DEVICE_IDS = (3,)
BATCH_SIZE = 128


def main(argv: list[str] | None = None) -> int:
    del argv
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    best = DIRECT / "best_model"
    final = DIRECT / "final_model"
    if not (best / "model.safetensors").is_file() and not (best / "config.json").is_file():
        raise FileNotFoundError(f"best_model missing under {best}")

    if not (final / "config.json").is_file():
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(best, final)
        print(f"promoted best_model → final_model (epoch meta below)", flush=True)
        meta = best / "best_meta.json"
        if meta.is_file():
            print(meta.read_text(encoding="utf-8"), flush=True)

    zsv_parsed = OUT_ROOT / "PARSED" / "zero-shot-validation"
    zsv_pred = OUT_ROOT / "PREDICT" / "zero-shot-validation"
    if not zsv_parsed.is_dir() or not zsv_pred.is_dir():
        raise FileNotFoundError(
            f"ZSV trees missing: {zsv_parsed} and/or {zsv_pred}"
        )

    # Map logical device 0 → physical DEVICE_IDS[0] via CUDA_VISIBLE_DEVICES.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in DEVICE_IDS)

    from src.pipeline.job_queue import (
        CLASS_GPU_TRAIN,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.zsv_eval import eval_zsv_from_train_outdir

    wait_until_launchable(
        peak_ram_gib=12.0,
        gpus=DEVICE_IDS,
        job_class=CLASS_GPU_TRAIN,
        label=f"{RUN_ID}_zsv_8k",
    )
    append_queue_entry(
        f"{RUN_ID}_zsv_8k",
        job=f"python -m src.runs.{RUN_ID}.run_zsv_8k",
        pid=os.getpid(),
        estimated_time="15-45m",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=12.0,
        gpus=DEVICE_IDS,
        resources=f"ZSV max_samples={MAX_SAMPLES} seed={SEED}; no adversarial",
        log=f"logs/{RUN_ID}_zsv_8k.log",
    )

    print(
        f"ZSV 8k: outdir={DIRECT} split_root={OUT_ROOT} "
        f"physical_gpus={DEVICE_IDS} max_samples={MAX_SAMPLES} seed={SEED} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    result = eval_zsv_from_train_outdir(
        model="caduceus",
        outdir=DIRECT,
        split_root=OUT_ROOT,
        device=0,  # first visible device after CUDA_VISIBLE_DEVICES remap
        device_ids=None,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )
    if result is None:
        raise RuntimeError("ZSV eval returned None (trees not resolved)")

    done = {
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "stage": "zsv_8k_only",
        "adversarial": False,
        "max_samples": MAX_SAMPLES,
        "seed": SEED,
        "device_ids": list(DEVICE_IDS),
        "direct": str(DIRECT),
        "zsv_metrics": result.get("metrics"),
        "zsv_json": str(DIRECT / "logs" / "zero_shot_metrics.json"),
    }
    (OUT_ROOT / "pipeline_done.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.get("metrics") or {}, sort_keys=True), flush=True)
    print(f"run14 ZSV-only COMPLETED (no adversarial) → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
