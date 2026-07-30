"""run13_7mer_legnet orchestrator: CPU 7-mer split → wait GPUs 2,3 → train.

- Split: k-mer k=7, engine=cpp, cluster=kmeans_elbow, mice ZSV
- Train: LegNet direct + adversarial + ZSV + viz (starts only when GPUs 2 and 3
  are free). Default train uses 1 visible GPU (ddp_spawn hang workaround);
  override with ``n_devices=2``.

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run13_7mer_legnet.pipeline_ready_legnet
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run13_7mer_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
WAIT_GPUS = (2, 3)
MEM_FREE_MIB = 500
POLL_SEC = 60


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


def wait_for_gpus(gpus: tuple[int, ...] = WAIT_GPUS, thresh: int = MEM_FREE_MIB) -> None:
    print(
        f"Waiting for GPUs {gpus} to be free (memory.used < {thresh} MiB); "
        f"poll every {POLL_SEC}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in gpus}
        print(f"GPU memory.used MiB: {used}", flush=True)
        if all(v is not None and v < thresh for v in used.values()):
            print(f"GPUs {gpus} free — starting train", flush=True)
            return
        time.sleep(POLL_SEC)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip_split = False
    skip_wait = False
    for tok in list(argv):
        if tok in {"skip_split=true", "--skip-split"}:
            skip_split = True
            argv.remove(tok)
        elif tok in {"skip_wait=true", "--skip-wait"}:
            skip_wait = True
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        CLASS_GPU_TRAIN,
        wait_until_launchable,
    )

    if not skip_split and not (OUT_ROOT / "split_cpu_done.json").is_file():
        wait_until_launchable(
            peak_ram_gib=40.0,
            job_class=CLASS_CPU_RAM_HEAVY,
            label="run13_7mer_split",
        )
        from src.runs.run13_7mer_legnet.run_split_cpu import main as split_main

        rc = split_main(argv)
        if rc != 0:
            return rc
    else:
        print(
            f"skip_split or split_cpu_done present → {OUT_ROOT / 'split_cpu_done.json'}",
            flush=True,
        )

    if not skip_wait:
        wait_for_gpus()
        wait_until_launchable(
            peak_ram_gib=12.0,
            gpus=WAIT_GPUS,
            job_class=CLASS_GPU_TRAIN,
            label="run13_7mer_train",
        )
    else:
        print("skip_wait=true — not polling GPUs", flush=True)

    # Prefer GPU 2 for 1-device train when both free (leave 3 spare / match wait).
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    from src.runs.run13_7mer_legnet.continue_from_split import main as train_main

    # Force single-GPU unless caller overrides n_devices=
    if not any(t.startswith("n_devices=") for t in argv):
        argv = ["n_devices=1", "batch_size=8192", *argv]
    # Bind physical GPU 2 as cuda:0 for the single-device case
    if any(t == "n_devices=1" for t in argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
