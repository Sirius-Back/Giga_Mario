"""Resume run8/12/14 from saved artifacts only (skip completed stages).

- run8: skip direct + reuse adversarial SPLIT/caduceus_input; train adv @ batch 256
- run12: keep split + caduceus_input; train direct+adv @ batch 256
- run14: full CPU split then train (alone-ish); prior runs OOM-killed under load

Does not kill other agents' jobs. Serializes GPU trains on devices 0,1.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAIT_GPUS = (0, 1)
MEM_FREE_MIB = 500
POLL_SEC = 60
# Safer than 480 after repeated CUDA OOM on adversarial / crowded GPUs
SAFE_BATCH = 256


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
        print(f"WARNING: nvidia-smi GPU {index}: {exc}", flush=True)
        return None


def wait_for_gpus(gpus: tuple[int, ...] = WAIT_GPUS, thresh: int = MEM_FREE_MIB) -> None:
    print(
        f"Waiting for GPUs {gpus} free (mem.used < {thresh} MiB); poll {POLL_SEC}s …",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in gpus}
        print(f"GPU memory.used MiB: {used}", flush=True)
        if all(v is not None and v < thresh for v in used.values()):
            print(f"GPUs {gpus} free", flush=True)
            return
        time.sleep(POLL_SEC)


def _run_mod(mod: str, *extra: str) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        "caduceus_env",
        "--no-capture-output",
        "python",
        "-m",
        mod,
        *extra,
    ]
    print("EXEC:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        raise RuntimeError(f"{mod} exited {rc}")


def _done(run_id: str) -> bool:
    return (ROOT / "runs" / run_id / "pipeline_done.json").is_file()


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # --- run14 CPU in background (no GPU); do not wait for it before GPU jobs ---
    run14_id = "run14_7mer_caduceus"
    if _done(run14_id):
        print(f"{run14_id} already DONE — skip", flush=True)
        proc14 = None
    else:
        log14 = ROOT / "logs" / "run14_7mer_caduceus_pipeline.log"
        with open(log14, "a", encoding="utf-8") as fh:
            fh.write("\n# RESUME relaunch\n")
        print("=== start run14 pipeline (CPU split → wait GPU → train) ===", flush=True)
        proc14 = subprocess.Popen(
            [
                "conda",
                "run",
                "-n",
                "caduceus_env",
                "--no-capture-output",
                "python",
                "-m",
                "src.runs.run14_7mer_caduceus.pipeline_ready_caduceus",
            ],
            cwd=str(ROOT),
            stdout=open(log14, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
        )
        print(f"run14 PID={proc14.pid} log={log14}", flush=True)

    # --- run8: only adversarial train ---
    run8 = "run8_2mer_caduceus"
    if _done(run8):
        print(f"{run8} already DONE — skip", flush=True)
    else:
        wait_for_gpus()
        print("=== run8 skip_direct + skip_adv_setup (batch 256) ===", flush=True)
        _run_mod(
            "src.runs.run8_2mer_caduceus.continue_from_split",
            "skip_direct=true",
            "skip_adv_setup=true",
            f"batch_size={SAFE_BATCH}",
        )

    # --- run12: split already done ---
    run12 = "run12_4mer_caduceus"
    if _done(run12):
        print(f"{run12} already DONE — skip", flush=True)
    else:
        wait_for_gpus()
        print("=== run12 continue (keep caduceus_input; batch 256) ===", flush=True)
        _run_mod(
            "src.runs.run12_4mer_caduceus.continue_from_split",
            f"batch_size={SAFE_BATCH}",
        )

    if proc14 is not None:
        print(f"run14 still running? poll={proc14.poll()}", flush=True)
    print("resume_caduceus_8_12_14: GPU path finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
