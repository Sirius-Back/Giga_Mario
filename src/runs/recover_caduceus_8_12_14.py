"""Serialize recovery for run8 (adv only), run12 (direct+adv), run14 (full).

Causes (2026-07-30):
- run8/run12: CUDA OOM from overlapping Caduceus on GPUs 0,1
- run14: OS killed during 7-mer CPU split

Does not kill other agents' jobs. Waits for physical GPUs 0,1 free before each train.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAIT_GPUS = (0, 1)
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


def _clean_failed_direct(run_dir: Path) -> None:
    direct = run_dir / "direct"
    if not direct.is_dir():
        return
    # Keep caduceus_input if present; drop empty/partial train artifacts
    for name in (
        "final_model",
        "best_model",
        "checkpoints",
        "logs",
        "tensorboard",
        "figures",
        "train_time.json",
        "run_config.json",
    ):
        p = direct / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    print(f"cleaned failed train artifacts under {direct}", flush=True)


def _run_mod(mod: str, *extra: str, env: dict[str, str] | None = None) -> None:
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
    merged = os.environ.copy()
    if env:
        merged.update(env)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=merged)
    if rc != 0:
        raise RuntimeError(f"{mod} exited {rc}")


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # --- run14 CPU first (no GPU) ---
    run14 = ROOT / "runs" / "run14_7mer_caduceus"
    run14.mkdir(parents=True, exist_ok=True)
    (run14 / ".agent_dead_alerted").unlink(missing_ok=True)
    print("=== relaunch run14 CPU split + later train (background) ===", flush=True)
    log14 = ROOT / "logs" / "run14_7mer_caduceus_pipeline.log"
    with open(log14, "a", encoding="utf-8") as fh:
        fh.write("\n# RECOVER relaunch\n")
    # Start run14 in background via nohup-equivalent
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
    )
    print(f"run14 PID={proc14.pid} log={log14}", flush=True)

    # --- run8 adversarial only ---
    run8 = ROOT / "runs" / "run8_2mer_caduceus"
    (run8 / ".agent_dead_alerted").unlink(missing_ok=True)
    wait_for_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print("=== run8 continue skip_direct (adversarial) ===", flush=True)
    _run_mod(
        "src.runs.run8_2mer_caduceus.continue_from_split",
        "skip_direct=true",
        env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )

    # --- run12 full continue from split ---
    run12 = ROOT / "runs" / "run12_4mer_caduceus"
    (run12 / ".agent_dead_alerted").unlink(missing_ok=True)
    _clean_failed_direct(run12)
    wait_for_gpus()
    print("=== run12 continue_from_split ===", flush=True)
    _run_mod(
        "src.runs.run12_4mer_caduceus.continue_from_split",
        env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )

    print("recover_caduceus_8_12_14: run8+run12 train path finished; run14 may still be running", flush=True)
    print(f"run14 poll returncode={proc14.poll()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
