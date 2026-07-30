"""run16_hashfrag_caduceus: hashFrag split (reuse run5) → wait GPUs 0,1 → Caduceus.

- Split: hashFrag orthogonal homology (threshold=60, reuse run5 BLAST)
- Panel: ready_caduceus; mice ZSV
- Train: Caduceus regression + adversarial classification; epochs 10–50;
  early_stopping_patience=10; checkpoint every 10 → best/final_model
- GPUs: wait until physical 0 and 1 are free, then ``CUDA_VISIBLE_DEVICES=0,1``

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run16_hashfrag_caduceus.pipeline_ready_caduceus
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run16_hashfrag_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
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
        print(f"WARNING: nvidia-smi failed for GPU {index}: {exc}", flush=True)
        return None


def wait_for_gpus(gpus: tuple[int, ...] = WAIT_GPUS, thresh: int = MEM_FREE_MIB) -> None:
    print(
        f"Waiting for GPUs {gpus} to be free (memory.used < {thresh} MiB); "
        f"poll every {POLL_SEC}s … (does not stop other jobs)",
        flush=True,
    )
    while True:
        used = {g: _gpu_used_mib(g) for g in gpus}
        print(f"GPU memory.used MiB: {used}", flush=True)
        if all(v is not None and v < thresh for v in used.values()):
            print(f"GPUs {gpus} free — starting Caduceus train", flush=True)
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

    if not skip_split and not (OUT_ROOT / "split_cpu_done.json").is_file():
        from src.runs.run16_hashfrag_caduceus.run_split_cpu import main as split_main

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
    else:
        print("skip_wait=true — not polling GPUs", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    from src.runs.run16_hashfrag_caduceus.continue_from_split import main as train_main

    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
