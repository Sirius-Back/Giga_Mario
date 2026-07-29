"""run1 pipeline: ready_caduceus → random split (mice ZSV) → Caduceus regression.

Scripts: ``src/runs/run1/``; artifacts: ``runs/run1/``.

Uses free GPUs via ``CUDA_VISIBLE_DEVICES`` (default ``2,3`` when 0–1 are busy).

Launch from project root::

  CUDA_VISIBLE_DEVICES=2,3 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run1.pipeline_ready_caduceus
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run1"
PANEL_ROOT = "ready_caduceus"
OUT_ROOT = "runs/run1"
MICE_GENOME = "GCF_000001635.27"
EPOCHS = 10
N_DEVICES = 2
# Overridden after probe; conservative default for Caduceus-PS @ max_length=256.
BATCH_SIZE = 192  # probed: 256 amp fwd+bwd fits @ max_length=256; 192 for DDP headroom
MAX_LENGTH = 256  # gene±100 → 200 bp windows
NUM_WORKERS = 8
SEED = 42


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    for tok in list(argv):
        if tok.startswith("batch_size="):
            batch = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)

    # Do not steal GPUs from run2 LegNet on 0,1 unless caller overrides.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Mice ZSV fold required by hydra_pipeline panel contract.
    from src.runs.run1.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_caduceus",
        "split=random",
        "train=caduceus",
        "task_type=regression",
        "adversarial=false",
        "zsv=true",
        f"epochs={epochs}",
        f"n_devices={N_DEVICES}",
        f"batch_size={batch}",
        f"max_length={MAX_LENGTH}",
        f"num_workers={NUM_WORKERS}",
        f"seed={SEED}",
        "ratios=[1,1,3]",
        f"panel_root={PANEL_ROOT}",
        f"out_root={OUT_ROOT}",
        "conda_env=caduceus_env",
        *argv,
    ]
    print("run1 hydra overrides:", overrides, flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
