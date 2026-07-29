"""run2 pipeline: ready_legnet → random split (mice ZSV) → LegNet regression.

Scripts live under ``src/runs/run2/``; artifacts under ``runs/run2/``.

Launch (from project root)::

  CUDA_VISIBLE_DEVICES=0 conda run -n legnet --no-capture-output \\
    python -m src.runs.run2.pipeline_ready_legnet

Note: 2-GPU Lightning DDP hangs after sanity check on this host (asymmetric
VRAM); production run uses 1×V100. Epochs sized for ~8h wall from epoch-1 ETR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Locked run2 settings (also mirrored in Hydra overrides below).
RUN_ID = "run2"
PANEL_ROOT = "ready_legnet"
OUT_ROOT = "runs/run2"
MICE_GENOME = "GCF_000001635.27"
# ~47s/epoch @ batch 8192 on 1×V100 → ~600 epochs ≈ 8h (refined after epoch 1).
EPOCHS = 600
N_DEVICES = 1
# Single-GPU: 12288 amp fwd+bwd fits V100-32GB (~32.5GB); 8192 leaves headroom.
BATCH_SIZE = 8192
NUM_WORKERS = 8
SEED = 42
RATIOS = (1, 1, 3)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Allow BATCH_SIZE=… / epochs=… override without Hydra for probes.
    batch = BATCH_SIZE
    epochs = EPOCHS
    for tok in list(argv):
        if tok.startswith("batch_size="):
            batch = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Ensure project root on path when executed as a file.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_legnet",
        "split=random",
        "train=legnet",
        "task_type=regression",
        "adversarial=false",
        "zsv=true",
        f"epochs={epochs}",
        f"n_devices={N_DEVICES}",
        f"batch_size={batch}",
        f"num_workers={NUM_WORKERS}",
        f"seed={SEED}",
        "ratios=[1,1,3]",
        f"panel_root={PANEL_ROOT}",
        f"out_root={OUT_ROOT}",
        "conda_env=legnet",
        *argv,
    ]
    print("run2 hydra overrides:", overrides)
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
