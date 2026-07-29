"""run3 pipeline: ready_caduceus → GC split (mice ZSV) → Caduceus regression.

Scripts: ``src/runs/run3/``; artifacts: ``runs/run3/``.

Launch from project root::

  CUDA_VISIBLE_DEVICES=0,1 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run3.pipeline_ready_caduceus
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run3"
PANEL_ROOT = "ready_caduceus"
OUT_ROOT = "runs/run3"
MICE_GENOME = "GCF_000001635.27"
EPOCHS = 100
N_DEVICES = 2
# Overridden after probe; short gene±100 windows (≤200 bp).
BATCH_SIZE = 480  # probed: 512 amp fwd+bwd fits @ max_length=208; 480 for DDP headroom
MAX_LENGTH = 208
NUM_WORKERS = 8
SEED = 42
# Stop when val_loss plateaus for this many epochs (enabled early stop).
EARLY_STOPPING_PATIENCE = 10


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    cluster_method = "kmeans_elbow"
    for tok in list(argv):
        if tok.startswith("batch_size="):
            batch = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("early_stopping_patience="):
            patience = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.runs.run3.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_caduceus",
        "split=gc",
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
        f"early_stopping_patience={patience}",
        "checkpoint_every_n_epochs=10",
        "plot_split=true",
        f"cluster_method={cluster_method}",
        *argv,
    ]
    print("run3 hydra overrides:", overrides, flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"mice_zsv_genome={MICE_GENOME}", flush=True)
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
