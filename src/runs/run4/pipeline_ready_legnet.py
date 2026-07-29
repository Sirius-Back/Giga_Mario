"""run4 pipeline: ready_legnet → GC split (mice ZSV) → LegNet + adversarial.

Scripts: ``src/runs/run4/``; artifacts: ``runs/run4/``.

- Direct: GC SBS split, LegNet regression, max 50 epochs, min 10, early stop
- Adversarial: random re-split + fold-class PREDICT, same early-stop settings
- ZSV: mice genome ``GCF_000001635.27``
- GPUs: default ``n_devices=1`` on ``CUDA_VISIBLE_DEVICES=2`` — LegNet
  ``ddp_spawn`` hangs after sanity check on this host (same as run2).
  Resume train-only via ``python -m src.runs.run4.continue_from_split``.

Launch from project root::

  CUDA_VISIBLE_DEVICES=2 conda run -n legnet --no-capture-output \\
    python -m src.runs.run4.pipeline_ready_legnet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run4"
PANEL_ROOT = "ready_legnet"
OUT_ROOT = "runs/run4"
MICE_GENOME = "GCF_000001635.27"
# epoch max; early stop may end sooner after min_epochs floor
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
# Locked workaround: 2-GPU LegNet ddp_spawn hangs after sanity (run2/run4).
N_DEVICES = 1
# Per-GPU batch; matches run2 V100 headroom for 230 bp LegNet
BATCH_SIZE = 8192
NUM_WORKERS = 8
SEED = 42


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    min_epochs = MIN_EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    n_devices = N_DEVICES
    cluster_method = "kmeans_elbow"
    for tok in list(argv):
        if tok.startswith("batch_size="):
            batch = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("min_epochs="):
            min_epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("early_stopping_patience="):
            patience = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("n_devices="):
            n_devices = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)

    # Single free GPU; override externally if needed.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.runs.run4.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_legnet",
        "split=gc",
        "train=legnet",
        "task_type=regression",
        "adversarial=true",
        "adversarial_task_type=classification",
        "zsv=true",
        f"epochs={epochs}",
        f"min_epochs={min_epochs}",
        f"early_stopping_patience={patience}",
        f"n_devices={n_devices}",
        f"batch_size={batch}",
        f"num_workers={NUM_WORKERS}",
        f"seed={SEED}",
        "ratios=[1,1,3]",
        f"panel_root={PANEL_ROOT}",
        f"out_root={OUT_ROOT}",
        "conda_env=legnet",
        "checkpoint_every_n_epochs=10",
        "plot_split=true",
        f"cluster_method={cluster_method}",
        *argv,
    ]
    print("run4 hydra overrides:", overrides, flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"mice_zsv_genome={MICE_GENOME}", flush=True)
    print(
        f"epochs={epochs} min_epochs={min_epochs} early_stopping_patience={patience} "
        f"n_devices={n_devices}",
        flush=True,
    )
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
