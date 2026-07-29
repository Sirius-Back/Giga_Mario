"""run5 pipeline: ready_legnet → hashFrag split (mice ZSV) → LegNet + adversarial.

Scripts: ``src/runs/run5/``; artifacts: ``runs/run5/``.

- Direct: hashFrag orthogonal homology split, LegNet regression, max 50 / min 10,
  early stop
- Adversarial: random re-split + fold-class PREDICT, same early-stop settings
- ZSV: mice genome ``GCF_000001635.27``
- GPUs: ``n_devices=4`` (override with ``n_devices=…``)

Launch from project root::

  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n legnet --no-capture-output \\
    python -m src.runs.run5.pipeline_ready_legnet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run5"
PANEL_ROOT = "ready_legnet"
OUT_ROOT = "runs/run5"
MICE_GENOME = "GCF_000001635.27"
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
N_DEVICES = 4
# Per-GPU batch; 4×V100 — keep global batch similar to run4 (8192×1)
BATCH_SIZE = 2048
NUM_WORKERS = 8
SEED = 42
# hashFrag alignment-score threshold (obligatory; smoke used 60)
THRESHOLD = 60
# BLAST threads for create_orthogonal_splits
THREADS = 16


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    min_epochs = MIN_EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    n_devices = N_DEVICES
    threshold = THRESHOLD
    threads = THREADS
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
        elif tok.startswith("threshold="):
            threshold = int(float(tok.split("=", 1)[1]))
            argv.remove(tok)
        elif tok.startswith("threads="):
            threads = int(tok.split("=", 1)[1])
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    # Prefer conda env bin + system BLAST+
    os.environ["PATH"] = (
        f"{ROOT / 'miniconda3' / 'envs' / 'legnet' / 'bin'}:"
        f"{ROOT / 'miniconda3' / 'bin'}:"
        f"{ROOT / 'bin'}:"
        + os.environ.get("PATH", "")
    )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.runs.run5.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_legnet",
        "split=hashfrag",
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
        "plot_split=false",
        "plot_train=true",
        "plot_sbs=false",
        f"threshold={threshold}",
        f"threads={threads}",
        "force=true",
        *argv,
    ]
    print("run5 hydra overrides:", overrides, flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"mice_zsv_genome={MICE_GENOME}", flush=True)
    print(
        f"epochs={epochs} min_epochs={min_epochs} early_stopping_patience={patience} "
        f"n_devices={n_devices} batch_size={batch} threshold={threshold} "
        f"threads={threads}",
        flush=True,
    )
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
