"""run7_2mer_legnet: ready_legnet → 2-mer SBS (cpp) → LegNet + adversarial.

Scripts: ``src/runs/run7_2mer_legnet/``; artifacts: ``runs/run7_2mer_legnet/``.

- Direct: k-mer SBS split (k=2, engine=cpp/native), LegNet regression,
  max 50 epochs, min 10, early stop
- Adversarial: random re-split + fold-class PREDICT, same early-stop settings
- ZSV: mice genome ``GCF_000001635.27``
- GPUs: ``n_devices=2`` on ``CUDA_VISIBLE_DEVICES=2,3``

Launch from project root::

  CUDA_VISIBLE_DEVICES=2,3 conda run -n legnet --no-capture-output \\
    python -m src.runs.run7_2mer_legnet.pipeline_ready_legnet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run7_2mer_legnet"
PANEL_ROOT = "ready_legnet"
OUT_ROOT = "runs/run7_2mer_legnet"
MICE_GENOME = "GCF_000001635.27"
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
N_DEVICES = 2
# Per-GPU batch; 2×V100 — global batch ~8192 like run4 single-GPU
BATCH_SIZE = 4096
NUM_WORKERS = 8
SEED = 42
KMER_SIZE = 2
KMER_ENGINE = "cpp"
CLUSTER_METHOD = "kmeans_elbow"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    min_epochs = MIN_EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    n_devices = N_DEVICES
    kmer_size = KMER_SIZE
    kmer_engine = KMER_ENGINE
    cluster_method = CLUSTER_METHOD
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
        elif tok.startswith("kmer_size="):
            kmer_size = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("kmer_engine="):
            kmer_engine = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.runs.run7_2mer_legnet.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_legnet",
        "split=kmer",
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
        "plot_train=true",
        "plot_sbs=true",
        f"kmer_size={kmer_size}",
        f"kmer_engine={kmer_engine}",
        f"cluster_method={cluster_method}",
        *argv,
    ]
    print("run7_2mer_legnet hydra overrides:", overrides, flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"mice_zsv_genome={MICE_GENOME}", flush=True)
    print(
        f"epochs={epochs} min_epochs={min_epochs} early_stopping_patience={patience} "
        f"n_devices={n_devices} batch_size={batch} kmer_size={kmer_size} "
        f"kmer_engine={kmer_engine} cluster_method={cluster_method}",
        flush=True,
    )
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
