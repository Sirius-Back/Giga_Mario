"""run8_2mer_caduceus: ready_caduceus → k-mer(k=2, C++) → Caduceus + adversarial.

Scripts: ``src/runs/run8_2mer_caduceus/``; artifacts: ``runs/run8_2mer_caduceus/``.

Replaces the accidental LegNet launcher (legacy ``pipeline_ready_legnet.py``).

- Direct: SBS k-mer split (``kmer_size=2``, ``engine=cpp``), Caduceus regression,
  max 50 epochs, min 10, early stop patience 10; best → ``final_model/``
- Adversarial: random re-split + fold-class PREDICT, same early-stop settings
- ZSV: mice genome ``GCF_000001635.27``
- GPUs: ``n_devices=2`` on ``CUDA_VISIBLE_DEVICES=0,1`` (user Locked)

Launch from project root::

  CUDA_VISIBLE_DEVICES=0,1 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run8_2mer_caduceus.pipeline_ready_caduceus
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run8_2mer_caduceus"
PANEL_ROOT = "ready_caduceus"
OUT_ROOT = "runs/run8_2mer_caduceus"
MICE_GENOME = "GCF_000001635.27"
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
N_DEVICES = 2
# Caduceus gene±100 windows (~200 bp); match run3 probed headroom on 2×V100.
BATCH_SIZE = 480
MAX_LENGTH = 208
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
    max_length = MAX_LENGTH
    cluster_method = CLUSTER_METHOD
    kmer_size = KMER_SIZE
    kmer_engine = KMER_ENGINE
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
        elif tok.startswith("max_length="):
            max_length = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("kmer_size="):
            kmer_size = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("kmer_engine=") or tok.startswith("engine="):
            kmer_engine = tok.split("=", 1)[1]
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.splits.sbs.backends.native import ensure_built

    lib = ensure_built()
    print(f"native_kmer_lib={lib}", flush=True)

    from src.runs.run8_2mer_caduceus.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.hydra_pipeline import main as hydra_main

    overrides = [
        f"run_id={RUN_ID}",
        "mode=run",
        "data=ready_caduceus",
        "split=kmer",
        "train=caduceus",
        "task_type=regression",
        "adversarial=true",
        "adversarial_task_type=classification",
        "zsv=true",
        f"epochs={epochs}",
        f"min_epochs={min_epochs}",
        f"early_stopping_patience={patience}",
        f"n_devices={n_devices}",
        f"batch_size={batch}",
        f"max_length={max_length}",
        f"num_workers={NUM_WORKERS}",
        f"seed={SEED}",
        "ratios=[1,1,3]",
        f"panel_root={PANEL_ROOT}",
        f"out_root={OUT_ROOT}",
        "conda_env=caduceus_env",
        "checkpoint_every_n_epochs=10",
        "plot_split=true",
        "plot_train=true",
        "plot_sbs=true",
        f"cluster_method={cluster_method}",
        f"kmer_size={kmer_size}",
        f"kmer_engine={kmer_engine}",
        *argv,
    ]
    print("run8_2mer_caduceus hydra overrides:", overrides, flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"mice_zsv_genome={MICE_GENOME}", flush=True)
    print(
        f"split=kmer k={kmer_size} engine={kmer_engine} cluster={cluster_method}",
        flush=True,
    )
    print(
        f"train=caduceus epochs={epochs} min_epochs={min_epochs} "
        f"early_stopping_patience={patience} n_devices={n_devices} "
        f"batch_size={batch} max_length={max_length}",
        flush=True,
    )
    return hydra_main(overrides)


if __name__ == "__main__":
    raise SystemExit(main())
