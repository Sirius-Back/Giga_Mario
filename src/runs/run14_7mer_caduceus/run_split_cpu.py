"""CPU-only: mice fold + 7-mer SBS (cpp) + materialize SPLIT for Caduceus.

No GPU train. Fast clustering: single MiniBatchKMeans (``cluster_method=kmeans``,
fixed ``n_clusters``) — not elbow / silhouette search.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run14_7mer_caduceus.run_split_cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run14_7mer_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
SEED = 42
RATIOS = (1, 1, 3)
KMER_SIZE = 7
KMER_ENGINE = "cpp"
# Fastest practical path on ~400k IDs: one MiniBatchKMeans fit.
CLUSTER_METHOD = "kmeans"
N_CLUSTERS = 8


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    kmer_size = KMER_SIZE
    kmer_engine = KMER_ENGINE
    cluster_method = CLUSTER_METHOD
    n_clusters: int | Literal["auto"] = N_CLUSTERS
    for tok in list(argv):
        if tok.startswith("kmer_size="):
            kmer_size = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("kmer_engine=") or tok.startswith("engine="):
            kmer_engine = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("n_clusters="):
            raw = tok.split("=", 1)[1]
            n_clusters = "auto" if raw.lower() == "auto" else int(raw)
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.splits.sbs.backends.native import ensure_built

    lib = ensure_built()
    print(f"native_kmer_lib={lib}", flush=True)

    from src.runs.run14_7mer_caduceus.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict

    marked = PANEL_ROOT / "MARKED"
    if not marked.is_dir():
        raise FileNotFoundError(f"MARKED missing: {marked}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "split": "kmer",
        "kmer_size": kmer_size,
        "kmer_engine": kmer_engine,
        "cluster_method": cluster_method,
        "n_clusters": n_clusters,
        "seed": SEED,
        "ratios": list(RATIOS),
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "status": "RUNNING",
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run14 split_cpu meta={meta}", flush=True)

    split_csv = run_split_predict(
        outdir=OUT_ROOT,
        type="kmer",
        seed=SEED,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
        marked_fasta=marked,
        plot=True,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        kmer_size=kmer_size,
        engine=kmer_engine,
        threads=max(4, (os.cpu_count() or 8) // 2),
    )
    split_root = run_split(
        split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        outdir=OUT_ROOT,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL_ROOT / "ID.csv",
    )
    done = {
        **meta,
        "status": "COMPLETED",
        "split_csv": str(split_csv),
        "split_root": str(split_root),
    }
    (OUT_ROOT / "split_cpu_done.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run14 split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"split_root={split_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
