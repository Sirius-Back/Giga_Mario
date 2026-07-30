"""Resume run14 from ``feature_table.npz``: assign → Caduceus SPLIT.

Skips multi-hour 7-mer recount. Prefer running after run13 assign frees RAM.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run14_7mer_caduceus.continue_from_features
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run14_7mer_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
SEED = 42
RATIOS = (1, 1, 3)
KMER_SIZE = 7
CLUSTER_METHOD = "kmeans"
N_CLUSTERS = 8
# Wait for sibling 7-mer LegNet split before loading another ~30–50 GiB matrix.
WAIT_FOR = ROOT / "runs" / "run13_7mer_legnet" / "split_cpu_done.json"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip_wait_run13 = False
    for tok in list(argv):
        if tok in {"skip_wait_run13=true", "--skip-wait-run13"}:
            skip_wait_run13 = True
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    npz = OUT_ROOT / "feature_table.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"missing {npz}")

    if not skip_wait_run13:
        print(f"Waiting for {WAIT_FOR} before run14 assign …", flush=True)
        while not WAIT_FOR.is_file():
            time.sleep(60)
        print("run13 split_cpu_done present — continuing", flush=True)

    from src.pipeline.job_queue import CLASS_CPU_RAM_HEAVY, wait_until_launchable
    from src.pipeline.mem_guard import wait_for_ram_headroom
    from src.pipeline.split import run_split
    from src.splits.kmer import run_kmer_split_assign

    wait_until_launchable(
        peak_ram_gib=48.0,
        job_class=CLASS_CPU_RAM_HEAVY,
        label="run14_assign_from_npz",
    )
    wait_for_ram_headroom(0.90, label="run14_pre_assign")

    marked = PANEL_ROOT / "MARKED"
    summary = run_kmer_split_assign(
        outdir=OUT_ROOT,
        fna=marked,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        seed=SEED,
        k=KMER_SIZE,
        cluster_method=CLUSTER_METHOD,
        n_clusters=N_CLUSTERS,
        ratios=RATIOS,
        plot=False,
        features_npz=npz,
        engine="native",
    )
    split_csv = Path(summary["split_csv"])
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
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "status": "COMPLETED",
        "resumed_from": str(npz),
        "cluster_method": CLUSTER_METHOD,
        "n_clusters": N_CLUSTERS,
        "split_csv": str(split_csv),
        "split_root": str(split_root),
        "assign_meta": summary.get("assign_meta"),
    }
    (OUT_ROOT / "split_cpu_done.json").write_text(
        json.dumps(done, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"run14 continue_from_features COMPLETED → {done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
