"""Resume run13 from ``feature_table.npz``: assign → SPLIT → LegNet TSV.

Skips multi-hour 7-mer recount. Serializes via job_queue (cpu_ram_heavy).

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run13_7mer_legnet.continue_from_features
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run13_7mer_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
SEED = 42
RATIOS = (1, 1, 3)
KMER_SIZE = 7
CLUSTER_METHOD = "kmeans"
# Hundreds of SBS folds (not 8): ~sqrt(n_assignable) capped — user expects hundreds.
N_CLUSTERS = 512


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cluster_method = CLUSTER_METHOD
    n_clusters: int | str = N_CLUSTERS
    for tok in list(argv):
        if tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("n_clusters="):
            raw = tok.split("=", 1)[1]
            n_clusters = "auto" if raw.lower() == "auto" else int(raw)
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    npz = OUT_ROOT / "feature_table.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"missing {npz}")

    from src.pipeline.job_queue import CLASS_CPU_RAM_HEAVY, wait_until_launchable
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.mem_guard import wait_for_ram_headroom
    from src.pipeline.split import run_split
    from src.splits.kmer import run_kmer_split_assign

    wait_until_launchable(
        peak_ram_gib=48.0,
        job_class=CLASS_CPU_RAM_HEAVY,
        label="run13_assign_from_npz",
    )
    wait_for_ram_headroom(0.90, label="run13_pre_assign")

    marked = PANEL_ROOT / "MARKED"
    summary = run_kmer_split_assign(
        outdir=OUT_ROOT,
        fna=marked,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        seed=SEED,
        k=KMER_SIZE,
        cluster_method=cluster_method,
        n_clusters=n_clusters,  # type: ignore[arg-type]
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
    tsv = build_legnet_tsv(
        split_root=split_root, out_tsv=OUT_ROOT / "legnet_input" / "all.tsv"
    )
    done = {
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "status": "COMPLETED",
        "resumed_from": str(npz),
        "cluster_method": cluster_method,
        "split_csv": str(split_csv),
        "split_root": str(split_root),
        "legnet_tsv": str(tsv),
        "assign_meta": summary.get("assign_meta"),
    }
    (OUT_ROOT / "split_cpu_done.json").write_text(
        json.dumps(done, indent=2, default=str) + "\n", encoding="utf-8"
    )
    am = done.get("assign_meta") or {}
    cl = (am.get("cluster") or {}) if isinstance(am, dict) else {}
    print(
        f"run13 continue_from_features COMPLETED split_csv={done.get('split_csv')} "
        f"method={cl.get('method_used')} n_clusters={cl.get('n_clusters')} "
        f"n_features={am.get('n_features')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
