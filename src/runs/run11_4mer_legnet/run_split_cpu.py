"""CPU-only stages for run11_4mer_legnet: mice fold + 4-mer SBS split + LegNet TSV.

No GPU train. Writes under ``runs/run11_4mer_legnet/``.

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run11_4mer_legnet.run_split_cpu
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run11_4mer_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
SEED = 42
RATIOS = (1, 1, 3)
KMER_SIZE = 4
KMER_ENGINE = "cpp"
CLUSTER_METHOD = "kmeans_elbow"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    kmer_size = KMER_SIZE
    kmer_engine = KMER_ENGINE
    cluster_method = CLUSTER_METHOD
    for tok in list(argv):
        if tok.startswith("kmer_size="):
            kmer_size = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("kmer_engine="):
            kmer_engine = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("cluster_method="):
            cluster_method = tok.split("=", 1)[1]
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.runs.run11_4mer_legnet.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.pipeline.legnet_input import build_legnet_tsv
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
        "seed": SEED,
        "ratios": list(RATIOS),
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run11 split_cpu meta={meta}", flush=True)

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
        kmer_size=kmer_size,
        engine=kmer_engine,
        threads=max(4, (os_cpu_count() or 8) // 2),
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
    tsv = build_legnet_tsv(
        split_root=split_root, out_tsv=OUT_ROOT / "legnet_input" / "all.tsv"
    )
    done = {
        **meta,
        "status": "COMPLETED",
        "split_csv": str(split_csv),
        "split_root": str(split_root),
        "legnet_tsv": str(tsv),
    }
    (OUT_ROOT / "split_cpu_done.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run11 split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"legnet_tsv={tsv}", flush=True)
    return 0


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


if __name__ == "__main__":
    raise SystemExit(main())
