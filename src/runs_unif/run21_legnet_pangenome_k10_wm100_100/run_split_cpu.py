"""CPU stages for run21: NEW adapt window -100..100 + k=10 C++ pangenome graph.

- Adapt ``MARKED_pangenome`` from ``raw/{gtf,fna}`` with window ``{pos1:-100, pos2:100}``
- Build **new** contingency graph with k=10 (cpp), modularity-refine large CCs
- Plot contingency graph; ratios train:test:val ≈ 3:1:1 (near enough)
- Materialize SPLIT + LegNet TSV under ``runs_unif/legnet/run21_legnet_pangenome_k10_wm100_100``

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs_unif.run21_legnet_pangenome_k10_wm100_100.run_split_cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_I = 21
MODEL = "legnet"
SPLIT = "pangenome"
SPLIT_PARAMS = "k10_wm100_100"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME
RAW_GTF = ROOT / "raw" / "gtf"
RAW_FNA = ROOT / "raw" / "fna"

SEED = 42
RATIOS = (3.0, 1.0, 1.0)  # train:test:val ≈ 3:1:1 (near enough OK)
ENVIRONMENT = "gene"
WINDOW = {"pos1": -100, "pos2": 100}
KMER_SIZE = 10
ENGINE = "cpp"
PEAK_RAM_GIB = 40.0  # adapt + contingency + modularity
MAX_FOLD_SIZE = 1000  # modularity refine oversized CCs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    max_ids: int | None = None
    force = False
    kmer_size = KMER_SIZE
    max_fold_size = MAX_FOLD_SIZE
    for tok in list(argv):
        if tok.startswith("max_ids="):
            max_ids = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("kmer_size="):
            kmer_size = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("max_fold_size="):
            max_fold_size = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok in {"force=true", "--force"}:
            force = True
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.environ["PATH"] = (
        f"{ROOT / 'bin'}:"
        f"{ROOT / 'miniconda3' / 'envs' / 'legnet' / 'bin'}:"
        + os.environ.get("PATH", "")
    )

    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.rerun_aligned import assert_fresh_out_root
    from src.runs_unif.run21_legnet_pangenome_k10_wm100_100.ensure_mice_fold import (
        main as ensure_fold,
    )

    ensure_fold()

    for req in (
        PANEL_ROOT / "ID.csv",
        PANEL_ROOT / "fold.csv",
        PANEL_ROOT / "PARSED",
        PANEL_ROOT / "PREDICT",
        RAW_GTF,
        RAW_FNA,
    ):
        if not req.exists():
            raise FileNotFoundError(f"missing required input: {req}")

    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    assert_fresh_out_root(OUT_ROOT)

    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB,
        gpus=(),
        job_class=CLASS_CPU_RAM_HEAVY,
        label=f"{RUN_NAME}_split_cpu",
    )

    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": RUN_NAME,
        "aligned_run": RUN_I,
        "stage": "split_cpu",
        "split": SPLIT,
        "engine": ENGINE,
        "environment": ENVIRONMENT,
        "window": WINDOW,
        "kmer_size": kmer_size,
        "modularity_refine": True,
        "max_fold_size": max_fold_size,
        "seed": SEED,
        "ratios": list(RATIOS),
        "force": force,
        "max_ids": max_ids,
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "gtf_dir": str(RAW_GTF),
        "fna_dir": str(RAW_FNA),
        "adapt": "from_raw_new",
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{RUN_NAME} split_cpu meta={meta}", flush=True)

    append_queue_entry(
        f"{RUN_NAME}_split",
        job=f"python -m src.runs_unif.{RUN_NAME}.run_split_cpu",
        pid=os.getpid(),
        estimated_time="6-18h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB,
        resources="NEW adapt window -100..100 + C++ contingency k=10 + modularity + plot",
        log=str(ROOT / "logs" / f"{RUN_NAME}_split.log"),
    )

    split_csv = run_split_predict(
        outdir=OUT_ROOT,
        type="pangenome",
        seed=SEED,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
        parsed=PANEL_ROOT / "PARSED",
        marked_fasta=PANEL_ROOT / "MARKED",
        reuse_panel_marked=False,
        # New adapt — do not pass marked_pangenome.
        gtf_dir=RAW_GTF,
        fna_dir=RAW_FNA,
        environment=ENVIRONMENT,
        window=WINDOW,
        kmer_size=kmer_size,
        engine=ENGINE,
        force=force,
        max_ids=max_ids,
        plot=True,
        modularity_refine=True,
        max_fold_size=max_fold_size,
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
    (OUT_ROOT / "split_done.json").write_text(
        json.dumps({"status": "ok", "split_csv": str(split_csv), "tsv": str(tsv)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"{RUN_NAME} split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"legnet_tsv={tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
