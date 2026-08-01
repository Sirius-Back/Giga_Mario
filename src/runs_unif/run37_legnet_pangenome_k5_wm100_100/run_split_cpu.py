"""CPU stages for run37: reuse MARKED (−100..100) + k=5 hash-majority pangenome.

- Reuse panel ``ready_legnet/MARKED`` (window −100..100 / 200 bp CRS)
- Hash-majority contingency graph k=5 (cpp), modularity-refine large CCs
- Plot contingency graph (color by cluster); ratios train:test:val ≈ 3:1:1
- Materialize SPLIT + LegNet TSV under
  ``runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100``

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs_unif.run37_legnet_pangenome_k5_wm100_100.run_split_cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_I = 37
MODEL = "legnet"
SPLIT = "pangenome"
SPLIT_PARAMS = "k5_wm100_100"
RUN_NAME = f"run{RUN_I}_{MODEL}_{SPLIT}_{SPLIT_PARAMS}"

PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs_unif" / MODEL / RUN_NAME

SEED = 42
RATIOS = (3.0, 1.0, 1.0)  # train:test:val ≈ 3:1:1
ENVIRONMENT = "gene"
WINDOW = {"pos1": -100, "pos2": 100}
KMER_SIZE = 5
ENGINE = "cpp"
PEAK_RAM_GIB = 48.0  # hash-majority k=5 + modularity can be heavy
MAX_FOLD_SIZE = 1000
CLUSTER_METHOD = "hash_majority"
MIN_DF = 2


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
    from src.runs_unif.run37_legnet_pangenome_k5_wm100_100.ensure_mice_fold import (
        main as ensure_fold,
    )

    ensure_fold()

    for req in (
        PANEL_ROOT / "ID.csv",
        PANEL_ROOT / "fold.csv",
        PANEL_ROOT / "PARSED",
        PANEL_ROOT / "PREDICT",
        PANEL_ROOT / "MARKED",
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

    from omegaconf import OmegaConf

    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": RUN_NAME,
        "aligned_run": RUN_I,
        "stage": "split_cpu",
        "mode": "run",
        "split": SPLIT,
        "engine": ENGINE,
        "environment": ENVIRONMENT,
        "window": WINDOW,
        "kmer_size": kmer_size,
        "cluster_method": CLUSTER_METHOD,
        "min_df": MIN_DF,
        "modularity_refine": True,
        "max_fold_size": max_fold_size,
        "seed": SEED,
        "ratios": list(RATIOS),
        "force": force,
        "max_ids": max_ids,
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "reuse_panel_marked": True,
        "train": {"name": MODEL},
        "task_type": "regression",
        "adversarial": True,
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_ROOT / "hydra_resolved_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(meta)), encoding="utf-8"
    )
    print(f"{RUN_NAME} split_cpu meta={meta}", flush=True)

    append_queue_entry(
        f"{RUN_NAME}_split",
        job=f"python -m src.runs_unif.{RUN_NAME}.run_split_cpu",
        pid=os.getpid(),
        estimated_time="4-12h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB,
        resources=(
            "reuse MARKED −100..100 + hash_majority k=5 + modularity + plot"
        ),
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
        reuse_panel_marked=True,
        environment=ENVIRONMENT,
        window=WINDOW,
        kmer_size=kmer_size,
        engine=ENGINE,
        force=force,
        max_ids=max_ids,
        plot=True,
        modularity_refine=True,
        max_fold_size=max_fold_size,
        min_df=MIN_DF,
        pangenome_cluster_method=CLUSTER_METHOD,
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
        json.dumps(
            {"status": "ok", "split_csv": str(split_csv), "tsv": str(tsv)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{RUN_NAME} split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"legnet_tsv={tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
