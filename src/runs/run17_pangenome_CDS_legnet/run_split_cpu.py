"""CPU stages for run17: mice fold + pangenome adapt (CDS 0..100) + C++ CC split.

No GPU train. Writes under ``runs/run17_pangenome_CDS_legnet/``.

A2A: adapt ``MARKED_pangenome`` from ``raw/{gtf,fna}`` with window
``{pos1:0, pos2:100}`` (may differ from panel LegNet MARKED), then ∩ PARSED →
C++ contingency graph (native/cpp).

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.run_split_cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run17_pangenome_CDS_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
RAW_GTF = ROOT / "raw" / "gtf"
RAW_FNA = ROOT / "raw" / "fna"
SEED = 42
# Caduceus-aligned default (~81% / 10% / 9% train/test/val). Do not use (1,1,3):
# that weight puts most samples in val and LegNet evaluates the full val set.
RATIOS = None
ENVIRONMENT = "gene"
WINDOW = {"pos1": 0, "pos2": 100}
# Pangenome contingency uses C++ native; single k (default 21).
KMER_SIZE = 21
ENGINE = "cpp"
PEAK_RAM_GIB = 24.0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    max_ids: int | None = None
    force = False
    kmer_size = KMER_SIZE
    for tok in list(argv):
        if tok.startswith("max_ids="):
            max_ids = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("kmer_size="):
            kmer_size = int(tok.split("=", 1)[1])
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
    from src.runs.run17_pangenome_CDS_legnet.ensure_mice_fold import main as ensure_fold

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

    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB,
        job_class=CLASS_CPU_RAM_HEAVY,
        label=f"{RUN_ID}_split_cpu",
    )

    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "split": "pangenome",
        "engine": ENGINE,
        "environment": ENVIRONMENT,
        "window": WINDOW,
        "kmer_size": kmer_size,
        "seed": SEED,
        "ratios": list(RATIOS) if RATIOS is not None else None,
        "force": force,
        "max_ids": max_ids,
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "gtf_dir": str(RAW_GTF),
        "fna_dir": str(RAW_FNA),
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    # Hydra-compatible snapshot for audit / re-run.
    hydra_cfg = {
        "run_id": RUN_ID,
        "mode": "run",
        "data": "ready_legnet",
        "split": "pangenome",
        "train": {"name": "legnet"},
        "task_type": "regression",
        "adversarial": True,
        "adversarial_task_type": "classification",
        "epochs": 50,
        "min_epochs": 25,
        "early_stopping_patience": 10,
        "checkpoint_every_n_epochs": 10,
        "n_devices": 4,
        "zsv": True,
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "gtf_dir": str(RAW_GTF),
        "fna_dir": str(RAW_FNA),
        "environment": ENVIRONMENT,
        "window": WINDOW,
        "kmer_size": kmer_size,
        "kmer_engine": ENGINE,
        "seed": SEED,
        "ratios": list(RATIOS) if RATIOS is not None else None,
    }
    try:
        from omegaconf import OmegaConf

        (OUT_ROOT / "hydra_resolved_config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(hydra_cfg)), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        (OUT_ROOT / "hydra_resolved_config.yaml").write_text(
            json.dumps(hydra_cfg, indent=2) + "\n", encoding="utf-8"
        )
    print(f"run17 split_cpu meta={meta}", flush=True)

    append_queue_entry(
        f"{RUN_ID}_split",
        job=f"python -m src.runs.{RUN_ID}.run_split_cpu",
        pid=os.getpid(),
        estimated_time="4-12h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB,
        resources="pangenome adapt window 0..100 + C++ contingency k=21",
        log=f"logs/{RUN_ID}_split.log",
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
        # Do NOT reuse panel MARKED — window differs (LegNet CRS vs CDS 0..100).
        reuse_panel_marked=False,
        gtf_dir=RAW_GTF,
        fna_dir=RAW_FNA,
        environment=ENVIRONMENT,
        window=WINDOW,
        kmer_size=kmer_size,
        engine=ENGINE,
        force=force,
        max_ids=max_ids,
        plot=False,
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
    print(f"run17 split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"legnet_tsv={tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
