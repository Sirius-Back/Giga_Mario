"""CPU stages for run15_blastp_legnet: mice fold + DIAMOND BLASTP split + LegNet TSV.

No GPU train. Writes under ``runs/run15_blastp_legnet/``.

Homology search uses DIAMOND ``blastp --sensitive`` (not NCBI BLASTP).

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run15_blastp_legnet.run_split_cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run15_blastp_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
RAW_GTF = ROOT / "raw" / "gtf"
RAW_FNA = ROOT / "raw" / "fna"
SEED = 42
RATIOS = (1, 1, 3)
GENETIC_CODE = "universal"
ENVIRONMENT = "gene"
WINDOW = {"pos1": 0, "pos2": 0}
THREADS = max(8, (os.cpu_count() or 8) // 2)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    genetic_code = GENETIC_CODE
    threads = THREADS
    force = False
    max_ids: int | None = None
    for tok in list(argv):
        if tok.startswith("genetic_code="):
            genetic_code = tok.split("=", 1)[1]
            argv.remove(tok)
        elif tok.startswith("threads="):
            threads = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("max_ids="):
            max_ids = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok == "force=true" or tok == "--force":
            force = True
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.environ["PATH"] = (
        f"{ROOT / 'bin'}:"
        f"{ROOT / 'miniconda3' / 'envs' / 'legnet' / 'bin'}:"
        + os.environ.get("PATH", "")
    )

    from src.runs.run15_blastp_legnet.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict

    for req in (PANEL_ROOT / "ID.csv", PANEL_ROOT / "fold.csv", PANEL_ROOT / "PARSED", RAW_GTF, RAW_FNA):
        if not req.exists():
            raise FileNotFoundError(f"missing required input: {req}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "split": "blastp",
        "homology_engine": "diamond",
        "genetic_code": genetic_code,
        "environment": ENVIRONMENT,
        "window": WINDOW,
        "seed": SEED,
        "ratios": list(RATIOS),
        "threads": threads,
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
    print(f"run15 split_cpu meta={meta}", flush=True)

    split_csv = run_split_predict(
        outdir=OUT_ROOT,
        type="blastp",
        seed=SEED,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
        parsed=PANEL_ROOT / "PARSED",
        marked_fasta=PANEL_ROOT / "MARKED",
        # MARKED is only the ID keep-set (∩ PARSED); proteins come from CDS/GTF.
        reuse_panel_marked=True,
        gtf_dir=RAW_GTF,
        fna_dir=RAW_FNA,
        environment=ENVIRONMENT,
        window=WINDOW,
        genetic_code=genetic_code,
        threads=threads,
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
    print(f"run15 split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"legnet_tsv={tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
