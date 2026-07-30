"""CPU-only: mice fold + hashFrag split (reuse run5 homology) + materialize SPLIT.

Reuses ``runs/run5/hashfrag_work`` (MARKED and fold.csv identical between
``ready_caduceus`` and ``ready_legnet``). No full-panel BLAST unless ``force=true``.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run16_hashfrag_caduceus.run_split_cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run16_hashfrag_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
RUN5_HASHFRAG_WORK = ROOT / "runs" / "run5" / "hashfrag_work"
SEED = 42
RATIOS = (1, 1, 3)
THRESHOLD = 60
THREADS = 16


def _setup_path() -> None:
    """hashFrag in legnet env; BLAST+ in project bin."""
    os.environ["PATH"] = (
        f"{ROOT / 'miniconda3' / 'envs' / 'legnet' / 'bin'}:"
        f"{ROOT / 'miniconda3' / 'bin'}:"
        f"{ROOT / 'bin'}:"
        + os.environ.get("PATH", "")
    )


def _link_hashfrag_work(out_work: Path, *, force: bool) -> None:
    """Symlink run5 hashfrag_work into run16 outdir (reuse BLAST homology)."""
    if out_work.exists() or out_work.is_symlink():
        if force:
            if out_work.is_symlink():
                out_work.unlink()
            elif out_work.is_dir():
                import shutil

                shutil.rmtree(out_work)
            else:
                out_work.unlink()
        else:
            print(f"hashfrag_work already present: {out_work}", flush=True)
            return
    if not RUN5_HASHFRAG_WORK.is_dir():
        raise FileNotFoundError(
            f"run5 hashfrag_work missing (required for reuse): {RUN5_HASHFRAG_WORK}"
        )
    groups = RUN5_HASHFRAG_WORK / "hashFrag.homologous_groups.tsv"
    if not groups.is_file() or groups.stat().st_size == 0:
        raise FileNotFoundError(f"run5 homologous groups missing: {groups}")
    out_work.parent.mkdir(parents=True, exist_ok=True)
    out_work.symlink_to(RUN5_HASHFRAG_WORK.resolve())
    print(f"Linked hashfrag_work → {RUN5_HASHFRAG_WORK}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    threshold = THRESHOLD
    threads = THREADS
    force = False
    for tok in list(argv):
        if tok.startswith("threshold="):
            threshold = int(float(tok.split("=", 1)[1]))
            argv.remove(tok)
        elif tok.startswith("threads="):
            threads = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok in {"force=true", "--force"}:
            force = True
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    _setup_path()

    marked = PANEL_ROOT / "MARKED"
    if not marked.is_dir():
        raise FileNotFoundError(f"MARKED missing: {marked}")
    for req in ("ID.csv", "fold.csv", "PARSED", "PREDICT"):
        p = PANEL_ROOT / req
        if not p.exists():
            raise FileNotFoundError(f"Panel missing {req}: {p}")

    from src.runs.run16_hashfrag_caduceus.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _link_hashfrag_work(OUT_ROOT / "hashfrag_work", force=force)

    meta = {
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "split": "hashfrag",
        "threshold": threshold,
        "seed": SEED,
        "threads": threads,
        "force": force,
        "ratios": list(RATIOS),
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "hashfrag_work_reuse": str(RUN5_HASHFRAG_WORK),
        "status": "RUNNING",
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run16 split_cpu meta={meta}", flush=True)

    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict

    split_csv = run_split_predict(
        outdir=OUT_ROOT,
        type="hashfrag",
        seed=SEED,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
        marked_fasta=marked,
        plot=False,
        threshold=float(threshold),
        threads=threads,
        force=force,
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
    print(f"run16 split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    print(f"split_csv={split_csv}", flush=True)
    print(f"split_root={split_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
