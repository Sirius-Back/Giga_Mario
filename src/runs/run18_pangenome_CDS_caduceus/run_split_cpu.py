"""CPU: reuse run17 pangenome split → materialize Caduceus SPLIT (no re-adapt).

Copies ``split.csv`` from ``runs/run17_pangenome_CDS_legnet`` (same region IDs /
window 0..100 / C++ contingency assignment) and materializes against
``ready_caduceus`` PARSED/PREDICT. Symlinks MARKED_* for provenance.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run18_pangenome_CDS_caduceus.run_split_cpu
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run18_pangenome_CDS_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
RUN17_ROOT = ROOT / "runs" / "run17_pangenome_CDS_legnet"
SEED = 42
WINDOW = {"pos1": 0, "pos2": 100}
PEAK_RAM_GIB = 12.0


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.pipeline.split import run_split
    from src.runs.run18_pangenome_CDS_caduceus.ensure_mice_fold import main as ensure_fold

    ensure_fold()

    for req in (
        PANEL_ROOT / "ID.csv",
        PANEL_ROOT / "fold.csv",
        PANEL_ROOT / "PARSED",
        PANEL_ROOT / "PREDICT",
        RUN17_ROOT / "split.csv",
        RUN17_ROOT / "split_cpu_done.json",
    ):
        if not req.exists():
            raise FileNotFoundError(f"missing required input: {req}")

    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB,
        job_class=CLASS_CPU_RAM_HEAVY,
        label=f"{RUN_ID}_reuse_split",
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    append_queue_entry(
        f"{RUN_ID}_split",
        job=f"python -m src.runs.{RUN_ID}.run_split_cpu",
        pid=os.getpid(),
        estimated_time="1-3h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB,
        resources="reuse run17 split.csv; materialize ready_caduceus SPLIT",
        log=f"logs/{RUN_ID}_split.log",
    )

    split_src = RUN17_ROOT / "split.csv"
    split_dst = OUT_ROOT / "split.csv"
    shutil.copy2(split_src, split_dst)

    # Provenance links (do not re-adapt).
    for name in ("MARKED_pangenome", "MARKED_parsed", "pangenome_assignment.csv", "pangenome_split_meta.json"):
        src = RUN17_ROOT / name
        if src.exists():
            _link_or_copy(src, OUT_ROOT / name)

    meta = {
        "run_id": RUN_ID,
        "stage": "split_cpu",
        "split": "pangenome",
        "reuse_from": str(RUN17_ROOT),
        "window": WINDOW,
        "engine": "cpp",
        "seed": SEED,
        "panel_root": str(PANEL_ROOT),
        "out_root": str(OUT_ROOT),
        "split_csv_source": str(split_src),
    }
    (OUT_ROOT / "split_cpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    try:
        from omegaconf import OmegaConf

        (OUT_ROOT / "hydra_resolved_config.yaml").write_text(
            OmegaConf.to_yaml(
                OmegaConf.create(
                    {
                        "run_id": RUN_ID,
                        "mode": "run",
                        "data": "ready_caduceus",
                        "split": "pangenome",
                        "train": {"name": "caduceus"},
                        "reuse_split_from": str(RUN17_ROOT),
                        "window": WINDOW,
                        "epochs": 50,
                        "min_epochs": 25,
                        "early_stopping_patience": 10,
                        "n_devices": 4,
                        "zsv": True,
                        "adversarial": True,
                        "panel_root": str(PANEL_ROOT),
                        "out_root": str(OUT_ROOT),
                    }
                )
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass

    print(f"run18 reused split.csv from {split_src}", flush=True)
    split_root = run_split(
        split_dst,
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
        "split_csv": str(split_dst),
        "split_root": str(split_root),
    }
    (OUT_ROOT / "split_cpu_done.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run18 split_cpu COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
