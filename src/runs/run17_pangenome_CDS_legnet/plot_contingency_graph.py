"""Rebuild run17 pangenome contingency figure (fold + train/test colours).

Edges were not saved during the original split (``plot=False``). This
recomputes C++ contingency edges from ``MARKED_parsed`` and renders connected
nodes only.

Launch::

  conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.plot_contingency_graph
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run17_pangenome_CDS_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
PEAK_RAM_GIB = 16.0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    max_edges = 100_000
    for tok in list(argv):
        if tok.startswith("max_edges="):
            max_edges = int(tok.split("=", 1)[1])
            argv.remove(tok)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.splits.pangenome import plot_pangenome_contingency_from_artifacts

    marked = OUT_ROOT / "MARKED_parsed"
    split_csv = OUT_ROOT / "split.csv"
    meta_path = OUT_ROOT / "pangenome_split_meta.json"
    k = 21
    min_shared = 1
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        k = int(meta.get("k", k))
        min_shared = int(meta.get("min_shared", min_shared))

    wait_until_launchable(
        peak_ram_gib=PEAK_RAM_GIB,
        job_class=CLASS_CPU_RAM_HEAVY,
        label=f"{RUN_ID}_plot_graph",
    )
    append_queue_entry(
        f"{RUN_ID}_plot_graph",
        job=f"python -m src.runs.{RUN_ID}.plot_contingency_graph",
        pid=os.getpid(),
        estimated_time="30-90m",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=PEAK_RAM_GIB,
        resources=f"rebuild contingency edges max_edges={max_edges}; fold+train_test figure",
        log=f"logs/{RUN_ID}_plot_graph.log",
    )

    summary = plot_pangenome_contingency_from_artifacts(
        marked_parsed=marked,
        split_csv=split_csv,
        outdir=OUT_ROOT,
        k=k,
        min_shared=min_shared,
        max_edges=max_edges,
    )
    print(json.dumps(summary.get("plot") or summary, indent=2), flush=True)
    print(f"figures → {OUT_ROOT / 'figures'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
