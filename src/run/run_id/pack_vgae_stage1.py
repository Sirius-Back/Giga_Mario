"""CPU pack Stage-1 VGAE features (GC + k=5) onto run37 region graph."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from src.pipeline.job_queue import CLASS_CPU_RAM_HEAVY, append_queue_entry
from src.splits.vgae.graph_data import pack_region_graph

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "VGAE" / "stage1_region_k5"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    append_queue_entry(
        "vgae_stage1_pack_k5_exec",
        job="python -m src.run.run_id.pack_vgae_stage1",
        pid=os.getpid(),
        estimated_time="1-3h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=24.0,
        log=str(OUT / "pack.log"),
        resources="n_workers=4 k=5 full panel",
    )
    t0 = time.time()
    try:
        pack = pack_region_graph(
            ROOT / "runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph",
            ROOT / "ready_legnet/MARKED",
            OUT / "pack",
            k=5,
            max_ids=None,
        )
        meta = {
            "ok": True,
            "n": pack.n_nodes,
            "e": pack.n_edges,
            "sec": time.time() - t0,
            "pack_dir": str(OUT / "pack"),
        }
        (OUT / "pack_done.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("PACK_DONE", meta, flush=True)
    except Exception as exc:
        (OUT / "pack_done.json").write_text(
            json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
