"""CPU export+pack Stage-2 hash-node graph (k=5) for VGAE."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from src.pipeline.job_queue import CLASS_CPU_RAM_HEAVY, append_queue_entry
from src.splits.vgae.hash_export import pack_hash_graph

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "VGAE" / "stage2_hash_k5"
IDS = ROOT / "runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph/ids.txt"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    append_queue_entry(
        "vgae_stage2_pack_hash_k5",
        job="python src/run/run_id/pack_vgae_stage2.py",
        pid=os.getpid(),
        estimated_time="1-4h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=32.0,
        log=str(OUT / "pack.log"),
        resources="hash export k=5 full panel",
    )
    ids = [ln.strip() for ln in IDS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    t0 = time.time()
    try:
        pack, incidence = pack_hash_graph(
            ROOT / "ready_legnet/MARKED",
            ids,
            OUT / "pack",
            k=5,
            max_edges=500_000,
        )
        meta = {
            "ok": True,
            "n_hash": pack.n_nodes,
            "n_edges": pack.n_edges,
            "n_regions": len(incidence["region_ids"]),
            "sec": time.time() - t0,
        }
        (OUT / "pack_done.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("PACK2_DONE", meta, flush=True)
    except Exception as exc:
        (OUT / "pack_done.json").write_text(
            json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
