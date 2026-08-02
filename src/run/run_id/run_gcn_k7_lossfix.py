"""GCN/VGAE on run29 k=7 pangenome with homology_first loss.

Stage1 — region contingency graph (k=7) + streamed projected 7-mers (2048-d).
Stage2 — hash-node graph (4**7 nodes) with the same projected compositional wrap.

Waits for RAM headroom and a free GPU. Does not clobber k=5 VGAE runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT / "runs_unif" / "legnet" / "run29_legnet_pangenome_k7_wm100_100" / "graph"
MARKED = ROOT / "ready_legnet" / "MARKED"
OUT1 = ROOT / "VGAE" / "stage1_region_k7_lossfix"
OUT2 = ROOT / "VGAE" / "stage2_hash_k7_lossfix"
CHECK_BIN = ROOT / "mag" / "src" / "split_check_othoparagroup" / "split_check_othoparagroup"
HASH_TABLE = ROOT / "mag" / "homology_graph" / "maps" / "gene_ortho_para_hash.tsv"

K = 7
PROJECT_DIM = 2048
PEAK_PACK_GIB = 16.0  # streamed projection — no full 28 GiB dense matrix


def _mem_used_frac() -> float:
    info: dict[str, float] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            key, val, *_ = line.split()
            info[key.rstrip(":")] = float(val)
    total = info["MemTotal"]
    avail = info["MemAvailable"]
    return (total - avail) / total


def _wait_ram(max_used: float = 0.85, poll: float = 60.0, label: str = "ram") -> None:
    while True:
        u = _mem_used_frac()
        if u <= max_used:
            print(f"[k7] {label}: RAM used={100 * u:.1f}% OK", flush=True)
            return
        print(
            f"[k7] {label}: RAM used={100 * u:.1f}% > {100 * max_used:.0f}%; "
            f"sleep {poll}s",
            flush=True,
        )
        time.sleep(poll)


def _run_checker(split_csv: Path, out_dir: Path, run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not CHECK_BIN.is_file():
        print(f"[k7] checker missing: {CHECK_BIN}", flush=True)
        return
    cmd = [
        str(CHECK_BIN),
        "--split",
        str(split_csv),
        "--hash-table",
        str(HASH_TABLE),
        "--outdir",
        str(out_dir),
        "--model",
        "vgae",
        "--run-id",
        run_id,
    ]
    print("[k7] checker:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def main() -> int:
    if not GRAPH.is_dir():
        raise FileNotFoundError(f"missing k=7 graph: {GRAPH}")
    if not MARKED.is_dir():
        raise FileNotFoundError(f"missing MARKED: {MARKED}")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    from src.pipeline.job_queue import (
        CLASS_CPU_RAM_HEAVY,
        CLASS_WAITER,
        append_queue_entry,
        wait_until_launchable,
    )
    from src.splits.gcn import run_gcn_split_assign

    append_queue_entry(
        "waiter_vgae_k7_lossfix",
        job="run_gcn_k7_lossfix wait RAM+GPU",
        pid=os.getpid(),
        estimated_time="until RAM<=85% + free GPU",
        job_class=CLASS_WAITER,
        peak_ram_gib=0.0,
        gpus=(),
        log=str(OUT1 / "train.log"),
    )
    _wait_ram(0.85, poll=120.0, label="pre_stage1")
    wait_until_launchable(
        peak_ram_gib=PEAK_PACK_GIB,
        gpus=(),
        job_class=CLASS_CPU_RAM_HEAVY,
        poll_sec=60.0,
        label="vgae_k7_stage1_pack_ram",
    )

    print("[k7] Stage1 region GCN/VGAE homology_first…", flush=True)
    OUT1.mkdir(parents=True, exist_ok=True)
    s1 = run_gcn_split_assign(
        outdir=OUT1,
        model="stage1_region_k7_lossfix",
        seed=42,
        graph_dir=GRAPH,
        marked_dir=MARKED,
        k=K,
        force_retrain=True,
        loss_mode="homology_first",
        feature_k=K,
        project_dim=PROJECT_DIM,
        peak_ram_gib=16.0,
        wait_poll_sec=600.0,
        max_gpu_used_mib=2048.0,
        min_epochs=25,
        patience=10,
        max_epochs=200,
    )
    print("[k7] Stage1 summary:", json.dumps(s1, default=str)[:800], flush=True)
    _run_checker(
        OUT1 / "split.csv",
        ROOT / "VGAE" / "checks" / "stage1_k7_lossfix",
        "stage1_k7_lossfix",
    )

    _wait_ram(0.85, poll=120.0, label="pre_stage2")
    print("[k7] Stage2 hash GCN/VGAE homology_first…", flush=True)
    OUT2.mkdir(parents=True, exist_ok=True)
    s2 = run_gcn_split_assign(
        outdir=OUT2,
        model="stage2_hash_k7_lossfix",
        seed=42,
        graph_dir=GRAPH,
        marked_dir=MARKED,
        k=K,
        force_retrain=True,
        loss_mode="homology_first",
        project_dim=PROJECT_DIM,
        peak_ram_gib=20.0,
        wait_poll_sec=600.0,
        max_gpu_used_mib=2048.0,
        min_epochs=25,
        patience=10,
        max_epochs=200,
    )
    print("[k7] Stage2 summary:", json.dumps(s2, default=str)[:800], flush=True)
    _run_checker(
        OUT2 / "split.csv",
        ROOT / "VGAE" / "checks" / "stage2_k7_lossfix",
        "stage2_k7_lossfix",
    )

    print(json.dumps({"stage1": s1, "stage2": s2}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
