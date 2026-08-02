"""Train VGAE k=7 homology_first reusing Stage1 pack; then Stage2.

Pinned to a free GPU with lowered peak_ram_gib so RAM politics allow launch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT1 = ROOT / "VGAE" / "stage1_region_k7_lossfix"
OUT2 = ROOT / "VGAE" / "stage2_hash_k7_lossfix"
GRAPH = ROOT / "runs_unif" / "legnet" / "run29_legnet_pangenome_k7_wm100_100" / "graph"
MARKED = ROOT / "ready_legnet" / "MARKED"
CHECK = ROOT / "mag" / "src" / "split_check_othoparagroup" / "split_check_othoparagroup"
HASH = ROOT / "mag" / "homology_graph" / "maps" / "gene_ortho_para_hash.tsv"


def _checker(split: Path, outdir: Path, run_id: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not CHECK.is_file():
        print(f"[k7] checker missing: {CHECK}", flush=True)
        return
    subprocess.run(
        [
            str(CHECK),
            "--split",
            str(split),
            "--hash-table",
            str(HASH),
            "--outdir",
            str(outdir),
            "--model",
            "vgae",
            "--run-id",
            run_id,
        ],
        check=False,
    )


def _pick_device() -> str:
    import torch
    from src.splits.vgae.train import _gpu_is_free, _pick_free_gpu

    # Prefer truly empty; allow up to 2 GiB leftover
    idx = _pick_free_gpu(max_used_mib=2048.0)
    if idx is None:
        # Fall back: lowest used among launchable-by-VRAM
        best = None
        best_used = None
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = (total - free) / (1024**2)
            if best_used is None or used < best_used:
                best_used = used
                best = i
        if best is None or best_used > 4000:
            raise RuntimeError(f"no usable GPU (best used_mib={best_used})")
        print(f"[k7] fallback GPU cuda:{best} used_mib={best_used:.0f}", flush=True)
        return f"cuda:{best}"
    print(f"[k7] free GPU cuda:{idx}", flush=True)
    return f"cuda:{idx}"


def main() -> int:
    if not (OUT1 / "pack" / "feature_meta.json").is_file():
        raise FileNotFoundError(f"missing Stage1 pack under {OUT1 / 'pack'}")

    from src.splits.vgae.split_assign import run_vgae_split_assign
    from src.splits.vgae.stage2 import run_stage2_hash_vgae

    device = _pick_device()
    print(f"[k7-resubmit] Stage1 train device={device} reuse pack", flush=True)
    meta1 = run_vgae_split_assign(
        outdir=OUT1,
        pack_dir=OUT1 / "pack",
        seed=42,
        k=7,
        device=device,
        loss_mode="homology_first",
        peak_ram_gib=6.0,
        wait_poll_sec=60.0,
        max_gpu_used_mib=2048.0,
        min_epochs=25,
        patience=10,
        max_epochs=200,
    )
    print(
        "[k7-resubmit] Stage1",
        json.dumps(
            {
                k: meta1.get(k)
                for k in ("best_l_hom", "best_epoch", "device", "counts", "loss_mode")
            },
            default=str,
        ),
        flush=True,
    )
    _checker(OUT1 / "split.csv", ROOT / "VGAE" / "checks" / "stage1_k7_lossfix", "stage1_k7_lossfix")

    ids = [
        ln.strip()
        for ln in (GRAPH / "ids.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    print("[k7-resubmit] Stage2 hash train FULL features (no projection)", flush=True)
    meta2 = run_stage2_hash_vgae(
        out_dir=OUT2,
        marked_dir=MARKED,
        region_ids=ids,
        k=7,
        seed=42,
        device=None,
        loss_mode="homology_first",
        # Hash grain: 4**7 nodes × (1+4**7) feats ≈ 1 GiB — full spectrum fits GPU.
        project_dim=None,
        peak_ram_gib=40.0,
        wait_poll_sec=60.0,
        max_gpu_used_mib=2048.0,
        min_epochs=25,
        patience=10,
        max_epochs=200,
    )
    print(
        "[k7-resubmit] Stage2",
        json.dumps(
            {
                k: meta2.get(k)
                for k in (
                    "best_l_hom",
                    "best_epoch",
                    "device",
                    "counts",
                    "loss_mode",
                    "n_hash_nodes",
                )
            },
            default=str,
        ),
        flush=True,
    )
    _checker(OUT2 / "split.csv", ROOT / "VGAE" / "checks" / "stage2_k7_lossfix", "stage2_k7_lossfix")
    print("[k7-resubmit] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
