"""Stage2 k=7 hash VGAE with full (non-projected) compositional features.

Stage1 region pack stays projected (4**7 dense does not fit GPU). Stage2 hash
grain has 4**7 nodes × (1+4**7) features ≈ 1 GiB on device — full spectrum fits.
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
COMPARE = ROOT / "VGAE" / "loss_comparison_k5_k7.json"


def _checker(split: Path, outdir: Path, run_id: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not CHECK.is_file():
        print(f"[k7-s2] checker missing: {CHECK}", flush=True)
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
    from src.splits.vgae.train import _pick_free_gpu

    idx = _pick_free_gpu(max_used_mib=2048.0)
    if idx is not None:
        print(f"[k7-s2] free GPU cuda:{idx}", flush=True)
        return f"cuda:{idx}"
    best, best_used = None, None
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1024**2)
        if best_used is None or used < best_used:
            best_used, best = used, i
    if best is None or best_used > 4000:
        raise RuntimeError(f"no usable GPU (best used_mib={best_used})")
    print(f"[k7-s2] fallback GPU cuda:{best} used_mib={best_used:.0f}", flush=True)
    return f"cuda:{best}"


def _collect_comparison() -> dict:
    rows = []
    for path in sorted((ROOT / "VGAE").glob("stage*/train_meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        pack_meta = {}
        pm = path.parent / "pack" / "feature_meta.json"
        if pm.is_file():
            pack_meta = json.loads(pm.read_text(encoding="utf-8"))
        proj = pack_meta.get("feature_projection") or {}
        rows.append(
            {
                "run": path.parent.name,
                "best_l_hom": meta.get("best_l_hom"),
                "best_epoch": meta.get("best_epoch"),
                "final_l_hom": meta.get("final_l_hom"),
                "random_baseline_l_hom": meta.get("random_baseline_l_hom"),
                "loss_mode": meta.get("loss_mode"),
                "k": meta.get("k") or pack_meta.get("k"),
                "grain": meta.get("grain") or pack_meta.get("grain") or "region",
                "n_nodes": meta.get("n_nodes")
                or meta.get("n_hash_nodes")
                or pack_meta.get("n_nodes"),
                "n_features": pack_meta.get("n_features"),
                "feature_projection_applied": bool(proj.get("applied")),
                "device": meta.get("device"),
                "counts": meta.get("counts"),
            }
        )
    rows.sort(key=lambda r: (r.get("k") or 0, r.get("grain") or "", r["run"]))
    return {"models": rows}


def main() -> int:
    if not (OUT1 / "train_meta.json").is_file():
        raise FileNotFoundError(
            f"Stage1 train_meta missing under {OUT1}; finish Stage1 first"
        )

    from src.splits.vgae.stage2 import run_stage2_hash_vgae

    # Drop any incomplete / projected pack so we rebuild full features
    pack = OUT2 / "pack"
    if (pack / "feature_meta.json").is_file():
        meta = json.loads((pack / "feature_meta.json").read_text(encoding="utf-8"))
        proj = meta.get("feature_projection") or {}
        n_feat = int(meta.get("n_features") or 0)
        full_dim = 1 + 4**7
        if proj.get("applied") or n_feat < full_dim:
            print(
                f"[k7-s2] removing non-full pack "
                f"(n_features={n_feat}, projected={proj.get('applied')})",
                flush=True,
            )
            import shutil

            shutil.rmtree(pack)

    device = _pick_device()
    ids = [
        ln.strip()
        for ln in (GRAPH / "ids.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    print(
        f"[k7-s2] Stage2 FULL features (project_dim=None) device={device} "
        f"n_regions={len(ids)}",
        flush=True,
    )
    meta2 = run_stage2_hash_vgae(
        out_dir=OUT2,
        marked_dir=MARKED,
        region_ids=ids,
        k=7,
        seed=42,
        device=device,
        loss_mode="homology_first",
        # Explicit: no projection — full GC + 4**7 k-mer spectrum
        project_dim=None,
        peak_ram_gib=40.0,
        wait_poll_sec=60.0,
        max_gpu_used_mib=2048.0,
        min_epochs=25,
        patience=10,
        max_epochs=200,
    )
    print(
        "[k7-s2] Stage2",
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

    cmp = _collect_comparison()
    COMPARE.write_text(json.dumps(cmp, indent=2, default=str) + "\n", encoding="utf-8")
    print("[k7-s2] loss comparison →", COMPARE, flush=True)
    for row in cmp["models"]:
        print(
            f"  {row['run']}: best_l_hom={row['best_l_hom']} "
            f"k={row['k']} grain={row['grain']} "
            f"n_feat={row['n_features']} proj={row['feature_projection_applied']} "
            f"mode={row['loss_mode']}",
            flush=True,
        )
    print("[k7-s2] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
