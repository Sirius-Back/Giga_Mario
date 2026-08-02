"""Train Stage1 GCL (+ optional GCL-GAT) with homology_first on k5 pack.

Uses the best up-to-date loss: ``homology_first`` + early-stop on legacy mean L_hom.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "VGAE" / "stage1_region_k5_lossfix" / "pack"
RUNS = (
    ("gcl", ROOT / "VGAE" / "stage1_region_k5_gcl_lossfix"),
    ("gcl_gat", ROOT / "VGAE" / "stage1_region_k5_gcl_gat_lossfix"),
)


def main() -> int:
    if not (PACK / "feature_meta.json").is_file():
        raise FileNotFoundError(PACK)
    from src.splits.vgae.train import run_vgae_train
    from src.run.run_id.compare_vae_vgae_architectures import main as cmp_main

    summaries = []
    for arch, out in RUNS:
        print(f"[gcl] === {arch} homology_first → {out} ===", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        pack = out / "pack"
        pack.mkdir(exist_ok=True)
        for name in (
            "node_features.npz",
            "edges_weighted.npz",
            "ids.txt",
            "feature_meta.json",
        ):
            shutil.copy2(PACK / name, pack / name)
        meta = run_vgae_train(
            pack=pack,
            out_dir=out,
            seed=42,
            device=None,
            loss_mode="homology_first",
            architecture=arch,
            gat_heads=4,
            lambda_gcl=1.0,
            early_stop_on_legacy=True,
            peak_ram_gib=8.0,
            wait_poll_sec=60.0,
            max_gpu_used_mib=2048.0,
            min_epochs=25,
            patience=10,
            max_epochs=200,
        )
        summaries.append(
            {
                "architecture": arch,
                "best_l_hom": meta.get("best_l_hom"),
                "best_l_hom_legacy": meta.get("best_l_hom_legacy"),
                "best_epoch": meta.get("best_epoch"),
                "final_l_hom_legacy": meta.get("final_l_hom_legacy"),
                "all_aggs": meta.get("all_aggs"),
                "early_stop_on_legacy": meta.get("early_stop_on_legacy"),
            }
        )
        print("[gcl]", arch, json.dumps(summaries[-1], default=str)[:800], flush=True)

    (ROOT / "VGAE" / "arch_gcl_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8"
    )
    cmp_main()
    print("[gcl] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
