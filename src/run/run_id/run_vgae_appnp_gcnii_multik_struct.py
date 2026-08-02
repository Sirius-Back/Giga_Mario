"""Stage1 A/B: APPNP, GCNII, multi-k GCN, structural-feat GCN (homology_first).

Reuses k5 topology from ``VGAE/stage1_region_k5_lossfix/pack``.
Homology never enters the encoder feature matrix.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_PACK = ROOT / "VGAE" / "stage1_region_k5_lossfix" / "pack"
GRAPH = ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100" / "graph"
MARKED = ROOT / "ready_legnet" / "MARKED"

OUT_APPNP = ROOT / "VGAE" / "stage1_region_k5_appnp_lossfix"
OUT_GCNII = ROOT / "VGAE" / "stage1_region_k5_gcnii_lossfix"
OUT_MULTIK = ROOT / "VGAE" / "stage1_region_k5_multik457_lossfix"
OUT_STRUCT = ROOT / "VGAE" / "stage1_region_k5_structfeat_lossfix"

SUMMARY = ROOT / "VGAE" / "arch_appnp_gcnii_multik_struct_summary.json"


def _copy_pack(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "node_features.npz",
        "edges_weighted.npz",
        "ids.txt",
        "feature_meta.json",
    ):
        shutil.copy2(src / name, dst / name)


def _train(pack: Path, out: Path, architecture: str, **kw) -> dict:
    from src.splits.vgae.train import run_vgae_train

    print(f"[arch] === {architecture} → {out} ===", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    meta = run_vgae_train(
        pack=pack,
        out_dir=out,
        seed=42,
        device=None,
        loss_mode="homology_first",
        architecture=architecture,
        peak_ram_gib=10.0,
        wait_poll_sec=60.0,
        max_gpu_used_mib=2048.0,
        min_epochs=25,
        patience=10,
        max_epochs=200,
        **kw,
    )
    return {
        "architecture": architecture,
        "out": str(out.relative_to(ROOT)),
        "best_l_hom": meta.get("best_l_hom"),
        "best_l_hom_legacy": meta.get("best_l_hom_legacy"),
        "best_epoch": meta.get("best_epoch"),
        "n_features": meta.get("n_features"),
        "all_aggs": meta.get("all_aggs"),
    }


def main() -> int:
    if not (BASE_PACK / "feature_meta.json").is_file():
        raise FileNotFoundError(BASE_PACK)

    from src.splits.vgae.graph_data import (
        append_structural_features_to_pack,
        pack_region_graph,
    )
    from src.run.run_id.compare_vae_vgae_architectures import main as cmp_main

    summaries: list[dict] = []

    # 1) APPNP on legacy k5 pack
    _copy_pack(BASE_PACK, OUT_APPNP / "pack")
    summaries.append(
        _train(OUT_APPNP / "pack", OUT_APPNP, "appnp", appnp_k=10, appnp_alpha=0.1)
    )

    # 2) GCNII on legacy k5 pack
    _copy_pack(BASE_PACK, OUT_GCNII / "pack")
    summaries.append(
        _train(OUT_GCNII / "pack", OUT_GCNII, "gcnii", gcnii_layers=8)
    )

    # 3) Structural features appended to k5 pack → GCN
    struct_pack = OUT_STRUCT / "pack"
    print("[arch] building structural pack…", flush=True)
    append_structural_features_to_pack(BASE_PACK, struct_pack, n_cc_hash=8, seed=42)
    summaries.append(_train(struct_pack, OUT_STRUCT, "gcn"))

    # 4) Multi-k concat (4+5+7) light projection → GCN
    multik_pack = OUT_MULTIK / "pack"
    if not (multik_pack / "feature_meta.json").is_file():
        print("[arch] packing multi-k 4+5+7 (projected)…", flush=True)
        pack_region_graph(
            GRAPH,
            MARKED,
            multik_pack,
            k=5,
            feature_ks=(4, 5, 7),
            per_k_project_dim=256,
            project_seed=42,
        )
    else:
        print(f"[arch] reuse existing multik pack {multik_pack}", flush=True)
    summaries.append(_train(multik_pack, OUT_MULTIK, "gcn"))

    SUMMARY.write_text(json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8")
    print("[arch] summary →", SUMMARY, flush=True)
    cmp_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
