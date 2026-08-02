"""Parallel launcher: APPNP+GCNII on free GPUs; struct+multik packs then GCN.

Does not kill foreign jobs. Uses caduceus_env via the invoking python.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_PACK = ROOT / "VGAE" / "stage1_region_k5_lossfix" / "pack"
GRAPH = ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100" / "graph"
MARKED = ROOT / "ready_legnet" / "MARKED"
LOG_DIR = ROOT / "logs" / "vgae_arch_ab"
SUMMARY = ROOT / "VGAE" / "arch_appnp_gcnii_multik_struct_summary.json"

OUTS = {
    "appnp": ROOT / "VGAE" / "stage1_region_k5_appnp_lossfix",
    "gcnii": ROOT / "VGAE" / "stage1_region_k5_gcnii_lossfix",
    "structfeat": ROOT / "VGAE" / "stage1_region_k5_structfeat_lossfix",
    "multik457": ROOT / "VGAE" / "stage1_region_k5_multik457_lossfix",
}


def _copy_pack(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "node_features.npz",
        "edges_weighted.npz",
        "ids.txt",
        "feature_meta.json",
    ):
        shutil.copy2(src / name, dst / name)


def _train_one(
    name: str, arch: str, out: Path, pack: Path, extra: list[str]
) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{name}.log"
    extra_s = (", " + ", ".join(extra)) if extra else ""
    cmd = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path\n"
            "from src.splits.vgae.train import run_vgae_train\n"
            f"run_vgae_train(pack=Path({str(pack)!r}), out_dir=Path({str(out)!r}), "
            "seed=42, device=None, loss_mode='homology_first', "
            f"architecture={arch!r}, peak_ram_gib=10.0, wait_poll_sec=60.0, "
            "max_gpu_used_mib=2048.0, min_epochs=25, patience=10, max_epochs=200"
            f"{extra_s})\n"
        ),
    ]
    out.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w", encoding="utf-8")
    print(f"[launch] {name} arch={arch} log={log}", flush=True)
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def main() -> int:
    if not (BASE_PACK / "feature_meta.json").is_file():
        raise FileNotFoundError(BASE_PACK)

    from src.splits.vgae.graph_data import (
        append_structural_features_to_pack,
        pack_region_graph,
    )

    # Prepare APPNP / GCNII packs (reuse k5 features)
    _copy_pack(BASE_PACK, OUTS["appnp"] / "pack")
    _copy_pack(BASE_PACK, OUTS["gcnii"] / "pack")

    # Structural pack (skip rebuild if already present)
    struct_meta = OUTS["structfeat"] / "pack" / "feature_meta.json"
    if struct_meta.is_file():
        import json as _json

        meta0 = _json.loads(struct_meta.read_text(encoding="utf-8"))
        if meta0.get("structural_features"):
            print(f"[launch] reuse structural pack {OUTS['structfeat']/ 'pack'}", flush=True)
        else:
            print("[launch] structural pack…", flush=True)
            append_structural_features_to_pack(
                BASE_PACK, OUTS["structfeat"] / "pack", n_cc_hash=8, seed=42
            )
    else:
        print("[launch] structural pack…", flush=True)
        append_structural_features_to_pack(
            BASE_PACK, OUTS["structfeat"] / "pack", n_cc_hash=8, seed=42
        )

    # Start APPNP + GCNII + struct GCN in parallel (GPU waiter inside train)
    procs: dict[str, subprocess.Popen] = {}
    procs["appnp"] = _train_one(
        "appnp",
        "appnp",
        OUTS["appnp"],
        OUTS["appnp"] / "pack",
        ["appnp_k=10", "appnp_alpha=0.1"],
    )
    procs["gcnii"] = _train_one(
        "gcnii",
        "gcnii",
        OUTS["gcnii"],
        OUTS["gcnii"] / "pack",
        ["gcnii_layers=8"],
    )
    procs["structfeat"] = _train_one(
        "structfeat", "gcn", OUTS["structfeat"], OUTS["structfeat"] / "pack", []
    )

    # Multi-k pack while GPUs train (CPU-bound)
    mk_pack = OUTS["multik457"] / "pack"
    if not (mk_pack / "feature_meta.json").is_file():
        print("[launch] multi-k pack 4+5+7…", flush=True)
        pack_region_graph(
            GRAPH,
            MARKED,
            mk_pack,
            k=5,
            feature_ks=(4, 5, 7),
            per_k_project_dim=256,
            project_seed=42,
        )
    else:
        print(f"[launch] reuse multik pack {mk_pack}", flush=True)

    procs["multik457"] = _train_one(
        "multik457", "gcn", OUTS["multik457"], mk_pack, []
    )

    # Wait all
    rc = 0
    for name, p in procs.items():
        code = p.wait()
        print(f"[launch] {name} exit={code}", flush=True)
        if code != 0:
            rc = code

    # Summaries from train_meta
    rows = []
    for key, out in OUTS.items():
        meta_p = out / "train_meta.json"
        if not meta_p.is_file():
            rows.append({"run": key, "status": "MISSING"})
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": key,
                "architecture": meta.get("architecture"),
                "best_l_hom": meta.get("best_l_hom"),
                "best_l_hom_legacy": meta.get("best_l_hom_legacy"),
                "best_epoch": meta.get("best_epoch"),
                "path": str(out.relative_to(ROOT)),
            }
        )
    SUMMARY.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("[launch] summary →", SUMMARY, flush=True)

    try:
        from src.run.run_id.compare_vae_vgae_architectures import main as cmp_main

        cmp_main()
    except Exception as exc:  # noqa: BLE001
        print(f"[launch] compare failed: {exc}", flush=True)

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
