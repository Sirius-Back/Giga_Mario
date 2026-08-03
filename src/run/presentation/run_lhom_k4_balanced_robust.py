"""Build k=4 metrics JSON (VAE + random + pangenome + all GCN-VAE Stage1 k4) and plot.

1. Score baselines (VAE k4, random, pure pangenome).
2. Pack Stage1 region graph with ``feature_k=4`` (reuse k5 contingency topology).
3. Train GCN-family VGAE encoders (homology_first) on that pack.
4. Write metrics JSON + call the presentation plotter.

Usage:
  conda run -n caduceus_env python -m src.run.presentation.run_lhom_k4_balanced_robust
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100" / "graph"
MARKED = ROOT / "ready_legnet" / "MARKED"
PACK = ROOT / "VGAE" / "stage1_region_k4_pack" / "pack"
OUT_ROOT = ROOT / "VGAE"
LOG_DIR = ROOT / "logs" / "vgae_k4_lhom"
METRICS_JSON = ROOT / "figures" / "presentation" / "lhom_balanced_robust_k4_metrics.json"

# All GCN-family VGAE encoders to train at k=4
ARCH_RUNS: list[tuple[str, str, str, dict[str, Any]]] = [
    # label, family, architecture, train kwargs
    ("GCN-VAE", "gcn", "gcn", {}),
    ("GAT-VAE", "gat", "gat", {"gat_heads": 4}),
    ("SAGE-VAE", "sage", "sage", {}),
    ("GCL-VAE", "gcl", "gcl", {}),
    ("APPNP-VAE", "appnp", "appnp", {"appnp_k": 10, "appnp_alpha": 0.1}),
    ("GCNII-VAE", "gcnii", "gcnii", {"gcnii_layers": 8}),
]

BASELINES: list[tuple[str, str, Path]] = [
    ("Random", "random", ROOT / "runs_unif" / "legnet" / "run2_legnet_random"),
    (
        "Pangenome",
        "pangenome",
        ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100",
    ),
    ("MLP-VAE (k=4)", "vae", ROOT / "VAE" / "mlp_vae_kmer_k4_lossfix"),
]


def _kill_indexers() -> None:
    from src.pipeline.mem_guard import kill_cursor_indexers, ram_snapshot

    snap = ram_snapshot()
    killed = kill_cursor_indexers(min_used_fraction=0.94)
    print(f"[k4] RAM used={snap['mem_used_pct']:.1f}% killed={killed}", flush=True)


def _eval(path: Path) -> dict[str, Any]:
    from src.run.run_id.eval_vgae_legacy_losses import eval_run

    return eval_run(Path(path))


def _ensure_pack() -> Path:
    meta = PACK / "feature_meta.json"
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if int(m.get("n_features") or 0) == 257 or m.get("feature_k") == 4:
            print(f"[k4] reuse pack {PACK} n_features={m.get('n_features')}", flush=True)
            return PACK
    from src.splits.vgae.graph_data import pack_region_graph

    print("[k4] packing feature_k=4 …", flush=True)
    pack_region_graph(
        GRAPH,
        MARKED,
        PACK,
        k=5,
        feature_k=4,
        project_dim=None,
        project_seed=42,
    )
    return PACK


def _train_arch(label: str, arch: str, out: Path, pack: Path, kw: dict[str, Any]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{arch}.log"
    out.mkdir(parents=True, exist_ok=True)
    pack_dst = out / "pack"
    pack_dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "node_features.npz",
        "edges_weighted.npz",
        "ids.txt",
        "feature_meta.json",
    ):
        shutil.copy2(pack / name, pack_dst / name)

    kw_s = ""
    for k, v in kw.items():
        kw_s += f", {k}={v!r}" if isinstance(v, str) else f", {k}={v}"
    code = (
        "from pathlib import Path\n"
        "from src.splits.vgae.train import run_vgae_train\n"
        f"run_vgae_train(pack=Path({str(pack_dst)!r}), out_dir=Path({str(out)!r}), "
        "seed=42, device=None, loss_mode='homology_first', "
        f"architecture={arch!r}, peak_ram_gib=10.0, wait_poll_sec=60.0, "
        "max_gpu_used_mib=2048.0, min_epochs=25, patience=10, max_epochs=200"
        f"{kw_s})\n"
    )
    print(f"[k4] train {label} → {out} log={log}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    return int(proc.returncode)


def main() -> int:
    _kill_indexers()
    rows: list[dict[str, Any]] = []

    # Baselines
    for label, family, path in BASELINES:
        print(f"[k4] eval baseline {label} …", flush=True)
        r = _eval(path)
        rows.append(
            {
                "label": label,
                "family": family,
                "architecture": family,
                "source": str(path.relative_to(ROOT)),
                "status": r.get("status"),
                "reason": r.get("reason"),
                "all_aggs": r.get("all_aggs"),
                "n_regions": r.get("n_regions"),
            }
        )
        _kill_indexers()

    pack = _ensure_pack()
    _kill_indexers()

    # Train + eval each GCN-VAE arch
    for label, family, arch, kw in ARCH_RUNS:
        out = OUT_ROOT / f"stage1_region_k4_{arch}_lossfix"
        split = out / "split.csv"
        if not split.is_file():
            rc = _train_arch(label, arch, out, pack, kw)
            if rc != 0:
                rows.append(
                    {
                        "label": label,
                        "family": family,
                        "architecture": arch,
                        "source": str(out.relative_to(ROOT)),
                        "status": "FAILED",
                        "reason": f"train_exit={rc}",
                    }
                )
                _kill_indexers()
                continue
        print(f"[k4] eval {label} …", flush=True)
        r = _eval(out)
        rows.append(
            {
                "label": label,
                "family": family,
                "architecture": arch,
                "source": str(out.relative_to(ROOT)),
                "status": r.get("status"),
                "reason": r.get("reason"),
                "all_aggs": r.get("all_aggs"),
                "n_regions": r.get("n_regions"),
                "reported_best_l_hom": r.get("reported_best_l_hom"),
            }
        )
        _kill_indexers()

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k_feature": 4,
        "metrics": {
            "balanced": "weighted (√n size-weighted L_hom)",
            "robust": "robust (winsorized mean of sd/√n)",
        },
        "rows": rows,
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[k4] wrote {METRICS_JSON}", flush=True)

    from src.run.presentation.plot_lhom_balanced_robust_k4 import main as plot_main

    return plot_main(["--metrics-json", str(METRICS_JSON)])


if __name__ == "__main__":
    raise SystemExit(main())
