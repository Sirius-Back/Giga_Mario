"""Parallel k=4 GCN-VAE train launcher (homology_first); then metrics + plot."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "VGAE" / "stage1_region_k4_pack" / "pack"
LOG_DIR = ROOT / "logs" / "vgae_k4_lhom"
METRICS_JSON = ROOT / "figures" / "presentation" / "lhom_balanced_robust_k4_metrics.json"

ARCHES = [
    ("GCN-VAE", "gcn", "gcn", {}),
    ("GAT-VAE", "gat", "gat", {"gat_heads": 4}),
    ("SAGE-VAE", "sage", "sage", {}),
    ("GCL-VAE", "gcl", "gcl", {}),
    ("APPNP-VAE", "appnp", "appnp", {"appnp_k": 10, "appnp_alpha": 0.1}),
    ("GCNII-VAE", "gcnii", "gcnii", {"gcnii_layers": 8}),
]

BASELINES = [
    ("Random", "random", ROOT / "runs_unif" / "legnet" / "run2_legnet_random"),
    (
        "Pangenome",
        "pangenome",
        ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100",
    ),
    ("MLP-VAE (k=4)", "vae", ROOT / "VAE" / "mlp_vae_kmer_k4_lossfix"),
]


def _copy_pack(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "node_features.npz",
        "edges_weighted.npz",
        "ids.txt",
        "feature_meta.json",
    ):
        shutil.copy2(PACK / name, dst / name)


def _spawn(
    label: str, arch: str, out: Path, kw: dict, *, device: str
) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{arch}.log"
    _copy_pack(out / "pack")
    kw_s = "".join(
        f", {k}={v!r}" if isinstance(v, str) else f", {k}={v}" for k, v in kw.items()
    )
    # Explicit physical GPU; do not remap CUDA_VISIBLE_DEVICES (breaks job_queue GPU ids).
    # max_gpu_used_mib high enough to allow our own train after claim; device is pinned.
    code = (
        "from pathlib import Path\n"
        "from src.splits.vgae.train import run_vgae_train\n"
        f"run_vgae_train(pack=Path({str(out / 'pack')!r}), out_dir=Path({str(out)!r}), "
        f"seed=42, device={device!r}, loss_mode='homology_first', "
        f"architecture={arch!r}, peak_ram_gib=10.0, wait_poll_sec=30.0, "
        "max_gpu_used_mib=32000.0, min_epochs=25, patience=10, max_epochs=200, "
        "register_queue=False"
        f"{kw_s})\n"
    )
    fh = open(log, "w", encoding="utf-8")
    print(f"[k4] spawn {label} device={device} log={log}", flush=True)
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def main() -> int:
    from src.pipeline.mem_guard import kill_cursor_indexers
    from src.run.run_id.eval_vgae_legacy_losses import eval_run

    if not (PACK / "feature_meta.json").is_file():
        raise FileNotFoundError(PACK)

    # Two waves of 4 GPUs: first four arches, then remaining
    wave1 = ARCHES[:4]
    wave2 = ARCHES[4:]
    outs: dict[str, Path] = {}
    for label, family, arch, kw in ARCHES:
        outs[arch] = ROOT / "VGAE" / f"stage1_region_k4_{arch}_lossfix"

    def _run_wave(wave: list, gpu_offset: int = 0) -> None:
        procs: dict[str, subprocess.Popen] = {}
        for i, (label, _family, arch, kw) in enumerate(wave):
            out = outs[arch]
            if (out / "split.csv").is_file() and (out / "train_meta.json").is_file():
                print(f"[k4] skip train {label} (exists)", flush=True)
                continue
            kill_cursor_indexers(min_used_fraction=0.94)
            device = f"cuda:{i + gpu_offset}"
            procs[arch] = _spawn(label, arch, out, kw, device=device)
        for arch, p in procs.items():
            rc = p.wait()
            kill_cursor_indexers(min_used_fraction=0.94)
            print(f"[k4] {arch} exit={rc}", flush=True)

    _run_wave(wave1, gpu_offset=0)
    _run_wave(wave2, gpu_offset=0)

    rows = []
    for label, family, path in BASELINES:
        kill_cursor_indexers(min_used_fraction=0.94)
        r = eval_run(path)
        rows.append(
            {
                "label": label,
                "family": family,
                "architecture": family,
                "source": str(path.relative_to(ROOT)),
                "status": r.get("status"),
                "all_aggs": r.get("all_aggs"),
                "n_regions": r.get("n_regions"),
            }
        )
    for label, family, arch, _kw in ARCHES:
        out = outs[arch]
        kill_cursor_indexers(min_used_fraction=0.94)
        r = eval_run(out)
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

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(
        json.dumps(
            {
                "k_feature": 4,
                "metrics": {
                    "balanced": "weighted (√n)",
                    "robust": "robust (sd/√n winsorized)",
                },
                "rows": rows,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[k4] metrics → {METRICS_JSON}", flush=True)

    from src.run.presentation.plot_lhom_balanced_robust_k4 import main as plot_main

    return plot_main(["--metrics-json", str(METRICS_JSON)])


if __name__ == "__main__":
    raise SystemExit(main())
