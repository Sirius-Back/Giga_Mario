"""Stage1 A/B: homology_robust + homology_log_balance on k=5 pack (reuse).

Also writes cross-agg legacy comparison + SD balance for each best split.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACK_SRC = ROOT / "VGAE" / "stage1_region_k5_lossfix" / "pack"
OUTS = {
    "homology_robust": ROOT / "VGAE" / "stage1_region_k5_hom_robust",
    "homology_log_balance": ROOT / "VGAE" / "stage1_region_k5_hom_logbal",
}
COMPARE = ROOT / "VGAE" / "stage1_loss_agg_comparison.json"


def _pick_device() -> str:
    import torch
    from src.splits.vgae.train import _pick_free_gpu

    idx = _pick_free_gpu(max_used_mib=4096.0)
    if idx is not None:
        return f"cuda:{idx}"
    best, best_used = None, None
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1024**2)
        if best_used is None or used < best_used:
            best_used, best = used, i
    if best is None or best_used > 8000:
        raise RuntimeError(f"no usable GPU (best used_mib={best_used})")
    return f"cuda:{best}"


def _copy_pack(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "node_features.npz",
        "edges_weighted.npz",
        "ids.txt",
        "feature_meta.json",
    ):
        src = PACK_SRC / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst / name)


def main() -> int:
    if not (PACK_SRC / "feature_meta.json").is_file():
        raise FileNotFoundError(f"missing k5 pack: {PACK_SRC}")

    from src.splits.vgae.train import run_vgae_train
    from src.run.run_id.eval_vgae_legacy_losses import eval_run

    # Wait for a free GPU via train.resolve_device (do not force a busy index)
    device = None
    print("[agg-ab] device=auto (wait for free GPU)", flush=True)
    summaries = []
    for mode, out in OUTS.items():
        print(f"[agg-ab] === {mode} → {out} ===", flush=True)
        _copy_pack(out / "pack")
        meta = run_vgae_train(
            pack=out / "pack",
            out_dir=out,
            seed=42,
            device=device,
            loss_mode=mode,
            peak_ram_gib=8.0,
            wait_poll_sec=120.0,
            max_gpu_used_mib=512.0,
            min_epochs=25,
            patience=10,
            max_epochs=200,
        )
        print(
            "[agg-ab]",
            mode,
            json.dumps(
                {
                    k: meta.get(k)
                    for k in (
                        "best_l_hom",
                        "best_l_hom_legacy",
                        "best_epoch",
                        "final_l_hom_legacy",
                        "hom_agg",
                        "all_aggs",
                    )
                },
                default=str,
            )[:1200],
            flush=True,
        )
        summaries.append({"mode": mode, "out": str(out), "meta": meta})

    # Re-score all Stage1 runs (incl. baselines) under every agg
    existing = []
    for p in sorted((ROOT / "VGAE").glob("stage1_region_k5*/")):
        if (p / "split.csv").is_file():
            existing.append(eval_run(p))
    payload = {
        "new_trains": [
            {
                "mode": s["mode"],
                "best_train_l_hom": s["meta"].get("best_l_hom"),
                "best_legacy_l_hom": s["meta"].get("best_l_hom_legacy"),
                "all_aggs": s["meta"].get("all_aggs"),
                "sd_balance": s["meta"].get("sd_balance"),
            }
            for s in summaries
        ],
        "all_stage1_k5_rescored": existing,
    }
    COMPARE.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[agg-ab] comparison → {COMPARE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
