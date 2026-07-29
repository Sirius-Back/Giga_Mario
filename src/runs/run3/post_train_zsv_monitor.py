"""Finish run3 post-train: ZSV eval + train_monitor (after early-stop train)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    outdir = ROOT / "runs" / "run3" / "direct"
    split_root = ROOT / "runs" / "run3"
    model_dir = outdir / "final_model"
    if not (model_dir / "config.json").is_file():
        model_dir = outdir / "best_model"

    from src.pipeline.zsv_eval import _resolve_zsv_trees, eval_caduceus_zsv

    trees = _resolve_zsv_trees(split_root)
    if trees is None:
        raise FileNotFoundError(f"ZSV trees missing under {split_root}")
    parsed, predict = trees

    # Inference-only: train used bs=480 (tight on 32GB); ZSV uses a safer batch.
    zsv_batch = int(os.environ.get("ZSV_BATCH_SIZE", "96"))
    print(f"ZSV eval starting (batch={zsv_batch}, max_length=208)…", flush=True)
    result = eval_caduceus_zsv(
        model_dir=model_dir,
        parsed_root=parsed,
        predict_root=predict,
        out_json=outdir / "logs" / "zero_shot_metrics.json",
        batch_size=zsv_batch,
        max_length=208,
        device=0,
        amp=True,
    )
    print("ZSV metrics:", json.dumps(result.get("metrics", {}), sort_keys=True), flush=True)

    from src.pipeline.train import _finalize_train_artifacts

    _finalize_train_artifacts(outdir)

    from src.train_viz.train_monitor import refresh_pipeline_monitors, refresh_train_monitor

    mon = refresh_train_monitor(
        outdir,
        model="run3_direct",
        title="run3 direct — train monitor",
        include_split_compare=True,
    )
    print(f"train_monitor status={mon.get('status')} → {mon.get('outdir')}", flush=True)
    pipe = refresh_pipeline_monitors(
        split_root, run_id="run3", include_split_compare=True
    )
    print(f"pipeline_monitors status={pipe.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
