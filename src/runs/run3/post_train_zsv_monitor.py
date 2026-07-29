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

    from src.pipeline.zsv_eval import eval_zsv_from_train_outdir

    print("ZSV eval starting…", flush=True)
    result = eval_zsv_from_train_outdir(
        model="caduceus",
        outdir=outdir,
        split_root=split_root,
        device=0,
    )
    if result is None:
        raise RuntimeError("ZSV eval produced no metrics")
    print("ZSV metrics:", json.dumps(result.get("metrics", {}), sort_keys=True), flush=True)

    from src.pipeline.train import _finalize_train_artifacts

    _finalize_train_artifacts(outdir)

    from src.train_viz.train_monitor import refresh_train_monitor, refresh_pipeline_monitors

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
