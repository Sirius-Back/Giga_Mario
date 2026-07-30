"""Finish run8 after adv train: ZSV (skip if classification) + viz + pipeline_done.

Does not retrain. Assumes ``adversarial/train/final_model`` exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run8_2mer_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    adv_train = OUT_ROOT / "adversarial" / "train"
    weights = adv_train / "final_model" / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing adv final_model: {weights}")

    from src.pipeline.pipeline_viz import run_pipeline_viz_auto
    from src.pipeline.zsv_eval import eval_zsv_from_train_outdir

    print("run8 finish: ZSV (classification → skip stub) …", flush=True)
    result = eval_zsv_from_train_outdir(
        model="caduceus",
        outdir=adv_train,
        split_root=OUT_ROOT / "adversarial",
    )
    if result is None:
        raise RuntimeError("ZSV produced no result (trees missing?)")
    print(
        f"ZSV skipped={result.get('skipped')} reason={result.get('reason')}",
        flush=True,
    )

    print("run8 finish: adversarial viz …", flush=True)
    try:
        run_pipeline_viz_auto(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            train_dir=adv_train,
            run_id=RUN_ID,
            seed=42,
            plot_train=True,
            plot_sbs=False,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: adv viz skipped: {type(exc).__name__}: {exc}", flush=True)

    done = {
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "direct": str(OUT_ROOT / "direct"),
        "adversarial": str(adv_train),
        "adv_best_epoch": 9,
        "adv_zsv": result,
        "note": "Finished after adv train; classification ZSV skipped by design",
    }
    path = OUT_ROOT / "pipeline_done.json"
    path.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    print(f"pipeline_done={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
