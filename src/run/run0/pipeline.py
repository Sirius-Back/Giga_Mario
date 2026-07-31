#!/usr/bin/env python3
"""Re-runnable run0 pipeline: split → direct train → adversarial → train."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import Counter
from pathlib import Path

from src.pipeline.common import read_csv
from src.run.run0.prokaryotes_random_adversarial import run as run_adversarial
from src.run.run0.prokaryotes_random_legnet_adversarial import run as run_adv_train
from src.run.run0.prokaryotes_random_legnet_direct import run as run_direct

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run0"
PANEL_ROOT = ROOT / "run" / RUN_ID
RESULT_ROOT = PANEL_ROOT
STATUS_PATH = ROOT / RUN_ID / "pipeline_status.md"


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required non-empty input is missing: {path}")


def _require_continuous_prepare() -> None:
    """Gate training on the completed continuous-target prepare panel."""
    status_path = ROOT / RUN_ID / "prepare_status.md"
    scale_path = PANEL_ROOT / "TARGET" / "get_mpra_scale.json"
    _require_nonempty(status_path)
    _require_nonempty(scale_path)
    status_text = status_path.read_text(encoding="utf-8")
    if "**Status:** FAILED" in status_text:
        raise RuntimeError("Prepare failed; refusing to use an incomplete panel.")
    if "**Status:** DONE" not in status_text:
        raise RuntimeError("Prepare is not DONE; wait before running the pipeline.")
    try:
        scale = json.loads(scale_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid continuous-target metadata: {scale_path}") from exc
    if scale.get("mode") != "continuous":
        raise ValueError(
            "run0 requires continuous log2 targets: "
            f"{scale_path} reports mode={scale.get('mode')!r}."
        )


def _write_status(status: str, *, commands: list[str], issue: str = "") -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = [
        f"# Pipeline status: {status}",
        "",
        f"- **run_id:** `{RUN_ID}`",
        f"- **results root:** `run/run0/`",
        "- **ratio:** train:test:val = `1:1:3`",
        "- **seed:** `42`",
        "- **GPUs:** `CUDA_VISIBLE_DEVICES=0,1,2,3`",
        "- **LegNet batches:** `1024` per process (existing human_legnet default; "
        "the largest documented in-repository setting for 230 bp inputs)",
        "- **train logs:** `run/run0/direct/logs/` and "
        "`run/run0/adversarial/train/logs/` when training completes",
        "- **ZSV:** retained by split under `PREDICT/` and `PARSED/` "
        "`zero-shot-validation/`; current `src.legnet` has no external final-model "
        "ZSV evaluator.",
        "",
        "## Commands",
        *[f"- `{command}`" for command in commands],
    ]
    if issue:
        text += ["", "## Issues", f"- {issue}"]
    STATUS_PATH.write_text("\n".join(text) + "\n", encoding="utf-8")


def _verify_ratios(split_csv: Path) -> Counter[str]:
    counts = Counter(
        row["train_test"] for row in read_csv(split_csv) if row["train_test"] != "zsv"
    )
    if not counts or counts["train"] + counts["test"] + counts["val"] == 0:
        raise ValueError(f"No train/test/val records in {split_csv}")
    total = counts["train"] + counts["test"] + counts["val"]
    expected = (total / 5, total / 5, 3 * total / 5)
    observed = (counts["train"], counts["test"], counts["val"])
    if any(abs(actual - target) > 1 for actual, target in zip(observed, expected)):
        raise RuntimeError(
            "Custom-ratio verification failed: "
            f"observed {observed}, expected within one record of {expected}"
        )
    return counts


def run_pipeline(*, mode: str, task_type: str) -> int:
    commands = [
        "python run0/pipeline.py --mode dry --type regression",
        "CUDA_VISIBLE_DEVICES=0,1,2,3 python run0/pipeline.py --mode run --type regression",
    ]
    try:
        if task_type != "regression":
            raise ValueError("run0 is locked to --type regression.")
        for required in ("parse.md", "fold.csv", "ID.csv"):
            _require_nonempty(PANEL_ROOT / required)
        _require_continuous_prepare()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        direct_tsv = run_direct(
            panel_root=PANEL_ROOT,
            out_root=RESULT_ROOT,
            epochs=3,
            run_training=mode == "run",
        )
        direct_counts = _verify_ratios(RESULT_ROOT / "split.csv")
        adversarial_root = RESULT_ROOT / "adversarial"
        run_adversarial(
            panel_root=PANEL_ROOT,
            direct_root=RESULT_ROOT,
            adversarial_root=adversarial_root,
        )
        adv_tsv = run_adv_train(
            adversarial_root=adversarial_root,
            epochs=3,
            run_training=mode == "run",
        )
        adv_counts = _verify_ratios(adversarial_root / "split.csv")
        _write_status(
            "DONE" if mode == "run" else "PARTIAL",
            commands=commands,
            issue=(
                "Dry run completed without training. Prepared target mode=continuous; "
                f"non-ZSV ratio counts direct={dict(direct_counts)}, adversarial={dict(adv_counts)}; "
                f"TSVs: {direct_tsv}, {adv_tsv}."
                if mode == "dry"
                else ""
            ),
        )
        return 0
    except Exception as exc:
        _write_status("FAILED", commands=commands, issue=f"{type(exc).__name__}: {exc}")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry", "run"), required=True)
    parser.add_argument("--type", choices=("regression",), required=True)
    args = parser.parse_args(argv)
    try:
        return run_pipeline(mode=args.mode, task_type=args.type)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
