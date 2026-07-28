#!/usr/bin/env python3
"""Re-runnable pipeline orchestrator (dry | run).

Template location: ``src/run/run_id/pipeline.py``.
Copy/adapt to ``src/run/<concrete_run_id>/pipeline.py`` per experiment.

Stage order (LOCKED):
  1. validate inputs
  2. /split          — src.pipeline.split_predict + src.pipeline.split
  3. /train          — src.pipeline.train (+ train_viz / caduceus TensorBoard)
  4. /adversarial    — src.pipeline.adversarial (optional)
  5. /train          — adversarial model on adversarial SPLIT (optional)

Prefer calling ``src.pipeline.*`` modules. Heavy training raises
``NotImplementedError`` until wired for a concrete run_id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


STAGE_ORDER = (
    "validate",
    "split",
    "train_direct",
    "adversarial",
    "train_adversarial",
)


def validate_inputs(args: argparse.Namespace) -> None:
    """Fail early when obligatory pipeline features are missing."""
    missing: list[str] = []
    if not args.data:
        missing.append("data")
    if not args.split:
        missing.append("split")
    if not args.train:
        missing.append("train")
    if not args.type:
        missing.append("type")
    if args.out_root is None:
        missing.append("out_root")
    if args.adversarial:
        if args.outdir_new is None:
            missing.append("outdir_new (required when --adversarial)")
        if not args.train_adversarial:
            missing.append("train_adversarial (required when --adversarial)")
    if missing:
        raise SystemExit(
            "Missing obligatory pipeline inputs: " + ", ".join(missing)
        )


def stage_split(args: argparse.Namespace) -> None:
    """Run random split-predict + materialize via src.pipeline.*."""
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.split import run_split

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    # Concrete wiring (id_csv, PARSED, PREDICT, strategy) belongs in a
    # run-specific copy of this stub. Keep import surface ready.
    _ = (run_split_predict, run_split, out)
    raise NotImplementedError(
        "stage_split: wire run_split_predict + run_split for this run_id "
        f"(data={args.data!r}, split={args.split!r})"
    )


def stage_train_direct(args: argparse.Namespace) -> None:
    """Direct /train via src.pipeline.train (+ viz)."""
    from src.pipeline.train import run_train
    from src.pipeline.train_viz import run_train_viz

    _ = (run_train, run_train_viz)
    if args.mode == "dry":
        print("[dry] skip full train_direct", flush=True)
        return
    raise NotImplementedError(
        "stage_train_direct: wire run_train / TensorBoard / optional ZSV "
        f"for model={args.train!r}"
    )


def stage_adversarial(args: argparse.Namespace) -> None:
    """Optional /adversarial combine + random re-split."""
    if not args.adversarial:
        print("[skip] adversarial not requested", flush=True)
        return
    from src.pipeline.adversarial import run_adversarial

    _ = run_adversarial
    raise NotImplementedError(
        "stage_adversarial: wire run_adversarial + random split for "
        f"outdir_new={args.outdir_new!r}"
    )


def stage_train_adversarial(args: argparse.Namespace) -> None:
    """Optional /train on adversarial SPLIT."""
    if not args.adversarial:
        print("[skip] train_adversarial not requested", flush=True)
        return
    from src.pipeline.train import run_train

    _ = run_train
    if args.mode == "dry":
        print("[dry] skip full train_adversarial", flush=True)
        return
    raise NotImplementedError(
        "stage_train_adversarial: wire run_train for "
        f"model={args.train_adversarial!r}"
    )


def run_pipeline(args: argparse.Namespace) -> int:
    print("pipeline mode=", args.mode, "stages=", " → ".join(STAGE_ORDER), flush=True)
    validate_inputs(args)
    if args.mode == "dry":
        print(
            "[dry] inputs OK; orchestrator importable. "
            "Generate/review child scripts; do not full-train.",
            flush=True,
        )
        # Import surface check only
        import src.pipeline.train  # noqa: F401
        import src.pipeline.adversarial  # noqa: F401
        import src.pipeline.split_predict  # noqa: F401
        import src.pipeline.split  # noqa: F401
        import src.pipeline.train_viz  # noqa: F401
        return 0

    stage_split(args)
    stage_train_direct(args)
    stage_adversarial(args)
    stage_train_adversarial(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pipeline orchestrator (dry|run). Stage order: validate → split → "
            "train_direct → adversarial → train_adversarial"
        )
    )
    p.add_argument(
        "--mode",
        choices=("dry", "run"),
        required=True,
        help="dry: validate/import only (no full train); run: execute stages",
    )
    p.add_argument("--data", default="", help="Data panel id (obligatory for run)")
    p.add_argument("--split", default="random", help="Split strategy id")
    p.add_argument(
        "--train",
        default="",
        help="Direct model: caduceus|legnet|human_legnet",
    )
    p.add_argument(
        "--train-adversarial",
        default="",
        help="Adversarial-branch model (obligatory when --adversarial)",
    )
    p.add_argument(
        "--type",
        default="regression",
        choices=("regression", "classification"),
        help="Prediction task type",
    )
    p.add_argument("--out-root", type=Path, default=None, help="Artifact root")
    p.add_argument(
        "--adversarial",
        action="store_true",
        help="Enable adversarial combine + adversarial train",
    )
    p.add_argument(
        "--outdir-new",
        type=Path,
        default=None,
        help="Adversarial panel destination (≠ source)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--zsv",
        action="store_true",
        help="Request final-model zero-shot-validation when ZSV trees exist",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
