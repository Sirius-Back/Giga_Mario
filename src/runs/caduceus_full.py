#!/usr/bin/env python3
"""Caduceus-full pipeline — thin orchestrator (no adapt; data already ready).

Reuses:
  src.splits.random.run_random_split   → splits/<strategy>/{M1,M2}
  src.caduceus.run                     → runs/caduceus/.../{logs,tensorboard,final_model}
  src.train_viz.viz.main               → figures/train-viz/...

Stages:
  1. /split   (M1=TPM, M2=predict M1 fold, stratified)
  2. /caduceus on M1 then M2
  3. /train-viz on trained run(s)

Re-run without subagents:
  python -m src.runs.caduceus_full --strategy random --seed 42 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

# --- package imports (canonical implementations) ---
from src.splits.random import run_random_split
from src.caduceus import DEFAULT_MODEL, run as run_caduceus
from src.train_viz.viz import main as run_train_viz


def _caduceus_args(
    *,
    root: Path,
    splits_dir: Path,
    out: Path,
    epochs: int,
    seed: int,
    max_length: int,
    batch_size: int,
    max_samples: int | None,
    model_name: str,
) -> Namespace:
    return Namespace(
        splits_dir=splits_dir,
        out=out,
        root=root,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=max(batch_size * 2, 4),
        lr=1e-4,
        max_length=max_length,
        seed=seed,
        num_workers=2,
        max_samples=max_samples,
        task="auto",
        num_labels=None,
    )


def run_pipeline(
    root: Path,
    *,
    strategy: str = "random",
    raw: Path = Path("raw"),
    ready: Path | None = None,
    seed: int = 42,
    epochs: int = 10,
    max_length: int = 8192,
    batch_size: int = 2,
    max_samples: int | None = None,
    model_name: str = DEFAULT_MODEL,
    skip_split: bool = False,
    skip_train: bool = False,
    skip_viz: bool = False,
    train_m2: bool = True,
) -> dict:
    """Execute split → caduceus(M1[,M2]) → train-viz. Returns path summary."""
    root = root.resolve()
    split_out = root / "splits" / strategy
    run_tag = f"{strategy}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    run_m1 = root / "runs" / "caduceus" / f"{run_tag}_M1"
    run_m2 = root / "runs" / "caduceus" / f"{run_tag}_M2"
    viz_m1 = root / "figures" / "train-viz" / f"{run_tag}_M1"
    viz_m2 = root / "figures" / "train-viz" / f"{run_tag}_M2"

    summary: dict = {
        "strategy": strategy,
        "seed": seed,
        "epochs": epochs,
        "split_out": str(split_out),
        "run_m1": str(run_m1),
        "run_m2": str(run_m2) if train_m2 else None,
        "viz_m1": str(viz_m1),
        "viz_m2": str(viz_m2) if train_m2 else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }

    # --- 1. split (M1 TPM + M2 stratified) ---
    if not skip_split:
        print("=== STAGE split ===", flush=True)
        if strategy != "random":
            raise SystemExit(
                f"Only strategy=random is wired in this runner; got {strategy!r}. "
                "Add src/splits/<id>.py and extend this script."
            )
        meta = run_random_split(
            root,
            raw_dir=raw,
            ready_dir=ready,
            out_dir=split_out,
            seed=seed,
            max_samples=max_samples,
        )
        summary["stages"].append({"stage": "split", "meta": meta})
        print(json.dumps({"split": "ok", "n": meta.get("n_samples")}, indent=2), flush=True)
    else:
        if not (split_out / "M1" / "train" / "labels.tsv").is_file():
            raise FileNotFoundError(f"--skip-split but missing {split_out}/M1/train/labels.tsv")
        summary["stages"].append({"stage": "split", "skipped": True})

    m1_dir = split_out / "M1"
    m2_dir = split_out / "M2"

    # --- 2. caduceus trains ---
    if not skip_train:
        print("=== STAGE caduceus M1 (TPM) ===", flush=True)
        rc = run_caduceus(
            _caduceus_args(
                root=root,
                splits_dir=m1_dir,
                out=run_m1,
                epochs=epochs,
                seed=seed,
                max_length=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
                model_name=model_name,
            )
        )
        if rc != 0:
            raise SystemExit(f"caduceus M1 failed with code {rc}")
        summary["stages"].append({"stage": "caduceus_M1", "out": str(run_m1), "rc": rc})

        if train_m2:
            print("=== STAGE caduceus M2 (predict M1 fold) ===", flush=True)
            if not (m2_dir / "train" / "labels.tsv").is_file():
                raise FileNotFoundError(f"M2 folds missing under {m2_dir}")
            rc2 = run_caduceus(
                _caduceus_args(
                    root=root,
                    splits_dir=m2_dir,
                    out=run_m2,
                    epochs=epochs,
                    seed=seed,
                    max_length=max_length,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    model_name=model_name,
                )
            )
            if rc2 != 0:
                raise SystemExit(f"caduceus M2 failed with code {rc2}")
            summary["stages"].append({"stage": "caduceus_M2", "out": str(run_m2), "rc": rc2})
    else:
        summary["stages"].append({"stage": "caduceus", "skipped": True})

    # --- 3. train-viz ---
    if not skip_viz:
        print("=== STAGE train-viz M1 ===", flush=True)
        rc_v = run_train_viz(
            [
                "--models",
                str(run_m1),
                "-o",
                str(viz_m1),
                "--title",
                f"Caduceus-full {strategy} M1 (TPM)",
            ]
        )
        if rc_v != 0:
            raise SystemExit(f"train-viz M1 failed with code {rc_v}")
        summary["stages"].append({"stage": "train_viz_M1", "out": str(viz_m1), "rc": rc_v})

        if train_m2 and (run_m2 / "logs" / "train_metrics.jsonl").is_file():
            print("=== STAGE train-viz M2 ===", flush=True)
            rc_v2 = run_train_viz(
                [
                    "--models",
                    str(run_m2),
                    "-o",
                    str(viz_m2),
                    "--title",
                    f"Caduceus-full {strategy} M2 (predict M1)",
                ]
            )
            if rc_v2 != 0:
                raise SystemExit(f"train-viz M2 failed with code {rc_v2}")
            summary["stages"].append(
                {"stage": "train_viz_M2", "out": str(viz_m2), "rc": rc_v2}
            )

            # comparison figure when both exist
            viz_cmp = root / "figures" / "train-viz" / f"{run_tag}_compare"
            print("=== STAGE train-viz compare M1 vs M2 ===", flush=True)
            rc_c = run_train_viz(
                [
                    "--models",
                    str(run_m1),
                    str(run_m2),
                    "-o",
                    str(viz_cmp),
                    "--title",
                    f"Caduceus-full {strategy} M1 vs M2",
                ]
            )
            summary["viz_compare"] = str(viz_cmp)
            summary["stages"].append(
                {"stage": "train_viz_compare", "out": str(viz_cmp), "rc": rc_c}
            )
    else:
        summary["stages"].append({"stage": "train_viz", "skipped": True})

    report = root / "docs" / "caduceus-full-report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Caduceus-full report",
                "",
                f"**Date:** {summary['created_at']}",
                f"**Strategy:** `{strategy}` (splits/{strategy}.md)",
                f"**Seed / epochs:** {seed} / {epochs}",
                "",
                "## Paths",
                "",
                f"- Split: `{split_out}` (M1=TPM, M2=predict M1 fold)",
                f"- Train M1: `{run_m1}`",
                f"- Train M2: `{run_m2 if train_m2 else 'skipped'}`",
                f"- Viz M1: `{viz_m1}`",
                f"- Viz M2: `{viz_m2 if train_m2 else 'skipped'}`",
                "",
                "## Notes",
                "",
                "- No `@adapt` — inputs are `raw/` + `ready/` (already adapted).",
                "- Orchestrator: `src/runs/caduceus_full.py` (import-only reuse of src.*).",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["report"] = str(report)
    (root / "runs" / "caduceus" / f"{run_tag}_pipeline.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "runs" / "caduceus" / f"{run_tag}_pipeline.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--strategy", default="random", help="splits/<id>.md id")
    ap.add_argument("--raw", type=Path, default=Path("raw"))
    ap.add_argument("--ready", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None, help="Smoke cap")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--no-m2", action="store_true", help="Skip M2 train+viz")
    args = ap.parse_args(argv)

    run_pipeline(
        args.root.resolve(),
        strategy=args.strategy,
        raw=args.raw,
        ready=args.ready,
        seed=args.seed,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        model_name=args.model_name,
        skip_split=args.skip_split,
        skip_train=args.skip_train,
        skip_viz=args.skip_viz,
        train_m2=not args.no_m2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
