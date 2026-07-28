#!/usr/bin/env python3
"""Caduceus-full pipeline — thin orchestrator (no adapt; data already ready).

Reuses:
  src.splits.random.run_random_split   → <out-root>/splits/{M1,M2,zero_shot}
  src.caduceus.run                     → <out-root>/runs/{M1,M2}
  src.train_viz.viz.main               → <out-root>/figures/...

Default layout (this run):
  output/random/{splits,runs,figures,zs_eval,report.md}

Re-run:
  python -m src.runs.caduceus_full \\
    --strategy random --out-root output/random \\
    --epochs-m1 10 --epochs-m2 5 \\
    --zs-genomes GCF_000001405.40
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.splits.random import run_random_split
from src.caduceus import (
    DEFAULT_MODEL,
    SplitWindowDataset,
    collate_pad,
    evaluate_regression,
    run as run_caduceus,
)
from src.train_viz.viz import main as run_train_viz

# Alias → RefSeq assembly id for zero-shot holdout
ZS_ALIASES = {
    "human": "GCF_000001405.40",
    "hsapiens": "GCF_000001405.40",
    "homo_sapiens": "GCF_000001405.40",
}


def _resolve_zs_genomes(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for g in raw:
        key = g.strip()
        out.append(ZS_ALIASES.get(key.lower(), key))
    return out


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


def _train_caduceus(
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
    nproc: int,
) -> int:
    """Train via torchrun (multi-GPU) or in-process src.caduceus.run."""
    if nproc > 1 and torch.cuda.is_available():
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            "-m",
            "src.caduceus",
            "--splits-dir",
            str(splits_dir),
            "--out",
            str(out),
            "--root",
            str(root),
            "--model-name",
            model_name,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--eval-batch-size",
            str(max(batch_size * 2, 4)),
            "--max-length",
            str(max_length),
            "--seed",
            str(seed),
        ]
        if max_samples is not None:
            cmd.extend(["--max-samples", str(max_samples)])
        print(f"[train] {' '.join(cmd)}", flush=True)
        return int(subprocess.call(cmd, cwd=str(root)))
    return int(
        run_caduceus(
            _caduceus_args(
                root=root,
                splits_dir=splits_dir,
                out=out,
                epochs=epochs,
                seed=seed,
                max_length=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
                model_name=model_name,
            )
        )
    )


def _eval_zero_shot_m1(
    *,
    model_dir: Path,
    zs_dir: Path,
    out_dir: Path,
    max_length: int,
    batch_size: int,
    max_samples: int | None,
) -> dict:
    """Evaluate saved M1 TPM model on zero_shot/all (reuse caduceus eval helpers)."""
    labels = zs_dir / "all" / "labels.tsv"
    if not labels.is_file():
        raise FileNotFoundError(f"ZS labels missing: {labels}")
    ckpt = model_dir / "final_model"
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(f"M1 final_model missing under {ckpt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
    )
    model.to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 4

    ds = SplitWindowDataset(
        zs_dir,
        "all",
        tokenizer,
        max_length,
        task="regression",
        max_samples=max_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )
    metrics = evaluate_regression(model, loader, device, torch.nn.MSELoss())
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "zs_dir": str(zs_dir),
        "n_eval": metrics.get("n"),
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_pipeline(
    root: Path,
    *,
    strategy: str = "random",
    out_root: Path = Path("output") / "random",
    raw: Path = Path("raw"),
    ready: Path | None = None,
    seed: int = 42,
    epochs_m1: int = 10,
    epochs_m2: int = 5,
    max_length: int = 8192,
    batch_size: int = 2,
    max_samples: int | None = None,
    model_name: str = DEFAULT_MODEL,
    zs_genomes: list[str] | None = None,
    nproc: int = 1,
    skip_split: bool = False,
    skip_train: bool = False,
    skip_viz: bool = False,
    skip_zs: bool = False,
    train_m2: bool = True,
) -> dict:
    """Execute split → caduceus(M1[,M2]) → ZS eval → train-viz under out_root."""
    root = root.resolve()
    out_root = out_root if out_root.is_absolute() else root / out_root
    out_root = out_root.resolve()
    split_out = out_root / "splits"
    run_m1 = out_root / "runs" / "M1"
    run_m2 = out_root / "runs" / "M2"
    viz_m1 = out_root / "figures" / "M1"
    viz_m2 = out_root / "figures" / "M2"
    viz_cmp = out_root / "figures" / "compare"
    zs_eval_dir = out_root / "zs_eval"
    zs_genomes = _resolve_zs_genomes(zs_genomes)

    # Confirm ready/raw before any stage
    ready_path = ready if ready is not None else Path("ready")
    ready_abs = ready_path if ready_path.is_absolute() else root / ready_path
    raw_abs = raw if raw.is_absolute() else root / raw
    if not ready_abs.exists():
        raise FileNotFoundError(
            f"ready data missing at {ready_abs} — caduceus-full does not run @adapt"
        )
    if not raw_abs.exists():
        raise FileNotFoundError(f"raw/ missing at {raw_abs}")

    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "strategy": strategy,
        "seed": seed,
        "epochs_m1": epochs_m1,
        "epochs_m2": epochs_m2,
        "out_root": str(out_root),
        "split_out": str(split_out),
        "run_m1": str(run_m1),
        "run_m2": str(run_m2) if train_m2 else None,
        "viz_m1": str(viz_m1),
        "viz_m2": str(viz_m2) if train_m2 else None,
        "zs_genomes": zs_genomes,
        "nproc": nproc,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }

    # --- 1. split (M1 TPM + M2 stratified + optional ZS holdout) ---
    if not skip_split:
        print("=== STAGE split ===", flush=True)
        if strategy != "random":
            raise SystemExit(
                f"Only strategy=random is wired in this runner; got {strategy!r}."
            )
        meta = run_random_split(
            root,
            raw_dir=raw,
            ready_dir=ready,
            out_dir=split_out,
            seed=seed,
            max_samples=max_samples,
            holdout_genomes=zs_genomes or None,
        )
        summary["stages"].append({"stage": "split", "meta": meta})
        print(
            json.dumps(
                {
                    "split": "ok",
                    "n": meta.get("n_samples"),
                    "n_zero_shot": meta.get("n_zero_shot"),
                    "holdout": meta.get("holdout_genomes"),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        if not (split_out / "M1" / "train" / "labels.tsv").is_file():
            raise FileNotFoundError(
                f"--skip-split but missing {split_out}/M1/train/labels.tsv"
            )
        summary["stages"].append({"stage": "split", "skipped": True})

    m1_dir = split_out / "M1"
    m2_dir = split_out / "M2"
    zs_dir = split_out / "zero_shot"

    # --- 2. caduceus trains ---
    if not skip_train:
        print(f"=== STAGE caduceus M1 (TPM, {epochs_m1} ep) ===", flush=True)
        rc = _train_caduceus(
            root=root,
            splits_dir=m1_dir,
            out=run_m1,
            epochs=epochs_m1,
            seed=seed,
            max_length=max_length,
            batch_size=batch_size,
            max_samples=max_samples,
            model_name=model_name,
            nproc=nproc,
        )
        if rc != 0:
            raise SystemExit(f"caduceus M1 failed with code {rc}")
        summary["stages"].append({"stage": "caduceus_M1", "out": str(run_m1), "rc": rc})

        if train_m2:
            print(f"=== STAGE caduceus M2 (predict M1 fold, {epochs_m2} ep) ===", flush=True)
            if not (m2_dir / "train" / "labels.tsv").is_file():
                raise FileNotFoundError(f"M2 folds missing under {m2_dir}")
            rc2 = _train_caduceus(
                root=root,
                splits_dir=m2_dir,
                out=run_m2,
                epochs=epochs_m2,
                seed=seed,
                max_length=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
                model_name=model_name,
                nproc=nproc,
            )
            if rc2 != 0:
                raise SystemExit(f"caduceus M2 failed with code {rc2}")
            summary["stages"].append(
                {"stage": "caduceus_M2", "out": str(run_m2), "rc": rc2}
            )
    else:
        summary["stages"].append({"stage": "caduceus", "skipped": True})

    # --- 3. zero-shot eval (human / holdout genomes) on M1 ---
    if not skip_zs and zs_genomes and (zs_dir / "all" / "labels.tsv").is_file():
        print("=== STAGE zero-shot eval (M1 TPM → holdout) ===", flush=True)
        zs_payload = _eval_zero_shot_m1(
            model_dir=run_m1,
            zs_dir=zs_dir,
            out_dir=zs_eval_dir,
            max_length=max_length,
            batch_size=max(batch_size * 2, 4),
            max_samples=max_samples,
        )
        summary["zs_eval"] = zs_payload
        summary["stages"].append(
            {"stage": "zs_eval", "out": str(zs_eval_dir), "n": zs_payload.get("n_eval")}
        )
        print(json.dumps({"zs": "ok", "metrics": zs_payload.get("metrics")}, indent=2), flush=True)
    elif skip_zs:
        summary["stages"].append({"stage": "zs_eval", "skipped": True})
    else:
        summary["stages"].append(
            {"stage": "zs_eval", "skipped": True, "reason": "no holdout folds"}
        )

    # --- 4. train-viz ---
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

    report = out_root / "report.md"
    zs_line = (
        f"- ZS holdout: `{', '.join(zs_genomes)}` → `{zs_eval_dir}`"
        if zs_genomes
        else "- ZS holdout: none"
    )
    report.write_text(
        "\n".join(
            [
                "# Caduceus-full report",
                "",
                f"**Date:** {summary['created_at']}",
                f"**Strategy:** `{strategy}`",
                f"**Out root:** `{out_root}`",
                f"**Seed / epochs:** {seed} / M1={epochs_m1}, M2={epochs_m2}",
                "",
                "## Paths",
                "",
                f"- Split: `{split_out}` (M1=TPM, M2=predict M1 fold)",
                f"- Train M1: `{run_m1}`",
                f"- Train M2: `{run_m2 if train_m2 else 'skipped'}`",
                f"- Viz M1: `{viz_m1}`",
                f"- Viz M2: `{viz_m2 if train_m2 else 'skipped'}`",
                zs_line,
                "",
                "## Notes",
                "",
                "- No `@adapt` — inputs are `raw/` + `ready/` (already adapted).",
                "- Orchestrator: `src/runs/caduceus_full.py` (import-only reuse of src.*).",
                "",
                "```json",
                json.dumps(summary, indent=2, default=str),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["report"] = str(report)
    (out_root / "pipeline.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--strategy", default="random", help="splits/<id>.md id")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("output") / "random",
        help="All pipeline outputs under this directory",
    )
    ap.add_argument("--raw", type=Path, default=Path("raw"))
    ap.add_argument("--ready", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="Legacy: set both M1 and M2")
    ap.add_argument("--epochs-m1", type=int, default=10)
    ap.add_argument("--epochs-m2", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None, help="Smoke cap")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument(
        "--zs-genomes",
        nargs="*",
        default=["human"],
        help="Holdout genome ids or aliases (default: human → GCF_000001405.40)",
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="GPUs for torchrun train stages (default: all visible CUDA devices; 1 = in-process)",
    )
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--skip-zs", action="store_true")
    ap.add_argument("--no-m2", action="store_true", help="Skip M2 train+viz")
    args = ap.parse_args(argv)

    epochs_m1 = args.epochs_m1 if args.epochs is None else args.epochs
    epochs_m2 = args.epochs_m2 if args.epochs is None else args.epochs

    if args.nproc is not None:
        nproc = max(1, args.nproc)
    elif torch.cuda.is_available():
        nproc = max(1, torch.cuda.device_count())
    else:
        nproc = 1

    run_pipeline(
        args.root.resolve(),
        strategy=args.strategy,
        out_root=args.out_root,
        raw=args.raw,
        ready=args.ready,
        seed=args.seed,
        epochs_m1=epochs_m1,
        epochs_m2=epochs_m2,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        model_name=args.model_name,
        zs_genomes=args.zs_genomes,
        nproc=nproc,
        skip_split=args.skip_split,
        skip_train=args.skip_train,
        skip_viz=args.skip_viz,
        skip_zs=args.skip_zs,
        train_m2=not args.no_m2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
