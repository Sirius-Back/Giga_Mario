#!/usr/bin/env python3
"""Write train/val(/test/ZSV) scalars via dual TB layout.

Writes into ``<run_dir>/tensorboard/summary/`` (SummaryWriter) and
``<run_dir>/tensorboard/lightning/`` (TensorBoardLogger API / stand-in).
Does **not** purge the sibling logger directory.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tb_logging import (
    LIGHTNING_NAME,
    SUMMARY_NAME,
    close_dual,
    log_split_metrics,
    open_summary_writer,
    open_tensorboard_logger,
    tensorboard_root,
)


def _scalar_items(split: str, metrics: dict[str, Any]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for k, v in metrics.items():
        if k == "n":
            continue
        try:
            out.append((f"{split}/{k}", float(v)))
        except (TypeError, ValueError):
            continue
    return out


def write_tensorboard_from_jsonl(
    run_dir: Path,
    *,
    tb_dirname: str = "tensorboard",
    purge: bool = True,
) -> dict[str, Any]:
    """Export epoch + final/ZSV metrics from jsonl into dual TensorBoard trees."""
    run_dir = Path(run_dir)
    logs = run_dir / "logs"
    jsonl = logs / "train_metrics.jsonl"
    tb_dir = run_dir / tb_dirname
    manifest: dict[str, Any] = {
        "run_dir": str(run_dir),
        "tensorboard": str(tb_dir),
        "tensorboard_summary": str(tb_dir / SUMMARY_NAME),
        "tensorboard_lightning": str(tb_dir / LIGHTNING_NAME),
        "status": "ok",
        "n_scalars": 0,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }

    if not jsonl.is_file() or jsonl.stat().st_size == 0:
        manifest["status"] = "no_metrics"
        return manifest

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError as exc:
        manifest["status"] = f"tensorboard_unavailable:{exc}"
        return manifest

    if purge:
        # Only purge summary backfill dir — keep live Lightning events.
        summary = tb_dir / SUMMARY_NAME
        if summary.exists():
            for p in summary.rglob("events.out.tfevents*"):
                p.unlink(missing_ok=True)

    writer = open_summary_writer(run_dir)
    tb_logger = open_tensorboard_logger(run_dir)
    n = 0
    try:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("smoke"):
                continue
            ep = rec.get("epoch")
            if isinstance(ep, int):
                step = ep
            elif ep == "final":
                step = int(rec.get("global_step", 10_000_000))
            else:
                continue

            for split_key, tag in (
                ("train", "train"),
                ("validation", "validation"),
                ("val", "validation"),
                ("test", "test"),
                ("zero-shot-validation", "zero-shot-validation"),
                ("zero_shot", "zero-shot-validation"),
            ):
                block = rec.get(split_key)
                if not isinstance(block, dict):
                    continue
                log_split_metrics(writer, tb_logger, tag, block, step)
                n += len(_scalar_items(tag, block))

        zsv_path = logs / "zero_shot_metrics.json"
        if zsv_path.is_file():
            try:
                zsv = json.loads(zsv_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                zsv = None
            metrics = zsv.get("metrics") if isinstance(zsv, dict) else None
            if isinstance(metrics, dict):
                log_split_metrics(
                    writer, tb_logger, "zero-shot-validation", metrics, 10_000_000
                )
                n += len(_scalar_items("zero-shot-validation", metrics))
    finally:
        close_dual(writer, tb_logger)

    manifest["n_scalars"] = n
    if n == 0:
        manifest["status"] = "smoke_only_or_empty"
    # Keep root alias for older consumers
    _ = tensorboard_root(run_dir)
    (logs / "tensorboard_export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
