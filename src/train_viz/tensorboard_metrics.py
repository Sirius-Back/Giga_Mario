#!/usr/bin/env python3
"""Write train/val(/test/ZSV) scalars to ``<run_dir>/tensorboard/``.

Caduceus already logs live via ``SummaryWriter``. LegNet/Lightning may log
during fit when ``tensorboard`` is installed; this helper also backfills from
``logs/train_metrics.jsonl`` (and ZSV JSON) so every train outdir has a
Caduceus-shaped TB tree for ``tensorboard --logdir …/tensorboard``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    """Export epoch + final/ZSV metrics from jsonl into TensorBoard events.

    Returns a small manifest dict (status, path, n_scalars).
    """
    run_dir = Path(run_dir)
    logs = run_dir / "logs"
    jsonl = logs / "train_metrics.jsonl"
    tb_dir = run_dir / tb_dirname
    manifest: dict[str, Any] = {
        "run_dir": str(run_dir),
        "tensorboard": str(tb_dir),
        "status": "ok",
        "n_scalars": 0,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }

    if not jsonl.is_file() or jsonl.stat().st_size == 0:
        manifest["status"] = "no_metrics"
        return manifest

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        manifest["status"] = f"tensorboard_unavailable:{exc}"
        return manifest

    if purge and tb_dir.exists():
        # Keep directory but drop prior event files so re-sync is deterministic
        for p in tb_dir.rglob("events.out.tfevents*"):
            p.unlink(missing_ok=True)

    tb_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
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
            step: int | None
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
                for name, val in _scalar_items(tag, block):
                    writer.add_scalar(name, val, step)
                    n += 1

        # Attach ZSV file if not already in jsonl
        zsv_path = logs / "zero_shot_metrics.json"
        if zsv_path.is_file():
            try:
                zsv = json.loads(zsv_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                zsv = None
            metrics = zsv.get("metrics") if isinstance(zsv, dict) else None
            if isinstance(metrics, dict):
                for name, val in _scalar_items("zero-shot-validation", metrics):
                    writer.add_scalar(name, val, 10_000_000)
                    n += 1

        writer.flush()

    manifest["n_scalars"] = n
    if n == 0:
        manifest["status"] = "smoke_only_or_empty"
    (logs / "tensorboard_export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
