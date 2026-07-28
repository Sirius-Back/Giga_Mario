#!/usr/bin/env python3
"""Dual TensorBoard logging for Caduceus and LegNet.

Every train outdir gets both:

- ``tensorboard/summary/`` — ``torch.utils.tensorboard.SummaryWriter``
- ``tensorboard/lightning/`` — Lightning ``TensorBoardLogger`` when importable,
  else a SummaryWriter-backed stand-in with the same ``log_metrics`` API

``tensorboard --logdir <outdir>/tensorboard`` shows both runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


SUMMARY_NAME = "summary"
LIGHTNING_NAME = "lightning"


class MetricsLogger(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def save(self) -> None: ...

    def finalize(self, status: str = "success") -> None: ...

    @property
    def log_dir(self) -> str: ...


class SummaryWriterLogger:
    """TensorBoardLogger-compatible wrapper around SummaryWriter (no Lightning)."""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._dir))

    @property
    def log_dir(self) -> str:
        return str(self._dir)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        s = 0 if step is None else int(step)
        for key, val in metrics.items():
            try:
                self._writer.add_scalar(str(key), float(val), s)
            except (TypeError, ValueError):
                continue
        self._writer.flush()

    def save(self) -> None:
        self._writer.flush()

    def finalize(self, status: str = "success") -> None:  # noqa: ARG002
        self._writer.flush()
        self._writer.close()


def tensorboard_root(out_dir: Path) -> Path:
    return Path(out_dir) / "tensorboard"


def summary_dir(out_dir: Path) -> Path:
    return tensorboard_root(out_dir) / SUMMARY_NAME


def lightning_dir(out_dir: Path) -> Path:
    return tensorboard_root(out_dir) / LIGHTNING_NAME


def open_summary_writer(out_dir: Path):
    """Return a ``SummaryWriter`` under ``<out>/tensorboard/summary``."""
    from torch.utils.tensorboard import SummaryWriter

    d = summary_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(d))


def open_tensorboard_logger(out_dir: Path) -> MetricsLogger:
    """Return Lightning ``TensorBoardLogger`` or a SummaryWriter stand-in."""
    out_dir = Path(out_dir)
    tensorboard_root(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger(
            save_dir=str(tensorboard_root(out_dir)),
            name=LIGHTNING_NAME,
            version="",
        )
    except Exception:
        return SummaryWriterLogger(lightning_dir(out_dir))


def log_scalar_pair(
    writer,
    tb_logger: MetricsLogger | None,
    tag: str,
    value: float,
    step: int,
) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if writer is not None:
        writer.add_scalar(tag, v, step)
    if tb_logger is not None:
        tb_logger.log_metrics({tag: v}, step=step)


def log_split_metrics(
    writer,
    tb_logger: MetricsLogger | None,
    split: str,
    metrics: dict[str, Any],
    step: int,
) -> None:
    """Log ``{split}/{key}`` scalars to both SummaryWriter and TensorBoardLogger."""
    for key, val in metrics.items():
        if key == "n":
            continue
        log_scalar_pair(writer, tb_logger, f"{split}/{key}", val, step)


def close_dual(writer, tb_logger: MetricsLogger | None) -> None:
    if writer is not None:
        writer.flush()
        writer.close()
    if tb_logger is not None:
        try:
            tb_logger.save()
        except Exception:
            pass
        try:
            tb_logger.finalize("success")
        except Exception:
            pass


def lightning_logger_for_pl(out_dir: Path):
    """Lightning-native TensorBoardLogger for human_legnet Trainer (raises if missing)."""
    from pytorch_lightning.loggers import TensorBoardLogger

    out_dir = Path(out_dir)
    tensorboard_root(out_dir).mkdir(parents=True, exist_ok=True)
    return TensorBoardLogger(
        save_dir=str(tensorboard_root(out_dir)),
        name=LIGHTNING_NAME,
        version="",
    )
