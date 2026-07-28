#!/usr/bin/env python3
"""Tests for dual TensorBoard helpers + Hydra /train compose."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tb_logging import (
    close_dual,
    lightning_dir,
    log_split_metrics,
    open_summary_writer,
    open_tensorboard_logger,
    summary_dir,
)
from src.train_viz.tensorboard_metrics import write_tensorboard_from_jsonl


def test_dual_tb_writers(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pytest.skip("tensorboard not installed")

    run = tmp_path / "run"
    writer = open_summary_writer(run)
    tb_logger = open_tensorboard_logger(run)
    log_split_metrics(
        writer,
        tb_logger,
        "validation",
        {"loss": 1.2, "pearson": 0.3, "n": 10},
        step=0,
    )
    close_dual(writer, tb_logger)

    assert summary_dir(run).is_dir()
    assert lightning_dir(run).is_dir()
    assert list(summary_dir(run).rglob("events.out.tfevents*"))
    assert list(lightning_dir(run).rglob("events.out.tfevents*"))


def test_jsonl_backfill_writes_summary_and_lightning(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pytest.skip("tensorboard not installed")

    run = tmp_path / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "train_metrics.jsonl").write_text(
        '{"epoch": 0, "train": {"loss": 1.0}, "validation": {"loss": 0.9, "pearson": 0.1}}\n',
        encoding="utf-8",
    )
    man = write_tensorboard_from_jsonl(run)
    assert man["status"] == "ok"
    assert man["n_scalars"] >= 3
    assert list(summary_dir(run).rglob("events.out.tfevents*"))
    assert list(lightning_dir(run).rglob("events.out.tfevents*"))


def test_hydra_train_compose() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    cfg_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="train_job",
            overrides=["mode=direct", "train=caduceus", "run_id=run0", "smoke=true"],
        )
    assert cfg.mode == "direct"
    assert cfg.train.name == "caduceus"
    assert "src.caduceus" in str(cfg.train.direct_cmd)
