#!/usr/bin/env python3
"""Tests for train_monitor sync + refresh."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.train_viz.train_monitor import (
    refresh_pipeline_monitors,
    refresh_train_monitor,
    sync_train_metrics_jsonl,
)
from src.train_viz.tensorboard_metrics import write_tensorboard_from_jsonl

RUN0_DIRECT = Path("run/run0/direct")
RUN0_ROOT = Path("run/run0")


def test_sync_from_synthetic_lightning(tmp_path: Path) -> None:
    run = tmp_path / "run"
    lightning = run / "model_x" / "lightning_logs" / "version_0"
    lightning.mkdir(parents=True)
    (lightning / "metrics.csv").write_text(
        "val_loss,val_pearson,epoch,step,train_loss\n"
        "1.2,0.1,0,0,\n"
        ",,0,0,1.5\n"
        "1.0,0.2,1,1,\n"
        ",,1,1,1.1\n",
        encoding="utf-8",
    )
    path = sync_train_metrics_jsonl(run)
    assert path is not None and path.is_file()
    text = path.read_text(encoding="utf-8")
    assert '"epoch": 0' in text and '"epoch": 1' in text
    assert "train" in text and "validation" in text


def test_tensorboard_export_from_jsonl(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pytest.skip("tensorboard not installed")
    run = tmp_path / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "train_metrics.jsonl").write_text(
        '{"epoch": 0, "train": {"loss": 1.5}, "validation": {"loss": 1.2, "pearson": 0.1}}\n'
        '{"epoch": 1, "train": {"loss": 1.1}, "validation": {"loss": 1.0, "pearson": 0.2}}\n',
        encoding="utf-8",
    )
    man = write_tensorboard_from_jsonl(run)
    assert man["status"] == "ok"
    assert man["n_scalars"] >= 4
    events = list((run / "tensorboard").rglob("events.out.tfevents*"))
    assert events


@pytest.mark.skipif(
    not any(RUN0_DIRECT.rglob("metrics.csv")),
    reason="run0 Lightning metrics.csv missing",
)
def test_refresh_train_monitor_run0() -> None:
    pytest.importorskip("cnsplots")
    pytest.importorskip("altair")
    out = RUN0_DIRECT / "figures" / "train_monitor"
    man = refresh_train_monitor(
        RUN0_DIRECT,
        outdir=out,
        model="run0_direct",
        title="run0 direct monitor",
        include_split_compare=True,
    )
    assert man["status"] == "ok"
    assert Path(man["jsonl"]).is_file()
    assert list(out.glob("Figure_*learning_curves*.pdf")) or list(
        out.glob("Figure_*.pdf")
    )
    tb = man.get("tensorboard") or {}
    assert tb.get("status") == "ok"
    assert Path(RUN0_DIRECT / "tensorboard").is_dir()


@pytest.mark.skipif(
    not (RUN0_ROOT / "adversarial" / "train").is_dir(),
    reason="run0 adversarial/train missing",
)
def test_refresh_pipeline_monitors_includes_adversarial() -> None:
    pytest.importorskip("cnsplots")
    man = refresh_pipeline_monitors(RUN0_ROOT, run_id="run0", include_split_compare=True)
    assert man.get("direct") is not None
    assert man.get("adversarial") is not None
    assert man["adversarial"].get("status") in {
        "ok",
        "smoke_only",
        "no_metrics",
    }
    assert (RUN0_ROOT / "pipeline_monitor_manifest.json").is_file()
