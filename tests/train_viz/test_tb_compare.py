"""Tests for TensorBoard compare export helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.train_viz.tb_compare import export_run_compare_events, write_index_html


def test_export_run_compare_events_writes_scalars(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    train = tmp_path / "run" / "direct"
    logs = train / "logs"
    logs.mkdir(parents=True)
    (logs / "train_metrics.jsonl").write_text(
        json.dumps(
            {
                "epoch": 0,
                "train": {"loss": 1.0, "pearson": 0.1},
                "validation": {"loss": 1.2, "pearson": 0.05},
            }
        )
        + "\n"
        + json.dumps({"epoch": "final", "zero-shot-validation": {"pearson": 0.2}})
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "tb" / "run_direct"
    man = export_run_compare_events(train, dest, run_label="run_direct")
    assert man["status"] == "ok"
    assert man["n_scalars"] >= 4
    assert any(dest.rglob("events.out.tfevents*"))


def test_write_index_html(tmp_path: Path) -> None:
    out = tmp_path / "figures"
    (out / "tb_compare" / "direct").mkdir(parents=True)
    (out / "all_completed_direct").mkdir(parents=True)
    html = (
        out
        / "all_completed_direct"
        / "Figure_11_multimodel_pearson_validation_altair.html"
    )
    html.write_text("<html></html>", encoding="utf-8")
    path = write_index_html(
        out,
        tb_manifest={
            "stages": {
                "direct": {
                    "runs": [{"run_label": "run1_direct", "status": "ok"}],
                },
                "adversarial": {"runs": []},
            }
        },
        tb_url="http://127.0.0.1:6006/",
        port=6006,
        host="127.0.0.1",
    )
    text = path.read_text(encoding="utf-8")
    assert "Open TensorBoard" in text
    assert "multimodel_pearson_validation" in text
