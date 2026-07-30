"""Tests for completed-run discovery used by unified train-viz."""
from __future__ import annotations

import json
from pathlib import Path

from src.train_viz.compare_completed_runs import (
    _jsonl_complete,
    discover_completed_stages,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )


def test_jsonl_complete_requires_final(tmp_path: Path) -> None:
    p = tmp_path / "train_metrics.jsonl"
    _write_jsonl(
        p,
        [
            {"epoch": 0, "train": {"loss": 1.0}, "validation": {"loss": 1.1}},
            {"epoch": 1, "train": {"loss": 0.5}, "validation": {"loss": 0.9}},
        ],
    )
    assert _jsonl_complete(p) is False
    _write_jsonl(
        p,
        [
            {"epoch": 0, "train": {"loss": 1.0}, "validation": {"loss": 1.1}},
            {"epoch": "final", "zero-shot-validation": {"pearson": 0.4}},
        ],
    )
    assert _jsonl_complete(p) is True


def test_discover_prefers_epochs_log_but_gates_on_full(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    direct = runs / "runX" / "direct"
    full = direct / "logs" / "train_metrics.jsonl"
    epochs = direct / "logs" / "train_metrics_epochs.jsonl"
    _write_jsonl(
        full,
        [
            {"epoch": 0, "train": {"loss": 1.0}, "validation": {"loss": 1.2}},
            {"epoch": "final", "zero-shot-validation": {"pearson": 0.3}},
        ],
    )
    _write_jsonl(
        epochs,
        [{"epoch": 0, "train": {"loss": 1.0}, "validation": {"loss": 1.2}}],
    )
    # incomplete sibling
    other = runs / "runY" / "direct" / "logs" / "train_metrics.jsonl"
    _write_jsonl(
        other,
        [{"epoch": 0, "train": {"loss": 1.0}, "validation": {"loss": 1.2}}],
    )
    found = discover_completed_stages(runs)
    assert [r for r, _, _ in found["direct"]] == ["runX"]
    assert found["direct"][0][2] == epochs
