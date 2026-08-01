"""Tests for src.pipeline.best_split_metrics (best-epoch / best-ckpt Spearman)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.best_split_metrics import (
    caduceus_best_from_jsonl,
    detect_model_family,
    write_best_split_metrics,
)


def test_detect_family_from_path(tmp_path: Path) -> None:
    p = tmp_path / "runs_unif" / "caduceus" / "runX" / "direct"
    p.mkdir(parents=True)
    assert detect_model_family(p) == "caduceus"
    q = tmp_path / "runs_unif" / "legnet" / "runY" / "direct"
    q.mkdir(parents=True)
    assert detect_model_family(q) == "legnet"


def test_caduceus_best_from_jsonl(tmp_path: Path) -> None:
    train = tmp_path / "runs_unif" / "caduceus" / "r" / "direct"
    logs = train / "logs"
    best = train / "best_model"
    logs.mkdir(parents=True)
    best.mkdir(parents=True)
    (best / "best_meta.json").write_text(
        json.dumps({"epoch": 2, "metric": "val_loss", "selection": "min_val_loss"}) + "\n",
        encoding="utf-8",
    )
    lines = [
        json.dumps(
            {
                "epoch": 1,
                "train": {"spearman": 0.1, "pearson": 0.2},
                "validation": {"spearman": 0.15, "pearson": 0.25},
                "test": {"spearman": 0.12, "pearson": 0.22},
            }
        ),
        json.dumps(
            {
                "epoch": 2,
                "train": {"spearman": 0.55, "pearson": 0.6},
                "validation": {"spearman": 0.44, "pearson": 0.5},
                "test": {"spearman": 0.41, "pearson": 0.48},
            }
        ),
        json.dumps(
            {
                "epoch": 3,
                "train": {"spearman": 0.9, "pearson": 0.91},
                "validation": {"spearman": 0.3, "pearson": 0.35},
                "test": {"spearman": 0.31, "pearson": 0.36},
            }
        ),
    ]
    (logs / "train_metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (logs / "zero_shot_metrics.json").write_text(
        json.dumps({"metrics": {"spearman": 0.39, "pearson": 0.4}}) + "\n",
        encoding="utf-8",
    )

    payload = caduceus_best_from_jsonl(train)
    assert payload["best_epoch"] == 2
    assert payload["spearman"]["train"] == pytest.approx(0.55)
    assert payload["spearman"]["val"] == pytest.approx(0.44)
    assert payload["spearman"]["test"] == pytest.approx(0.41)
    assert payload["spearman"]["zsv"] == pytest.approx(0.39)
    # Must not pick last epoch (0.9)
    assert payload["spearman"]["train"] != pytest.approx(0.9)

    out = write_best_split_metrics(train, payload)
    assert out.is_file()
