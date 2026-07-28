#!/usr/bin/env python3
"""Tests for train/val/test/ZSV comparison viz."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.train_viz.split_compare import collect_split_metrics, run_split_compare

RUN0_DIRECT = Path("run/run0/direct")


def test_collect_split_metrics_synthetic(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "test": {
                    "n": 10,
                    "pearson": 0.5,
                    "spearman": 0.4,
                    "mse": 1.0,
                    "rmse": 1.0,
                    "mae": 0.8,
                    "r2": 0.2,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "logs" / "zero_shot_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "n": 5,
                    "pearson": 0.1,
                    "spearman": 0.05,
                    "mse": 2.0,
                    "rmse": 1.4,
                    "mae": 1.1,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "logs" / "train_metrics.jsonl").write_text(
        json.dumps(
            {
                "epoch": 1,
                "train": {"loss": 3.0},
                "validation": {"loss": 2.5, "pearson": 0.2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    df = collect_split_metrics(run, model="toy")
    splits = set(df["split"].astype(str))
    assert {"train", "validation", "test", "zero_shot"} <= splits
    assert "pearson" in set(df["metric"])


@pytest.mark.skipif(
    not (RUN0_DIRECT / "metrics_summary.json").is_file(),
    reason="run0 direct metrics_summary missing",
)
@pytest.mark.skipif(
    not (RUN0_DIRECT / "logs" / "zero_shot_metrics.json").is_file(),
    reason="run0 ZSV metrics missing",
)
def test_split_compare_run0_direct(tmp_path: Path) -> None:
    cnsplots = pytest.importorskip("cnsplots")
    altair = pytest.importorskip("altair")
    _ = (cnsplots, altair)
    out = tmp_path / "figs"
    manifest = run_split_compare(RUN0_DIRECT, out, model="run0_direct")
    assert Path(manifest["csv"]).is_file()
    assert Path(manifest["json"]).is_file()
    assert "zero_shot" in manifest["splits"] or "test" in manifest["splits"]
    pdfs = list(out.glob("Figure_*split_compare*.pdf"))
    assert pdfs, "expected cnsplots PDF"
    assert list(out.glob("Figure_*_altair.html")) or list(
        out.glob("Figure_*_altair.vl.json")
    ), "expected Altair export"
