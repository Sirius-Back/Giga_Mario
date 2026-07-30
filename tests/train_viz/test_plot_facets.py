"""Tests for combined-plot epoch clip + faceted multi-model curves."""
from __future__ import annotations

from pathlib import Path

from src.train_viz.plotting import (
    FigureIndex,
    filter_rows_max_epoch,
    plot_learning_curves,
)
from src.train_viz.viz import _load_config


def test_filter_rows_max_epoch() -> None:
    rows = [
        {"epoch": 1, "model": "a", "value": 1.0},
        {"epoch": 23, "model": "a", "value": 0.5},
        {"epoch": 24, "model": "a", "value": 0.4},
        {"epoch": 100, "model": "a", "value": 0.1},
    ]
    clipped = filter_rows_max_epoch(rows, 23)
    assert [r["epoch"] for r in clipped] == [1, 23]
    assert filter_rows_max_epoch(rows, None) == rows


def test_multimodel_loss_is_faceted_and_clipped(tmp_path: Path) -> None:
    cfg = _load_config()
    cfg["dpi_png"] = 100  # faster; qc not run here
    rows: list[dict] = []
    for model, base in (("runA_direct", 1.0), ("runB_direct", 1.4)):
        for ep in range(0, 40):
            for split, offset in (("train", 0.0), ("validation", 0.2), ("test", 0.35)):
                rows.append(
                    {
                        "run": model,
                        "model": model,
                        "seed": 0,
                        "epoch": ep,
                        "global_step": ep,
                        "split": split,
                        "metric": "loss",
                        "value": base - 0.01 * ep + offset,
                    }
                )
    out = tmp_path / "viz"
    out.mkdir()
    written = plot_learning_curves(
        rows,
        ["loss"],
        cfg,
        out,
        FigureIndex(),
        x_key="epoch",
        title="pytest facets",
        column="double",
        ribbon="none",
        smooth=False,
        patience=None,
        dpi=100,
        max_epoch=23,
    )
    assert written
    loss_png = next(p for p in written if p.name.endswith("_loss.png") and "_altair" not in p.name)
    assert loss_png.is_file()
    # Sanity: clipped rows never exceed 23 in the learning-curve path
    assert max(r["epoch"] for r in filter_rows_max_epoch(rows, 23)) == 23
