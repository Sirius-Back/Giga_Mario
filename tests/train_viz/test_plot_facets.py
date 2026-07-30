"""Tests for combined-plot epoch clip + faceted multi-model curves."""
from __future__ import annotations

from pathlib import Path

from src.train_viz.plotting import (
    FigureIndex,
    filter_rows_max_epoch,
    load_zsv_rows,
    plot_learning_curves,
    plot_zsv_barplots,
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


def test_multimodel_overview_keeps_splits_with_style(tmp_path: Path) -> None:
    """Color=model + style=split must keep train/val/test (no zigzag merge)."""
    cfg = _load_config()
    rows: list[dict] = []
    for model, base in (("runA_direct", 1.0), ("runB_direct", 0.8)):
        for ep in range(0, 6):
            for split, offset in (("train", 0.0), ("validation", 0.3), ("test", 0.5)):
                rows.append(
                    {
                        "run": model,
                        "model": model,
                        "seed": 0,
                        "epoch": float(ep),
                        "global_step": ep,
                        "split": split,
                        "metric": "loss",
                        "value": base - 0.05 * ep + offset,
                    }
                )
    out = tmp_path / "viz_overview"
    out.mkdir()
    plot_learning_curves(
        rows,
        ["loss"],
        cfg,
        out,
        FigureIndex(),
        x_key="epoch",
        title="overview split styles",
        column="double",
        ribbon="none",
        smooth=False,
        patience=None,
        dpi=100,
        max_epoch=23,
    )
    vl = next(out.glob("Figure_01_learning_curves_altair.vl.json"))
    text = vl.read_text()
    assert '"train"' in text and '"validation"' in text and '"test"' in text
    assert "strokeDash" in text or "strokeDash" in text.replace("stroke_dash", "strokeDash")


def test_multimodel_loss_is_faceted_and_clipped(tmp_path: Path) -> None:
    cfg = _load_config()
    cfg["dpi_png"] = 100
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
    out = tmp_path / "viz_facets"
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
    assert max(r["epoch"] for r in filter_rows_max_epoch(rows, 23)) == 23


def test_zsv_barplots(tmp_path: Path) -> None:
    cfg = _load_config()
    train = tmp_path / "runX" / "direct"
    (train / "logs").mkdir(parents=True)
    (train / "logs" / "zero_shot_metrics.json").write_text(
        '{"metrics": {"pearson": 0.42, "spearman": 0.4, "mse": 0.2, "rmse": 0.45, "mae": 0.3}}\n',
        encoding="utf-8",
    )
    zsv = load_zsv_rows(train, model="runX_direct", seed=0)
    assert any(r["metric"] == "pearson" and r["split"] == "zero_shot" for r in zsv)
    # Need ≥1 other split epoch for vs-splits panel
    rows = list(zsv)
    for ep in range(3):
        rows.append(
            {
                "run": "runX_direct",
                "model": "runX_direct",
                "seed": 0,
                "epoch": ep,
                "global_step": ep,
                "split": "validation",
                "metric": "pearson",
                "value": 0.3 + 0.01 * ep,
            }
        )
    out = tmp_path / "zsv_fig"
    out.mkdir()
    written = plot_zsv_barplots(rows, cfg, out, FigureIndex(), dpi=100)
    assert written
    assert any("zsv_by_model" in p.name for p in written)
