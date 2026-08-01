"""Tests for src.train_viz (cnsplots + Altair backend)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.train_viz.viz import _series, flatten_epochs, main, parse_log, resolve_one_input

REAL_M1 = Path("archive/results/output/small_random/runs/M1/logs/train_metrics.jsonl")
REAL_M2 = Path("archive/results/output/small_random/runs/M2/logs/train_metrics.jsonl")
REAL_LEGNET = Path("archive/results/runs/legnet/GRCh38_4gpu/logs/train_metrics.jsonl")


def _assert_pub_bundle(outdir: Path, *, min_figures: int = 1) -> None:
    assert (outdir / "visualization_config.yaml").is_file()
    assert (outdir / "train_metrics.csv").is_file()
    assert (outdir / "training_summary.csv").is_file()
    assert (outdir / "training_summary.md").is_file()
    pdfs = sorted(outdir.glob("Figure_*.pdf"))
    svgs = sorted(outdir.glob("Figure_*.svg"))
    pngs = sorted(
        p for p in outdir.glob("Figure_*.png") if "_altair" not in p.name
    )
    assert len(pdfs) >= min_figures
    assert len(svgs) >= min_figures
    assert len(pngs) >= min_figures
    # Altair interactive exports for at least the first learning-curve figure
    assert list(outdir.glob("Figure_*_altair.html")), "expected Altair HTML"
    assert list(outdir.glob("Figure_*_altair.vl.json")), "expected Vega-Lite JSON"
    assert (outdir / "manuscript").is_dir()


def test_series_skips_non_numeric_epoch() -> None:
    rows = [
        {"run": "r", "model": "m", "seed": 1, "epoch": 1, "split": "train", "metric": "loss", "value": 1.0},
        {"run": "r", "model": "m", "seed": 1, "epoch": "final", "split": "train", "metric": "loss", "value": 0.5},
        {"run": "r", "model": "m", "seed": 1, "epoch": 2, "split": "train", "metric": "loss", "value": 0.8},
    ]
    xs, ys = _series(rows, model="m", seed=1, split="train", metric="loss", x_key="epoch")
    assert xs.tolist() == [1.0, 2.0]
    assert ys.tolist() == [1.0, 0.8]


def test_resolve_prefers_epochs_jsonl(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "train_metrics.jsonl").write_text(
        '{"epoch":"final","zero-shot-validation":{"spearman":0.1}}\n', encoding="utf-8"
    )
    (logs / "train_metrics_epochs.jsonl").write_text(
        '{"epoch":1,"train":{"loss":1.0},"validation":{"loss":1.1}}\n', encoding="utf-8"
    )
    picked = resolve_one_input(tmp_path)
    assert picked.name == "train_metrics_epochs.jsonl"


def test_parse_and_flatten_synthetic(tmp_path: Path) -> None:
    log = tmp_path / "train_metrics.jsonl"
    log.write_text(
        "\n".join(
            [
                '{"model_name":"toy","seed":7,"epochs":3,"batch_size":2}',
                '{"epoch":1,"train":{"loss":1.5,"pearson":0.1},"validation":{"loss":1.2,"pearson":0.2}}',
                '{"epoch":2,"train":{"loss":1.1,"pearson":0.3},"validation":{"loss":1.0,"pearson":0.35}}',
                '{"epoch":3,"train":{"loss":0.9,"pearson":0.4},"validation":{"loss":0.95,"pearson":0.38}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config, epochs = parse_log(log)
    assert config is not None
    assert config["seed"] == 7
    assert len(epochs) == 3
    rows, metrics, splits, nan_only = flatten_epochs("toy", epochs, model="toy", seed=7)
    assert not nan_only
    assert "loss" in metrics and "pearson" in metrics
    assert {"train", "validation"} <= splits
    assert len(rows) >= 8


@pytest.mark.skipif(not REAL_M1.is_file(), reason="real small_random M1 log missing")
def test_train_viz_real_small_random_m1(tmp_path: Path) -> None:
    out = tmp_path / "viz_m1"
    rc = main(
        [
            "--models",
            str(REAL_M1.parent.parent),
            "-o",
            str(out),
            "--title",
            "pytest small_random M1",
        ]
    )
    assert rc == 0
    _assert_pub_bundle(out, min_figures=3)
    # Learning curves + final performance expected for this log
    names = " ".join(p.name for p in out.glob("Figure_*.pdf"))
    assert "learning_curves" in names
    assert "final_performance" in names or "loss" in names


@pytest.mark.skipif(
    not (REAL_M1.is_file() and REAL_M2.is_file()),
    reason="real small_random M1/M2 logs missing",
)
def test_train_viz_real_compare_m1_m2(tmp_path: Path) -> None:
    out = tmp_path / "viz_cmp"
    rc = main(
        [
            "--models",
            str(REAL_M1.parent.parent),
            str(REAL_M2.parent.parent),
            "-o",
            str(out),
            "--title",
            "pytest M1 vs M2",
        ]
    )
    assert rc == 0
    _assert_pub_bundle(out, min_figures=2)
    names = " ".join(p.name for p in out.glob("Figure_*.pdf"))
    assert "multimodel" in names or "learning_curves" in names


@pytest.mark.skipif(not REAL_LEGNET.is_file(), reason="real LegNet log missing")
def test_train_viz_real_legnet(tmp_path: Path) -> None:
    out = tmp_path / "viz_legnet"
    rc = main(
        [
            str(REAL_LEGNET),
            "-o",
            str(out),
            "--label",
            "legnet_GRCh38",
            "--title",
            "pytest LegNet",
        ]
    )
    assert rc == 0
    _assert_pub_bundle(out, min_figures=1)
