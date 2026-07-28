#!/usr/bin/env python3
"""Tests for fold-class adversarial PREDICT rewrite, ZSV helpers, Hydra dry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
from src.pipeline.common import read_csv, write_csv
from src.pipeline.zsv_eval import load_zsv_pairs, metrics_from_preds
from src.splits.random import M1_FOLD_TO_CLASS


def _write_panel(tmp: Path, *, n: int = 6) -> Path:
    panel = tmp / "panel"
    (panel / "PREDICT").mkdir(parents=True)
    (panel / "PARSED").mkdir(parents=True)
    rows = []
    for i in range(1, n + 1):
        rid = str(i)
        (panel / "PREDICT" / f"{rid}.ext").write_text(f"{10.0 + i}\n", encoding="utf-8")
        (panel / "PARSED" / f"{rid}.ext").write_text("A" * 50 + "\n", encoding="utf-8")
        rows.append({"id": rid, "predict_var1": f"{10.0 + i}"})
    write_csv(panel / "PREDICT" / "predict.csv", rows, ["id", "predict_var1"])
    write_csv(
        panel / "split.csv",
        [
            {"ID": "1", "train_test": "train", "fold": "0"},
            {"ID": "2", "train_test": "train", "fold": "0"},
            {"ID": "3", "train_test": "test", "fold": "0"},
            {"ID": "4", "train_test": "val", "fold": "0"},
            {"ID": "5", "train_test": "zsv", "fold": "zsv"},
            {"ID": "6", "train_test": "zsv", "fold": "zsv"},
        ],
        ["ID", "train_test", "fold"],
    )
    return panel


def test_apply_fold_class_targets_rewrites_and_keeps_zsv(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path)
    adv = tmp_path / "adv"
    run_adversarial(outdir=panel, outdir_new=adv, intersect_allow=True)
    # New split assignment for adversarial *training folds* (different from previous)
    write_csv(
        adv / "split.csv",
        [
            {"ID": "1", "train_test": "test", "fold": "0"},
            {"ID": "2", "train_test": "val", "fold": "0"},
            {"ID": "3", "train_test": "train", "fold": "0"},
            {"ID": "4", "train_test": "train", "fold": "0"},
            {"ID": "5", "train_test": "zsv", "fold": "zsv"},
            {"ID": "6", "train_test": "zsv", "fold": "zsv"},
        ],
        ["ID", "train_test", "fold"],
    )
    # Source continuous must survive hardlink-safe rewrite
    before_src = (panel / "PREDICT" / "1.ext").read_text(encoding="utf-8").strip()
    # Labels come from **previous (direct) split**, not adv/split.csv
    meta = apply_fold_class_targets(
        predict_root=adv / "PREDICT",
        label_split_csv=panel / "split.csv",
    )
    assert meta["mode"] == "fold_class"
    assert meta["label_source"] == "previous_split_m1"
    assert meta["n_mapped"] == 4
    assert meta["n_zsv_kept_continuous"] == 2
    assert meta["class_map"] == M1_FOLD_TO_CLASS

    by_id = {r["id"]: r["predict_var1"] for r in read_csv(adv / "PREDICT" / "predict.csv")}
    # Previous: 1=train→0, 2=train→0, 3=test→2, 4=val→1
    assert by_id["1"] == "0"
    assert by_id["2"] == "0"
    assert by_id["3"] == "2"
    assert by_id["4"] == "1"
    assert float(by_id["5"]) == pytest.approx(15.0)  # zsv kept
    assert (adv / "PREDICT" / "1.ext").read_text(encoding="utf-8").strip() == "0"
    assert (panel / "PREDICT" / "1.ext").read_text(encoding="utf-8").strip() == before_src
    side = json.loads((adv / "PREDICT" / "predict_target.json").read_text(encoding="utf-8"))
    assert side["mode"] == "fold_class"
    assert side["label_source"] == "previous_split_m1"
    # New adv train bucket (IDs 3,4) must mix previous classes {2,1} — not constant
    assert {by_id["3"], by_id["4"]} == {"2", "1"}


def test_zsv_load_pairs_and_metrics(tmp_path: Path) -> None:
    parsed = tmp_path / "PARSED" / "zero-shot-validation"
    predict = tmp_path / "PREDICT" / "zero-shot-validation"
    parsed.mkdir(parents=True)
    predict.mkdir(parents=True)
    (parsed / "a.ext").write_text("ACGT\n", encoding="utf-8")
    (parsed / "b.ext").write_text("TTTT\n", encoding="utf-8")
    (predict / "a.ext").write_text("1.5\n", encoding="utf-8")
    (predict / "b.ext").write_text("2.5\n", encoding="utf-8")
    pairs = load_zsv_pairs(parsed_root=tmp_path / "PARSED", predict_root=tmp_path / "PREDICT")
    assert len(pairs) == 2
    preds = [1.5, 2.5]
    targets = [1.5, 2.5]
    m = metrics_from_preds(preds, targets)
    assert m["n"] == 2
    assert m["pearson"] == pytest.approx(1.0)
    assert m["mse"] == pytest.approx(0.0)


def test_hydra_pipeline_imports_and_config() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    cfg_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="pipeline", overrides=["mode=dry", "adversarial=true"])
    assert cfg.mode == "dry"
    assert cfg.train.name == "legnet"
    assert "src.legnet" in str(cfg.train.direct_cmd)
    assert "src.pipeline.zsv_eval" in str(cfg.train.zsv_cmd)
    assert list(cfg.ratios) == [1, 1, 3]
