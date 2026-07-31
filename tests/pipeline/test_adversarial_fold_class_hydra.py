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


def test_apply_fold_class_skips_panel_ids_outside_label_split(tmp_path: Path) -> None:
    from src.pipeline.adversarial import (
        apply_fold_class_targets,
        write_fold_csv_from_split,
        write_id_csv_from_split,
    )

    panel = _write_panel(tmp_path, n=8)
    # Direct/M1 labels only for 1..6; panel has extras 7,8
    label = panel / "split.csv"
    id_csv = write_id_csv_from_split(label, tmp_path / "id_sub.csv")
    fold_csv = write_fold_csv_from_split(label, tmp_path / "fold_sub.csv")
    assert id_csv.is_file() and fold_csv.is_file()
    ids = {r["ID"] for r in read_csv(id_csv)}
    assert ids == {"1", "2", "3", "4", "5", "6"}
    assert "7" not in ids

    adv = tmp_path / "adv"
    run_adversarial(outdir=panel, outdir_new=adv, intersect_allow=True)
    meta = apply_fold_class_targets(
        predict_root=adv / "PREDICT",
        label_split_csv=label,
    )
    assert meta["n_mapped"] == 4
    assert meta["n_skipped_unlabeled"] == 2
    by_id = {r["id"]: r["predict_var1"] for r in read_csv(adv / "PREDICT" / "predict.csv")}
    assert by_id["1"] == "0"
    assert float(by_id["7"]) == pytest.approx(17.0)  # unlabeled kept continuous


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


def test_caduceus_evaluate_zsv_root_writes_universal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path + artifact contract without loading the real Caduceus weights."""
    import torch
    from torch import nn

    from src import caduceus

    parsed = tmp_path / "PARSED" / "zero-shot-validation"
    predict = tmp_path / "PREDICT" / "zero-shot-validation"
    parsed.mkdir(parents=True)
    predict.mkdir(parents=True)
    for i, y in enumerate((1.0, 2.0, 3.0)):
        (parsed / f"s{i}.ext").write_text("ACGTACGT\n", encoding="utf-8")
        (predict / f"s{i}.ext").write_text(f"{y}\n", encoding="utf-8")

    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    class _Tok:
        pad_token_id = 4

        def __call__(self, seq, **kwargs):
            ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _Model(nn.Module):
        def forward(self, **batch):
            b = batch["labels"].shape[0]
            # Predict label itself → perfect metrics
            return _Out(batch["labels"].unsqueeze(-1))

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        caduceus,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: _Tok())}),
    )
    monkeypatch.setattr(
        caduceus,
        "AutoModelForSequenceClassification",
        type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: _Model())}),
    )

    out_json = tmp_path / "logs" / "zero_shot_metrics.json"
    payload = caduceus.evaluate_zsv_root(
        model_dir=model_dir,
        zsv_root=tmp_path,
        out_json=out_json,
        batch_size=2,
        max_length=16,
        device="cpu",
        amp=False,
    )
    assert out_json.is_file()
    assert payload["model"] == "caduceus"
    assert payload["metrics"]["n"] == 3
    assert payload["metrics"]["pearson"] == pytest.approx(1.0)
    assert payload["metrics"]["mse"] == pytest.approx(0.0)
    assert (tmp_path / "logs" / "train_metrics.jsonl").is_file()


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
    assert cfg.ratios is None


def test_hydra_pipeline_gc_caduceus_overrides() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    cfg_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "mode=run",
                "split=gc",
                "train=caduceus",
                "adversarial=false",
                "early_stopping_patience=10",
                "max_length=208",
                "panel_root=ready_caduceus",
                "out_root=runs/run3",
            ],
        )
    assert cfg.split == "gc"
    assert cfg.train.name == "caduceus"
    assert cfg.adversarial is False
    assert int(cfg.early_stopping_patience) == 10
    assert int(cfg.max_length) == 208
    assert "early-stopping-patience" in str(cfg.train.direct_cmd)
    assert "max-length" in str(cfg.train.direct_cmd)


def test_hydra_pipeline_run4_legnet_early_stop_overrides() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    cfg_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "mode=run",
                "split=gc",
                "train=legnet",
                "adversarial=true",
                "epochs=50",
                "min_epochs=10",
                "early_stopping_patience=10",
                "n_devices=2",
                "panel_root=ready_legnet",
                "out_root=runs/run4",
            ],
        )
    assert cfg.split == "gc"
    assert cfg.train.name == "legnet"
    assert cfg.adversarial is True
    assert int(cfg.epochs) == 50
    assert int(cfg.min_epochs) == 10
    assert int(cfg.early_stopping_patience) == 10
    assert "early-stopping-patience" in str(cfg.train.direct_cmd)
    assert "min-epochs" in str(cfg.train.direct_cmd)
    assert "early-stopping-patience" in str(cfg.train.adversarial_cmd)
