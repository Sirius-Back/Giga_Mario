"""Best-checkpoint metadata + final-model selection helpers."""
from __future__ import annotations

import json
from pathlib import Path

from src.legnet import _copy_checkpoints, _parse_epoch_from_ckpt_name
from src.train_viz.viz import load_best_checkpoint_meta, resolve_best_epoch_for_log


def test_parse_epoch_from_ckpt_name() -> None:
    assert _parse_epoch_from_ckpt_name("pearson-epoch=12-val_pearson=0.55.ckpt") == 12
    assert _parse_epoch_from_ckpt_name("epoch-09.ckpt") == 9
    assert _parse_epoch_from_ckpt_name("nope.ckpt") is None


def test_load_best_checkpoint_meta(tmp_path: Path) -> None:
    run = tmp_path / "run"
    best = run / "best_model"
    best.mkdir(parents=True)
    (best / "best_meta.json").write_text(
        json.dumps({"epoch": 10, "metric": "val_loss", "value": 0.5}) + "\n",
        encoding="utf-8",
    )
    meta = load_best_checkpoint_meta(run)
    assert meta is not None
    assert meta["epoch"] == 10
    logs = run / "logs"
    logs.mkdir()
    jsonl = logs / "train_metrics.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    assert resolve_best_epoch_for_log(jsonl) == 10.0


def test_legnet_copy_checkpoints_promotes_best_to_final(tmp_path: Path) -> None:
    run = tmp_path / "legnet_run"
    lightning = run / "model_2_1" / "lightning_logs" / "version_0" / "checkpoints"
    lightning.mkdir(parents=True)
    (lightning / "pearson-epoch=07-val_pearson=0.40.ckpt").write_bytes(b"best")
    (lightning / "pearson-epoch=03-val_pearson=0.20.ckpt").write_bytes(b"worse")
    (lightning / "last_model-epoch=09.ckpt").write_bytes(b"last")
    (lightning / "epoch-10.ckpt").write_bytes(b"periodic")

    info = _copy_checkpoints(run)
    assert info["best_model"] is not None
    assert info["final_model"] is not None
    assert "0.40" in Path(info["final_model"]).name
    assert (run / "best_model" / "best_meta.json").is_file()
    assert (run / "final_model" / "best_meta.json").is_file()
    meta = json.loads((run / "final_model" / "best_meta.json").read_text(encoding="utf-8"))
    assert meta["epoch"] == 7
    assert meta["promoted_to_final"] is True
    assert (run / "checkpoints" / "epoch-10.ckpt").is_file()
