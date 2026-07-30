"""Tests for continue_direct_model_25 candidate discovery."""
from __future__ import annotations

import json
from pathlib import Path

from src.runs.continue_direct_model_25 import discover_candidates


def _write_direct(
    root: Path,
    run_id: str,
    *,
    model: str,
    max_epoch: int,
    epochs_requested: int = 50,
    with_last_ckpt: bool = True,
) -> Path:
    direct = root / run_id / "direct"
    (direct / "best_model").mkdir(parents=True)
    (direct / "final_model").mkdir(parents=True)
    (direct / "logs").mkdir(parents=True)
    (direct / "best_model" / "best_meta.json").write_text(
        json.dumps({"epoch": 1, "val_loss": 0.2}) + "\n", encoding="utf-8"
    )
    lines = [
        json.dumps({"epoch": i, "validation": {"loss": 0.1}})
        for i in range(1 if model == "caduceus" else 0, max_epoch + 1)
    ]
    (direct / "logs" / "train_metrics.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if model == "caduceus":
        (direct / "caduceus_input" / "train").mkdir(parents=True)
        (direct / "run_config.json").write_text(
            json.dumps(
                {
                    "epochs": epochs_requested,
                    "batch_size": 256,
                    "seed": 42,
                    "splits_dir": str(direct / "caduceus_input"),
                    "model_name": "caduceus",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (direct / "model_2_1" / "lightning_logs" / "version_0" / "checkpoints").mkdir(
            parents=True
        )
        tsv = root / run_id / "legnet_input" / "all.tsv"
        tsv.parent.mkdir(parents=True)
        tsv.write_text("seq\ty\tfold\nACGT\t1.0\t1\n", encoding="utf-8")
        (direct / "logs" / "run_config.json").write_text(
            json.dumps(
                {
                    "skill": "legnet",
                    "producer": "src/legnet.py",
                    "epochs": epochs_requested,
                    "train_batch_size": 8192,
                    "seed": 42,
                    "data_path": str(tsv),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        if with_last_ckpt:
            ckpt = (
                direct
                / "model_2_1"
                / "lightning_logs"
                / "version_0"
                / "checkpoints"
                / f"last_model-epoch={max_epoch}.ckpt"
            )
            ckpt.write_bytes(b"ckpt")
            (direct / "best_model" / "pearson-epoch=01-val_pearson=0.40.ckpt").write_bytes(
                b"best"
            )
    return direct


def test_discover_under_25_only(tmp_path: Path) -> None:
    _write_direct(tmp_path, "run_short", model="caduceus", max_epoch=16)
    _write_direct(tmp_path, "run_long", model="caduceus", max_epoch=30)
    _write_direct(tmp_path, "run_leg", model="legnet", max_epoch=11)
    _write_direct(
        tmp_path,
        "run_probe",
        model="legnet",
        max_epoch=0,
        epochs_requested=1,
        with_last_ckpt=True,
    )
    found = discover_candidates(tmp_path)
    ids = {c.run_id for c in found}
    assert "run_short" in ids
    assert "run_leg" in ids
    assert "run_long" not in ids
    assert "run_probe" not in ids
    leg = next(c for c in found if c.run_id == "run_leg")
    assert leg.model == "legnet"
    assert leg.batch_size == 8192  # scaled at launch by n_devices
    assert leg.resume_ckpt is not None
