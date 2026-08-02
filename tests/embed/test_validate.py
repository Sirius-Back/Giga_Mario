"""Validation fail-fast and good-panel checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.embed.discover import LegNetRun
from src.embed.validate import validate_run, write_validation_report


def _write_minimal_unit(tmp: Path, *, good: bool = True) -> LegNetRun:
    root = tmp / "run_x"
    direct = root / "direct"
    best = direct / "best_model"
    best.mkdir(parents=True)
    (direct / "final_model").mkdir(parents=True)
    tsv_dir = root / "legnet_input"
    tsv_dir.mkdir(parents=True)

    # Fake ckpt file
    ckpt = best / "pearson-epoch=01-val_pearson=0.50.ckpt"
    ckpt.write_bytes(b"not-a-real-ckpt")
    (best / "best_meta.json").write_text(
        json.dumps({"epoch": 1, "val_pearson": 0.5}) + "\n", encoding="utf-8"
    )
    cfg = {
        "stem_ch": 64,
        "stem_ks": 11,
        "ef_ks": 9,
        "ef_block_sizes": [80, 96, 112, 128],
        "resize_factor": 4,
        "pool_sizes": [2, 2, 2, 2],
        "use_reverse_channel": False,
        "reverse_augment": False,
        "use_shift": False,
        "max_shift": None,
        "max_lr": 0.01,
        "weight_decay": 0.1,
        "model_dir": str(direct),
        "data_path": str(tsv_dir / "all.tsv"),
        "epoch_num": 1,
        "device": 0,
        "seed": 42,
        "train_batch_size": 8,
        "valid_batch_size": 8,
        "num_workers": 0,
    }
    (direct / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    split = root / "split.csv"
    if good:
        split.write_text(
            "ID|train_test|fold\n"
            "1|train|0\n"
            "2|train|0\n"
            "3|test|0\n"
            "4|val|0\n",
            encoding="utf-8",
        )
        seq = "A" * 230
        tsv = tsv_dir / "all.tsv"
        tsv.write_text(
            "seq_id\tseq\tmean_value\tfold\trev\n"
            f"1\t{seq}\t1.0\t3\t0\n"
            f"2\t{seq}\t2.0\t3\t0\n"
            f"3\t{seq}\t3.0\t1\t0\n"
            f"4\t{seq}\t4.0\t2\t0\n",
            encoding="utf-8",
        )
    else:
        split.write_text(
            "ID|train_test|fold\n1|train|0\n2|test|0\n",
            encoding="utf-8",
        )
        # missing ID 2; bad length for ID 1
        (tsv_dir / "all.tsv").write_text(
            "seq_id\tseq\tmean_value\tfold\trev\n"
            "1\tAAAA\t1.0\t3\t0\n",
            encoding="utf-8",
        )

    return LegNetRun(
        run_name="run_x",
        fold=None,
        root=root,
        train_dir=direct,
        split_csv=split,
        legnet_tsv=tsv_dir / "all.tsv",
        config_json=direct / "config.json",
        ckpt_path=ckpt,
        best_meta=best / "best_meta.json",
    )


def test_validate_good_panel(tmp_path: Path):
    run = _write_minimal_unit(tmp_path, good=True)
    res = validate_run(run, load_ckpt=False)
    assert res.status == "READY"
    assert res.n_train == 2 and res.n_test == 1 and res.n_val == 1


def test_validate_bad_panel_failed(tmp_path: Path):
    run = _write_minimal_unit(tmp_path, good=False)
    res = validate_run(run, load_ckpt=False)
    assert res.status in {"FAILED", "SKIPPED"}
    assert res.reasons


def test_validate_missing_ckpt_skipped(tmp_path: Path):
    run = _write_minimal_unit(tmp_path, good=True)
    run.ckpt_path.unlink()
    # reconstruct with missing ckpt path still pointing to deleted file
    res = validate_run(run, load_ckpt=False)
    assert res.status == "SKIPPED"


def test_write_validation_report(tmp_path: Path):
    run = _write_minimal_unit(tmp_path, good=True)
    res = validate_run(run)
    path = write_validation_report([res], tmp_path / "validation_report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_ready"] == 1
