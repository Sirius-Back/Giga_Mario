"""Tests for aligned pipeline rerun helpers + Hydra wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.common import write_csv
from src.pipeline.rerun_aligned import (
    ALIGNED_EPOCHS,
    ALIGNED_MIN_EPOCHS,
    ALIGNED_RATIOS,
    apply_rerun_schedule,
    assert_fresh_out_root,
    is_aligned_ratios,
    parse_override_keys,
    require_aligned_ratios,
    resolve_source_artifacts,
    stage_source_into_out_root,
)


def _write_split(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        path,
        [
            {"ID": "a", "train_test": "train", "fold": "0"},
            {"ID": "b", "train_test": "test", "fold": "0"},
            {"ID": "c", "train_test": "val", "fold": "0"},
        ],
        ["ID", "train_test", "fold"],
    )
    return path


def test_aligned_ratios_accept_3_1_1_and_fractions() -> None:
    assert is_aligned_ratios((3, 1, 1))
    assert is_aligned_ratios((0.6, 0.2, 0.2))
    assert is_aligned_ratios((6, 2, 2))
    assert not is_aligned_ratios((1, 1, 3))
    assert not is_aligned_ratios((0.81, 0.1, 0.09))
    assert require_aligned_ratios(None) == ALIGNED_RATIOS
    with pytest.raises(ValueError, match="3:1:1"):
        require_aligned_ratios((1, 1, 3))


def test_resolve_and_stage_does_not_mutate_source(tmp_path: Path) -> None:
    src_root = tmp_path / "runs" / "prior"
    split = _write_split(src_root / "split.csv")
    (src_root / "gc_features").mkdir()
    (src_root / "gc_features" / "feat.txt").write_text("x\n", encoding="utf-8")
    src_text = split.read_text(encoding="utf-8")

    arts = resolve_source_artifacts(src_root, project_root=tmp_path)
    assert arts.split_csv == split.resolve()
    assert "gc_features" in arts.intermediates

    out = tmp_path / "runs_aligned" / "new"
    info = stage_source_into_out_root(arts, out, include_split_tree=False)
    assert (out / "split.csv").is_file()
    assert (out / "gc_features" / "feat.txt").is_file()
    assert not (out / "SPLIT").exists()
    assert info["reuse_folds"] is True
    # Source inode content unchanged
    assert split.read_text(encoding="utf-8") == src_text
    # Dest split is a real copy (overwrite source text must not change dest)
    split.write_text(src_text + "# mutated\n", encoding="utf-8")
    assert "# mutated" not in (out / "split.csv").read_text(encoding="utf-8")


def test_assert_fresh_out_root_refuses_existing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    assert_fresh_out_root(out)
    (out / "split.csv").write_text("ID|train_test|fold\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        assert_fresh_out_root(out)


def test_apply_rerun_schedule_defaults_and_overrides() -> None:
    e, m, p = apply_rerun_schedule(
        epochs=50, min_epochs=25, early_stopping_patience=0, overridden=set()
    )
    assert (e, m, p) == (ALIGNED_EPOCHS, ALIGNED_MIN_EPOCHS, 10)
    assert m == 15
    e2, m2, p2 = apply_rerun_schedule(
        epochs=20,
        min_epochs=12,
        early_stopping_patience=5,
        overridden={"epochs", "min_epochs", "early_stopping_patience"},
    )
    assert (e2, m2, p2) == (20, 12, 5)


def test_rewrite_split_table_swap_train_val(tmp_path: Path) -> None:
    from src.pipeline.common import write_csv
    from src.pipeline.rerun_aligned import rewrite_split_table_aligned

    src = tmp_path / "legacy" / "split.csv"
    rows = (
        [{"ID": f"t{i}", "train_test": "train", "fold": "0"} for i in range(20)]
        + [{"ID": f"e{i}", "train_test": "test", "fold": "0"} for i in range(20)]
        + [{"ID": f"v{i}", "train_test": "val", "fold": "0"} for i in range(60)]
        + [{"ID": "z0", "train_test": "zsv", "fold": "zsv"}]
    )
    write_csv(src, rows, ["ID", "train_test", "fold"])
    dest = tmp_path / "unif" / "split.csv"
    info = rewrite_split_table_aligned(src, dest, prefer_label_swap=True)
    assert info["method"] == "swap_train_val"
    assert info["counts_after"]["train"] == 60
    assert info["counts_after"]["test"] == 20
    assert info["counts_after"]["val"] == 20
    assert info["counts_after"]["zsv"] == 1
    # source untouched
    assert "t0|train|" in src.read_text(encoding="utf-8")


def test_rewrite_refuses_id_reassign_without_assignment(tmp_path: Path) -> None:
    from src.pipeline.common import write_csv
    from src.pipeline.rerun_aligned import rewrite_split_table_aligned

    src = tmp_path / "legacy" / "split.csv"
    # Sizes that no pairwise swap maps to ≈3:1:1
    rows = (
        [{"ID": f"t{i}", "train_test": "train", "fold": "0"} for i in range(27)]
        + [{"ID": f"e{i}", "train_test": "test", "fold": "0"} for i in range(56)]
        + [{"ID": f"v{i}", "train_test": "val", "fold": "0"} for i in range(17)]
        + [{"ID": "z0", "train_test": "zsv", "fold": "zsv"}]
    )
    write_csv(src, rows, ["ID", "train_test", "fold"])
    dest = tmp_path / "unif" / "split.csv"
    with pytest.raises(RuntimeError, match="ID-level reassign is disabled"):
        rewrite_split_table_aligned(
            src, dest, prefer_label_swap=True, allow_id_reassign=False
        )


def test_rewrite_from_sbs_assignment_cluster_grain(tmp_path: Path) -> None:
    from src.pipeline.common import write_csv
    from src.pipeline.rerun_aligned import (
        is_aligned_ratios,
        rewrite_split_from_sbs_assignment,
    )
    from src.splits.sbs.assign import ASSIGNMENT_COLUMNS

    assign = tmp_path / "legacy" / "sbs_assignment.csv"
    # Ten equal clusters of 10 → fold-grain greedy can hit exact 6:2:2 (=3:1:1).
    rows = []
    n = 0
    for fold in range(10):
        for _ in range(10):
            rows.append(
                {
                    "region": str(n),
                    "cluster": str(fold),
                    "train_test": "train",
                    "fold": str(fold),
                    "additional": "",
                }
            )
            n += 1
    rows.append(
        {
            "region": "z0",
            "cluster": "zsv",
            "train_test": "zsv",
            "fold": "zsv",
            "additional": "",
        }
    )
    write_csv(assign, rows, ASSIGNMENT_COLUMNS)
    dest = tmp_path / "unif" / "split.csv"
    info = rewrite_split_from_sbs_assignment(assign, dest, seed=42)
    assert info["method"] == "sbs_cluster_to_train_test_val"
    after = info["counts_after"]
    assert is_aligned_ratios((after["train"], after["test"], after["val"]))
    assert after["zsv"] == 1
    assert (tmp_path / "unif" / "sbs_assignment.csv").is_file()
    # each ID once; no train/test/val intersections
    by_tt: dict[str, set[str]] = {"train": set(), "test": set(), "val": set()}
    seen: set[str] = set()
    for line in dest.read_text(encoding="utf-8").strip().splitlines()[1:]:
        rid, tt, _fold = line.split("|")
        assert rid not in seen
        seen.add(rid)
        if tt in by_tt:
            by_tt[tt].add(rid)
    assert not (by_tt["train"] & by_tt["test"])
    assert not (by_tt["train"] & by_tt["val"])
    assert not (by_tt["test"] & by_tt["val"])
    # whole-cluster integrity: all members of a fold share train_test
    fold_labels: dict[str, set[str]] = {}
    for line in dest.read_text(encoding="utf-8").strip().splitlines()[1:]:
        _rid, tt, fold = line.split("|")
        if tt == "zsv":
            continue
        fold_labels.setdefault(fold, set()).add(tt)
    assert all(len(v) == 1 for v in fold_labels.values())


def test_parse_override_keys() -> None:
    keys = parse_override_keys(
        ["mode=run", "rerun=true", "+epochs=20", "train=caduceus"]
    )
    assert "epochs" in keys
    assert "rerun" in keys
    assert "train" in keys


def test_hydra_pipeline_rerun_compose_defaults() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    cfg_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "mode=dry",
                "rerun=true",
                "source_split=runs/run3",
                "run_id=run3_aligned",
                "adversarial=false",
            ],
        )
    assert cfg.rerun is True
    assert cfg.source_split == "runs/run3"
    assert cfg.mode == "dry"
