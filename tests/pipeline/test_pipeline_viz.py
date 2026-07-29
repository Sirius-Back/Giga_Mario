"""Tests for pipeline visualization helpers."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.pipeline_viz import (
    assignment_rows_from_split_csv,
    has_viz_deps,
    resolve_sequence_dir,
)
from src.splits.sbs.fna_io import load_fna_directory


def test_assignment_rows_from_split_csv(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    split.write_text(
        "ID|train_test|fold\n"
        "a|train|0\n"
        "b|val|1\n"
        "c|zsv|zsv\n",
        encoding="utf-8",
    )
    rows = assignment_rows_from_split_csv(split)
    assert len(rows) == 3
    assert rows[0]["region"] == "a"
    assert rows[0]["train_test"] == "train"
    assert rows[0]["cluster"] == "0"
    assert rows[2]["train_test"] == "zsv"


def test_load_parsed_ext_raw_sequence(tmp_path: Path) -> None:
    d = tmp_path / "PARSED"
    d.mkdir()
    (d / "r1.ext").write_text("ACGTACGTAAA\n", encoding="utf-8")
    (d / "r2.ext").write_text(">hdr\nGGGCCC\n", encoding="utf-8")
    seqs = load_fna_directory(d)
    assert seqs["r1"] == "ACGTACGTAAA"
    assert seqs["r2"] == "GGGCCC"


def test_resolve_sequence_dir_prefers_marked(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    (panel / "MARKED").mkdir(parents=True)
    (panel / "MARKED" / "x.fa").write_text(">x\nAAA\n", encoding="utf-8")
    (panel / "PARSED").mkdir()
    (panel / "PARSED" / "x.ext").write_text("TTT\n", encoding="utf-8")
    assert resolve_sequence_dir(panel).name == "MARKED"


def test_has_viz_deps_bool() -> None:
    assert isinstance(has_viz_deps(), bool)
