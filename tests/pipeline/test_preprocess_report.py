"""Tests for non-agentic preprocess_report / parse.md writer."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.common import write_csv
from src.preprocess_report import collect_preprocess_checks, write_parse_md


def test_write_parse_md_mini_panel(tmp_path: Path) -> None:
    id_csv = tmp_path / "ID.csv"
    write_csv(
        id_csv,
        [
            {
                "genome": "G1",
                "chr": "c",
                "pos1": "1",
                "pos2": "10",
                "gene_nameORnon_coding_ID": "g1",
                "raw_target_ID": "t1",
                "ID": "1",
            }
        ],
        [
            "genome",
            "chr",
            "pos1",
            "pos2",
            "gene_nameORnon_coding_ID",
            "raw_target_ID",
            "ID",
        ],
    )
    marked = tmp_path / "MARKED"
    marked.mkdir()
    (marked / "1.fa").write_text(">|G1|c|1|10|g1|t1|1\nACGT\n", encoding="utf-8")
    parsed = tmp_path / "PARSED"
    parsed.mkdir()
    (parsed / "1.ext").write_text("ACGT\n", encoding="utf-8")
    predict = tmp_path / "PREDICT"
    predict.mkdir()
    (predict / "1.ext").write_text("1.5\n", encoding="utf-8")
    write_csv(
        predict / "predict.csv",
        [{"id": "1", "predict_var1": "1.5"}],
        ["id", "predict_var1"],
    )

    out = write_parse_md(tmp_path, id_csv=id_csv)
    text = out.read_text(encoding="utf-8")
    assert out.name == "parse.md"
    assert "Overall OK: True" in text
    assert "id_csv" in text
    checks = collect_preprocess_checks(outdir=tmp_path, id_csv=id_csv)
    assert checks["ok"] is True


def test_write_parse_md_real_pipeline_prok_if_present() -> None:
    root = Path(__file__).resolve().parents[1] / "output" / "pipeline_prok"
    if not root.is_dir():
        return
    # Prefer archived or live panel layout
    id_csv = root / "id_gen" / "ID.csv"
    if not id_csv.is_file():
        return
    # Build a thin view dir with symlinks/copies of stage roots for the checker
    # collect_preprocess_checks already searches common subdir names.
    checks = collect_preprocess_checks(outdir=root, id_csv=id_csv)
    assert checks["id_csv"]["ok"] is True
    assert checks["id_csv"]["n_rows"] > 1000
    md = write_parse_md(root, id_csv=id_csv, filename="parse_pytest.md")
    assert md.is_file()
    assert "Preprocess report" in md.read_text(encoding="utf-8")
