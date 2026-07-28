"""Integration coverage for the universal split-predict random output."""
from __future__ import annotations

import csv
from collections import Counter
import os
from pathlib import Path

from src.pipeline.common import read_csv
from src.pipeline.split import run_split
from src.pipeline.split_predict import run_split_predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROKARYOTE_IDS = PROJECT_ROOT / "output/pipeline_prok/id_gen/ID.csv"


def test_random_split_predict_real_prokaryote_ids_format_and_counts(tmp_path: Path) -> None:
    """Use the real 37,791-row ID table and verify the public split contract."""
    assert PROKARYOTE_IDS.is_file()

    split_path = run_split_predict(
        outdir=tmp_path / "split_predict_random",
        id_csv=PROKARYOTE_IDS,
        seed=42,
    )

    lines = split_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ID|train_test|fold"

    with split_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))

    assert len(rows) == 37_791
    assert set(row["ID"] for row in rows) == {str(i) for i in range(1, 37_792)}
    assert {row["fold"] for row in rows} == {"0"}
    assert Counter(row["train_test"] for row in rows) == {
        "train": 30_611,
        "test": 3_779,
        "val": 3_401,
    }


def test_split_materialize_real_prokaryote_artifacts(tmp_path: Path) -> None:
    """Materialize the complete Caduceus panel from actual pipeline outputs."""
    root = PROJECT_ROOT / "output/pipeline_prok"
    split_root = run_split(
        root / "split_predict_random/split.csv",
        root / "parse_target_merged",
        root / "parse_data_caduceus",
        outdir=tmp_path / "split_caduceus",
        strategy="traintestval",
    )

    expected = {"TRAIN": 30_611, "TEST": 3_779, "VAL": 3_401}
    for bucket, expected_count in expected.items():
        predict_dir = split_root / "PREDICT" / bucket
        fasta_dir = split_root / "FASTA" / bucket
        rows = read_csv(predict_dir / "predict.csv")
        assert len(rows) == expected_count
        assert len(list(predict_dir.glob("*.ext"))) == expected_count
        assert len(list(fasta_dir.glob("*.ext"))) == expected_count

    first_split_row = read_csv(root / "split_predict_random/split.csv")[0]
    first_id = first_split_row["ID"]
    assert os.path.samefile(
        root / "parse_target_merged/PREDICT" / f"{first_id}.ext",
        split_root / "PREDICT" / first_split_row["train_test"].upper() / f"{first_id}.ext",
    )
