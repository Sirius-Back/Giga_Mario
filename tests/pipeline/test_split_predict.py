"""Coverage for generate_fold, mapped PREDICT split, ZSV, intersect_allow."""
from __future__ import annotations

import csv
import warnings
from collections import Counter
from pathlib import Path

import pytest

from src.pipeline.common import read_csv, write_csv
from src.pipeline.generate_fold import run_generate_fold
from src.pipeline.split import run_split
from src.pipeline.split_predict import run_split_predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tests.pipeline.conftest import resolve_pipeline_prok

_PROK = resolve_pipeline_prok()
PROKARYOTE_IDS = (_PROK / "id_gen" / "ID.csv") if _PROK else PROJECT_ROOT / "output/pipeline_prok/id_gen/ID.csv"


def test_generate_fold_from_prepare_rules(tmp_path: Path) -> None:
    id_csv = tmp_path / "ID.csv"
    id_csv.write_text(
        "genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID\n"
        "GCF_A|chr1|1|10|g1|t1|1\n"
        "GCF_A|chr1|20|30|g2|t2|2\n"
        "GCF_B|chr1|1|10|g3|t3|3\n",
        encoding="utf-8",
    )
    prep = tmp_path / "prepare_fold.csv"
    prep.write_text(
        "identificator|column|fold\n"
        "GCF_B|genome|zsv\n"
        "GCF_A|genome|1\n",
        encoding="utf-8",
    )
    out = run_generate_fold(id_csv, prep, outdir=tmp_path / "folds")
    rows = {r["ID"]: r["fold"] for r in read_csv(out)}
    assert rows == {"1": "1", "2": "1", "3": "zsv"}


def test_generate_fold_semicolon_prepare(tmp_path: Path) -> None:
    id_csv = tmp_path / "ID.csv"
    id_csv.write_text(
        "genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID\n"
        "GENOME_ID|chr1|1|10|g1|t1|1\n",
        encoding="utf-8",
    )
    prep = tmp_path / "prepare_fold.csv"
    prep.write_text("identificator;column;fold\nGENOME_ID;genome;zsv\n", encoding="utf-8")
    out = run_generate_fold(id_csv, prep, outdir=tmp_path / "folds")
    assert read_csv(out)[0]["fold"] == "zsv"


def test_generate_fold_uses_id_rule(tmp_path: Path) -> None:
    """Rules resolve IDs via id_rule; unknown identificator leaves default_fold."""
    id_csv = tmp_path / "ID.csv"
    id_csv.write_text(
        "genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID\n"
        "GCF_A|chr1|1|10|g1|t1|1\n"
        "GCF_A|chr1|20|30|g2|t2|2\n"
        "GCF_B|chr1|1|10|g3|t3|3\n",
        encoding="utf-8",
    )
    prep = tmp_path / "prepare_fold.csv"
    prep.write_text(
        "identificator|column|fold\n"
        "GCF_A|genome|train\n"
        "MISSING_GENOME|genome|zsv\n",
        encoding="utf-8",
    )
    out = run_generate_fold(id_csv, prep, outdir=tmp_path / "folds", default_fold="0")
    rows = {r["ID"]: r["fold"] for r in read_csv(out)}
    assert rows == {"1": "train", "2": "train", "3": "0"}


def test_split_predict_warns_without_fold_and_holds_zsv(tmp_path: Path) -> None:
    id_csv = tmp_path / "ID.csv"
    write_csv(
        id_csv,
        [
            {
                "genome": "g",
                "chr": "c",
                "pos1": "1",
                "pos2": "2",
                "gene_nameORnon_coding_ID": f"g{i}",
                "raw_target_ID": f"t{i}",
                "ID": str(i),
            }
            for i in range(1, 7)
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
    fold_csv = tmp_path / "fold.csv"
    write_csv(
        fold_csv,
        [{"ID": "1", "fold": "zsv"}, {"ID": "2", "fold": "0"}, {"ID": "3", "fold": "0"},
         {"ID": "4", "fold": "0"}, {"ID": "5", "fold": "0"}, {"ID": "6", "fold": "0"}],
        ["ID", "fold"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_split_predict(outdir=tmp_path / "no_fold", id_csv=id_csv, seed=42)
    assert any("Warning: folds are not included" in str(w.message) for w in caught)

    split_csv = run_split_predict(
        outdir=tmp_path / "with_fold", id_csv=id_csv, fold_csv=fold_csv, seed=42
    )
    rows = read_csv(split_csv)
    by_id = {r["ID"]: r for r in rows}
    assert by_id["1"]["train_test"] == "zsv"
    assert by_id["1"]["fold"] == "zsv"
    assert Counter(r["train_test"] for r in rows if r["train_test"] != "zsv")["train"] >= 1
    assert all(r["train_test"] in {"train", "test", "val", "zsv"} for r in rows)


def test_split_predict_stratifies_all_columns(tmp_path: Path) -> None:
    ids = [str(i) for i in range(1, 13)]
    id_csv = tmp_path / "ID.csv"
    write_csv(
        id_csv,
        [
            {
                "genome": "g",
                "chr": "c",
                "pos1": "1",
                "pos2": "2",
                "gene_nameORnon_coding_ID": f"g{i}",
                "raw_target_ID": f"t{i}",
                "ID": i,
            }
            for i in ids
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
    strat = tmp_path / "strat.csv"
    write_csv(
        strat,
        [
            {"ID": i, "strat1": "A" if int(i) <= 6 else "B", "strat2": str(int(i) % 2)}
            for i in ids
        ],
        ["ID", "strat1", "strat2"],
    )
    split_csv = run_split_predict(
        outdir=tmp_path / "strat",
        id_csv=id_csv,
        stratification_csv=strat,
        fold_csv=None,
        seed=42,
    )
    assert len(read_csv(split_csv)) == 12


def test_split_predict_accepts_custom_train_test_val_ratios(tmp_path: Path) -> None:
    """Explicit 1:1:3 weights yield the requested five-way allocation."""
    id_csv = tmp_path / "ID.csv"
    write_csv(
        id_csv,
        [{"ID": str(index)} for index in range(1, 101)],
        ["ID"],
    )
    split_csv = run_split_predict(
        outdir=tmp_path / "custom_ratios",
        id_csv=id_csv,
        seed=42,
        ratios=(1, 1, 3),
    )
    counts = Counter(row["train_test"] for row in read_csv(split_csv))
    assert counts == {"train": 20, "test": 20, "val": 60}


def test_split_mapped_predict_and_zsv_aside(tmp_path: Path) -> None:
    """Mapped PREDICT flattens to composite unique ids shared by PREDICT/FASTA/predict.csv."""
    from src.pipeline.common import make_mapped_unique_id
    from src.pipeline.train import adapt_split_for_caduceus

    parsed = tmp_path / "PARSED"
    predict = tmp_path / "PREDICT"
    parsed.mkdir()
    predict.mkdir()
    samples = ["S1", "S2"]
    ids = ["1", "2", "3", "4"]
    pred_rows = []
    for rid in ids:
        (parsed / f"{rid}.ext").write_text(f"SEQ{rid}\n", encoding="utf-8")
        for sample in samples:
            sdir = predict / sample
            sdir.mkdir(exist_ok=True)
            (sdir / f"{rid}.ext").write_text(f"{rid}:{sample}\n", encoding="utf-8")
            pred_rows.append({"id": rid, "sample_id": sample, "predict_var1": rid})
    write_csv(predict / "predict.csv", pred_rows, ["id", "sample_id", "predict_var1"])

    split_csv = tmp_path / "split.csv"
    write_csv(
        split_csv,
        [
            {"ID": "1", "train_test": "train", "fold": "0"},
            {"ID": "2", "train_test": "test", "fold": "0"},
            {"ID": "3", "train_test": "val", "fold": "0"},
            {"ID": "4", "train_test": "zsv", "fold": "zsv"},
        ],
        ["ID", "train_test", "fold"],
    )

    outdir = tmp_path / "out"
    split_root = run_split(
        split_csv, predict, parsed, outdir=outdir, strategy="traintestval", intersect_allow=False
    )
    uid_s1 = make_mapped_unique_id("S1", "1")
    uid_s2 = make_mapped_unique_id("S2", "1")
    assert (split_root / "PREDICT" / "TRAIN" / f"{uid_s1}.ext").is_file()
    assert (split_root / "PREDICT" / "TRAIN" / f"{uid_s2}.ext").is_file()
    assert (split_root / "FASTA" / "TRAIN" / f"{uid_s1}.ext").is_file()
    assert (split_root / "FASTA" / "TRAIN" / f"{uid_s2}.ext").is_file()
    assert not (split_root / "PREDICT" / "TRAIN" / f"{make_mapped_unique_id('S1', '4')}.ext").exists()
    assert (outdir / "PREDICT" / "zero-shot-validation" / f"{make_mapped_unique_id('S1', '4')}.ext").is_file()
    assert (outdir / "PARSED" / "zero-shot-validation" / f"{make_mapped_unique_id('S1', '4')}.ext").is_file()
    train_pred = read_csv(split_root / "PREDICT" / "TRAIN" / "predict.csv")
    assert {r["id"] for r in train_pred} == {uid_s1, uid_s2}
    assert len(train_pred) == 2
    assert len({r["id"] for r in train_pred}) == len(train_pred)
    assert all(r["region_id"] == "1" for r in train_pred)

    adapted, counts = adapt_split_for_caduceus(
        split_root, outdir=tmp_path / "cad_adapt", task_type="regression"
    )
    assert counts["train"] == 2
    assert (adapted / "train" / "sequences" / f"{uid_s1}.txt").is_file()


def test_split_intersect_allow_skips_missing(tmp_path: Path) -> None:
    parsed = tmp_path / "PARSED"
    predict = tmp_path / "PREDICT"
    parsed.mkdir()
    predict.mkdir()
    (parsed / "1.ext").write_text("A\n", encoding="utf-8")
    (predict / "1.ext").write_text("1.0\n", encoding="utf-8")
    # ID 2 missing PARSED
    (predict / "2.ext").write_text("2.0\n", encoding="utf-8")
    write_csv(
        predict / "predict.csv",
        [{"id": "1", "predict_var1": "1"}, {"id": "2", "predict_var1": "2"}],
        ["id", "predict_var1"],
    )
    split_csv = tmp_path / "split.csv"
    write_csv(
        split_csv,
        [
            {"ID": "1", "train_test": "train", "fold": "0"},
            {"ID": "2", "train_test": "train", "fold": "0"},
        ],
        ["ID", "train_test", "fold"],
    )
    with pytest.raises(FileNotFoundError):
        run_split(
            split_csv,
            predict,
            parsed,
            outdir=tmp_path / "strict",
            intersect_allow=False,
        )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        split_root = run_split(
            split_csv,
            predict,
            parsed,
            outdir=tmp_path / "allow",
            intersect_allow=True,
        )
    assert (split_root / "PREDICT" / "TRAIN" / "1.ext").is_file()
    assert not (split_root / "PREDICT" / "TRAIN" / "2.ext").exists()


def test_random_split_predict_real_prokaryote_ids_format_and_counts(tmp_path: Path) -> None:
    """Use the real 37,791-row ID table and verify the public split contract."""
    assert PROKARYOTE_IDS.is_file()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
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
    root = resolve_pipeline_prok() or PROJECT_ROOT / "output/pipeline_prok"
    split_root = run_split(
        root / "split_predict_random/split.csv",
        root / "parse_target_merged",
        root / "parse_data_caduceus",
        outdir=tmp_path / "split_caduceus",
        strategy="traintestval",
        intersect_allow=True,
    )

    expected = {"TRAIN": 30_611, "TEST": 3_779, "VAL": 3_401}
    for bucket, expected_count in expected.items():
        predict_dir = split_root / "PREDICT" / bucket
        fasta_dir = split_root / "FASTA" / bucket
        rows = read_csv(predict_dir / "predict.csv")
        assert len(rows) == expected_count
        assert len(list(predict_dir.glob("*.ext"))) == expected_count
        assert len(list(fasta_dir.glob("*.ext"))) == expected_count
