"""SBS contract tests: FNA → feature table; features → assignment table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.common import read_csv, write_csv
from src.splits.sbs.assign import (
    ASSIGNMENT_COLUMNS,
    assign_from_features,
    assignment_rows_to_split_csv,
)
from src.splits.sbs.backends.gc import (
    GcAaaFeatureBackend,
    aaa_percent,
    gc_fraction,
    gc_percent,
)
from src.splits.sbs.features import FeatureTable, compute_feature_table
from src.splits.sbs.fna_io import load_fna_sequences


def _write_marked_dir(root: Path, records: dict[str, str]) -> Path:
    marked = root / "MARKED"
    marked.mkdir(parents=True)
    for rid, seq in records.items():
        (marked / f"{rid}.fa").write_text(
            f">genome|chr|1|10|g{rid}|t{rid}|{rid}\n{seq}\n",
            encoding="utf-8",
        )
    return marked


def test_contract_fna_to_feature_table_gc_aaa(tmp_path: Path) -> None:
    """C1: directory of per-ID FASTA → FeatureTable (GC_pct, AAA_pct)."""
    marked = _write_marked_dir(
        tmp_path,
        {
            "1": "AAAAAAAAAA",  # GC=0, high AAA
            "2": "CCCCCCCCCC",  # GC=100, AAA=0
            "3": "ACGTACGTAC",  # GC=50, AAA=0
        },
    )
    ft = compute_feature_table(marked, GcAaaFeatureBackend())
    assert isinstance(ft, FeatureTable)
    assert ft.n == 3
    assert ft.feature_names == ("GC_pct", "AAA_pct")
    assert ft.matrix.shape == (3, 2)
    i = {rid: idx for idx, rid in enumerate(ft.ids)}
    assert ft.matrix[i["1"], 0] == pytest.approx(0.0)
    assert ft.matrix[i["2"], 0] == pytest.approx(100.0)
    assert ft.matrix[i["3"], 0] == pytest.approx(50.0)
    assert ft.matrix[i["1"], 1] > 50.0
    assert ft.matrix[i["2"], 1] == pytest.approx(0.0)


def test_contract_fna_single_file_mode_features(tmp_path: Path) -> None:
    fa = tmp_path / "all.fna"
    fa.write_text(
        ">g|c|1|2|a|t|a\nAAAA\n>g|c|1|2|b|t|b\nGGGG\n>g|c|1|2|c|t|c\nACGT\n",
        encoding="utf-8",
    )
    seqs = load_fna_sequences(fa, mode="file")
    assert set(seqs) == {"a", "b", "c"}
    ft = compute_feature_table(fa, GcAaaFeatureBackend(), mode="file")
    assert ft.n == 3


def test_contract_features_to_assignment_table(tmp_path: Path) -> None:
    """C2: FeatureTable → region|cluster|train_test|fold|additional."""
    ids = tuple(str(i) for i in range(1, 13))
    # Two clear groups in GC%/AAA% space
    mat = np.zeros((12, 2), dtype=float)
    mat[:6, 0] = 10.0
    mat[:6, 1] = 80.0
    mat[6:, 0] = 90.0
    mat[6:, 1] = 5.0
    ft = FeatureTable(
        ids=ids,
        feature_names=("GC_pct", "AAA_pct"),
        matrix=mat,
        backend="gc_aaa",
    )
    fold_csv = tmp_path / "fold.csv"
    write_csv(
        fold_csv,
        [{"ID": "1", "fold": "zsv"}]
        + [{"ID": str(i), "fold": "0"} for i in range(2, 13)],
        ["ID", "fold"],
    )
    strat = tmp_path / "strat.csv"
    write_csv(
        strat,
        [
            {"ID": str(i), "strat1": "A" if i <= 6 else "B", "count": "1"}
            for i in range(1, 13)
        ],
        ["ID", "strat1", "count"],
    )
    rows, meta = assign_from_features(
        ft,
        fold_csv=fold_csv,
        stratification_csv=strat,
        seed=42,
        n_clusters=2,
        cluster_method="kmeans",
    )
    assert set(ASSIGNMENT_COLUMNS) <= set(rows[0])
    by_id = {r["region"]: r for r in rows}
    assert by_id["1"]["train_test"] == "zsv"
    assert by_id["1"]["fold"] == "zsv"
    non_zsv = [r for r in rows if r["train_test"] != "zsv"]
    assert len(non_zsv) == 11
    by_fold: dict[str, set[str]] = {}
    for r in non_zsv:
        by_fold.setdefault(r["fold"], set()).add(r["train_test"])
    assert all(len(v) == 1 for v in by_fold.values())
    assert meta["n_zsv"] == 1

    split_csv = assignment_rows_to_split_csv(rows, tmp_path / "out")
    split_rows = read_csv(split_csv)
    assert {r["ID"] for r in split_rows} == set(ids)


def test_composition_helpers() -> None:
    assert gc_fraction("AT") == pytest.approx(0.0)
    assert gc_percent("GC") == pytest.approx(100.0)
    assert aaa_percent("AAA") == pytest.approx(100.0)
    assert aaa_percent("AAAA") == pytest.approx(100.0)  # 2/2 overlapping
    assert aaa_percent("ACGT") == pytest.approx(0.0)
    assert aaa_percent("NN") == pytest.approx(0.0)
