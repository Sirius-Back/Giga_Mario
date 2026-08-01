"""Tests for paralogs_only orthogroup-representative split."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.common import read_csv, write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.paralogs_only import run_paralogs_only_split_assign
from src.splits.paralogs_only_native import ensure_built


def _write_mock_homology(tmp: Path) -> tuple[Path, Path]:
    """Two OGs + one unmapped panel id.

    OG0: A1--ortholog--A2 (A1 higher paralog degree via self-ish para edges)
    OG1: B1 alone (no ortho edges)
    Unmapped: U1 (in panel, not in nodes)
    """
    nodes = tmp / "nodes.tsv"
    nodes.write_text(
        "ensembl_species\tensembl_gene\tgene_symbol\tmarked_id\tgcf\n"
        "sp_a\tGENE_A1\tA1\tA1\tg1\n"
        "sp_b\tGENE_A2\tA2\tA2\tg2\n"
        "sp_a\tGENE_A1p\tA1p\tA1p\tg1\n"  # paralog of A1
        "sp_c\tGENE_B1\tB1\tB1\tg3\n"
        "sp_c\tGENE_B1p\tB1p\tB1p\tg3\n"  # paralog of B1 (same species)
        ,
        encoding="utf-8",
    )
    edges = tmp / "edges.tsv"
    edges.write_text(
        "gene1\tgenome1\tgene2\tgenome2\trelation\n"
        "GENE_A1\tsp_a\tGENE_A2\tsp_b\tortholog\n"
        "GENE_A1\tsp_a\tGENE_A1p\tsp_a\tparalog\n"
        "GENE_A1\tsp_a\tGENE_B1p\tsp_c\tparalog\n"  # boost A1 degree
        "GENE_B1\tsp_c\tGENE_B1p\tsp_c\tparalog\n"
        ,
        encoding="utf-8",
    )
    return edges, nodes


def _write_panel(tmp: Path, ids: list[str], zsv: set[str] | None = None) -> tuple[Path, Path]:
    zsv = zsv or set()
    id_rows = []
    fold_rows = []
    for i, rid in enumerate(ids, start=1):
        id_rows.append(
            {
                "genome": "G",
                "chr": "1",
                "pos1": str(i),
                "pos2": str(i + 1),
                "gene_nameORnon_coding_ID": rid,
                "raw_target_ID": rid,
                "ID": rid,
            }
        )
        fold_rows.append({"ID": rid, "fold": "zsv" if rid in zsv else "0"})
    id_csv = tmp / "ID.csv"
    fold_csv = tmp / "fold.csv"
    write_csv(
        id_csv,
        id_rows,
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
    write_csv(fold_csv, fold_rows, ["ID", "fold"])
    return id_csv, fold_csv


def test_native_builds() -> None:
    path = ensure_built(force=True)
    assert path.is_file()


def test_one_rep_per_og_and_unmapped_not_train(tmp_path: Path) -> None:
    edges, nodes = _write_mock_homology(tmp_path)
    # Panel: A1,A2,A1p,B1,B1p,U1(unmapped), ZSV_X
    ids = ["A1", "A2", "A1p", "B1", "B1p", "U1", "ZSV_X"]
    id_csv, fold_csv = _write_panel(tmp_path, ids, zsv={"ZSV_X"})
    outdir = tmp_path / "out"
    summary = run_paralogs_only_split_assign(
        outdir=outdir,
        id_csv=id_csv,
        homology_edges=edges,
        homology_nodes=nodes,
        fold_csv=fold_csv,
        seed=42,
    )
    rows = read_csv(Path(summary["split_csv"]))
    by = {r["ID"]: r for r in rows}

    assert by["ZSV_X"]["train_test"] == "zsv"
    # Unmapped never train
    assert by["U1"]["train_test"] in {"test", "val"}
    assert by["U1"]["fold"] == "unmapped"
    assert by["U1"]["train_test"] != "train"

    train = {r["ID"] for r in rows if r["train_test"] == "train"}
    # A1 and A2 share an OG → exactly one of them in train
    assert len(train & {"A1", "A2"}) == 1
    # A1 has higher paralog degree → preferred
    assert "A1" in train
    assert "A2" not in train
    # B1 alone in its OG (B1p only connected via paralog) → B1 is own OG, B1p own OG
    # B1 and B1p have no ortholog edge → two OGs → both can be train reps
    assert "B1" in train
    assert "B1p" in train

    # No train has fold=unmapped
    assert all(by[t]["fold"] != "unmapped" for t in train)

    # Remainder 50/50 among non-train non-zsv
    rem = [r for r in rows if r["train_test"] in {"test", "val"}]
    assert rem
    n_test = sum(1 for r in rem if r["train_test"] == "test")
    n_val = sum(1 for r in rem if r["train_test"] == "val")
    assert abs(n_test - n_val) <= 1


def test_split_predict_type_paralogs_only(tmp_path: Path) -> None:
    edges, nodes = _write_mock_homology(tmp_path)
    id_csv, fold_csv = _write_panel(tmp_path, ["A1", "A2", "U1", "A1p"])
    out = run_split_predict(
        outdir=tmp_path / "sp",
        type="paralogs_only",
        seed=7,
        id_csv=id_csv,
        fold_csv=fold_csv,
        homology_edges=edges,
        homology_nodes=nodes,
    )
    rows = read_csv(out)
    assert {r["ID"] for r in rows} == {"A1", "A2", "U1", "A1p"}
    assert all(r["train_test"] in {"train", "test", "val"} for r in rows)
    unmapped = [r for r in rows if r["ID"] == "U1"][0]
    assert unmapped["train_test"] != "train"
