"""hashFrag strategy: unit helpers + optional CLI smoke (BLAST+hashFrag)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.pipeline.common import read_csv, write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.hashfrag import (
    from_fasta_token,
    marked_to_multifasta,
    parse_hashfrag_split_tsv,
    run_hashfrag_split_assign,
    to_fasta_token,
)


def _hashfrag_available() -> bool:
    return all(
        shutil.which(x) for x in ("hashFrag", "blastn", "makeblastdb")
    )


def _mock_panel(tmp_path: Path, n: int = 20) -> tuple[Path, Path, Path]:
    marked = tmp_path / "MARKED"
    marked.mkdir()
    id_rows = []
    fold_rows = []
    for i in range(1, n + 1):
        rid = str(i)
        if i <= n // 2:
            # Near-identical AT-rich family (homology within group)
            seq = ("A" * 40) + ("T" * 40)
            if i % 2 == 0:
                seq = ("A" * 38) + "GG" + ("T" * 40)
            genome = "GCF_A"
        else:
            seq = ("G" * 40) + ("C" * 40)
            genome = "GCF_B"
        (marked / f"{rid}.fa").write_text(
            f">{genome}|chr1|{i}|{i + 80}|g{i}|t{i}|{rid}\n{seq}\n",
            encoding="utf-8",
        )
        id_rows.append(
            {
                "genome": genome,
                "chr": "chr1",
                "pos1": str(i),
                "pos2": str(i + 80),
                "gene_nameORnon_coding_ID": f"g{i}",
                "raw_target_ID": f"t{i}",
                "ID": rid,
            }
        )
        fold_rows.append({"ID": rid, "fold": "zsv" if i == 1 else "0"})
    id_csv = tmp_path / "ID.csv"
    fold_csv = tmp_path / "fold.csv"
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
    return marked, id_csv, fold_csv


def test_fasta_token_roundtrip() -> None:
    assert to_fasta_token("100000") == "hf_100000"
    assert from_fasta_token("hf_100000") == "100000"
    assert from_fasta_token("hf_100000_Reversed") is None
    assert from_fasta_token("100000") is None


def test_assign_orthogonal_off_by_one_fix(tmp_path: Path) -> None:
    """Upstream round(N*0.9)+round(N*0.1) fails for some N; our assigner always sums to N."""
    from src.splits.hashfrag import assign_orthogonal_from_groups

    path = tmp_path / "groups.tsv"
    lines = []
    for i in range(1, 6):
        lines.append(f"{to_fasta_token(str(i))}\t{i}")
        lines.append(f"{to_fasta_token(str(i))}_Reversed\t{i}")
    # N=5 (.5 rounding case for 0.9/0.1)
    path.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    labels = assign_orthogonal_from_groups(path, p_train=0.9, p_test=0.1, seed=42)
    assert set(labels.values()) <= {"train", "test"}
    assert "train" in labels.values() and "test" in labels.values()
    assert len(labels) >= 2


def test_assign_orthogonal_group_grain(tmp_path: Path) -> None:
    from src.splits.hashfrag import assign_orthogonal_from_groups

    path = tmp_path / "groups.tsv"
    rows = []
    for i in range(1, 7):
        gid = "0" if i <= 3 else "1"
        rows.append(f"{to_fasta_token(str(i))}\t{gid}")
        rows.append(f"{to_fasta_token(str(i))}_Reversed\t{gid}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    labels = assign_orthogonal_from_groups(path, p_train=0.5, p_test=0.5, seed=0)
    assert set(labels) == {str(i) for i in range(1, 7)}
    assert labels["1"] == labels["2"] == labels["3"]
    assert labels["4"] == labels["5"] == labels["6"]
    assert labels["1"] != labels["4"]


def test_marked_to_multifasta_headers(tmp_path: Path) -> None:
    marked, id_csv, _ = _mock_panel(tmp_path, n=5)
    ids = [r["ID"] for r in read_csv(id_csv)]
    out = marked_to_multifasta(marked, tmp_path / "all.fa", ids=ids)
    text = out.read_text(encoding="utf-8")
    assert ">hf_1\n" in text
    assert ">1\n" not in text


def test_parse_hashfrag_split_tsv(tmp_path: Path) -> None:
    path = tmp_path / "hashFrag.train_2.test_2.split_001.tsv"
    path.write_text(
        "id\tsplit\n"
        "hf_2\ttrain\n"
        "hf_2_Reversed\ttrain\n"
        "hf_3\ttest\n"
        "hf_3_Reversed\ttest\n",
        encoding="utf-8",
    )
    labels = parse_hashfrag_split_tsv(path)
    assert labels == {"2": "train", "3": "test"}


def test_hashfrag_requires_threshold(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path)
    with pytest.raises(ValueError, match="threshold"):
        run_split_predict(
            outdir=tmp_path / "out",
            type="hashfrag",
            id_csv=id_csv,
            fold_csv=fold_csv,
            marked_fasta=marked,
            threshold=None,
            seed=42,
        )


@pytest.mark.skipif(not _hashfrag_available(), reason="hashFrag/BLAST+ not on PATH")
def test_hashfrag_split_on_mock(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path, n=20)
    summary = run_hashfrag_split_assign(
        outdir=tmp_path / "hf_out",
        marked=marked,
        threshold=40,
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=42,
        threads=2,
        force=True,
    )
    split_rows = read_csv(Path(summary["split_csv"]))
    by_id = {r["ID"]: r for r in split_rows}
    assert by_id["1"]["train_test"] == "zsv"
    labels = {r["train_test"] for r in split_rows}
    assert "train" in labels
    assert "test" in labels
    assert "val" in labels
    assert Path(summary["all_fa"]).is_file()
    assert Path(summary["hashfrag_work"]).is_dir()


@pytest.mark.skipif(not _hashfrag_available(), reason="hashFrag/BLAST+ not on PATH")
def test_split_predict_type_hashfrag(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path, n=20)
    out = run_split_predict(
        outdir=tmp_path / "sp_hf",
        type="hashfrag",
        id_csv=id_csv,
        fold_csv=fold_csv,
        marked_fasta=marked,
        threshold=40,
        seed=42,
        threads=2,
        force=True,
    )
    rows = read_csv(out)
    assert rows[0].keys() >= {"ID", "train_test", "fold"}
    assert any(r["train_test"] == "zsv" for r in rows)
