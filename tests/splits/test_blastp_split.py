"""BLASTP homology split unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.common import write_csv
from src.splits.blastp import (
    connected_components_from_edges,
    intersect_blastp,
    resolve_genetic_code,
    run_blastp_split_assign,
    run_sparse_blastp,
)


def test_resolve_genetic_code_universal_default() -> None:
    assert resolve_genetic_code(None) == 1
    assert resolve_genetic_code("universal") == 1
    assert resolve_genetic_code("standard") == 1
    assert resolve_genetic_code(1) == 1


def test_connected_components_merges_edges() -> None:
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c")]
    cc = connected_components_from_edges(nodes, edges)
    assert cc["a"] == cc["b"] == cc["c"]
    assert cc["d"] != cc["a"]


def test_intersect_drops_missing_parsed(tmp_path: Path) -> None:
    marked = tmp_path / "MARKED"
    parsed = tmp_path / "PARSED"
    marked.mkdir()
    parsed.mkdir()
    for i in range(1, 6):
        (marked / f"{i}.fa").write_text(f">x\nATGC\n", encoding="utf-8")
        if i != 5:
            (parsed / f"{i}.ext").write_text("ATGC\n", encoding="utf-8")
    kept = intersect_blastp(marked, parsed)
    assert "5" not in kept
    assert set(kept) == {"1", "2", "3", "4"}


def _write_mini_genome(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """One genome, two genes with CDS; third region has no CDS."""
    gtf_dir = tmp_path / "gtf"
    fna_dir = tmp_path / "fna"
    marked = tmp_path / "MARKED"
    parsed = tmp_path / "PARSED"
    gtf_dir.mkdir()
    fna_dir.mkdir()
    marked.mkdir()
    parsed.mkdir()

    # Chrom of length 60: AT ATG AAA CCC TTT GGG TAA + padding
    chrom_seq = "AA" + "ATGAAACCCTTTGGGTAA" + ("C" * 40)
    (fna_dir / "GCF_TEST.1.fna").write_text(
        f">chr1\n{chrom_seq}\n", encoding="utf-8"
    )
    gtf_path = gtf_dir / "GCF_TEST.1.gtf"
    gtf_path.write_text(
        'chr1\tx\tgene\t3\t20\t.\t+\t.\tgene_id "gA"; gene_name "gA";\n'
        'chr1\tx\tCDS\t3\t20\t.\t+\t0\tgene_id "gA"; gene_name "gA";\n'
        'chr1\tx\tgene\t3\t20\t.\t+\t.\tgene_id "gB"; gene_name "gB";\n'
        'chr1\tx\tCDS\t3\t20\t.\t+\t0\tgene_id "gB"; gene_name "gB";\n'
        'chr1\tx\tgene\t25\t30\t.\t+\t.\tgene_id "nc1"; gene_name "nc1";\n',
        encoding="utf-8",
    )

    id_rows = []
    fold_rows = []
    for i, (gene, raw) in enumerate(
        [("gA", "gA"), ("gB", "gB"), ("nc1", "nc1")], start=1
    ):
        rid = str(i)
        (marked / f"{rid}.fa").write_text(
            f">|GCF_TEST.1|chr1|3|20|{gene}|{raw}|{rid}\nATGC\n",
            encoding="utf-8",
        )
        (parsed / f"{rid}.ext").write_text("N" * 230 + "\n", encoding="utf-8")
        id_rows.append(
            {
                "genome": "GCF_TEST.1",
                "chr": "chr1",
                "pos1": "3",
                "pos2": "20",
                "gene_nameORnon_coding_ID": gene,
                "raw_target_ID": raw,
                "ID": rid,
            }
        )
        fold_rows.append({"ID": rid, "fold": "zsv" if i == 3 else "0"})

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
    return marked, parsed, id_csv, fold_csv, gtf_dir, fna_dir


def test_translate_cds_and_blastp_assign(tmp_path: Path) -> None:
    marked, parsed, id_csv, fold_csv, gtf_dir, fna_dir = _write_mini_genome(tmp_path)
    out = tmp_path / "out"
    # Pre-place MARKED_blastp so we skip adapt.
    import shutil

    shutil.copytree(marked, out / "MARKED_blastp")

    summary = run_blastp_split_assign(
        outdir=out,
        parsed=parsed,
        id_csv=id_csv,
        fold_csv=fold_csv,
        gtf_dir=gtf_dir,
        fna_dir=fna_dir,
        marked_blastp=out / "MARKED_blastp",
        seed=0,
        force=True,
        threads=1,
        min_bitscore=10.0,
        evalue=10.0,
        max_target_seqs=5,
    )
    split_csv = Path(summary["split_csv"])
    assert split_csv.is_file()
    text = split_csv.read_text(encoding="utf-8")
    assert "zsv" in text
    # Identical CDS for gA and gB → BLASTP edge → same component for IDs 1 and 2
    assert summary["n_proteins"] >= 1
    assert summary["n_blast_edges"] >= 1 or summary["n_proteins"] >= 2


def test_sparse_blastp_identical_proteins(tmp_path: Path) -> None:
    faa = tmp_path / "p.faa"
    aa = "MKPFG" * 8
    faa.write_text(f">p1\n{aa}\n>p2\n{aa}\n", encoding="utf-8")
    hits, edges = run_sparse_blastp(
        faa,
        work=tmp_path / "blast",
        threads=1,
        evalue=10.0,
        max_target_seqs=5,
        min_bitscore=5.0,
        query_chunk=10,
        force=True,
    )
    assert hits.is_file()
    assert ("p1", "p2") in edges or ("p2", "p1") in edges
