"""Tests for GEO counts/FPKM → wide TPM conversion."""
from __future__ import annotations

import gzip
from pathlib import Path

from src.geo_expr_to_tpm import (
    fpkm_column_to_symbol_tpm,
    horse_counts_to_symbol_tpm,
    parse_ensembl_gene_table,
    parse_refseq_gene_tables,
    read_htseq_counts,
)


def _write(path: Path, text: str, *, gz: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode()
    if gz:
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    else:
        path.write_bytes(data)
    return path


def test_read_htseq_skips_meta(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.txt",
        "ENSECAG00000000001\t10\n__no_feature\t5\nENSECAG00000000002\t20\n",
    )
    assert read_htseq_counts(p) == {
        "ENSECAG00000000001": 10.0,
        "ENSECAG00000000002": 20.0,
    }


def test_horse_counts_to_symbol_tpm(tmp_path: Path) -> None:
    ens_lines = [
        '1\tensembl\tgene\t1\t1000\t.\t+\t.\tgene_id "ENSECAG00000000001"; gene_name "GENEA";\n',
        '1\tensembl\tgene\t1\t1000\t.\t+\t.\tgene_id "ENSECAG00000000002"; gene_name "GENEB";\n',
    ]
    ref_lines = [
        'NC_1\tGnomon\tgene\t1\t1000\t.\t+\t.\tgene_id "GENEA"; gene "GENEA"; db_xref "GeneID:1";\n',
        'NC_1\tGnomon\tgene\t1\t1000\t.\t+\t.\tgene_id "GENEB"; gene "GENEB"; db_xref "GeneID:2";\n',
    ]
    count_lines = ["ENSECAG00000000001\t1000\n", "ENSECAG00000000002\t2000\n"]
    for i in range(3, 1103):
        name = f"G{i}"
        ens_lines.append(
            f'1\tensembl\tgene\t1\t1000\t.\t+\t.\tgene_id "ENSECAG{i:011d}"; gene_name "{name}";\n'
        )
        ref_lines.append(
            f'NC_1\tGnomon\tgene\t1\t1000\t.\t+\t.\tgene_id "{name}"; gene "{name}"; db_xref "GeneID:{i}";\n'
        )
        count_lines.append(f"ENSECAG{i:011d}\t10\n")

    ens = _write(tmp_path / "ens.gtf", "".join(ens_lines))
    ref = _write(tmp_path / "ref.gtf", "".join(ref_lines))
    counts = _write(tmp_path / "counts.txt.gz", "".join(count_lines), gz=True)
    tpm = horse_counts_to_symbol_tpm(counts, ens, ref)
    assert len(tpm) >= 1000
    assert abs(sum(tpm.values()) - 1e6) < 1.0
    assert tpm["GENEB"] > tpm["GENEA"]


def test_fpkm_column_to_symbol_tpm(tmp_path: Path) -> None:
    ref = tmp_path / "ref.gtf"
    with ref.open("w") as rh:
        for i in range(1, 1101):
            rh.write(
                f'NC_1\tGnomon\tgene\t1\t1000\t.\t+\t.\t'
                f'gene_id "G{i}"; gene "G{i}"; db_xref "GeneID:{i}";\n'
            )
    fpkm = tmp_path / "fpkm.tsv"
    with fpkm.open("w") as fh:
        fh.write("gene_id\tBlank-1_FPKM\tBlank-2_FPKM\n")
        for i in range(1, 1101):
            fh.write(f"{i}\t{float(i)}\t0.0\n")
    tpm = fpkm_column_to_symbol_tpm(fpkm, "Blank-1_FPKM", ref)
    assert len(tpm) == 1100
    assert abs(sum(tpm.values()) - 1e6) < 1.0
    assert tpm["G1100"] > tpm["G1"]


def test_parse_tables(tmp_path: Path) -> None:
    ens = _write(
        tmp_path / "e.gtf",
        '1\tx\tgene\t1\t10\t.\t+\t.\tgene_id "E1"; gene_name "A";\n',
    )
    assert parse_ensembl_gene_table(ens)["E1"]["gene_name"] == "A"
    ref = _write(
        tmp_path / "r.gtf",
        '1\tx\tgene\t1\t10\t.\t+\t.\t'
        'gene_id "A"; gene "A"; db_xref "GeneID:99";\n',
    )
    gid_map, lengths = parse_refseq_gene_tables(ref)
    assert gid_map["99"] == "A"
    assert lengths["A"] == 10
