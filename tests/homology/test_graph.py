"""Tests for homology graph construction."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from src.homology.graph import (
    build_homology_graph,
    classify_relation,
    write_edge_table,
)
from src.homology.visualize import compute_graph_stats, connected_components


def test_classify_relation() -> None:
    assert classify_relation("ortholog_one2one") == "ortholog"
    assert classify_relation("within_species_paralog") == "paralog"
    assert classify_relation("other_paralog") == "paralog"
    assert classify_relation("gene_split") is None


def test_build_graph_from_tiny_compara(tmp_path: Path) -> None:
    header = (
        "gene_stable_id\tprotein_stable_id\tspecies\tidentity\thomology_type\t"
        "homology_gene_stable_id\thomology_protein_stable_id\thomology_species\t"
        "homology_identity\tdn\tds\tgoc_score\twga_coverage\tis_high_confidence\thomology_id\n"
    )
    rows = [
        # ortholog human-mouse
        "G1\tP1\thomo_sapiens\t90\tortholog_one2one\tG2\tP2\tmus_musculus\t90\tNULL\tNULL\tNULL\tNULL\t1\t1\n",
        # duplicate reverse should collapse
        "G2\tP2\tmus_musculus\t90\tortholog_one2one\tG1\tP1\thomo_sapiens\t90\tNULL\tNULL\tNULL\tNULL\t1\t2\n",
        # paralog human
        "G1\tP1\thomo_sapiens\t50\twithin_species_paralog\tG3\tP3\thomo_sapiens\t50\tNULL\tNULL\tNULL\tNULL\t1\t3\n",
        # out-of-panel species ignored
        "G1\tP1\thomo_sapiens\t80\tortholog_one2one\tGX\tPX\tdanio_rerio\t80\tNULL\tNULL\tNULL\tNULL\t1\t4\n",
    ]
    for sp in ("homo_sapiens", "mus_musculus"):
        d = tmp_path / sp / "compara_homology"
        d.mkdir(parents=True)
        path = d / "Compara.116.protein_default.homologies.tsv.gz"
        with gzip.open(path, "wt") as fh:
            fh.write(header)
            if sp == "homo_sapiens":
                fh.writelines(rows)
            else:
                fh.write(rows[1])

    result = build_homology_graph(
        tmp_path,
        species=("homo_sapiens", "mus_musculus"),
        release=116,
    )
    assert len(result.edges) == 2
    rels = {e[4] for e in result.edges}
    assert rels == {"ortholog", "paralog"}

    out = write_edge_table(result.sorted_edges(), tmp_path / "edges.tsv.gz")
    assert out.is_file() and out.stat().st_size > 0

    stats = compute_graph_stats(result.sorted_edges())
    assert not stats["component_sizes"].empty
    comps = connected_components(result.sorted_edges())
    assert len(comps) == 1
    assert len(comps[0]) == 3
