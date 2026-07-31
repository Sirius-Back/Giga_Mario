"""Tests for orthoparagroups cluster distribution viz helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.homology.orthoparagroups_viz import expand_nodes_per_species, load_clusters


def test_load_and_expand_clusters(tmp_path: Path) -> None:
    tsv = tmp_path / "clusters.tsv"
    pd.DataFrame(
        [
            {
                "fna_name": "cluster_0.fna",
                "component_id": 0,
                "n_nodes": 20,
                "n_distinct_orthology_groups": 2,
                "n_ortholog_edges": 30,
                "n_paralog_edges": 12,
                "n_written_orthologs": 9,
                "n_written_paralogs": 3,
                "nodes_per_species": "homo_sapiens:2;mus_musculus:3",
            },
            {
                "fna_name": "cluster_1.fna",
                "component_id": 1,
                "n_nodes": 15,
                "n_distinct_orthology_groups": 1,
                "n_ortholog_edges": 10,
                "n_paralog_edges": 8,
                "n_written_orthologs": 8,
                "n_written_paralogs": 1,
                "nodes_per_species": "homo_sapiens:1;bos_taurus:4",
            },
        ]
    ).to_csv(tsv, sep="\t", index=False)

    df = load_clusters(tsv)
    assert len(df) == 2
    assert set(df["n_species"]) == {2}
    assert "log10_n_nodes" in df.columns

    long = expand_nodes_per_species(df)
    assert len(long) == 4
    assert set(long["species"]) == {"homo_sapiens", "mus_musculus", "bos_taurus"}


def test_load_clusters_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_clusters(tmp_path / "missing.tsv")
