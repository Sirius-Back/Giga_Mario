"""Tests for meta-cluster consensus visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.homology.align_consensus import SeqRecord, write_fasta
from src.homology.align_consensus_meta_viz import (
    add_bins_on_full_length,
    choose_k_and_cluster,
    filter_well_aligned,
    find_atg_rel_positions,
    per_cluster_bin_similarity,
    profile_matrix,
)


def test_filter_and_bins() -> None:
    df = pd.DataFrame(
        {
            "cluster": ["c0"] * 10,
            "position": list(range(1, 11)),
            "n_seqs": [10] * 10,
            "n_non_gap": [2, 2, 6, 7, 8, 9, 8, 3, 2, 9],
            "aln_length": [10] * 10,
            "overall_consensus_rate": np.linspace(0.2, 0.9, 10),
            "orthologs_consensus_rate": np.linspace(0.3, 0.95, 10),
            "paralogs_consensus_rate": np.linspace(0.1, 0.8, 10),
        }
    )
    kept = filter_well_aligned(df, min_frac=0.5)
    assert (kept["n_non_gap"] / kept["n_seqs"] >= 0.5).all()
    assert len(kept) == 6
    binned = add_bins_on_full_length(kept, n_bins=5)
    assert binned["pos_bin"].between(0, 4).all()


def test_profile_kmeans_and_atg(tmp_path: Path) -> None:
    rows = []
    for ci in range(12):
        for b in range(10):
            for scope, base in [("full", 0.4), ("orthologs", 0.5), ("paralogs", 0.3)]:
                # two profile shapes
                sim = base + (0.3 if ci < 6 else -0.05) * (b / 10) + 0.01 * ci
                rows.append(
                    {
                        "cluster": f"cluster_{ci}",
                        "pos_bin": b,
                        "scope": scope,
                        "similarity": float(np.clip(sim, 0.05, 0.99)),
                        "n_positions": 5,
                        "rel_pos_mid": (b + 0.5) / 10,
                    }
                )
    bin_sim = pd.DataFrame(rows)
    wide, mat, clusters = profile_matrix(bin_sim, n_bins=10)
    assert wide.shape == (12, 30)
    labels, k, meta = choose_k_and_cluster(mat, max_k=5, seed=0)
    assert k <= 5
    assert len(labels) == 12
    assert meta["k"] == k

    # ATG in middle of ungapped
    aln = tmp_path / "cluster_x.aln.fa"
    write_fasta(
        [
            SeqRecord("ortholog|a", "AAA---ATGCCC", "ortholog"),
            SeqRecord("paralog|b", "AAAATG---CCC", "paralog"),
        ],
        aln,
    )
    info = find_atg_rel_positions(aln)
    assert info["n_with_atg"] == 2
    assert 0.0 <= info["atg_rel_mean"] <= 1.0
