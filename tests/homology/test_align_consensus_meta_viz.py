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
    ortho_minus_para_minmax_profiles,
    ortho_minus_para_profiles,
    per_cluster_bin_similarity,
    profile_matrix,
    scope_bin_significance,
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
    labels, k, meta = choose_k_and_cluster(mat, min_k=10, max_k=15, seed=0)
    assert 10 <= k <= 15
    assert len(labels) == 12
    assert meta["k"] == k
    assert meta["min_k"] == 10

    labels_f, k_f, meta_f = choose_k_and_cluster(mat, fixed_k=10, seed=0)
    assert k_f == 10
    assert meta_f["fixed_k"] == 10
    assert len(set(labels_f)) == 10

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


def test_scope_bin_significance_has_winner_column() -> None:
    rows = []
    for ci in range(20):
        for b in (0, 1):
            # bin0: paralogs higher; bin1: orthologs higher
            o = 0.4 + 0.01 * ci if b == 0 else 0.7 + 0.01 * ci
            p = 0.7 + 0.01 * ci if b == 0 else 0.4 + 0.01 * ci
            rows.append(
                {
                    "cluster": f"c{ci}",
                    "pos_bin": b,
                    "scope": "orthologs",
                    "similarity": o,
                    "meta_cluster": 0,
                }
            )
            rows.append(
                {
                    "cluster": f"c{ci}",
                    "pos_bin": b,
                    "scope": "paralogs",
                    "similarity": p,
                    "meta_cluster": 0,
                }
            )
            rows.append(
                {
                    "cluster": f"c{ci}",
                    "pos_bin": b,
                    "scope": "full",
                    "similarity": 0.5,
                    "meta_cluster": 0,
                }
            )
    out = scope_bin_significance(pd.DataFrame(rows), min_n=5, fdr_q=0.05)
    assert not out.empty
    assert set(out["winner"]).issubset({"paralogs", "orthologs", "none"})
    w0 = out.loc[out["pos_bin"] == 0, "winner"].iloc[0]
    w1 = out.loc[out["pos_bin"] == 1, "winner"].iloc[0]
    assert w0 == "paralogs"
    assert w1 == "orthologs"


def test_ortho_minus_para_profiles() -> None:
    rows = []
    for ci in range(6):
        for b in (0, 1):
            # meta0: ortho > para by ~0.2; meta1: para > ortho by ~0.1
            mc = 0 if ci < 3 else 1
            o = 0.7 if mc == 0 else 0.4
            p = 0.5 if mc == 0 else 0.5
            rows.extend(
                [
                    {
                        "cluster": f"c{ci}",
                        "pos_bin": b,
                        "scope": "orthologs",
                        "similarity": o + 0.01 * ci,
                        "meta_cluster": mc,
                    },
                    {
                        "cluster": f"c{ci}",
                        "pos_bin": b,
                        "scope": "paralogs",
                        "similarity": p + 0.01 * ci,
                        "meta_cluster": mc,
                    },
                    {
                        "cluster": f"c{ci}",
                        "pos_bin": b,
                        "scope": "full",
                        "similarity": 0.55,
                        "meta_cluster": mc,
                    },
                ]
            )
    prof = ortho_minus_para_profiles(pd.DataFrame(rows))
    assert set(prof.columns) >= {
        "meta_cluster",
        "pos_bin",
        "delta_ortho_minus_para",
        "n_opg",
    }
    d0 = float(prof.loc[prof["meta_cluster"] == 0, "delta_ortho_minus_para"].mean())
    d1 = float(prof.loc[prof["meta_cluster"] == 1, "delta_ortho_minus_para"].mean())
    assert d0 > 0.15
    assert d1 < -0.05


def test_ortho_minus_para_minmax_profiles() -> None:
    rows = []
    # One OPG: ortho rises 0.1→0.9 across bins; para flat 0.5 → after minmax ortho high at end
    for b in range(5):
        rows.extend(
            [
                {
                    "cluster": "c0",
                    "pos_bin": b,
                    "scope": "orthologs",
                    "similarity": 0.1 + 0.2 * b,
                    "meta_cluster": 0,
                },
                {
                    "cluster": "c0",
                    "pos_bin": b,
                    "scope": "paralogs",
                    "similarity": 0.5,
                    "meta_cluster": 0,
                },
            ]
        )
    # Second OPG: para rises, ortho flat
    for b in range(5):
        rows.extend(
            [
                {
                    "cluster": "c1",
                    "pos_bin": b,
                    "scope": "orthologs",
                    "similarity": 0.5,
                    "meta_cluster": 1,
                },
                {
                    "cluster": "c1",
                    "pos_bin": b,
                    "scope": "paralogs",
                    "similarity": 0.1 + 0.2 * b,
                    "meta_cluster": 1,
                },
            ]
        )
    prof = ortho_minus_para_minmax_profiles(pd.DataFrame(rows))
    assert "delta_minmax_ortho_minus_para" in prof.columns
    # meta0 last bin: ortho≈1, para≈0.5 → positive; meta1 last: ortho≈0.5, para≈1 → negative
    d0_last = float(
        prof.loc[(prof.meta_cluster == 0) & (prof.pos_bin == 4), "delta_minmax_ortho_minus_para"].iloc[0]
    )
    d1_last = float(
        prof.loc[(prof.meta_cluster == 1) & (prof.pos_bin == 4), "delta_minmax_ortho_minus_para"].iloc[0]
    )
    assert d0_last > 0.4
    assert d1_last < -0.4
