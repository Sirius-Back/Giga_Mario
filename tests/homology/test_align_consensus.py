"""Tests for orthoparagroups MAFFT consensus scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.homology.align_consensus import (
    SeqRecord,
    apply_size_norm,
    column_metrics,
    consensus_rate,
    fit_size_norm_models,
    parse_fasta,
    write_fasta,
)


def test_consensus_rate_gaps_dilute() -> None:
    assert consensus_rate(["A", "A", "A", "A"]) == 1.0
    assert consensus_rate(["A", "A", "-", "-"]) == 0.5
    assert np.isnan(consensus_rate(["A"]))
    assert np.isnan(consensus_rate(["-", "-"]))


def test_column_metrics_roles() -> None:
    aln = [
        SeqRecord("ortholog|a", "AC-T", "ortholog"),
        SeqRecord("ortholog|b", "ACCT", "ortholog"),
        SeqRecord("paralog|c", "AG-T", "paralog"),
        SeqRecord("paralog|d", "AGTT", "paralog"),
    ]
    df = column_metrics(aln)
    assert len(df) == 4
    # pos1: A,A,A,A → 1.0
    assert df.loc[0, "overall_consensus_rate"] == 1.0
    # pos2: C,C,G,G → 0.5
    assert df.loc[1, "overall_consensus_rate"] == 0.5
    assert df.loc[1, "orthologs_consensus_rate"] == 1.0
    assert df.loc[1, "paralogs_consensus_rate"] == 1.0


def test_fit_and_apply_norm(tmp_path: Path) -> None:
    tables: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(0)
    for i in range(12):
        n = 4 + (i % 5)
        L = 30
        # synthetic: rate decreases with log(n)
        rate = 0.9 - 0.08 * np.log(n) + rng.normal(0, 0.02, size=L)
        rate = np.clip(rate, 0.2, 1.0)
        tables[f"cluster_{i}"] = pd.DataFrame(
            {
                "cluster": f"cluster_{i}",
                "position": np.arange(1, L + 1),
                "n_seqs": n,
                "n_orthologs": max(2, n - 1),
                "n_paralogs": max(2, n // 2),
                "n_non_gap": n,
                "gap_fraction": 0.05,
                "overall_consensus_rate": rate,
                "orthologs_consensus_rate": rate,
                "paralogs_consensus_rate": rate * 0.95,
            }
        )
    models, meta = fit_size_norm_models(tables, train_fraction=0.7, seed=42)
    assert set(models) == {"overall", "orthologs", "paralogs"}
    assert models["overall"].coef_log_n < 0  # larger n → lower expected rate
    assert meta["held_out_clusters"]
    normed = apply_size_norm(tables["cluster_0"], models)
    assert "overall_consensus_rate_norm_residual" in normed.columns
    assert np.isfinite(normed["overall_consensus_rate_norm_z"]).all()


def test_parse_write_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.fa"
    write_fasta(
        [
            SeqRecord("ortholog|s=hs|g=1", "ACGT", "ortholog"),
            SeqRecord("paralog|s=mm|g=2", "ACGA", "paralog"),
        ],
        path,
    )
    recs = parse_fasta(path)
    assert [r.role for r in recs] == ["ortholog", "paralog"]
