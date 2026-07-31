"""Tests for alignment consensus visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.homology.align_consensus_viz import (
    median_per_position_bin,
    pairwise_correlation_table,
    similar_length_table,
)


def _toy_metrics(tmp_path: Path, n_clusters: int = 6, L: int = 40) -> Path:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_clusters):
        pos = np.arange(1, L + 1)
        base = 0.4 + 0.3 * (pos / L) + rng.normal(0, 0.05, size=L)
        base = np.clip(base, 0.05, 0.99)
        ortho = np.clip(base + 0.1 + rng.normal(0, 0.03, size=L), 0.05, 0.99)
        para = np.clip(base - 0.05 + rng.normal(0, 0.04, size=L), 0.05, 0.99)
        df = pd.DataFrame(
            {
                "cluster": f"cluster_{i}",
                "position": pos,
                "overall_consensus_rate": base,
                "orthologs_consensus_rate": ortho,
                "paralogs_consensus_rate": para,
                "overall_consensus_rate_norm_residual": base - 0.5,
                "orthologs_consensus_rate_norm_residual": ortho - 0.5,
                "paralogs_consensus_rate_norm_residual": para - 0.5,
                "overall_consensus_rate_norm_ratio": base / 0.5,
                "orthologs_consensus_rate_norm_ratio": ortho / 0.5,
                "paralogs_consensus_rate_norm_ratio": para / 0.5,
                "overall_consensus_rate_norm_z": (base - 0.5) / 0.1,
                "orthologs_consensus_rate_norm_z": (ortho - 0.5) / 0.1,
                "paralogs_consensus_rate_norm_z": (para - 0.5) / 0.1,
            }
        )
        df.to_csv(metrics / f"cluster_{i}.pos.tsv.gz", sep="\t", index=False, compression="gzip")
    return metrics


def test_pairwise_and_lengths(tmp_path: Path) -> None:
    metrics = _toy_metrics(tmp_path)
    frames = [pd.read_csv(p, sep="\t") for p in sorted(metrics.glob("*.pos.tsv.gz"))]
    df = pd.concat(frames, ignore_index=True)
    corr = pairwise_correlation_table(df)
    assert set(corr["pair"]) == {"full_vs_orthologs", "full_vs_paralogs", "orthologs_vs_paralogs"}
    assert set(corr["variant"]) >= {"raw", "norm_residual", "norm_ratio", "norm_z"}
    assert corr["pearson_r"].notna().any()

    lengths = similar_length_table(df, thresholds=(0.5, 0.7))
    assert set(lengths["scope"]) == {"full", "orthologs", "paralogs"}
    assert lengths["similar_length_total"].ge(0).all()
    assert lengths["similar_length_longest_run"].le(lengths["aln_length"]).all()

    med = median_per_position_bin(df, n_bins=10)
    assert set(med["scope"]) == {"full", "orthologs", "paralogs"}
    assert med["median_rate"].between(0, 1).all()
