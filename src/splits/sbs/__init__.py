"""Split-by-similarity (SBS) shared architecture.

Contracts:
  1. FNA → FeatureTable (preferred; O(n·d))
  2. FeatureTable → assignment table (region|cluster|train_test|fold|additional)
  3. assignment (+ prepared panel) → split.csv / SPLIT/ via pipeline hooks

Dense DistanceMatrix remains available only for small-n / legacy backends.
"""
from __future__ import annotations

from .assign import (
    ASSIGNMENT_COLUMNS,
    assign_from_features,
    assignment_rows_to_split_csv,
)
from .features import FeatureBackend, FeatureTable, compute_feature_table
from .fna_io import load_fna_sequences
from .visualize import plot_feature_pca, plot_sbs_pca_diagnostics

__all__ = [
    "ASSIGNMENT_COLUMNS",
    "FeatureBackend",
    "FeatureTable",
    "assign_from_features",
    "assignment_rows_to_split_csv",
    "compute_feature_table",
    "load_fna_sequences",
    "plot_feature_pca",
    "plot_sbs_pca_diagnostics",
]
