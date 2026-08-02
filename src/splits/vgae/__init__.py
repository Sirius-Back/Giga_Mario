"""VGAE (classic GCN-VAE) train/test/val split strategy.

Encoder inputs: weighted adjacency + compositional node features (GC, k-mer).
Ortholog/paralog labels are NEVER fed to the GCN/VAE encoder — only to the
post-assignment homology loss ``L_hom`` and offline checkers.
"""
from __future__ import annotations

from src.splits.vgae.assign import size_constrained_assign
from src.splits.vgae.graph_data import (
    FORBIDDEN_FEATURE_PATTERN,
    PackedGraph,
    assert_no_homology_features,
    pack_region_graph,
)
from src.splits.vgae.homology_loss import compute_l_hom, load_homology_groups
from src.splits.vgae.model import ClassicVGAE
from src.splits.vgae.train import run_vgae_train
from src.splits.vgae.split_assign import run_vgae_split_assign

__all__ = [
    "ClassicVGAE",
    "FORBIDDEN_FEATURE_PATTERN",
    "PackedGraph",
    "assert_no_homology_features",
    "compute_l_hom",
    "load_homology_groups",
    "pack_region_graph",
    "run_vgae_split_assign",
    "run_vgae_train",
    "size_constrained_assign",
]
