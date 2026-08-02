"""MLP-VAE (no GCN) for k-mer feature panels — additive sibling of VGAE."""
from __future__ import annotations

from src.splits.vae.features import PackedFeatures, load_feature_table, pack_feature_table
from src.splits.vae.model import MlpVAE
from src.splits.vae.train import run_vae_train

__all__ = [
    "MlpVAE",
    "PackedFeatures",
    "load_feature_table",
    "pack_feature_table",
    "run_vae_train",
]
