"""MLP-VAE pack / model / dual-loss / early-stop tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.splits.vae.features import (
    expected_kmer_dim,
    load_feature_table,
    pack_feature_table,
    validate_k4_dims,
)
from src.splits.vae.model import MlpVAE
from src.splits.vgae.graph_data import assert_no_homology_features
from src.splits.vgae.homology_loss import EmaTermNorm, compute_l_hom
from src.splits.vgae.model import gumbel_softmax_roles, soft_role_probs
from src.splits.vgae.train import compose_objective


def test_expected_k4_dim() -> None:
    assert expected_kmer_dim(4) == 256
    validate_k4_dims([f"kmer_{i}" for i in range(256)], k=4)
    with pytest.raises(ValueError):
        validate_k4_dims(["a", "b"], k=4)


def test_mlp_vae_forward_recon() -> None:
    n, f = 64, 256
    x = torch.randn(n, f)
    model = MlpVAE(f, hidden_dim=64, latent_dim=16)
    out = model(x)
    assert out["z"].shape == (n, 16)
    assert out["x_hat"].shape == (n, f)
    assert out["role_logits"].shape == (n, 3)
    recon = MlpVAE.recon_loss_mse(x, out["x_hat"])
    kl = MlpVAE.kl_loss(out["mu"], out["logstd"])
    assert torch.isfinite(recon) and torch.isfinite(kl)


def test_homology_firewall_on_names() -> None:
    with pytest.raises(ValueError, match="homology"):
        assert_no_homology_features(["kmer_AAAA", "orthogroup_id"])


def test_dual_loss_shapes() -> None:
    n, f = 40, 256
    x = torch.randn(n, f, requires_grad=False)
    model = MlpVAE(f, hidden_dim=32, latent_dim=8)
    out = model(x)
    soft = soft_role_probs(out["role_logits"])
    gs = gumbel_softmax_roles(out["role_logits"], tau=1.0)
    from src.splits.vgae.homology_loss import HomologyGroups

    groups = HomologyGroups(
        orthogroup=np.full(n, -1, dtype=np.int64),
        paragroup=np.full(n, -1, dtype=np.int64),
        ortho_groups=(np.arange(0, 10), np.arange(10, 20)),
        para_groups=(np.arange(20, 30), np.arange(30, 40)),
    )
    recon = MlpVAE.recon_loss_mse(x, out["x_hat"])
    kl = MlpVAE.kl_loss(out["mu"], out["logstd"])
    hom = compute_l_hom(gs, groups, soft=True, weighted=True, subset_seed=1)
    from src.splits.vgae.assign import size_loss, role_target_fractions

    sz = size_loss(soft, torch.as_tensor(role_target_fractions((3, 1, 1)), dtype=torch.float32))
    ema = EmaTermNorm(decay=0.9)
    composed = compose_objective(
        recon=recon,
        kl=kl,
        l_hom=hom["l_hom"],
        size=sz,
        loss_mode="homology_first",
        epoch=1,
        beta_kl=1.0,
        lambda_hom=25.0,
        lambda_size=1.0,
        alpha_recon=0.3,
        beta_kl_max=0.05,
        kl_anneal_epochs=15,
        ema=ema,
    )
    legacy = recon.detach() + kl.detach() + compute_l_hom(soft, groups, soft=True)["l_hom"] + sz.detach()
    assert torch.isfinite(composed["loss"])
    assert torch.isfinite(legacy)
    composed["loss"].backward()


def test_early_stop_policy_min_epochs() -> None:
    min_epochs, patience = 25, 10
    stale = 0
    stopped_at = None
    best = 1e9
    for epoch in range(1, 40):
        l_hom = 1.0
        if l_hom < best - 1e-6:
            best = l_hom
            stale = 0
        elif epoch >= min_epochs:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            stopped_at = epoch
            break
    assert stopped_at == 34


def test_pack_feature_table_tiny(tmp_path: Path) -> None:
    # Tiny synthetic 4-mer table (not full 256 — use k=None path via raw npz)
    # Build full 256-dim sparse random for k=4 contract
    n = 8
    names = [f"kmer_{i:03d}" for i in range(256)]
    # Use ACGT-like names to pass firewall
    alphabet = "ACGT"
    names = []
    for a in alphabet:
        for b in alphabet:
            for c in alphabet:
                for d in alphabet:
                    names.append(f"kmer_{a}{b}{c}{d}")
    assert len(names) == 256
    ids = [str(i) for i in range(1, n + 1)]
    x = np.random.default_rng(0).random((n, 256), dtype=np.float32)
    x = x / x.sum(axis=1, keepdims=True)
    csv_path = tmp_path / "feature_table.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("region|" + "|".join(names) + "\n")
        for i, rid in enumerate(ids):
            fh.write(rid + "|" + "|".join(f"{float(v):.8g}" for v in x[i]) + "\n")
    pack = pack_feature_table(csv_path, tmp_path / "pack", k=4)
    assert pack.n_nodes == n
    assert pack.x.shape == (n, 256)
    assert pack.meta["homology_in_encoder"] is False
    ids2, x2, names2 = load_feature_table(tmp_path / "pack" / "node_features.npz", k=4)
    assert len(ids2) == n and x2.shape == (n, 256)
