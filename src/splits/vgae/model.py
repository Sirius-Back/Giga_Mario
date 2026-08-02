"""Classic VGAE (Kipf & Welling) with pure PyTorch sparse ops — no PyG required."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetric_normalized_adjacency(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build sparse Â = D^{-1/2} (A+I) D^{-1/2} as a coalesced sparse tensor."""
    if edge_index.numel() == 0:
        idx = torch.arange(n, device=device)
        eye_i = torch.stack([idx, idx], dim=0)
        eye_v = torch.ones(n, device=device, dtype=dtype)
        return torch.sparse_coo_tensor(eye_i, eye_v, (n, n)).coalesce()

    u = edge_index[0]
    v = edge_index[1]
    w = edge_weight.to(dtype=dtype)
    # Undirected: both directions + self-loops
    u2 = torch.cat([u, v, torch.arange(n, device=device)])
    v2 = torch.cat([v, u, torch.arange(n, device=device)])
    w2 = torch.cat([w, w, torch.ones(n, device=device, dtype=dtype)])
    adj = torch.sparse_coo_tensor(
        torch.stack([u2, v2], dim=0), w2, (n, n)
    ).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-12)
    deg_inv_sqrt = deg.pow(-0.5)
    vals = adj.values() * deg_inv_sqrt[adj.indices()[0]] * deg_inv_sqrt[adj.indices()[1]]
    return torch.sparse_coo_tensor(adj.indices(), vals, (n, n)).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = self.lin(x)
        return torch.sparse.mm(adj_norm, support)


class ClassicVGAE(nn.Module):
    """Two-layer GCN encoder → μ, logσ; inner-product decoder with neg. sampling."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.1,
        n_roles: int = 3,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1; got {in_dim}")
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv_mu = GCNLayer(hidden_dim, latent_dim)
        self.conv_logstd = GCNLayer(hidden_dim, latent_dim)
        self.dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        self.role_head = nn.Linear(latent_dim, int(n_roles))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = x.size(0)
        adj = _symmetric_normalized_adjacency(
            n, edge_index, edge_weight, device=x.device, dtype=x.dtype
        )
        h = self.conv1(x, adj)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.conv_mu(h, adj)
        logstd = self.conv_logstd(h, adj).clamp(-5.0, 5.0)
        return mu, logstd, adj

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(logstd)
        return mu

    def role_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.role_head(z)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logstd, _adj = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logstd)
        logits = self.role_logits(z)
        return {"mu": mu, "logstd": logstd, "z": z, "role_logits": logits}

    @staticmethod
    def kl_loss(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        # KL(q||N(0,I)) averaged over nodes
        return -0.5 * torch.mean(
            torch.sum(1.0 + 2.0 * logstd - mu.pow(2) - torch.exp(2.0 * logstd), dim=1)
        )

    @staticmethod
    def recon_loss_neg_sample(
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        *,
        num_neg: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """BCE on positive edges vs random negatives (never dense zzᵀ)."""
        n = z.size(0)
        if edge_index.numel() == 0 or n < 2:
            return z.new_tensor(0.0)
        u = edge_index[0]
        v = edge_index[1]
        pos_score = (z[u] * z[v]).sum(dim=-1)
        if edge_weight is not None and edge_weight.numel() == u.numel():
            # Weight positives lightly by normalized edge weight
            pos_weight = edge_weight.to(dtype=z.dtype).clamp_min(1e-6)
        else:
            pos_weight = torch.ones_like(pos_score)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score), weight=pos_weight, reduction="mean"
        )

        n_pos = int(u.numel())
        n_neg = int(num_neg if num_neg is not None else n_pos)
        if generator is None:
            neg_u = torch.randint(0, n, (n_neg,), device=z.device)
            neg_v = torch.randint(0, n, (n_neg,), device=z.device)
        else:
            neg_u = torch.randint(0, n, (n_neg,), device=z.device, generator=generator)
            neg_v = torch.randint(0, n, (n_neg,), device=z.device, generator=generator)
        # Avoid trivial self-loops as negatives when possible
        same = neg_u == neg_v
        if same.any():
            neg_v = neg_v.clone()
            neg_v[same] = (neg_v[same] + 1) % n
        neg_score = (z[neg_u] * z[neg_v]).sum(dim=-1)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score), reduction="mean"
        )
        return pos_loss + neg_loss


def soft_role_probs(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return F.softmax(logits / t, dim=-1)
