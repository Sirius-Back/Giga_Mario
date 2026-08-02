"""Graph contrastive learning helpers for VGAE (GRACE-style InfoNCE).

Additive — used when ``architecture`` is ``gcl`` / ``gcl_gat``. Homology labels
never enter augmentations or the contrastive head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def augment_graph_view(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    edge_drop: float = 0.2,
    feat_mask: float = 0.2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random edge dropout + feature masking (independent Bernoulli)."""
    device = x.device

    def _rand(n: int) -> torch.Tensor:
        if generator is None:
            return torch.rand(n, device=device)
        # CPU generator → sample on CPU, move (CUDA generators are device-bound)
        return torch.rand(n, generator=generator).to(device=device)

    if float(feat_mask) > 0.0:
        keep = _rand(x.size(1)) >= float(feat_mask)
        x_aug = x * keep.to(dtype=x.dtype).unsqueeze(0)
    else:
        x_aug = x

    if edge_index.numel() == 0 or float(edge_drop) <= 0.0:
        return x_aug, edge_index, edge_weight

    e = edge_index.size(1)
    mask = _rand(e) >= float(edge_drop)
    if not bool(mask.any()):
        mask = mask.clone()
        mask[0] = True
    ei = edge_index[:, mask]
    ew = edge_weight[mask] if edge_weight.numel() == e else edge_weight
    return x_aug, ei, ew


def info_nce_pairwise(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    temperature: float = 0.5,
    max_nodes: int = 8192,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE between two views (GRACE / SimCLR on nodes).

    For large panels, samples up to ``max_nodes`` nodes (seeded via ``generator``).
    """
    if z1.size(0) != z2.size(0):
        raise ValueError("z1/z2 node count mismatch")
    n = z1.size(0)
    if n < 2:
        return z1.new_tensor(0.0)

    if max_nodes is not None and n > int(max_nodes):
        if generator is None:
            idx = torch.randperm(n, device=z1.device)[: int(max_nodes)]
        else:
            idx = torch.randperm(n, generator=generator)[: int(max_nodes)].to(z1.device)
        z1 = z1.index_select(0, idx)
        z2 = z2.index_select(0, idx)
        n = z1.size(0)

    h1 = F.normalize(z1, dim=-1)
    h2 = F.normalize(z2, dim=-1)
    tau = max(float(temperature), 1e-6)
    logits_12 = (h1 @ h2.T) / tau
    logits_21 = (h2 @ h1.T) / tau
    labels = torch.arange(n, device=z1.device)
    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)
    return 0.5 * (loss_12 + loss_21)
