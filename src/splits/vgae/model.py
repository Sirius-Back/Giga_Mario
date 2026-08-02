"""Classic VGAE (Kipf & Welling) + GAT / GraphSAGE encoders — pure PyTorch, no PyG."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_GCN = "gcn"
ARCH_GAT = "gat"
ARCH_SAGE = "sage"
ARCH_GCL = "gcl"
ARCH_GCL_GAT = "gcl_gat"
ARCH_APPNP = "appnp"
ARCH_GCNII = "gcnii"
ARCH_CHOICES = (
    ARCH_GCN,
    ARCH_GAT,
    ARCH_SAGE,
    ARCH_GCL,
    ARCH_GCL_GAT,
    ARCH_APPNP,
    ARCH_GCNII,
)


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


def _undirected_edge_index(
    n: int,
    edge_index: torch.Tensor,
    *,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Bidirectional edges (+ optional self-loops) as ``(2, E')`` long tensor."""
    device = edge_index.device if edge_index.numel() else torch.device("cpu")
    if edge_index.numel() == 0:
        if not add_self_loops:
            return torch.zeros((2, 0), device=device, dtype=torch.long)
        idx = torch.arange(n, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)
    u = edge_index[0]
    v = edge_index[1]
    u2 = torch.cat([u, v])
    v2 = torch.cat([v, u])
    if add_self_loops:
        idx = torch.arange(n, device=u.device, dtype=torch.long)
        u2 = torch.cat([u2, idx])
        v2 = torch.cat([v2, idx])
    return torch.stack([u2, v2], dim=0)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = self.lin(x)
        return torch.sparse.mm(adj_norm, support)


class GATConv(nn.Module):
    """Multi-head GAT on sparse edges only (Velickovic et al.; no PyG)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1; got {heads}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        if bias:
            bout = out_dim * heads if concat else out_dim
            self.bias = nn.Parameter(torch.empty(bout))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_dim)
        if edge_index.numel() == 0:
            out = h.mean(dim=1) if not self.concat else h.reshape(n, -1)
            if self.bias is not None:
                out = out + self.bias
            return out

        src, dst = edge_index[0], edge_index[1]
        alpha_src = (h * self.att_src).sum(dim=-1)  # (n, heads)
        alpha_dst = (h * self.att_dst).sum(dim=-1)
        e = F.leaky_relu(
            alpha_src[src] + alpha_dst[dst], negative_slope=self.negative_slope
        )
        e_max = torch.full((n, self.heads), -1e9, device=x.device, dtype=x.dtype)
        e_max.scatter_reduce_(
            0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=True
        )
        e = e - e_max[dst]
        exp_e = torch.exp(e)
        denom = torch.zeros((n, self.heads), device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / denom[dst].clamp_min(1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros((n, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        if self.concat:
            out = out.reshape(n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out


class SAGEConv(nn.Module):
    """GraphSAGE mean aggregator (Hamilton et al.) on sparse edges."""

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if edge_index.numel() == 0:
            return self.lin_self(x)
        src, dst = edge_index[0], edge_index[1]
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        ones = torch.ones(src.numel(), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, ones)
        neigh_mean = neigh_sum / deg.clamp_min(1.0).unsqueeze(1)
        return self.lin_self(x) + self.lin_neigh(neigh_mean)


class _VGAEBase(nn.Module):
    """Shared reparam / role head / KL / edge recon for graph VAEs."""

    latent_dim: int
    role_head: nn.Linear
    architecture: str

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(logstd)
        return mu

    def role_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.role_head(z)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
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
        same = neg_u == neg_v
        if same.any():
            neg_v = neg_v.clone()
            neg_v[same] = (neg_v[same] + 1) % n
        neg_score = (z[neg_u] * z[neg_v]).sum(dim=-1)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score), reduction="mean"
        )
        return pos_loss + neg_loss


class ClassicVGAE(_VGAEBase):
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
        self.architecture = ARCH_GCN
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


class GATVGAE(_VGAEBase):
    """Two-layer GATConv encoder (multi-head) → μ, logσ + role head."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.1,
        n_roles: int = 3,
        heads: int = 4,
        heads_out: int = 1,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1; got {in_dim}")
        self.architecture = ARCH_GAT
        h1 = max(1, int(hidden_dim) // int(heads))
        self.conv1 = GATConv(
            in_dim, h1, heads=heads, concat=True, dropout=dropout
        )
        hid = h1 * heads
        self.conv_mu = GATConv(
            hid, latent_dim, heads=heads_out, concat=False, dropout=dropout
        )
        self.conv_logstd = GATConv(
            hid, latent_dim, heads=heads_out, concat=False, dropout=dropout
        )
        self.dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        self.role_head = nn.Linear(latent_dim, int(n_roles))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_weight  # attention uses topology; weights unused (additive)
        ei = _undirected_edge_index(x.size(0), edge_index, add_self_loops=True)
        h = self.conv1(x, ei)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.conv_mu(h, ei)
        logstd = self.conv_logstd(h, ei).clamp(-5.0, 5.0)
        return mu, logstd

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logstd = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logstd)
        return {
            "mu": mu,
            "logstd": logstd,
            "z": z,
            "role_logits": self.role_logits(z),
        }


class SAGEVGAE(_VGAEBase):
    """Two-layer GraphSAGE mean-encoder → μ, logσ + role head."""

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
        self.architecture = ARCH_SAGE
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv_mu = SAGEConv(hidden_dim, latent_dim)
        self.conv_logstd = SAGEConv(hidden_dim, latent_dim)
        self.dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        self.role_head = nn.Linear(latent_dim, int(n_roles))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_weight
        ei = _undirected_edge_index(x.size(0), edge_index, add_self_loops=True)
        h = self.conv1(x, ei)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.conv_mu(h, ei)
        logstd = self.conv_logstd(h, ei).clamp(-5.0, 5.0)
        return mu, logstd

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logstd = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logstd)
        return {
            "mu": mu,
            "logstd": logstd,
            "z": z,
            "role_logits": self.role_logits(z),
        }


class APPNPVGAE(_VGAEBase):
    """MLP features + APPNP teleport propagation (Klicpera et al.)."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.1,
        n_roles: int = 3,
        k_hops: int = 10,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1; got {in_dim}")
        self.architecture = ARCH_APPNP
        self.k_hops = int(k_hops)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lin_mu = nn.Linear(hidden_dim, latent_dim)
        self.lin_logstd = nn.Linear(hidden_dim, latent_dim)
        self.role_head = nn.Linear(latent_dim, int(n_roles))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        adj = _symmetric_normalized_adjacency(
            n, edge_index, edge_weight, device=x.device, dtype=x.dtype
        )
        h0 = self.feat_mlp(x)
        h0 = F.dropout(h0, p=self.dropout, training=self.training)
        z = h0
        for _ in range(self.k_hops):
            z = (1.0 - self.alpha) * torch.sparse.mm(adj, z) + self.alpha * h0
        mu = self.lin_mu(z)
        logstd = self.lin_logstd(z).clamp(-5.0, 5.0)
        return mu, logstd

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logstd = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logstd)
        return {
            "mu": mu,
            "logstd": logstd,
            "z": z,
            "role_logits": self.role_logits(z),
        }


class GCNIILayer(nn.Module):
    """One GCNII layer: ((1-α)ÂH + αH0)((1-β)I + βW)."""

    def __init__(self, dim: int, *, layer_idx: int, n_layers: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        # β_l = log(λ/l + 1) style — use 0.5/log for stability
        self.alpha = 0.1
        self.beta = 0.5 / (1.0 + float(layer_idx))
        self.layer_idx = int(layer_idx)

    def forward(
        self, h: torch.Tensor, h0: torch.Tensor, adj_norm: torch.Tensor
    ) -> torch.Tensor:
        prop = (1.0 - self.alpha) * torch.sparse.mm(adj_norm, h) + self.alpha * h0
        return (1.0 - self.beta) * prop + self.beta * self.lin(prop)


class GCNIIVGAE(_VGAEBase):
    """Deep residual GCNII encoder (Chen et al.) → μ, logσ."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.1,
        n_roles: int = 3,
        n_layers: int = 8,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1; got {in_dim}")
        self.architecture = ARCH_GCNII
        self.dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        n_layers = max(2, int(n_layers))
        self.layers = nn.ModuleList(
            [GCNIILayer(hidden_dim, layer_idx=i + 1, n_layers=n_layers) for i in range(n_layers)]
        )
        self.lin_mu = nn.Linear(hidden_dim, latent_dim)
        self.lin_logstd = nn.Linear(hidden_dim, latent_dim)
        self.role_head = nn.Linear(latent_dim, int(n_roles))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        adj = _symmetric_normalized_adjacency(
            n, edge_index, edge_weight, device=x.device, dtype=x.dtype
        )
        h0 = F.relu(self.input_lin(x))
        h0 = F.dropout(h0, p=self.dropout, training=self.training)
        h = h0
        for layer in self.layers:
            h = F.relu(layer(h, h0, adj))
            h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.lin_mu(h)
        logstd = self.lin_logstd(h).clamp(-5.0, 5.0)
        return mu, logstd

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logstd = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logstd)
        return {
            "mu": mu,
            "logstd": logstd,
            "z": z,
            "role_logits": self.role_logits(z),
        }


def build_vgae(
    architecture: str,
    in_dim: int,
    *,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    dropout: float = 0.1,
    n_roles: int = 3,
    gat_heads: int = 4,
    appnp_k: int = 10,
    appnp_alpha: float = 0.1,
    gcnii_layers: int = 8,
) -> _VGAEBase:
    """Factory for GCN / GAT / SAGE / GCL / APPNP / GCNII encoders."""
    arch = str(architecture).lower().strip()
    if arch in ("classic", "vgae", "gcn", ARCH_GCN, ARCH_GCL):
        model = ClassicVGAE(
            in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            n_roles=n_roles,
        )
        if arch == ARCH_GCL:
            model.architecture = ARCH_GCL
        return model
    if arch in ("gat", "gatconv", ARCH_GAT, ARCH_GCL_GAT):
        model = GATVGAE(
            in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            n_roles=n_roles,
            heads=int(gat_heads),
        )
        if arch == ARCH_GCL_GAT:
            model.architecture = ARCH_GCL_GAT
        return model
    if arch in ("sage", "graphsage", ARCH_SAGE):
        return SAGEVGAE(
            in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            n_roles=n_roles,
        )
    if arch in ("appnp", ARCH_APPNP):
        return APPNPVGAE(
            in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            n_roles=n_roles,
            k_hops=int(appnp_k),
            alpha=float(appnp_alpha),
        )
    if arch in ("gcnii", ARCH_GCNII):
        return GCNIIVGAE(
            in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            n_roles=n_roles,
            n_layers=int(gcnii_layers),
        )
    raise ValueError(
        f"unknown architecture={architecture!r}; expected one of {ARCH_CHOICES}"
    )


def uses_contrastive(architecture: str) -> bool:
    """True when train should add GRACE-style InfoNCE on two augmented views."""
    a = str(architecture).lower().strip()
    return a in (ARCH_GCL, ARCH_GCL_GAT, "grace", "contrastive")


def soft_role_probs(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return F.softmax(logits / t, dim=-1)


def gumbel_softmax_roles(
    logits: torch.Tensor,
    *,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """Gumbel-Softmax role probs (additive; does not replace :func:`soft_role_probs`)."""
    t = max(float(tau), 1e-6)
    return F.gumbel_softmax(logits, tau=t, hard=bool(hard), dim=dim)


def gumbel_tau_schedule(
    epoch: int,
    *,
    tau_start: float = 1.0,
    tau_end: float = 0.3,
    t_anneal: int = 20,
) -> float:
    """Linear ``τ: tau_start → tau_end`` over the first ``t_anneal`` epochs."""
    if int(t_anneal) <= 0:
        return float(tau_end)
    t = min(1.0, max(0.0, float(epoch - 1) / float(t_anneal)))
    return float(tau_start) + t * (float(tau_end) - float(tau_start))


def kl_beta_schedule(
    epoch: int,
    *,
    beta_max: float = 0.05,
    t_anneal: int = 15,
) -> float:
    """``β(t) = β_max · min(1, t / T_anneal)`` with ``t`` = epoch (1-indexed)."""
    if int(t_anneal) <= 0:
        return float(beta_max)
    return float(beta_max) * min(1.0, float(epoch) / float(t_anneal))
