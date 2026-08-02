"""MLP VAE encoder/decoder + role head (no graph ops)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splits.vgae.model import (
    gumbel_softmax_roles,
    gumbel_tau_schedule,
    kl_beta_schedule,
    soft_role_probs,
)


class MlpVAE(nn.Module):
    """Classic MLP-VAE on dense features with a 3-way role head for splits."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.1,
        n_roles: int = 3,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1; got {in_dim}")
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, in_dim),
        )
        self.role_head = nn.Linear(latent_dim, int(n_roles))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.fc_mu(h)
        logstd = self.fc_logstd(h).clamp(-5.0, 5.0)
        return mu, logstd

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(logstd)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def role_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.role_head(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        x_hat = self.decode(z)
        logits = self.role_logits(z)
        return {
            "mu": mu,
            "logstd": logstd,
            "z": z,
            "x_hat": x_hat,
            "role_logits": logits,
        }

    @staticmethod
    def kl_loss(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(
            torch.sum(1.0 + 2.0 * logstd - mu.pow(2) - torch.exp(2.0 * logstd), dim=1)
        )

    @staticmethod
    def recon_loss_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Feature reconstruction (replaces VGAE edge BCE)."""
        return F.mse_loss(x_hat, x)


# Re-export helpers for callers / tests
__all__ = [
    "MlpVAE",
    "gumbel_softmax_roles",
    "gumbel_tau_schedule",
    "kl_beta_schedule",
    "soft_role_probs",
]
