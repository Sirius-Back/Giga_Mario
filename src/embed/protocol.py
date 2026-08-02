"""Shared embedding-extractor contract (LegNet now; Caduceus later)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Backend that maps sequences → layer embedding matrices."""

    name: str

    def layer_dims(self) -> dict[str, int]:
        """Expected feature dim per layer key."""

    def extract_batch(
        self, sequences: list[str], *, layers: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        """Return ``{layer: float32 [B, D]}`` (RC-averaged when applicable)."""
