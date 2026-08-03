"""Layer embeddings + geometry compare (LegNet phase-1, Caduceus phase-2).

Caduceus: ``python -m src.embed.run_caduceus`` → ``results/embed_caduceus/``.
Shared contract: ``src.embed.protocol.EmbeddingExtractor``.
"""

from __future__ import annotations

__all__ = [
    "LAYER_DIMS",
    "DEFAULT_LAYERS",
]

LAYER_DIMS: dict[str, int] = {
    "stage0": 160,
    "stage1_2": 416,
    "pooled": 256,
    "head_h": 256,
    "pred": 1,
}

DEFAULT_LAYERS: tuple[str, ...] = (
    "stage0",
    "stage1_2",
    "pooled",
    "head_h",
    "pred",
)

SEQ_LEN = 230
ROLE_TRAIN = 0
ROLE_TEST = 1
ROLE_VAL = 2
ROLE_NAMES = ("train", "test", "val")
