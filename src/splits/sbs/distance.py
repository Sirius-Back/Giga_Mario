"""Distance-matrix contract for split-by-similarity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .fna_io import FastaMode, load_fna_sequences


@dataclass(frozen=True)
class DistanceMatrix:
    """Square pairwise distance matrix indexed by region ids."""

    ids: tuple[str, ...]
    matrix: np.ndarray
    metric: str
    backend: str
    extras: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"distance matrix must be square; got shape {mat.shape}")
        if mat.shape[0] != len(self.ids):
            raise ValueError(
                f"matrix size {mat.shape[0]} != n_ids {len(self.ids)}"
            )
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("DistanceMatrix.ids must be unique")
        object.__setattr__(self, "matrix", mat)

    @property
    def n(self) -> int:
        return len(self.ids)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.matrix, index=list(self.ids), columns=list(self.ids))

    def subset(self, ids: list[str]) -> "DistanceMatrix":
        index = {rid: i for i, rid in enumerate(self.ids)}
        missing = [rid for rid in ids if rid not in index]
        if missing:
            raise KeyError(f"ids absent from distance matrix: {missing[:3]}")
        ix = [index[rid] for rid in ids]
        return DistanceMatrix(
            ids=tuple(ids),
            matrix=self.matrix[np.ix_(ix, ix)],
            metric=self.metric,
            backend=self.backend,
            extras=self.extras,
        )


@runtime_checkable
class DistanceBackend(Protocol):
    """Pluggable pairwise (or cluster-aware) distance from sequences."""

    name: str

    def compute(self, sequences: Mapping[str, str]) -> DistanceMatrix:
        """Return a DistanceMatrix over ``sequences`` keys (stable sorted order)."""


def compute_distance_matrix(
    fna: Path,
    backend: DistanceBackend,
    *,
    mode: FastaMode = "auto",
    ids: list[str] | None = None,
    max_ids: int | None = None,
) -> DistanceMatrix:
    """Contract C1: FNA → DistanceMatrix via a selected backend."""
    selected = list(ids) if ids is not None else None
    if selected is not None and max_ids is not None:
        selected = selected[: max(0, int(max_ids))]
        max_ids = None  # already truncated
    sequences = load_fna_sequences(fna, mode=mode, ids=selected, max_ids=max_ids)
    if len(sequences) < 2:
        raise ValueError(f"need >=2 sequences for a distance matrix; got {len(sequences)}")
    return backend.compute(sequences)
