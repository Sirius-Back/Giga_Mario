"""Feature-table contract for split-by-similarity (preferred over dense distances)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .fna_io import FastaMode, load_fna_sequences


@dataclass(frozen=True)
class FeatureTable:
    """Per-region numeric feature matrix (n × d), id-indexed."""

    ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    matrix: np.ndarray
    backend: str
    extras: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=float)
        if mat.ndim != 2:
            raise ValueError(f"feature matrix must be 2D; got shape {mat.shape}")
        if mat.shape[0] != len(self.ids):
            raise ValueError(
                f"matrix rows {mat.shape[0]} != n_ids {len(self.ids)}"
            )
        if mat.shape[1] != len(self.feature_names):
            raise ValueError(
                f"matrix cols {mat.shape[1]} != n_features {len(self.feature_names)}"
            )
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("FeatureTable.ids must be unique")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FeatureTable.feature_names must be unique")
        object.__setattr__(self, "matrix", mat)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def to_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(self.matrix, columns=list(self.feature_names))
        df.insert(0, "region", list(self.ids))
        return df

    def subset(self, ids: Sequence[str]) -> "FeatureTable":
        index = {rid: i for i, rid in enumerate(self.ids)}
        missing = [rid for rid in ids if rid not in index]
        if missing:
            raise KeyError(f"ids absent from feature table: {missing[:3]}")
        ix = [index[rid] for rid in ids]
        return FeatureTable(
            ids=tuple(ids),
            feature_names=self.feature_names,
            matrix=self.matrix[ix, :],
            backend=self.backend,
            extras=self.extras,
        )

    def scaled_matrix(self) -> np.ndarray:
        """Column z-score (safe for constant columns)."""
        x = np.asarray(self.matrix, dtype=float)
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (x - mu) / sd

    def write_csv(self, path: Path) -> Path:
        from src.pipeline.common import write_csv

        rows = []
        for i, rid in enumerate(self.ids):
            row = {"region": rid}
            for j, name in enumerate(self.feature_names):
                row[name] = f"{float(self.matrix[i, j]):.8g}"
            rows.append(row)
        path = Path(path)
        write_csv(path, rows, ["region", *self.feature_names])
        return path


@runtime_checkable
class FeatureBackend(Protocol):
    """Pluggable FNA → per-region features."""

    name: str

    def compute(self, sequences: Mapping[str, str]) -> FeatureTable:
        ...


def compute_feature_table(
    fna: Path,
    backend: FeatureBackend,
    *,
    mode: FastaMode = "auto",
    ids: list[str] | None = None,
    max_ids: int | None = None,
) -> FeatureTable:
    """Contract C1: FNA → FeatureTable via a selected feature backend."""
    selected = list(ids) if ids is not None else None
    if selected is not None and max_ids is not None:
        selected = selected[: max(0, int(max_ids))]
        max_ids = None
    sequences = load_fna_sequences(fna, mode=mode, ids=selected, max_ids=max_ids)
    if len(sequences) < 2:
        raise ValueError(f"need >=2 sequences for features; got {len(sequences)}")
    return backend.compute(sequences)
