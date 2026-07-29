"""GC% + AAA% feature backend (and legacy delta-GC distance helper)."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from src.splits.sbs.distance import DistanceMatrix
from src.splits.sbs.features import FeatureTable

FEATURE_GC_PCT = "GC_pct"
FEATURE_AAA_PCT = "AAA_pct"
DEFAULT_FEATURE_NAMES = (FEATURE_GC_PCT, FEATURE_AAA_PCT)


def gc_fraction(sequence: str) -> float:
    """GC / (A+C+G+T); ambiguous bases ignored. Empty ACGT → 0.0."""
    seq = sequence.upper()
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    for base in seq:
        if base in counts:
            counts[base] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts["G"] + counts["C"]) / total


def gc_percent(sequence: str) -> float:
    """GC composition in percent [0, 100]."""
    return 100.0 * gc_fraction(sequence)


def aaa_percent(sequence: str) -> float:
    """Overlapping AAA triplet frequency in percent of possible trinucleotide starts.

    For length < 3 returns 0.0.
    """
    seq = sequence.upper()
    if len(seq) < 3:
        return 0.0
    hits = 0
    for i in range(len(seq) - 2):
        if seq[i : i + 3] == "AAA":
            hits += 1
    return 100.0 * hits / (len(seq) - 2)


class GcAaaFeatureBackend:
    """Per-region features: GC (%) and AAA (%). Primary SBS GC backend."""

    name = "gc_aaa"

    def compute(self, sequences: Mapping[str, str]) -> FeatureTable:
        ids = tuple(sorted(sequences))
        mat = np.zeros((len(ids), 2), dtype=float)
        for i, rid in enumerate(ids):
            seq = sequences[rid]
            mat[i, 0] = gc_percent(seq)
            mat[i, 1] = aaa_percent(seq)
        return FeatureTable(
            ids=ids,
            feature_names=DEFAULT_FEATURE_NAMES,
            matrix=mat,
            backend=self.name,
            extras=None,
        )


class GcDistanceBackend:
    """Legacy pairwise |ΔGC| distance (small-n diagnostics only; O(n²))."""

    name = "gc"

    def compute(self, sequences: Mapping[str, str]) -> DistanceMatrix:
        ids = tuple(sorted(sequences))
        gc = np.asarray([gc_fraction(sequences[i]) for i in ids], dtype=float)
        matrix = np.abs(gc[:, None] - gc[None, :])
        np.fill_diagonal(matrix, 0.0)
        return DistanceMatrix(
            ids=ids,
            matrix=matrix,
            metric="delta_gc",
            backend=self.name,
            extras={"gc_fraction": {i: float(g) for i, g in zip(ids, gc)}},
        )
