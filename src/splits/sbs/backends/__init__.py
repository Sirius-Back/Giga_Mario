"""Distance backends for SBS (legacy O(n²) path; prefer FeatureBackend)."""
from __future__ import annotations

from .gc import (
    DEFAULT_FEATURE_NAMES,
    FEATURE_AAA_PCT,
    FEATURE_GC_PCT,
    GcAaaFeatureBackend,
    GcDistanceBackend,
    aaa_percent,
    gc_fraction,
    gc_percent,
)
from .mmseqs import MMseqsDistanceBackend

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "FEATURE_AAA_PCT",
    "FEATURE_GC_PCT",
    "GcAaaFeatureBackend",
    "GcDistanceBackend",
    "MMseqsDistanceBackend",
    "aaa_percent",
    "gc_fraction",
    "gc_percent",
]
