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
from .kmer import (
    DSK_MIN_K,
    NATIVE_MAX_K,
    KmerFeatureBackend,
    count_kmers,
    count_kmers_dsk,
    count_kmers_local,
    find_dsk,
    find_dsk2ascii,
    normalize_k_list,
    parse_dsk_ascii,
)
from .mmseqs import MMseqsDistanceBackend

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "DSK_MIN_K",
    "NATIVE_MAX_K",
    "FEATURE_AAA_PCT",
    "FEATURE_GC_PCT",
    "GcAaaFeatureBackend",
    "GcDistanceBackend",
    "KmerFeatureBackend",
    "MMseqsDistanceBackend",
    "aaa_percent",
    "count_kmers",
    "count_kmers_dsk",
    "count_kmers_local",
    "find_dsk",
    "find_dsk2ascii",
    "gc_fraction",
    "gc_percent",
    "normalize_k_list",
    "parse_dsk_ascii",
]
