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
from .mmseqs import (
    DEFAULT_MIN_SEQ_ID,
    DEFAULT_SENSITIVITY,
    MMseqsDistanceBackend,
    cluster_map_to_dense_ids,
    find_mmseqs,
    parse_cluster_tsv,
    run_mmseqs_easy_cluster,
    write_multifasta,
)

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "DEFAULT_MIN_SEQ_ID",
    "DEFAULT_SENSITIVITY",
    "DSK_MIN_K",
    "NATIVE_MAX_K",
    "FEATURE_AAA_PCT",
    "FEATURE_GC_PCT",
    "GcAaaFeatureBackend",
    "GcDistanceBackend",
    "KmerFeatureBackend",
    "MMseqsDistanceBackend",
    "aaa_percent",
    "cluster_map_to_dense_ids",
    "count_kmers",
    "count_kmers_dsk",
    "count_kmers_local",
    "find_dsk",
    "find_dsk2ascii",
    "find_mmseqs",
    "gc_fraction",
    "gc_percent",
    "normalize_k_list",
    "parse_cluster_tsv",
    "parse_dsk_ascii",
    "run_mmseqs_easy_cluster",
    "write_multifasta",
]
