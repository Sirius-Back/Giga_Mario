---
id: kmer
name: k-mer composition split (SBS)
aliases:
  - kmer
  - k-mer
  - dsk_kmer
---

# k-mer similarity split

**Status:** current contract  
**Date:** 2026-07-29  
**Architecture:** **MUST** use the generic **Split-by-Similarity (SBS)** architecture described in `splits-by-similarity.md`. This strategy only replaces the feature extraction backend.

---

## Overview

This strategy performs similarity-aware splitting using **k-mer composition** instead of handcrafted sequence statistics.

The complete pipeline, assignment logic, clustering, diagnostics, contracts, visualization, and output formats **must remain identical to SBS**.

Only the feature backend changes.

---

## SBS backend

Use:

```
src.splits.sbs.backends.kmer
```

instead of

```
src.splits.sbs.backends.gc
```

All downstream components are shared with SBS.

---

## Input

In addition to the standard SBS parameters, this strategy accepts:

### k

One or more k-mer sizes.

Examples

```
k = 5
```

or

```
k = [4,5,6]
```

Features from all requested k values are concatenated into a single `FeatureTable`.

---

## Feature extraction

Feature extraction **must** use **DSK**.

DSK is responsible only for efficient k-mer counting.

Pipeline:

```
FNA
    ↓
DSK
    ↓
k-mer counts
    ↓
normalization
    ↓
FeatureTable
```

The implementation should never enumerate all possible k-mers in memory.

Instead, DSK should produce observed k-mer counts which are then converted into feature vectors.

Recommended normalization:

- relative abundance
- optional log transform
- optional TF-IDF (future)

---

## FeatureTable

Rows

```
region
```

Columns

```
kmer_AAAA
kmer_AAAC
...
```

or, for multiple k,

```
k4_AAAA
...
k5_AAAAA
...
k6_AAAAAA
...
```

The output **must** satisfy SBS Contract C1.

---

## Clustering

Identical to SBS.

Supported methods:

- dbscan (default)
- kmeans
- kmeans_elbow
- hierarchical
- pca_kmeans
- auto

---

## Assignment

Identical to SBS.

Including:

- ZSV handling
- fold assignment
- train / validation / test
- optional zero-shot
- stratification

The output **must** satisfy SBS Contract C2.

---

## Diagnostics

Reuse the SBS diagnostics without modification.

PCA is computed from the k-mer feature table.

Produce:

1. clusters
2. train/test/validation
3. genome
4. custom metadata

using the standard SBS visualization pipeline.

---

## Required implementation

Strategy wrapper

```
src/splits/kmer.py
```

Backend

```
src/splits/sbs/backends/kmer.py
```

The backend must expose

```
KmerFeatureBackend
```

implementing

```
compute_feature_table(...)
```

returning the standard SBS `FeatureTable`.

---

## External dependency

Feature extraction must use

- DSK

The implementation should automatically invoke DSK, collect its output and convert it into the SBS `FeatureTable`.

---

## Output

Exactly the same outputs as SBS.

```
FeatureTable
```

```
assignment table
```

```
split.csv
```

```
SPLIT/
```

```
PCA diagnostics
```

No additional output formats should be introduced.
