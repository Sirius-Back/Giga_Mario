#ifndef GIGAMARIO_PANGENOME_REPEAT_GRAPH_H
#define GIGAMARIO_PANGENOME_REPEAT_GRAPH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* k-mer length for rolling 2-bit codes; supports 1..32. */
#define PANGENOME_MAX_K 32

/*
 * DEPRECATED path: union-find on *regions* that share a k-mer (first-owner
 * contingency). Prefer ``pangenome_hash_majority_clusters``.
 *
 * Sequences are laid out in ``seq_blob``; region i occupies
 * ``seq_blob[offsets[i] .. offsets[i+1])``. ``offsets`` has length
 * ``n_regions + 1``.
 *
 * Returns 0 on success, non-zero on error:
 *   1 invalid args
 *   2 allocation failure
 */
int pangenome_contingency_clusters(
    const char *seq_blob,
    const int64_t *offsets,
    int32_t n_regions,
    int k,
    int32_t min_shared,
    int32_t *cluster_out,
    int32_t *n_clusters_out,
    int32_t *edge_u_out,
    int32_t *edge_v_out,
    int32_t *edge_w_out,
    int32_t max_edges,
    int32_t *n_edges_out
);

/*
 * Hash-graph → majority → region folds.
 *
 * 1. Extract ACGT k-mer hashes per region.
 * 2. Keep *repeat* hashes with document frequency >= ``min_df`` (default 2).
 * 3. Union-find on hash nodes: two repeat hashes are united only if they
 *    co-occur in ≥2 sequences (so a single multi-motif bridge does not
 *    collapse distinct repeat families). Within each sequence, star/chain
 *    pairs are counted toward that co-occurrence.
 * 4. Each region gets the majority hash-cluster among its repeat hashes
 *    (ties → smaller cluster id). Regions with no repeat hashes get a
 *    unique singleton fold.
 *
 * Optional region–region edges (viz): weight = shared repeat hashes,
 * written when weight >= 1 up to ``max_edges``.
 *
 * Returns 0 on success, non-zero on error (same codes as above).
 */
int pangenome_hash_majority_clusters(
    const char *seq_blob,
    const int64_t *offsets,
    int32_t n_regions,
    int k,
    int32_t min_df,
    int32_t *cluster_out,
    int32_t *n_clusters_out,
    int32_t *edge_u_out,
    int32_t *edge_v_out,
    int32_t *edge_w_out,
    int32_t max_edges,
    int32_t *n_edges_out
);

#ifdef __cplusplus
}
#endif

#endif /* GIGAMARIO_PANGENOME_REPEAT_GRAPH_H */
