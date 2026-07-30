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
 * Build region contingency clusters from a concatenated sequence blob.
 *
 * Sequences are laid out in ``seq_blob``; region i occupies
 * ``seq_blob[offsets[i] .. offsets[i+1])``. ``offsets`` has length
 * ``n_regions + 1``.
 *
 * Clustering: union-find on regions that share at least one ACGT k-mer
 * (connected components of the bipartite region↔k-mer projection). Does
 * **not** materialize pairwise distance matrices or resolve bubbles.
 *
 * Optional region–region edges (for visualization): when a k-mer already
 * seen in region A is observed in region B, increment weight(A,B). Only
 * edges with weight >= ``min_shared`` are written (up to ``max_edges``).
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

#ifdef __cplusplus
}
#endif

#endif /* GIGAMARIO_PANGENOME_REPEAT_GRAPH_H */
