#ifndef GIGAMARIO_PARALOGS_ONLY_H
#define GIGAMARIO_PARALOGS_ONLY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Orthogroup-representative train / remainder 50-50 test-val.
 *
 * edges_path / nodes_path: TSV (not gzip).
 * panel_ids_path: one panel region ID per line (assignable only; no ZSV).
 * Writes tab-separated out_assignment_path: ID\ttrain_test\tfold
 * Optional out_meta_json_path may be NULL.
 *
 * Returns 0 on success, non-zero on error.
 */
int paralogs_only_assign(
    const char *edges_path,
    const char *nodes_path,
    const char *panel_ids_path,
    uint64_t seed,
    const char *out_assignment_path,
    const char *out_meta_json_path
);

#ifdef __cplusplus
}
#endif

#endif /* GIGAMARIO_PARALOGS_ONLY_H */
