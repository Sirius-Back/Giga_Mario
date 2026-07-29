#ifndef GIGAMARIO_KMER_COUNT_H
#define GIGAMARIO_KMER_COUNT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum k for dense 4^k counting (4^12 = 16,777,216 uint32 ≈ 64 MiB). */
#define KMER_COUNT_MAX_K 12

/*
 * Count overlapping ACGT k-mers into a dense abundance table of length 4^k.
 * Non-ACGT bases break the current window (same semantics as Python local).
 *
 * Returns 0 on success, non-zero on error:
 *   1 invalid args (null / k out of range / out_len != 4^k)
 */
int kmer_count_dense(
    const char *seq,
    size_t seq_len,
    int k,
    uint32_t *out,
    size_t out_len
);

/* Convenience: k must be 2; out must have length 16. */
int kmer_count_dimers(const char *seq, size_t seq_len, uint32_t out[16]);

/* 4^k for 1 <= k <= KMER_COUNT_MAX_K; 0 if out of range. */
size_t kmer_vocab_size(int k);

#ifdef __cplusplus
}
#endif

#endif /* GIGAMARIO_KMER_COUNT_H */
