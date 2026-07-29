#include "kmer_count.h"

#include <cstring>

namespace {

inline int base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

}  // namespace

extern "C" size_t kmer_vocab_size(int k) {
    if (k < 1 || k > KMER_COUNT_MAX_K) {
        return 0;
    }
    size_t n = 1;
    for (int i = 0; i < k; ++i) {
        n *= 4;
    }
    return n;
}

extern "C" int kmer_count_dense(
    const char *seq,
    size_t seq_len,
    int k,
    uint32_t *out,
    size_t out_len
) {
    if (seq == nullptr || out == nullptr || k < 1 || k > KMER_COUNT_MAX_K) {
        return 1;
    }
    const size_t need = kmer_vocab_size(k);
    if (need == 0 || out_len != need) {
        return 1;
    }
    std::memset(out, 0, out_len * sizeof(uint32_t));
    if (seq_len < static_cast<size_t>(k)) {
        return 0;
    }

    const uint32_t mask = (k == 32) ? 0xffffffffu : ((1u << (2 * k)) - 1u);
    // For k <= 12, 2*k <= 24 bits fit in uint32.
    uint32_t state = 0;
    int filled = 0;
    for (size_t i = 0; i < seq_len; ++i) {
        const int b = base_code(seq[i]);
        if (b < 0) {
            filled = 0;
            state = 0;
            continue;
        }
        state = ((state << 2) | static_cast<uint32_t>(b)) & mask;
        if (filled + 1 < k) {
            ++filled;
            continue;
        }
        filled = k;
        ++out[state];
    }
    return 0;
}

extern "C" int kmer_count_dimers(const char *seq, size_t seq_len, uint32_t out[16]) {
    return kmer_count_dense(seq, seq_len, 2, out, 16);
}
