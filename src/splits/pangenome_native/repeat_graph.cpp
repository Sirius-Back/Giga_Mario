#include "repeat_graph.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

inline int base_code(char c) {
    switch (c) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            return -1;
    }
}

struct UnionFind {
    std::vector<int32_t> parent;
    std::vector<int32_t> rank;

    explicit UnionFind(int32_t n) : parent(static_cast<size_t>(n)), rank(static_cast<size_t>(n), 0) {
        for (int32_t i = 0; i < n; ++i) {
            parent[static_cast<size_t>(i)] = i;
        }
    }

    int32_t find(int32_t x) {
        int32_t r = x;
        while (parent[static_cast<size_t>(r)] != r) {
            r = parent[static_cast<size_t>(r)];
        }
        while (parent[static_cast<size_t>(x)] != x) {
            const int32_t p = parent[static_cast<size_t>(x)];
            parent[static_cast<size_t>(x)] = r;
            x = p;
        }
        return r;
    }

    void unite(int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)]) {
            parent[static_cast<size_t>(a)] = b;
        } else if (rank[static_cast<size_t>(a)] > rank[static_cast<size_t>(b)]) {
            parent[static_cast<size_t>(b)] = a;
        } else {
            parent[static_cast<size_t>(b)] = a;
            ++rank[static_cast<size_t>(a)];
        }
    }
};

inline uint64_t pack_edge_key(int32_t u, int32_t v) {
    if (u > v) {
        std::swap(u, v);
    }
    return (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(v));
}

}  // namespace

extern "C" int pangenome_contingency_clusters(
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
) {
    if (seq_blob == nullptr || offsets == nullptr || cluster_out == nullptr ||
        n_clusters_out == nullptr || n_regions <= 0 || k < 1 || k > PANGENOME_MAX_K ||
        min_shared < 1) {
        return 1;
    }
    if (n_edges_out == nullptr) {
        return 1;
    }
    const bool want_edges =
        edge_u_out != nullptr && edge_v_out != nullptr && edge_w_out != nullptr && max_edges > 0;

    UnionFind uf(n_regions);
    // First region index that observed each k-mer (contingency join key).
    std::unordered_map<uint64_t, int32_t> kmer_owner;
    kmer_owner.reserve(static_cast<size_t>(n_regions) * 64u);

    std::unordered_map<uint64_t, int32_t> edge_weight;
    if (want_edges) {
        edge_weight.reserve(static_cast<size_t>(std::min(max_edges, n_regions * 8)));
    }

    const uint64_t mask =
        (k == 32) ? ~uint64_t{0} : ((uint64_t{1} << (2 * k)) - uint64_t{1});

    for (int32_t r = 0; r < n_regions; ++r) {
        const int64_t start = offsets[r];
        const int64_t end = offsets[r + 1];
        if (start < 0 || end < start) {
            return 1;
        }
        const size_t len = static_cast<size_t>(end - start);
        if (len < static_cast<size_t>(k)) {
            continue;
        }
        uint64_t state = 0;
        int filled = 0;
        for (size_t i = 0; i < len; ++i) {
            const int b = base_code(seq_blob[static_cast<size_t>(start) + i]);
            if (b < 0) {
                filled = 0;
                state = 0;
                continue;
            }
            state = ((state << 2) | static_cast<uint64_t>(b)) & mask;
            if (filled + 1 < k) {
                ++filled;
                continue;
            }
            filled = k;
            const auto it = kmer_owner.find(state);
            if (it == kmer_owner.end()) {
                kmer_owner.emplace(state, r);
            } else {
                const int32_t owner = it->second;
                if (owner != r) {
                    uf.unite(owner, r);
                    if (want_edges) {
                        const uint64_t ek = pack_edge_key(owner, r);
                        ++edge_weight[ek];
                    }
                }
            }
        }
    }

    // Compress cluster ids to 0..n_clusters-1 in first-seen region order.
    std::unordered_map<int32_t, int32_t> root_to_cid;
    root_to_cid.reserve(static_cast<size_t>(n_regions));
    int32_t next_cid = 0;
    for (int32_t r = 0; r < n_regions; ++r) {
        const int32_t root = uf.find(r);
        auto it = root_to_cid.find(root);
        if (it == root_to_cid.end()) {
            root_to_cid.emplace(root, next_cid);
            cluster_out[r] = next_cid;
            ++next_cid;
        } else {
            cluster_out[r] = it->second;
        }
    }
    *n_clusters_out = next_cid;

    int32_t n_edges = 0;
    if (want_edges) {
        for (const auto &kv : edge_weight) {
            if (kv.second < min_shared) {
                continue;
            }
            if (n_edges >= max_edges) {
                break;
            }
            const int32_t u = static_cast<int32_t>(kv.first >> 32);
            const int32_t v = static_cast<int32_t>(kv.first & 0xffffffffu);
            edge_u_out[n_edges] = u;
            edge_v_out[n_edges] = v;
            edge_w_out[n_edges] = kv.second;
            ++n_edges;
        }
    }
    *n_edges_out = n_edges;
    return 0;
}
