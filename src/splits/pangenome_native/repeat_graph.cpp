#include "repeat_graph.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
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

inline void extract_unique_kmers(
    const char *seq_blob,
    int64_t start,
    int64_t end,
    int k,
    uint64_t mask,
    std::unordered_set<uint64_t> *out
) {
    out->clear();
    if (start < 0 || end < start) {
        return;
    }
    const size_t len = static_cast<size_t>(end - start);
    if (len < static_cast<size_t>(k)) {
        return;
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
        out->insert(state);
    }
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

extern "C" int pangenome_hash_majority_clusters(
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
) {
    if (seq_blob == nullptr || offsets == nullptr || cluster_out == nullptr ||
        n_clusters_out == nullptr || n_regions <= 0 || k < 1 || k > PANGENOME_MAX_K ||
        min_df < 1) {
        return 1;
    }
    if (n_edges_out == nullptr) {
        return 1;
    }
    const bool want_edges =
        edge_u_out != nullptr && edge_v_out != nullptr && edge_w_out != nullptr && max_edges > 0;

    const uint64_t mask =
        (k == 32) ? ~uint64_t{0} : ((uint64_t{1} << (2 * k)) - uint64_t{1});

    // Pass 1: document frequency (#sequences containing each k-mer ≥1).
    std::unordered_map<uint64_t, int32_t> df;
    df.reserve(static_cast<size_t>(n_regions) * 64u);
    std::unordered_set<uint64_t> local;
    local.reserve(256u);

    for (int32_t r = 0; r < n_regions; ++r) {
        extract_unique_kmers(seq_blob, offsets[r], offsets[r + 1], k, mask, &local);
        for (uint64_t h : local) {
            ++df[h];
        }
    }

    // Index repeat hashes (df >= min_df).
    std::unordered_map<uint64_t, int32_t> hash_to_idx;
    hash_to_idx.reserve(df.size() / 2u + 1u);
    int32_t n_repeat = 0;
    for (const auto &kv : df) {
        if (kv.second >= min_df) {
            hash_to_idx.emplace(kv.first, n_repeat);
            ++n_repeat;
        }
    }

    UnionFind hash_uf(n_repeat > 0 ? n_repeat : 1);
    std::vector<std::vector<int32_t>> seq_hash_ids(static_cast<size_t>(n_regions));

    // First region that contributed each repeat hash (for region–region viz edges).
    std::vector<int32_t> hash_first_region(static_cast<size_t>(std::max(n_repeat, 1)), -1);
    std::unordered_map<uint64_t, int32_t> edge_weight;
    if (want_edges) {
        edge_weight.reserve(static_cast<size_t>(std::min(max_edges, n_regions * 8)));
    }

    // Pass 2a: collect per-seq repeat hash lists + hash-pair co-occurrence counts.
    // Unite hash pairs only when they co-occur in ≥2 sequences (so a single
    // multi-motif bridge does not collapse distinct repeat families).
    std::unordered_map<uint64_t, int32_t> hash_pair_count;
    hash_pair_count.reserve(static_cast<size_t>(std::max(n_repeat, 1)) * 4u);

    for (int32_t r = 0; r < n_regions; ++r) {
        extract_unique_kmers(seq_blob, offsets[r], offsets[r + 1], k, mask, &local);
        auto &ids = seq_hash_ids[static_cast<size_t>(r)];
        ids.clear();
        ids.reserve(local.size());
        for (uint64_t h : local) {
            const auto it = hash_to_idx.find(h);
            if (it == hash_to_idx.end()) {
                continue;
            }
            const int32_t hid = it->second;
            ids.push_back(hid);
            if (hash_first_region[static_cast<size_t>(hid)] < 0) {
                hash_first_region[static_cast<size_t>(hid)] = r;
            } else if (want_edges) {
                const int32_t owner = hash_first_region[static_cast<size_t>(hid)];
                if (owner != r) {
                    const uint64_t ek = pack_edge_key(owner, r);
                    ++edge_weight[ek];
                }
            }
        }
        // Count all unordered pairs among this seq's repeat hashes.
        if (ids.size() >= 2) {
            std::sort(ids.begin(), ids.end());
            for (size_t i = 0; i < ids.size(); ++i) {
                for (size_t j = i + 1; j < ids.size(); ++j) {
                    const uint64_t pk = pack_edge_key(ids[i], ids[j]);
                    ++hash_pair_count[pk];
                }
            }
        }
    }

    // Pass 2b: unite hash pairs seen together in ≥2 sequences.
    for (const auto &kv : hash_pair_count) {
        if (kv.second < 2) {
            continue;
        }
        const int32_t u = static_cast<int32_t>(kv.first >> 32);
        const int32_t v = static_cast<int32_t>(kv.first & 0xffffffffu);
        hash_uf.unite(u, v);
    }

    // Compress hash CCs → 0..n_hash_clusters-1
    std::unordered_map<int32_t, int32_t> hash_root_to_cid;
    hash_root_to_cid.reserve(static_cast<size_t>(std::max(n_repeat, 1)));
    int32_t n_hash_clusters = 0;
    std::vector<int32_t> hash_cluster(static_cast<size_t>(std::max(n_repeat, 1)), -1);
    for (int32_t h = 0; h < n_repeat; ++h) {
        const int32_t root = hash_uf.find(h);
        auto it = hash_root_to_cid.find(root);
        if (it == hash_root_to_cid.end()) {
            hash_root_to_cid.emplace(root, n_hash_clusters);
            hash_cluster[static_cast<size_t>(h)] = n_hash_clusters;
            ++n_hash_clusters;
        } else {
            hash_cluster[static_cast<size_t>(h)] = it->second;
        }
    }

    // Majority vote per region; singletons for regions with no repeat hashes.
    int32_t next_singleton = n_hash_clusters;
    for (int32_t r = 0; r < n_regions; ++r) {
        const auto &ids = seq_hash_ids[static_cast<size_t>(r)];
        if (ids.empty()) {
            cluster_out[r] = next_singleton;
            ++next_singleton;
            continue;
        }
        std::unordered_map<int32_t, int32_t> counts;
        counts.reserve(ids.size());
        for (int32_t hid : ids) {
            ++counts[hash_cluster[static_cast<size_t>(hid)]];
        }
        int32_t best_cid = -1;
        int32_t best_count = -1;
        for (const auto &kv : counts) {
            if (kv.second > best_count ||
                (kv.second == best_count && (best_cid < 0 || kv.first < best_cid))) {
                best_count = kv.second;
                best_cid = kv.first;
            }
        }
        cluster_out[r] = best_cid;
    }

    // Compact region fold labels to 0..n-1 in first-seen order (stable for assign).
    std::unordered_map<int32_t, int32_t> fold_remap;
    fold_remap.reserve(static_cast<size_t>(n_regions));
    int32_t n_folds = 0;
    for (int32_t r = 0; r < n_regions; ++r) {
        const int32_t raw = cluster_out[r];
        auto it = fold_remap.find(raw);
        if (it == fold_remap.end()) {
            fold_remap.emplace(raw, n_folds);
            cluster_out[r] = n_folds;
            ++n_folds;
        } else {
            cluster_out[r] = it->second;
        }
    }
    *n_clusters_out = n_folds;

    int32_t n_edges = 0;
    if (want_edges) {
        for (const auto &kv : edge_weight) {
            if (kv.second < 1) {
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
