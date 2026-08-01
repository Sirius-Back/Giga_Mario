#pragma once
#include "types.hpp"
#include <utility>
#include <vector>

struct GlobalRoleFrac {
  double p_train = 0;
  double p_test = 0;
  double p_val = 0;
  int64_t n_train = 0;
  int64_t n_test = 0;
  int64_t n_val = 0;
  int64_t n_other = 0;
};

/** Sort split rows by id_hash, then id (stable merge key). */
void sort_split_by_hash(std::vector<SplitRow>& split);

/** Empirical train/test/val fractions from split (Other/zsv excluded from denom). */
GlobalRoleFrac compute_role_fractions(const std::vector<SplitRow>& split);

/**
 * Merge sorted split with sorted hash table on (id_hash == id_MARKED_hash) + id match.
 * Accumulate per-orthogroup and per-paragroup role counts for train/test/val only.
 * Genes with empty orthogroup/paragroup are skipped for that table.
 * Multi-Ensembl hash rows for one MARKED id contribute to each listed group.
 */
void merge_and_count(const std::vector<SplitRow>& split_sorted,
                     const std::vector<HashRow>& hash_sorted,
                     std::vector<GroupCounts>& orthologs_out,
                     std::vector<GroupCounts>& paralogs_out,
                     GlobalRoleFrac fracs);

/** sd_random = sqrt(sum_r (O_r - n*p_r)^2) using empirical split fractions. */
double sd_random_deviation(int64_t n_train, int64_t n_test, int64_t n_val,
                           const GlobalRoleFrac& fracs);
