#include "merge_summarize.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>

void sort_split_by_hash(std::vector<SplitRow>& split) {
  std::sort(split.begin(), split.end(), [](const SplitRow& a, const SplitRow& b) {
    if (a.id_hash != b.id_hash) return a.id_hash < b.id_hash;
    return a.id < b.id;
  });
}

GlobalRoleFrac compute_role_fractions(const std::vector<SplitRow>& split) {
  GlobalRoleFrac f;
  for (const auto& r : split) {
    switch (r.role) {
      case Role::Train: ++f.n_train; break;
      case Role::Test: ++f.n_test; break;
      case Role::Val: ++f.n_val; break;
      default: ++f.n_other; break;
    }
  }
  const double denom = static_cast<double>(f.n_train + f.n_test + f.n_val);
  if (denom <= 0.0)
    throw std::runtime_error("split.csv has no train/test/val rows");
  f.p_train = f.n_train / denom;
  f.p_test = f.n_test / denom;
  f.p_val = f.n_val / denom;
  return f;
}

double sd_random_deviation(int64_t n_train, int64_t n_test, int64_t n_val,
                           const GlobalRoleFrac& fracs) {
  const double n = static_cast<double>(n_train + n_test + n_val);
  if (n <= 0.0) return 0.0;
  const double e_tr = n * fracs.p_train;
  const double e_te = n * fracs.p_test;
  const double e_va = n * fracs.p_val;
  const double d0 = static_cast<double>(n_train) - e_tr;
  const double d1 = static_cast<double>(n_test) - e_te;
  const double d2 = static_cast<double>(n_val) - e_va;
  return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

struct Acc {
  int64_t n_train = 0;
  int64_t n_test = 0;
  int64_t n_val = 0;
};

static void bump(Acc& a, Role role) {
  switch (role) {
    case Role::Train: ++a.n_train; break;
    case Role::Test: ++a.n_test; break;
    case Role::Val: ++a.n_val; break;
    default: break;
  }
}

static std::vector<GroupCounts> finalize(std::map<std::string, Acc>& m,
                                         const GlobalRoleFrac& fracs) {
  std::vector<GroupCounts> out;
  out.reserve(m.size());
  for (auto& [gid, a] : m) {
    if (gid.empty()) continue;
    if (a.n_train + a.n_test + a.n_val == 0) continue;
    GroupCounts g;
    g.group_id = gid;
    g.n_train = a.n_train;
    g.n_test = a.n_test;
    g.n_val = a.n_val;
    g.sd_random = sd_random_deviation(a.n_train, a.n_test, a.n_val, fracs);
    out.push_back(std::move(g));
  }
  std::sort(out.begin(), out.end(), [](const GroupCounts& a, const GroupCounts& b) {
    const int64_t na = a.n_train + a.n_test + a.n_val;
    const int64_t nb = b.n_train + b.n_test + b.n_val;
    if (na != nb) return na > nb;
    return a.group_id < b.group_id;
  });
  return out;
}

void merge_and_count(const std::vector<SplitRow>& split_sorted,
                     const std::vector<HashRow>& hash_sorted,
                     std::vector<GroupCounts>& orthologs_out,
                     std::vector<GroupCounts>& paralogs_out,
                     GlobalRoleFrac fracs) {
  std::map<std::string, Acc> ortho;
  std::map<std::string, Acc> para;

  size_t i = 0;  // split
  size_t j = 0;  // hash
  const size_t n = split_sorted.size();
  const size_t m = hash_sorted.size();

  while (i < n && j < m) {
    const uint32_t hs = split_sorted[i].id_hash;
    const uint32_t hh = hash_sorted[j].id_marked_hash;
    if (hs < hh) {
      ++i;
      continue;
    }
    if (hh < hs) {
      ++j;
      continue;
    }
    // Equal hash bucket: walk both sides with matching hash.
    size_t i1 = i;
    while (i1 < n && split_sorted[i1].id_hash == hs) ++i1;
    size_t j1 = j;
    while (j1 < m && hash_sorted[j1].id_marked_hash == hh) ++j1;

    // Nested match on id string within the hash bucket.
    size_t a = i;
    size_t b = j;
    while (a < i1 && b < j1) {
      const std::string& sid = split_sorted[a].id;
      const std::string& hid = hash_sorted[b].id_marked;
      if (sid < hid) {
        ++a;
        continue;
      }
      if (hid < sid) {
        ++b;
        continue;
      }
      // sid == hid: one split row may pair with multiple hash rows (multi-Ensembl).
      size_t a1 = a;
      while (a1 < i1 && split_sorted[a1].id == sid) ++a1;
      size_t b1 = b;
      while (b1 < j1 && hash_sorted[b1].id_marked == hid) ++b1;
      for (size_t aa = a; aa < a1; ++aa) {
        const Role role = split_sorted[aa].role;
        if (role == Role::Other) continue;
        for (size_t bb = b; bb < b1; ++bb) {
          if (!hash_sorted[bb].orthogroup.empty())
            bump(ortho[hash_sorted[bb].orthogroup], role);
          if (!hash_sorted[bb].paragroup.empty())
            bump(para[hash_sorted[bb].paragroup], role);
        }
      }
      a = a1;
      b = b1;
    }
    i = i1;
    j = j1;
  }

  orthologs_out = finalize(ortho, fracs);
  paralogs_out = finalize(para, fracs);
}
