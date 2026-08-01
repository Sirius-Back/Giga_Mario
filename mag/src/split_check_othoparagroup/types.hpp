#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class Role : uint8_t { Train = 0, Test = 1, Val = 2, Other = 3 };

struct HashRow {
  uint32_t id_marked_hash = 0;
  std::string id_marked;
  std::string ensembl_id;   // may be empty (NULL)
  std::string genome;
  std::string orthogroup;   // empty if unknown
  std::string paragroup;    // empty if unknown
};

struct SplitRow {
  uint32_t id_hash = 0;
  std::string id;
  Role role = Role::Other;
};

struct GroupCounts {
  std::string group_id;
  int64_t n_train = 0;
  int64_t n_test = 0;
  int64_t n_val = 0;
  double sd_random = 0.0;
};
