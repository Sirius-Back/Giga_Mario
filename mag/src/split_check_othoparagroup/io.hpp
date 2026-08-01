#pragma once
#include "types.hpp"
#include <string>
#include <vector>

std::vector<HashRow> load_hash_table(const std::string& path);
std::vector<SplitRow> load_split_csv(const std::string& path);
void write_summary_csv(const std::string& path, const std::string& group_col,
                       const std::vector<GroupCounts>& rows);
