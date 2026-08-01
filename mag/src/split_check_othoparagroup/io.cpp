#include "io.hpp"
#include "stable_hash.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

static std::vector<std::string> split_pipe(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == '|') {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

static Role parse_role(const std::string& s) {
  if (s == "train") return Role::Train;
  if (s == "test") return Role::Test;
  if (s == "val" || s == "validation") return Role::Val;
  return Role::Other;  // zsv, zeroshotvalidation, etc.
}

std::vector<HashRow> load_hash_table(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open hash table: " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("Empty hash table: " + path);
  auto hdr = split_pipe(line);
  std::unordered_map<std::string, int> col;
  for (int i = 0; i < static_cast<int>(hdr.size()); ++i) col[hdr[i]] = i;
  for (const char* need : {"id_MARKED", "id_MARKED_hash", "id", "genome", "orthogroup",
                           "paragroup"}) {
    if (!col.count(need))
      throw std::runtime_error(std::string("hash table missing column: ") + need);
  }
  std::vector<HashRow> rows;
  rows.reserve(500000);
  uint32_t prev_h = 0;
  bool first = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto c = split_pipe(line);
    auto at = [&](const char* name) -> const std::string& {
      int i = col[name];
      if (i >= static_cast<int>(c.size()))
        throw std::runtime_error("short row in hash table");
      return c[i];
    };
    HashRow r;
    r.id_marked = at("id_MARKED");
    r.id_marked_hash = static_cast<uint32_t>(std::stoul(at("id_MARKED_hash")));
    r.ensembl_id = at("id");
    r.genome = at("genome");
    r.orthogroup = at("orthogroup");
    r.paragroup = at("paragroup");
    if (!first && r.id_marked_hash < prev_h)
      throw std::runtime_error("hash table not sorted by id_MARKED_hash");
    prev_h = r.id_marked_hash;
    first = false;
    rows.push_back(std::move(r));
  }
  return rows;
}

std::vector<SplitRow> load_split_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open split.csv: " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("Empty split.csv: " + path);
  auto hdr = split_pipe(line);
  std::unordered_map<std::string, int> col;
  for (int i = 0; i < static_cast<int>(hdr.size()); ++i) col[hdr[i]] = i;
  if (!col.count("ID") || !col.count("train_test"))
    throw std::runtime_error("split.csv needs ID|train_test|...");
  std::vector<SplitRow> rows;
  rows.reserve(500000);
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto c = split_pipe(line);
    int i_id = col["ID"];
    int i_tt = col["train_test"];
    if (i_id >= static_cast<int>(c.size()) || i_tt >= static_cast<int>(c.size()))
      throw std::runtime_error("short row in split.csv");
    SplitRow r;
    r.id = c[i_id];
    r.role = parse_role(c[i_tt]);
    r.id_hash = stable_hash(r.id);
    rows.push_back(std::move(r));
  }
  return rows;
}

void write_summary_csv(const std::string& path, const std::string& group_col,
                       const std::vector<GroupCounts>& rows) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write: " + path);
  out << group_col << "|n_train|n_test|n_val|sd_random\n";
  out.setf(std::ios::fixed);
  out.precision(6);
  for (const auto& r : rows) {
    out << r.group_id << '|' << r.n_train << '|' << r.n_test << '|' << r.n_val << '|'
        << r.sd_random << '\n';
  }
}
