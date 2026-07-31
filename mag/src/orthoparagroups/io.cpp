#include "io.hpp"
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace {

class LineReader {
 public:
  explicit LineReader(const std::string& path) {
    ifs_.open(path);
    if (!ifs_) throw std::runtime_error("Cannot open: " + path);
  }
  bool getline(std::string& out) {
    if (!std::getline(ifs_, out)) return false;
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
  }

 private:
  std::ifstream ifs_;
};

std::vector<std::string> split_tab(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == '\t') {
      out.push_back(cur);
      cur.clear();
    } else
      cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

int col_index(const std::vector<std::string>& hdr, const std::string& name) {
  for (size_t i = 0; i < hdr.size(); ++i)
    if (hdr[i] == name) return static_cast<int>(i);
  throw std::runtime_error("Missing column: " + name);
}

}  // namespace

void load_nodes_enriched(const std::string& path, Graph& g, bool require_marked) {
  LineReader r(path);
  std::string line;
  if (!r.getline(line)) throw std::runtime_error("Empty nodes file: " + path);
  auto hdr = split_tab(line);
  const int i_sp = col_index(hdr, "ensembl_species");
  const int i_gene = col_index(hdr, "ensembl_gene");
  const int i_sym = col_index(hdr, "gene_symbol");
  const int i_mark = col_index(hdr, "marked_id");
  const int need = std::max(std::max(i_sp, i_gene), std::max(i_sym, i_mark));

  while (r.getline(line)) {
    if (line.empty()) continue;
    auto c = split_tab(line);
    if (static_cast<int>(c.size()) <= need) continue;
    if (require_marked && c[i_mark].empty()) continue;
    int id = g.get_or_add(c[i_sp], c[i_gene]);
    g.nodes[id].gene_symbol = c[i_sym];
    g.nodes[id].marked_id = c[i_mark];
  }
}

void load_edges_tsv(const std::string& path, Graph& g, bool both_ends_required) {
  LineReader r(path);
  std::string line;
  if (!r.getline(line)) throw std::runtime_error("Empty edges file: " + path);
  auto hdr = split_tab(line);
  const int i_g1 = col_index(hdr, "gene1");
  const int i_s1 = col_index(hdr, "genome1");
  const int i_g2 = col_index(hdr, "gene2");
  const int i_s2 = col_index(hdr, "genome2");
  const int i_rel = col_index(hdr, "relation");

  while (r.getline(line)) {
    if (line.empty() || line[0] == '#') continue;
    auto c = split_tab(line);
    int need = i_g1;
    need = std::max(need, i_s1);
    need = std::max(need, i_g2);
    need = std::max(need, i_s2);
    need = std::max(need, i_rel);
    if (static_cast<int>(c.size()) <= need) continue;
    std::string g1 = c[i_g1], g2 = c[i_g2];
    auto d1 = g1.find('.');
    if (d1 != std::string::npos) g1.resize(d1);
    auto d2 = g2.find('.');
    if (d2 != std::string::npos) g2.resize(d2);
    auto it1 = g.key_to_idx.find(node_key(c[i_s1], g1));
    auto it2 = g.key_to_idx.find(node_key(c[i_s2], g2));
    if (it1 == g.key_to_idx.end() || it2 == g.key_to_idx.end()) continue;
    (void)both_ends_required;
    g.add_edge(it1->second, it2->second, c[i_rel] == "ortholog");
  }
}

std::string read_marked_sequence(const std::string& marked_dir,
                                 const std::string& marked_id) {
  const std::string path = marked_dir + "/" + marked_id + ".fa";
  std::ifstream in(path);
  if (!in) return {};
  std::string line, seq;
  bool first = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (first) {
      first = false;
      continue;
    }
    for (char ch : line) {
      if (!std::isspace(static_cast<unsigned char>(ch))) seq.push_back(ch);
    }
  }
  return seq;
}
