#include "paralogs_only.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Node {
  std::string ensembl_gene;
  std::string species;
  std::string marked_id;
  int paralog_degree = 0;
};

struct Edge {
  int u = -1;
  int v = -1;
  bool is_ortholog = false;
};

struct Graph {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::unordered_map<std::string, int> key_to_idx;
  std::vector<std::vector<int>> ortho_adj;
  std::vector<std::vector<int>> para_adj;

  static std::string node_key(const std::string& species, const std::string& gene) {
    return species + "|" + gene;
  }

  int get_or_add(const std::string& species, const std::string& gene) {
    const std::string key = node_key(species, gene);
    auto it = key_to_idx.find(key);
    if (it != key_to_idx.end()) return it->second;
    int idx = static_cast<int>(nodes.size());
    Node n;
    n.ensembl_gene = gene;
    n.species = species;
    nodes.push_back(std::move(n));
    key_to_idx.emplace(key, idx);
    return idx;
  }

  void add_edge(int u, int v, bool ortholog) {
    if (u == v) return;
    if (u > v) std::swap(u, v);
    edges.push_back(Edge{u, v, ortholog});
  }

  void finalize_adj() {
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
      if (a.u != b.u) return a.u < b.u;
      if (a.v != b.v) return a.v < b.v;
      return a.is_ortholog > b.is_ortholog;
    });
    edges.erase(std::unique(edges.begin(), edges.end(),
                            [](const Edge& a, const Edge& b) {
                              return a.u == b.u && a.v == b.v &&
                                     a.is_ortholog == b.is_ortholog;
                            }),
                edges.end());
    const int n = static_cast<int>(nodes.size());
    ortho_adj.assign(n, {});
    para_adj.assign(n, {});
    for (const auto& e : edges) {
      if (e.is_ortholog) {
        ortho_adj[e.u].push_back(e.v);
        ortho_adj[e.v].push_back(e.u);
      } else {
        para_adj[e.u].push_back(e.v);
        para_adj[e.v].push_back(e.u);
      }
    }
  }

  void compute_paralog_degrees() {
    for (size_t i = 0; i < nodes.size(); ++i)
      nodes[i].paralog_degree = static_cast<int>(para_adj[i].size());
  }
};

std::vector<std::string> split_tab(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == '\t') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

int col_index(const std::vector<std::string>& hdr, const std::string& name) {
  for (size_t i = 0; i < hdr.size(); ++i)
    if (hdr[i] == name) return static_cast<int>(i);
  return -1;
}

bool load_nodes(const std::string& path, Graph& g) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  auto hdr = split_tab(line);
  const int i_sp = col_index(hdr, "ensembl_species");
  const int i_gene = col_index(hdr, "ensembl_gene");
  const int i_mark = col_index(hdr, "marked_id");
  if (i_sp < 0 || i_gene < 0 || i_mark < 0) return false;
  const int need = std::max(std::max(i_sp, i_gene), i_mark);
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    auto c = split_tab(line);
    if (static_cast<int>(c.size()) <= need) continue;
    if (c[i_mark].empty()) continue;
    int id = g.get_or_add(c[i_sp], c[i_gene]);
    g.nodes[id].marked_id = c[i_mark];
  }
  return true;
}

bool load_edges(const std::string& path, Graph& g) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  auto hdr = split_tab(line);
  const int i_g1 = col_index(hdr, "gene1");
  const int i_s1 = col_index(hdr, "genome1");
  const int i_g2 = col_index(hdr, "gene2");
  const int i_s2 = col_index(hdr, "genome2");
  const int i_rel = col_index(hdr, "relation");
  if (i_g1 < 0 || i_s1 < 0 || i_g2 < 0 || i_s2 < 0 || i_rel < 0) return false;
  const int need = std::max({i_g1, i_s1, i_g2, i_s2, i_rel});
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    auto c = split_tab(line);
    if (static_cast<int>(c.size()) <= need) continue;
    std::string g1 = c[i_g1], g2 = c[i_g2];
    auto d1 = g1.find('.');
    if (d1 != std::string::npos) g1.resize(d1);
    auto d2 = g2.find('.');
    if (d2 != std::string::npos) g2.resize(d2);
    auto it1 = g.key_to_idx.find(Graph::node_key(c[i_s1], g1));
    auto it2 = g.key_to_idx.find(Graph::node_key(c[i_s2], g2));
    if (it1 == g.key_to_idx.end() || it2 == g.key_to_idx.end()) continue;
    g.add_edge(it1->second, it2->second, c[i_rel] == "ortholog");
  }
  return true;
}

bool load_id_lines(const std::string& path, std::vector<std::string>& out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    out.push_back(line);
  }
  return true;
}

std::vector<int> orthogroups_eligible(const Graph& g,
                                      const std::vector<char>& eligible) {
  const int n = static_cast<int>(g.nodes.size());
  std::vector<int> parent(n);
  for (int i = 0; i < n; ++i) parent[i] = i;
  auto find = [&](auto&& self, int x) -> int {
    return parent[x] == x ? x : (parent[x] = self(self, parent[x]));
  };
  auto unite = [&](int a, int b) {
    a = find(find, a);
    b = find(find, b);
    if (a != b) parent[b] = a;
  };
  for (const auto& e : g.edges) {
    if (!e.is_ortholog) continue;
    if (!eligible[e.u] || !eligible[e.v]) continue;
    unite(e.u, e.v);
  }
  std::vector<int> og(n, -1);
  std::unordered_map<int, int> remap;
  int next = 0;
  for (int i = 0; i < n; ++i) {
    if (!eligible[i]) continue;
    int r = find(find, i);
    auto it = remap.find(r);
    if (it == remap.end()) {
      remap.emplace(r, next);
      og[i] = next++;
    } else {
      og[i] = it->second;
    }
  }
  return og;
}

}  // namespace

extern "C" int paralogs_only_assign(
    const char *edges_path,
    const char *nodes_path,
    const char *panel_ids_path,
    uint64_t seed,
    const char *out_assignment_path,
    const char *out_meta_json_path
) {
  if (!edges_path || !nodes_path || !panel_ids_path || !out_assignment_path)
    return 1;

  Graph g;
  if (!load_nodes(nodes_path, g)) return 2;
  if (!load_edges(edges_path, g)) return 3;
  g.finalize_adj();
  g.compute_paralog_degrees();

  std::vector<std::string> panel_ids;
  if (!load_id_lines(panel_ids_path, panel_ids)) return 4;
  std::unordered_set<std::string> panel_set(panel_ids.begin(), panel_ids.end());

  // marked_id -> list of node indices (eligible candidates)
  std::unordered_map<std::string, std::vector<int>> marked_to_nodes;
  std::vector<char> eligible(g.nodes.size(), 0);
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    const std::string& mid = g.nodes[i].marked_id;
    if (mid.empty()) continue;
    if (!panel_set.count(mid)) continue;
    eligible[i] = 1;
    marked_to_nodes[mid].push_back(static_cast<int>(i));
  }

  auto og_of = orthogroups_eligible(g, eligible);
  int n_og = 0;
  for (int v : og_of) if (v + 1 > n_og) n_og = v + 1;

  std::vector<std::vector<int>> members(n_og);
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    if (!eligible[i]) continue;
    members[og_of[i]].push_back(static_cast<int>(i));
  }

  std::mt19937_64 rng(seed);
  std::unordered_set<std::string> train_ids;
  std::unordered_map<std::string, int> id_to_og;  // mapped panel IDs

  for (int og = 0; og < n_og; ++og) {
    auto& mem = members[og];
    if (mem.empty()) continue;
    // Prefer unique marked_ids within OG when selecting rep node.
    std::shuffle(mem.begin(), mem.end(), rng);
    std::stable_sort(mem.begin(), mem.end(), [&](int a, int b) {
      if (g.nodes[a].paralog_degree != g.nodes[b].paralog_degree)
        return g.nodes[a].paralog_degree > g.nodes[b].paralog_degree;
      return g.nodes[a].ensembl_gene < g.nodes[b].ensembl_gene;
    });
    int rep = mem.front();
    const std::string& rid = g.nodes[rep].marked_id;
    if (!rid.empty() && panel_set.count(rid)) train_ids.insert(rid);
    for (int u : mem) {
      const std::string& mid = g.nodes[u].marked_id;
      if (!mid.empty()) id_to_og[mid] = og;
    }
  }

  // Remainder = all panel IDs not selected as train (includes unmapped).
  std::vector<std::string> remainder;
  remainder.reserve(panel_ids.size());
  for (const auto& id : panel_ids) {
    if (train_ids.count(id)) continue;
    remainder.push_back(id);
  }
  std::shuffle(remainder.begin(), remainder.end(), rng);
  const size_t n_test = (remainder.size() + 1) / 2;  // ceil → test

  std::ofstream out(out_assignment_path);
  if (!out) return 5;
  out << "ID\ttrain_test\tfold\n";
  for (const auto& id : train_ids) {
    int fold = id_to_og.count(id) ? id_to_og[id] : -1;
    out << id << "\ttrain\t" << fold << "\n";
  }
  for (size_t i = 0; i < remainder.size(); ++i) {
    const std::string& id = remainder[i];
    const char* lab = (i < n_test) ? "test" : "val";
    std::string fold = "unmapped";
    if (id_to_og.count(id)) fold = std::to_string(id_to_og[id]);
    out << id << "\t" << lab << "\t" << fold << "\n";
  }

  if (out_meta_json_path && out_meta_json_path[0]) {
    std::ofstream meta(out_meta_json_path);
    if (meta) {
      meta << "{\n"
           << "  \"n_panel\": " << panel_ids.size() << ",\n"
           << "  \"n_graph_nodes\": " << g.nodes.size() << ",\n"
           << "  \"n_edges\": " << g.edges.size() << ",\n"
           << "  \"n_orthogroups\": " << n_og << ",\n"
           << "  \"n_train\": " << train_ids.size() << ",\n"
           << "  \"n_remainder\": " << remainder.size() << ",\n"
           << "  \"n_test\": " << n_test << ",\n"
           << "  \"n_val\": " << (remainder.size() - n_test) << ",\n"
           << "  \"n_mapped_panel\": " << marked_to_nodes.size() << ",\n"
           << "  \"n_unmapped_panel\": "
           << (panel_ids.size() - marked_to_nodes.size()) << ",\n"
           << "  \"seed\": " << seed << "\n"
           << "}\n";
    }
  }
  return 0;
}
