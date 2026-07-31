#include "graph.hpp"
#include <algorithm>
#include <unordered_map>

std::string node_key(const std::string& species, const std::string& gene) {
  return species + "|" + gene;
}

int Graph::get_or_add(const std::string& species, const std::string& gene) {
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

void Graph::add_edge(int u, int v, bool ortholog) {
  if (u == v) return;
  if (u > v) std::swap(u, v);
  edges.push_back(Edge{u, v, ortholog});
}

void Graph::finalize_adj() {
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

void Graph::compute_paralog_degrees() {
  for (size_t i = 0; i < nodes.size(); ++i)
    nodes[i].paralog_degree = static_cast<int>(para_adj[i].size());
}

std::vector<int> Graph::connected_components(bool ortholog_only) const {
  const int n = static_cast<int>(nodes.size());
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
  for (const auto& e : edges) {
    if (ortholog_only && !e.is_ortholog) continue;
    unite(e.u, e.v);
  }
  std::vector<int> comp(n, -1);
  std::unordered_map<int, int> remap;
  int next = 0;
  for (int i = 0; i < n; ++i) {
    int r = find(find, i);
    auto it = remap.find(r);
    if (it == remap.end()) {
      remap.emplace(r, next);
      comp[i] = next++;
    } else {
      comp[i] = it->second;
    }
  }
  return comp;
}
