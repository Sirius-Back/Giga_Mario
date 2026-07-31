#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Node {
  std::string ensembl_gene;
  std::string species;
  std::string gene_symbol;
  std::string marked_id;
  int paralog_degree = 0;
  int component_id = -1;
  int orthogroup_id = -1;  // within full graph, on ortholog edges only
};

struct Edge {
  int u = -1;
  int v = -1;
  bool is_ortholog = false;  // else paralog
};

struct Graph {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::unordered_map<std::string, int> key_to_idx;  // species|gene -> idx
  std::vector<std::vector<int>> ortho_adj;
  std::vector<std::vector<int>> para_adj;

  int get_or_add(const std::string& species, const std::string& gene);
  void add_edge(int u, int v, bool ortholog);
  void finalize_adj();
  void compute_paralog_degrees();
  std::vector<int> connected_components(bool ortholog_only) const;
};

std::string node_key(const std::string& species, const std::string& gene);
