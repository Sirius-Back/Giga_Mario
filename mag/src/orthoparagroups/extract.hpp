#pragma once
#include "graph.hpp"
#include "io.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct ClusterStats {
  std::string fna_name;
  int component_id = -1;
  int n_nodes = 0;
  int n_ortholog_groups = 0;
  int n_ortholog_edges = 0;
  int n_paralog_edges = 0;
  int n_written_orthologs = 0;
  int n_written_paralogs = 0;
  std::unordered_map<std::string, int> nodes_per_species;
};

std::vector<ClusterStats> extract_orthoparagroups(Graph& g,
                                                  const ExtractOptions& opt);
