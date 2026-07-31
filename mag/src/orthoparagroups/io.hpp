#pragma once
#include "graph.hpp"
#include <string>
#include <unordered_map>

struct ExtractOptions {
  std::string edges_path;
  std::string nodes_path;
  std::string marked_dir;
  std::string outdir;
  int min_nodes = 11;          // nodes(a) > 10
  int min_paralog_edges = 6;   // n(paralog_edges) > 5
  uint64_t seed = 42;
};

void load_nodes_enriched(const std::string& path, Graph& g,
                         bool require_marked);
void load_edges_tsv(const std::string& path, Graph& g,
                    bool both_ends_required);
std::string read_marked_sequence(const std::string& marked_dir,
                                 const std::string& marked_id);
