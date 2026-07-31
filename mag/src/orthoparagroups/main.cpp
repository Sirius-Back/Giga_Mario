#include "extract.hpp"
#include "graph.hpp"
#include "io.hpp"
#include <iostream>
#include <string>

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --edges EDGE.tsv[.gz] --nodes NODES.tsv --marked-dir DIR --outdir DIR\n"
      << "  [--min-nodes 11] [--min-paralog-edges 6] [--seed 42]\n";
}

int main(int argc, char** argv) {
  ExtractOptions opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
      return argv[++i];
    };
    try {
      if (a == "--edges") opt.edges_path = need("--edges");
      else if (a == "--nodes") opt.nodes_path = need("--nodes");
      else if (a == "--marked-dir") opt.marked_dir = need("--marked-dir");
      else if (a == "--outdir") opt.outdir = need("--outdir");
      else if (a == "--min-nodes") opt.min_nodes = std::stoi(need("--min-nodes"));
      else if (a == "--min-paralog-edges")
        opt.min_paralog_edges = std::stoi(need("--min-paralog-edges"));
      else if (a == "--seed") opt.seed = std::stoull(need("--seed"));
      else if (a == "-h" || a == "--help") {
        usage(argv[0]);
        return 0;
      } else {
        std::cerr << "Unknown arg: " << a << "\n";
        usage(argv[0]);
        return 2;
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return 2;
    }
  }
  if (opt.edges_path.empty() || opt.nodes_path.empty() || opt.marked_dir.empty() ||
      opt.outdir.empty()) {
    usage(argv[0]);
    return 2;
  }

  try {
    Graph g;
    std::cout << "[load] nodes " << opt.nodes_path << "\n";
    load_nodes_enriched(opt.nodes_path, g, true);
    std::cout << "[load] nodes with MARKED: " << g.nodes.size() << "\n";
    std::cout << "[load] edges " << opt.edges_path << "\n";
    load_edges_tsv(opt.edges_path, g, true);
    g.finalize_adj();
    g.compute_paralog_degrees();
    std::cout << "[load] edges kept: " << g.edges.size() << "\n";
    extract_orthoparagroups(g, opt);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
