#include "extract.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

struct WrittenSeq {
  int node = -1;
  std::string role;  // ortholog | paralog
  std::string seq;
};

std::vector<int> members_of_component(const std::vector<int>& comp_of, int cid) {
  std::vector<int> out;
  for (int i = 0; i < static_cast<int>(comp_of.size()); ++i)
    if (comp_of[i] == cid) out.push_back(i);
  return out;
}

int count_paralog_edges(const Graph& g, const std::vector<int>& nodes) {
  std::unordered_set<int> S(nodes.begin(), nodes.end());
  int n = 0;
  for (const auto& e : g.edges) {
    if (e.is_ortholog) continue;
    if (S.count(e.u) && S.count(e.v)) ++n;
  }
  return n;
}

int count_ortholog_edges(const Graph& g, const std::vector<int>& nodes) {
  std::unordered_set<int> S(nodes.begin(), nodes.end());
  int n = 0;
  for (const auto& e : g.edges) {
    if (!e.is_ortholog) continue;
    if (S.count(e.u) && S.count(e.v)) ++n;
  }
  return n;
}

// Orthogroup ids restricted to a node subset (reindexed locally via UF on ortho edges).
std::unordered_map<int, int> orthogroup_within(const Graph& g,
                                               const std::vector<int>& nodes) {
  std::unordered_set<int> S(nodes.begin(), nodes.end());
  std::unordered_map<int, int> parent;
  for (int u : nodes) parent[u] = u;
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
    if (S.count(e.u) && S.count(e.v)) unite(e.u, e.v);
  }
  std::unordered_map<int, int> remap;
  std::unordered_map<int, int> og;
  int next = 0;
  for (int u : nodes) {
    int r = find(find, u);
    auto it = remap.find(r);
    if (it == remap.end()) {
      remap.emplace(r, next);
      og[u] = next++;
    } else {
      og[u] = it->second;
    }
  }
  return og;
}

std::vector<int> select_one_per_species(const Graph& g,
                                        const std::vector<int>& group,
                                        std::mt19937_64& rng) {
  // Unique species; within species pick max paralog_degree; ties broken randomly.
  std::unordered_map<std::string, std::vector<int>> by_sp;
  for (int u : group) by_sp[g.nodes[u].species].push_back(u);
  std::vector<int> reps;
  for (auto& [sp, vec] : by_sp) {
    std::shuffle(vec.begin(), vec.end(), rng);
    std::stable_sort(vec.begin(), vec.end(), [&](int a, int b) {
      return g.nodes[a].paralog_degree > g.nodes[b].paralog_degree;
    });
    reps.push_back(vec.front());
  }
  // Order reps by descending paralog degree for stable FASTA order
  std::sort(reps.begin(), reps.end(), [&](int a, int b) {
    if (g.nodes[a].paralog_degree != g.nodes[b].paralog_degree)
      return g.nodes[a].paralog_degree > g.nodes[b].paralog_degree;
    return g.nodes[a].ensembl_gene < g.nodes[b].ensembl_gene;
  });
  return reps;
}

std::vector<int> orthogroup_members(const std::unordered_map<int, int>& og_of,
                                    int og_id,
                                    const std::vector<int>& universe) {
  std::vector<int> out;
  for (int u : universe)
    if (og_of.at(u) == og_id) out.push_back(u);
  return out;
}

}  // namespace

std::vector<ClusterStats> extract_orthoparagroups(Graph& g,
                                                  const ExtractOptions& opt) {
  fs::create_directories(opt.outdir);
  auto comp_of = g.connected_components(false);
  for (size_t i = 0; i < g.nodes.size(); ++i) g.nodes[i].component_id = comp_of[i];

  if (comp_of.empty()) return {};
  const int ncomp = 1 + *std::max_element(comp_of.begin(), comp_of.end());
  std::vector<ClusterStats> all_stats;
  std::mt19937_64 rng(opt.seed);

  std::ofstream table(opt.outdir + "/clusters.tsv");
  table << "fna_name\tcomponent_id\tn_nodes\tn_distinct_orthology_groups\t"
           "n_ortholog_edges\tn_paralog_edges\tn_written_orthologs\t"
           "n_written_paralogs\tnodes_per_species\n";

  for (int cid = 0; cid < ncomp; ++cid) {
    auto members = members_of_component(comp_of, cid);
    if (static_cast<int>(members.size()) < opt.min_nodes) continue;
    const int n_para_e = count_paralog_edges(g, members);
    if (n_para_e < opt.min_paralog_edges) continue;

    auto og_of = orthogroup_within(g, members);
    int n_og = 0;
    for (int u : members) n_og = std::max(n_og, og_of[u] + 1);

    // Shared mask across orthology-group seeds in this component.
    std::vector<char> masked(g.nodes.size(), 0);
    std::vector<WrittenSeq> written;

    // Process OGs in descending size / paralog connectivity.
    std::vector<std::pair<int, int>> og_order;  // score, og_id
    for (int og = 0; og < n_og; ++og) {
      auto mem = orthogroup_members(og_of, og, members);
      int score = 0;
      for (int u : mem) score += 1000 + g.nodes[u].paralog_degree;
      score += static_cast<int>(mem.size()) * 10000;
      og_order.push_back({score, og});
    }
    std::sort(og_order.begin(), og_order.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    for (auto [score, og] : og_order) {
      auto group = orthogroup_members(og_of, og, members);
      bool any_unmasked = false;
      for (int u : group)
        if (!masked[u]) {
          any_unmasked = true;
          break;
        }
      if (!any_unmasked) continue;

      // Select 1 per species by paralog degree (covers "more than #species" case).
      auto reps = select_one_per_species(g, group, rng);
      for (int u : reps) {
        if (masked[u]) continue;
        if (g.nodes[u].marked_id.empty()) continue;
        std::string seq =
            read_marked_sequence(opt.marked_dir, g.nodes[u].marked_id);
        if (seq.empty()) continue;
        written.push_back(WrittenSeq{u, "ortholog", std::move(seq)});
      }
      // Mask entire orthology group b
      for (int u : group) masked[u] = 1;

      // Expand: d = unmasked paralog of any node from b (original group)
      while (true) {
        std::vector<int> candidates;
        std::unordered_set<int> seen;
        for (int u : group) {
          for (int v : g.para_adj[u]) {
            if (masked[v]) continue;
            if (og_of.count(v) == 0) continue;  // outside component
            if (seen.insert(v).second) candidates.push_back(v);
          }
        }
        if (candidates.empty()) break;
        // Prefer high paralog degree; random among ties
        std::shuffle(candidates.begin(), candidates.end(), rng);
        std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
          return g.nodes[a].paralog_degree > g.nodes[b].paralog_degree;
        });
        int d = candidates.front();
        // NO orthologs between written paralogs: mask whole OG(d)
        int og_d = og_of[d];
        auto k = orthogroup_members(og_of, og_d, members);
        if (!g.nodes[d].marked_id.empty()) {
          std::string seq =
              read_marked_sequence(opt.marked_dir, g.nodes[d].marked_id);
          if (!seq.empty())
            written.push_back(WrittenSeq{d, "paralog", std::move(seq)});
        }
        for (int u : k) masked[u] = 1;
      }
    }

    if (written.empty()) continue;

    ClusterStats st;
    st.component_id = cid;
    st.n_nodes = static_cast<int>(members.size());
    st.n_ortholog_groups = n_og;
    st.n_ortholog_edges = count_ortholog_edges(g, members);
    st.n_paralog_edges = n_para_e;
    for (int u : members) st.nodes_per_species[g.nodes[u].species]++;
    for (const auto& w : written) {
      if (w.role == "ortholog") st.n_written_orthologs++;
      else st.n_written_paralogs++;
    }

    std::ostringstream name;
    name << "cluster_" << cid << ".fna";
    st.fna_name = name.str();
    const std::string out_path = opt.outdir + "/" + st.fna_name;
    std::ofstream out(out_path);
    for (const auto& w : written) {
      const auto& n = g.nodes[w.node];
      out << ">" << w.role
          << "|species=" << n.species
          << "|gene=" << n.ensembl_gene
          << "|symbol=" << n.gene_symbol
          << "|marked_id=" << n.marked_id
          << "|component=" << cid
          << "|paralog_degree=" << n.paralog_degree
          << "\n";
      // wrap 80
      for (size_t i = 0; i < w.seq.size(); i += 80)
        out << w.seq.substr(i, 80) << "\n";
    }

    // nodes_per_species compact
    std::ostringstream nps;
    bool first = true;
    for (const auto& [sp, cnt] : st.nodes_per_species) {
      if (!first) nps << ";";
      first = false;
      nps << sp << ":" << cnt;
    }
    table << st.fna_name << "\t" << st.component_id << "\t" << st.n_nodes << "\t"
          << st.n_ortholog_groups << "\t" << st.n_ortholog_edges << "\t"
          << st.n_paralog_edges << "\t" << st.n_written_orthologs << "\t"
          << st.n_written_paralogs << "\t" << nps.str() << "\n";
    all_stats.push_back(std::move(st));
  }
  std::cout << "[extract] wrote " << all_stats.size()
            << " cluster FASTA files to " << opt.outdir << "\n";
  return all_stats;
}
