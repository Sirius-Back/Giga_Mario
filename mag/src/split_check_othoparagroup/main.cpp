#include "io.hpp"
#include "merge_summarize.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --split SPLIT.csv --hash-table HASH.tsv --outdir DIR\n"
      << "  [--model MODEL] [--run-id RUN_DIR_NAME]\n\n"
      << "If --outdir omitted: runs_unif/splits/{model}_{run-id}/\n"
      << "  inferred from --split path .../runs_unif/{model}/{run-id}/split.csv\n\n"
      << "Writes:\n"
      << "  othologs.csv   orthogroup|n_train|n_test|n_val|sd_random\n"
      << "  paralogs.csv   paragroup|n_train|n_test|n_val|sd_random\n"
      << "  summary.json   role fractions + counts\n";
}

/** Infer model + run_id from .../runs_unif/<model>/<run_id>/split.csv */
static bool infer_from_split(const fs::path& split, std::string& model, std::string& run_id) {
  fs::path p = fs::weakly_canonical(split);
  // .../run_id/split.csv
  if (p.filename() != "split.csv") return false;
  run_id = p.parent_path().filename().string();
  model = p.parent_path().parent_path().filename().string();
  return !model.empty() && !run_id.empty();
}

int main(int argc, char** argv) {
  std::string split_path, hash_path, outdir, model, run_id;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
      return argv[++i];
    };
    try {
      if (a == "--split") split_path = need("--split");
      else if (a == "--hash-table") hash_path = need("--hash-table");
      else if (a == "--outdir") outdir = need("--outdir");
      else if (a == "--model") model = need("--model");
      else if (a == "--run-id") run_id = need("--run-id");
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

  if (split_path.empty() || hash_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  try {
    if (outdir.empty()) {
      std::string inf_model, inf_run;
      if (!infer_from_split(split_path, inf_model, inf_run))
        throw std::runtime_error(
            "--outdir required when --split is not under runs_unif/{model}/{run}/split.csv");
      if (model.empty()) model = inf_model;
      if (run_id.empty()) run_id = inf_run;
      // Prefer repo-relative runs_unif/splits/...
      fs::path split_p = fs::path(split_path);
      fs::path runs_unif = split_p.parent_path().parent_path().parent_path();  // .../runs_unif
      if (runs_unif.filename() != "runs_unif") {
        // fallback: CWD-relative
        outdir = (fs::path("runs_unif") / "splits" / (model + "_" + run_id)).string();
      } else {
        outdir = (runs_unif / "splits" / (model + "_" + run_id)).string();
      }
    }

    std::cout << "[load] hash-table " << hash_path << "\n";
    auto hash_rows = load_hash_table(hash_path);
    std::cout << "[load] hash rows " << hash_rows.size() << "\n";
    std::cout << "[load] split " << split_path << "\n";
    auto split_rows = load_split_csv(split_path);
    std::cout << "[load] split rows " << split_rows.size() << "\n";

    auto fracs = compute_role_fractions(split_rows);
    std::cout << "[frac] train=" << fracs.p_train << " test=" << fracs.p_test
              << " val=" << fracs.p_val << " (exclude other=" << fracs.n_other << ")\n";

    sort_split_by_hash(split_rows);
    std::cout << "[sort] split by id_hash\n";

    std::vector<GroupCounts> orthologs, paralogs;
    merge_and_count(split_rows, hash_rows, orthologs, paralogs, fracs);
    std::cout << "[merge] orthogroups=" << orthologs.size()
              << " paragroups=" << paralogs.size() << "\n";

    fs::create_directories(outdir);
    const std::string otho = (fs::path(outdir) / "othologs.csv").string();
    const std::string para = (fs::path(outdir) / "paralogs.csv").string();
    write_summary_csv(otho, "orthogroup", orthologs);
    write_summary_csv(para, "paragroup", paralogs);

    // summary.json (minimal, no JSON lib)
    std::ofstream js((fs::path(outdir) / "summary.json").string());
    js << "{\n"
       << "  \"split\": \"" << split_path << "\",\n"
       << "  \"hash_table\": \"" << hash_path << "\",\n"
       << "  \"outdir\": \"" << outdir << "\",\n"
       << "  \"n_split\": " << split_rows.size() << ",\n"
       << "  \"n_hash\": " << hash_rows.size() << ",\n"
       << "  \"n_train\": " << fracs.n_train << ",\n"
       << "  \"n_test\": " << fracs.n_test << ",\n"
       << "  \"n_val\": " << fracs.n_val << ",\n"
       << "  \"n_other\": " << fracs.n_other << ",\n"
       << "  \"p_train\": " << fracs.p_train << ",\n"
       << "  \"p_test\": " << fracs.p_test << ",\n"
       << "  \"p_val\": " << fracs.p_val << ",\n"
       << "  \"n_orthogroups\": " << orthologs.size() << ",\n"
       << "  \"n_paragroups\": " << paralogs.size() << ",\n"
       << "  \"sd_random\": \"sqrt(sum_r (O_r - n*p_r)^2) with empirical p_r from split\"\n"
       << "}\n";

    std::cout << "[write] " << otho << "\n";
    std::cout << "[write] " << para << "\n";
    std::cout << "[done] " << outdir << "\n";
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
