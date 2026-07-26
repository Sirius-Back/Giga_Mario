#!/usr/bin/env Rscript
# metagenomic-import skill — WGS: metadata → all Bracken data → tree → complete phyloseq
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ape", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  library(ape)
  library(data.table)
})

root <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) {
    script <- normalizePath(sub("^--file=", "", f[[1]]), mustWork = FALSE)
    # .cursor/skills/<skill>/scripts/*.R → five dirname levels to project root
    return(dirname(dirname(dirname(dirname(dirname(script))))))
  }
  getwd()
})
source(file.path(root, ".cursor/skills/_shared/import/import_common.R"))
source(file.path(root, ".cursor/skills/_shared/import/bracken_parse.R"))
source(file.path(root, ".cursor/skills/_shared/import/taxon_format.R"))

build_lineage_from_names <- function(tax_ids, names_vec, ranks = RANK_COLS_LC) {
  tax <- data.frame(
    matrix(NA_character_, nrow = length(tax_ids), ncol = length(ranks),
           dimnames = list(paste0("tax_", tax_ids), ranks)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(tax_ids)) {
    nm <- names_vec[[i]]
    tax$genus[[i]] <- nm
    tax$species[[i]] <- nm
  }
  tax
}

run_metagenomic_import <- function(indir, outdir, metadata_path = NULL, max_files = NULL) {
  ensure_dir(outdir)
  message("metagenomic-import: metadata → all available data → reconstruct tree")
  message("Analyzing WGS/Bracken input directory: ", indir)
  d <- discover_wgs(indir)
  if (!is.null(metadata_path) && nzchar(metadata_path)) d$metadata <- metadata_path
  write_json(d, file.path(outdir, "discovery.json"))
  # 1) metadata (required)
  validate_wgs_discovery(d)

  # 2) all available abundance reports
  files <- if (length(d$bracken_genus)) d$bracken_genus
  else if (length(d$bracken_species_report)) d$bracken_species_report
  else d$bracken_genus_report
  if (!is.null(max_files)) files <- head(files, as.integer(max_files))
  message("Parsing ", length(files), " abundance files (all available",
          if (!is.null(max_files)) paste0(", capped at ", max_files) else "", ")…")
  long_dt <- data.table::rbindlist(lapply(files, detect_and_parse_one), use.names = TRUE, fill = TRUE)
  parsed <- cleanup_host_rows(build_wide_matrix(long_dt))

  meta <- read_sample_metadata(d$metadata)
  samples <- parsed$samples
  if (!is.na(d$sample_map) && file.exists(d$sample_map)) {
    sm <- utils::read.csv(d$sample_map, stringsAsFactors = FALSE)
    if (all(c("bracken_id", "sample_id") %in% names(sm))) {
      idx <- match(samples, sm$bracken_id)
      map_samples <- ifelse(!is.na(idx), sm$sample_id[idx], samples)
      colnames(parsed$counts) <- map_samples
      samples <- map_samples
    }
  }

  common <- intersect(samples, meta$sampleID)
  if (!length(common)) {
    fail(
      "Metadata not aligned with Bracken samples. Run skill @fix-metadata. ",
      "Bracken: ", paste(head(samples, 5), collapse = ","),
      "; metadata sampleID: ", paste(head(meta$sampleID, 5), collapse = ",")
    )
  }
  counts <- parsed$counts[, common, drop = FALSE]
  meta <- meta[common, , drop = FALSE]

  tax <- build_lineage_from_names(parsed$taxonomy_id, parsed$name)
  tax <- tax[rownames(counts), , drop = FALSE]
  tax <- finalize_taxonomy_for_phyloseq(tax, otu_mat = counts)

  OTU <- phyloseq::otu_table(counts, taxa_are_rows = TRUE)
  TAX <- phyloseq::tax_table(as.matrix(tax))
  SAM <- phyloseq::sample_data(meta)
  ps <- phyloseq::phyloseq(OTU, TAX, SAM)

  # 3) reconstruct tree
  lineage_ids <- parsed$taxonomy_id[match(phyloseq::taxa_names(ps), paste0("tax_", parsed$taxonomy_id))]
  ps <- require_tree(ps, tree_path = NA_character_, lineage_taxids = lineage_ids, outdir = outdir)
  structure <- assert_complete_phyloseq(ps)

  saveRDS(parsed, file.path(outdir, "bracken_parsed.rds"))
  rds_path <- file.path(outdir, "phyloseq.rds")
  saveRDS(ps, rds_path)
  summary <- list(
    indir = indir,
    outdir = outdir,
    n_files = length(files),
    n_taxa = structure$n_taxa,
    n_samples = structure$n_samples,
    has_tree = structure$tree_data,
    phyloseq_structure = structure,
    sample_names = structure$sample_names,
    phyloseq_rds = rds_path
  )
  write_json(summary, file.path(outdir, "metagenomic-import-report.json"))
  message("Saved ", rds_path)
  invisible(summary)
}

self_test <- function() {
  setwd(project_root())
  system2("python3", c(".cursor/skills/mock-data/scripts/mock_data.py", "--out", "test", "--target", "wgs", "--self-test"))
  s1 <- run_metagenomic_import("test/wgs", "test/metagenomic-import/mock")
  if (!isTRUE(s1$phyloseq_structure$complete)) stop("mock phyloseq incomplete")

  honey_k2 <- "/mnt/tank/scratch/dsmutin/bee/honey/data/annotations/k2"
  sra <- "/mnt/tank/scratch/dsmutin/bee/honey/data/legends/sra.csv"
  if (dir.exists(honey_k2) && file.exists(sra)) {
    tmp <- "test/metagenomic-import/honey-subset"
    ensure_dir(tmp)
    files <- file.path(honey_k2, c("ERR2592241.nt.G.bracken", "ERR2592240.nt.G.bracken"))
    files <- files[file.exists(files)]
    file.copy(files, tmp, overwrite = TRUE)
    md <- utils::read.csv(sra, stringsAsFactors = FALSE, check.names = FALSE)
    md$sampleID <- md$Run
    runs <- sub("\\.nt\\.G\\.bracken$", "", basename(files))
    md <- md[md$Run %in% runs, , drop = FALSE]
    utils::write.csv(md, file.path(tmp, "sample-metadata.csv"), row.names = FALSE)
    s2 <- run_metagenomic_import(tmp, file.path(tmp, "out"), max_files = 2)
    if (!isTRUE(s2$phyloseq_structure$complete)) stop("honey phyloseq incomplete")
    message("honey real subset OK: ", s2$n_taxa, " × ", s2$n_samples)
  }

  k2 <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/Metagenome/final/k2"
  smap <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/bracken_sample_map.csv"
  if (dir.exists(k2) && file.exists(smap)) {
    tmp <- "test/metagenomic-import/kristina-subset"
    ensure_dir(tmp)
    rep <- file.path(k2, "32212.nt.bracken.S.report")
    if (file.exists(rep)) {
      file.copy(rep, tmp, overwrite = TRUE)
      sm <- utils::read.csv(smap, stringsAsFactors = FALSE)
      sm <- sm[sm$bracken_id == "32212", , drop = FALSE]
      if (nrow(sm)) {
        md <- data.frame(sampleID = sm$sample_id, bracken_id = sm$bracken_id, stringsAsFactors = FALSE)
        utils::write.csv(md, file.path(tmp, "sample-metadata.csv"), row.names = FALSE)
        file.copy(smap, file.path(tmp, "bracken_sample_map.csv"), overwrite = TRUE)
        s3 <- run_metagenomic_import(tmp, file.path(tmp, "out"))
        if (!isTRUE(s3$phyloseq_structure$complete)) stop("kristina phyloseq incomplete")
        message("kristina real subset OK: ", s3$n_taxa, " × ", s3$n_samples)
      }
    }
  }

  tmp <- "test/metagenomic-import/missing-bracken"
  ensure_dir(tmp)
  utils::write.csv(data.frame(sampleID = "x"), file.path(tmp, "sample-metadata.csv"), row.names = FALSE)
  script <- file.path(project_root(), ".cursor/skills/metagenomic-import/scripts/metagenomic_import.R")
  st <- system2("Rscript", c(script, "--indir", tmp, "--outdir", file.path(tmp, "out")),
                stdout = TRUE, stderr = TRUE)
  if (is.null(attr(st, "status"))) attr(st, "status") <- 0
  if (identical(as.integer(attr(st, "status")), 0L)) stop("expected failure for missing bracken")
  message("missing-bracken error path OK")
  message("SELF-TEST OK")
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  indir <- args$indir %||% args$input %||% "test/wgs"
  outdir <- args$outdir %||% "test/metagenomic-import/run"
  max_files <- if (!is.null(args$max_files)) as.integer(args$max_files) else NULL
  run_metagenomic_import(indir, outdir, metadata_path = args$metadata, max_files = max_files)
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
