#!/usr/bin/env Rscript
# metabarcoding-import skill — 16S: metadata → all data → tree → complete phyloseq
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ape", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  library(ape)
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
source(file.path(root, ".cursor/skills/_shared/import/taxon_format.R"))

read_feature_table_tsv <- function(path) {
  dt <- utils::read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  idcol <- names(dt)[1]
  ids <- as.character(dt[[idcol]])
  if (anyDuplicated(ids)) fail("Duplicate feature IDs in ", path)
  dt[[idcol]] <- NULL
  mat <- as.matrix(data.frame(lapply(dt, as.numeric), check.names = FALSE,
                              row.names = ids))
  storage.mode(mat) <- "double"
  mat[is.na(mat)] <- 0
  mat
}

read_taxonomy_tsv <- function(path, taxa_ids) {
  dt <- utils::read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  id_col <- intersect(c("Feature ID", "FeatureID", "OTU", "#OTU ID", "ASV"), names(dt))
  tax_col <- intersect(c("Taxon", "taxonomy"), names(dt))
  if (length(id_col) && length(tax_col)) {
    ids <- as.character(dt[[id_col[[1]]]])
    parsed <- lapply(dt[[tax_col[[1]]]], parse_qiime_taxon_string)
    tax_mat <- do.call(rbind, parsed)
    rownames(tax_mat) <- ids
  } else if (all(RANK_COLS %in% names(dt)) || all(RANK_COLS_LC %in% names(dt))) {
    ranks <- if (all(RANK_COLS %in% names(dt))) RANK_COLS else RANK_COLS_LC
    id_col <- setdiff(names(dt), ranks)[1]
    tax_mat <- as.matrix(dt[, ranks, drop = FALSE])
    rownames(tax_mat) <- as.character(dt[[id_col]])
    colnames(tax_mat) <- RANK_COLS
  } else {
    fail("Unrecognized taxonomy table format: ", path)
  }
  missing <- setdiff(taxa_ids, rownames(tax_mat))
  if (length(missing)) {
    fill <- matrix(NA_character_, nrow = length(missing), ncol = ncol(tax_mat),
                   dimnames = list(missing, colnames(tax_mat)))
    tax_mat <- rbind(tax_mat, fill)
  }
  tax_mat <- tax_mat[taxa_ids, , drop = FALSE]
  colnames(tax_mat) <- RANK_COLS
  tax_mat[] <- apply(tax_mat, 2, normalize_unclassified_vec)
  tax_mat
}

import_16s_qza <- function(d) {
  if (!requireNamespace("qiime2R", quietly = TRUE)) {
    fail("qiime2R is required for .qza import")
  }
  if (is.na(d$features_qza)) fail("table.qza / features qza required for qza import")
  args <- list(features = d$features_qza, metadata = d$metadata)
  if (!is.na(d$taxonomy_qza)) args$taxonomy <- d$taxonomy_qza
  if (!is.na(d$tree_qza)) args$tree <- d$tree_qza
  message("Importing via qiime2R::qza_to_phyloseq with: ", paste(names(args), collapse = ", "))
  do.call(qiime2R::qza_to_phyloseq, args)
}

import_16s_plain <- function(d) {
  if (is.na(d$feature_table_tsv)) fail("feature-table.tsv required for plain 16S import")
  otu <- read_feature_table_tsv(d$feature_table_tsv)
  meta <- read_sample_metadata(d$metadata)
  common <- intersect(colnames(otu), rownames(meta))
  if (!length(common)) {
    fail(
      "Metadata not aligned with feature-table samples. Run skill @fix-metadata. ",
      "Feature samples: ", paste(head(colnames(otu), 5), collapse = ","),
      "; metadata: ", paste(head(rownames(meta), 5), collapse = ",")
    )
  }
  otu <- otu[, common, drop = FALSE]
  meta <- meta[common, , drop = FALSE]

  if (!is.na(d$taxonomy_tsv)) {
    tax <- read_taxonomy_tsv(d$taxonomy_tsv, rownames(otu))
  } else {
    tax <- matrix("Unclassified", nrow = nrow(otu), ncol = length(RANK_COLS),
                  dimnames = list(rownames(otu), RANK_COLS))
    message("No taxonomy file — filling Unclassified ranks")
  }

  keep <- rowSums(otu) > 0
  otu <- otu[keep, , drop = FALSE]
  tax <- tax[keep, , drop = FALSE]

  tax_df <- as.data.frame(tax, stringsAsFactors = FALSE)
  rownames(tax_df) <- rownames(tax)
  tax_df <- finalize_taxonomy_for_phyloseq(tax_df, otu_mat = otu)
  # tip_rank kept in tax_table for plot formatting
  tax <- as.matrix(tax_df)

  OTU <- phyloseq::otu_table(otu, taxa_are_rows = TRUE)
  TAX <- phyloseq::tax_table(tax)
  SAM <- phyloseq::sample_data(meta)
  phyloseq::phyloseq(OTU, TAX, SAM)
}

cleanup_16s_phyloseq <- function(ps) {
  ps <- phyloseq::prune_taxa(phyloseq::taxa_sums(ps) > 0, ps)
  tt <- as.data.frame(phyloseq::tax_table(ps), stringsAsFactors = FALSE)
  otu <- as(phyloseq::otu_table(ps), "matrix")
  if (!phyloseq::taxa_are_rows(ps)) otu <- t(otu)
  tt <- finalize_taxonomy_for_phyloseq(tt, otu_mat = otu)
  drop <- rep(FALSE, nrow(tt))
  for (cn in setdiff(names(tt), "tip_rank")) {
    drop <- drop | grepl("Chloroplast|Mitochondria", tt[[cn]], ignore.case = TRUE)
  }
  if (any(drop)) {
    message("Cleanup: dropping ", sum(drop), " Chloroplast/Mitochondria taxa")
    ps <- phyloseq::prune_taxa(!drop, ps)
    tt <- tt[!drop, , drop = FALSE]
  }
  phyloseq::tax_table(ps) <- phyloseq::tax_table(as.matrix(tt))
  ps
}

run_metabarcoding_import <- function(indir, outdir, metadata_path = NULL) {
  ensure_dir(outdir)
  message("metabarcoding-import: metadata → all available data → reconstruct tree")
  message("Analyzing 16S input directory: ", indir)
  d <- discover_16s(indir)
  if (!is.null(metadata_path) && nzchar(metadata_path)) d$metadata <- metadata_path
  write_json(d, file.path(outdir, "discovery.json"))
  # 1) metadata (required)
  validate_16s_discovery(d)
  # 2) all available data → phyloseq
  use_qza <- !is.na(d$features_qza) && nzchar(d$features_qza)
  ps <- if (use_qza) import_16s_qza(d) else import_16s_plain(d)
  ps <- cleanup_16s_phyloseq(ps)
  # 3) reconstruct tree if missing
  tree_try <- if (!is.na(d$tree_nwk)) d$tree_nwk else NA_character_
  if (is.null(phyloseq::phy_tree(ps, errorIfNULL = FALSE))) {
    ps <- require_tree(ps, tree_path = tree_try, lineage_taxids = NULL, outdir = outdir)
  }
  structure <- assert_complete_phyloseq(ps)

  rds_path <- file.path(outdir, "phyloseq.rds")
  saveRDS(ps, rds_path)
  summary <- list(
    indir = indir,
    outdir = outdir,
    method = if (use_qza) "qza_to_phyloseq" else "plain_tsv",
    n_taxa = structure$n_taxa,
    n_samples = structure$n_samples,
    has_tree = structure$tree_data,
    phyloseq_structure = structure,
    sample_names = structure$sample_names,
    discovery = d,
    phyloseq_rds = rds_path
  )
  write_json(summary, file.path(outdir, "metabarcoding-import-report.json"))
  message("Saved ", rds_path)
  invisible(summary)
}

self_test <- function() {
  setwd(project_root())
  system2("python3", c(".cursor/skills/mock-data/scripts/mock_data.py", "--out", "test", "--target", "16s", "--self-test"))
  s1 <- run_metabarcoding_import("test/16s", "test/metabarcoding-import/mock")
  if (!isTRUE(s1$phyloseq_structure$complete)) stop("mock phyloseq incomplete")
  grazing <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2025/grazing_article/data/qza"
  if (dir.exists(grazing) && file.exists(file.path(grazing, "table.qza"))) {
    s2 <- run_metabarcoding_import(grazing, "test/metabarcoding-import/grazing")
    if (!isTRUE(s2$phyloseq_structure$complete)) stop("grazing phyloseq incomplete")
    message("grazing real import OK: ", s2$n_taxa, " × ", s2$n_samples)
  }
  tmp <- file.path("test/metabarcoding-import", "missing-meta")
  ensure_dir(tmp)
  file.copy("test/16s/feature-table.tsv", tmp, overwrite = TRUE)
  file.copy("test/16s/taxonomy.tsv", tmp, overwrite = TRUE)
  script <- file.path(project_root(), ".cursor/skills/metabarcoding-import/scripts/metabarcoding_import.R")
  st <- system2("Rscript", c(script, "--indir", tmp, "--outdir", file.path(tmp, "out")),
                stdout = TRUE, stderr = TRUE)
  code <- attr(st, "status"); if (is.null(code)) code <- 0
  if (identical(as.integer(code), 0L)) stop("expected failure for missing metadata")
  message("missing-metadata error path OK")
  message("SELF-TEST OK")
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  indir <- args$indir %||% args$input %||% "test/16s"
  outdir <- args$outdir %||% "test/metabarcoding-import/run"
  run_metabarcoding_import(indir, outdir, metadata_path = args$metadata)
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
