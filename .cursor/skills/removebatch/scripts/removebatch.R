#!/usr/bin/env Rscript
# removebatch skill — MMUPHin::adjust_batch on phyloseq (BATCH | covariate = TARGET)
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("MMUPHin", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  stopifnot(requireNamespace("ape", quietly = TRUE))
  library(phyloseq)
  library(MMUPHin)
})

root <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) {
    script <- normalizePath(sub("^--file=", "", f[[1]]), mustWork = FALSE)
    return(dirname(dirname(dirname(dirname(dirname(script))))))
  }
  getwd()
})
source(file.path(root, ".cursor/skills/_shared/import/import_common.R"))

load_phyloseq_rds <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(target = NA_character_, batch = NA_character_, source_rds = path)
  if (inherits(obj, "phyloseq")) {
    return(list(ps = obj, meta = meta))
  }
  if (is.list(obj) && !is.null(obj$phyloseq) && inherits(obj$phyloseq, "phyloseq")) {
    meta$target <- obj$target %||% NA_character_
    meta$batch <- obj$batch %||% NA_character_
    return(list(ps = obj$phyloseq, meta = meta, wrapper = obj))
  }
  fail("RDS must be a phyloseq or a list with $phyloseq: ", path)
}

to_rel_abd_features_x_samples <- function(ps) {
  otu <- as(otu_table(ps), "matrix")
  if (!taxa_are_rows(ps)) otu <- t(otu)
  # drop empty taxa/samples
  otu <- otu[rowSums(otu) > 0, , drop = FALSE]
  otu <- otu[, colSums(otu) > 0, drop = FALSE]
  if (!nrow(otu) || !ncol(otu)) fail("Empty OTU table after dropping zeros")
  # relative abundance per sample (columns sum to 1) — ticks_metaanalyse path
  rel <- sweep(otu, 2, colSums(otu), "/")
  rel[is.na(rel)] <- 0
  rel
}

structure_slots <- function(ps) {
  otu <- otu_table(ps, errorIfNULL = FALSE)
  tax <- tax_table(ps, errorIfNULL = FALSE)
  sam <- sample_data(ps, errorIfNULL = FALSE)
  tre <- phy_tree(ps, errorIfNULL = FALSE)
  list(
    tax_table = !is.null(tax),
    otu_table = !is.null(otu),
    sam_data = !is.null(sam),
    tree_data = !is.null(tre),
    n_taxa = if (!is.null(otu)) ntaxa(ps) else 0L,
    n_samples = if (!is.null(sam)) nsamples(ps) else 0L,
    n_tree_tips = if (!is.null(tre)) length(tre$tip.label) else 0L,
    complete = !is.null(otu) && !is.null(tax) && !is.null(sam) && !is.null(tre)
  )
}

run_removebatch <- function(rds_path, outdir,
                            batch_var = NULL, covariate = NULL,
                            diagnostic_plot = NULL) {
  ensure_dir(outdir)
  loaded <- load_phyloseq_rds(rds_path)
  ps <- loaded$ps

  batch_var <- batch_var %||% loaded$meta$batch %||% "batch"
  covariate <- covariate %||% loaded$meta$target %||% "grazing"
  # If meta$target is the column name (e.g. "grazing"), use it; if missing column, fail
  sam <- as(sample_data(ps), "data.frame")
  if (!batch_var %in% names(sam)) {
    fail("BATCH variable '", batch_var, "' not in sample_data columns: ", paste(names(sam), collapse = ","))
  }
  if (!covariate %in% names(sam)) {
    fail(
      "Real/TARGET variable '", covariate, "' not in sample_data columns: ",
      paste(names(sam), collapse = ","),
      ". Pass --covariate <column>."
    )
  }

  sam[[batch_var]] <- as.factor(as.character(sam[[batch_var]]))
  sam[[covariate]] <- as.factor(as.character(sam[[covariate]]))
  n_batch <- nlevels(sam[[batch_var]])
  if (n_batch < 2) {
    fail("BATCH variable '", batch_var, "' has <2 levels — cannot adjust_batch")
  }

  message("removebatch: MMUPHin::adjust_batch(batch=", batch_var,
          ", covariates=", covariate, ")")
  rel <- to_rel_abd_features_x_samples(ps)
  # align metadata to columns
  common <- intersect(colnames(rel), rownames(sam))
  if (length(common) < 2) fail("Insufficient overlapping samples between OTU and sample_data")
  rel <- rel[, common, drop = FALSE]
  meta <- sam[common, , drop = FALSE]

  if (is.null(diagnostic_plot) || !nzchar(diagnostic_plot)) {
    diagnostic_plot <- file.path(outdir, "mmuphin_diagnostic.pdf")
  }
  ensure_dir(dirname(diagnostic_plot))

  fit <- tryCatch(
    adjust_batch(
      feature_abd = rel,
      batch = batch_var,
      covariates = covariate,
      data = meta,
      control = list(verbose = TRUE, diagnostic_plot = diagnostic_plot)
    ),
    error = function(e) fail("MMUPHin::adjust_batch failed: ", conditionMessage(e))
  )

  adj <- as.matrix(fit$feature_abd_adj)
  storage.mode(adj) <- "double"
  # rebuild phyloseq: keep tax/sam/tree for common taxa/samples
  taxa_keep <- intersect(rownames(adj), taxa_names(ps))
  samp_keep <- intersect(colnames(adj), sample_names(ps))
  if (length(taxa_keep) < 2 || length(samp_keep) < 2) {
    fail("Adjusted matrix does not overlap phyloseq taxa/samples")
  }
  adj <- adj[taxa_keep, samp_keep, drop = FALSE]
  ps2 <- prune_taxa(taxa_keep, ps)
  ps2 <- prune_samples(samp_keep, ps2)
  otu_table(ps2) <- otu_table(adj, taxa_are_rows = TRUE)

  # Locked: keep original tax/sam/tree — prune tips to adjusted taxa; never drop/rebuild
  tr <- phy_tree(ps2, errorIfNULL = FALSE)
  if (is.null(tr)) {
    fail(
      "Input phyloseq has no phy_tree; removebatch must keep tax/sam/tree. ",
      "Provide a complete phyloseq (or run taxonomy-tree before removebatch)."
    )
  }
  tips <- intersect(tr$tip.label, taxa_names(ps2))
  if (length(tips) < 2L) {
    fail(
      "Cannot keep phylogenetic tree after batch adjustment: only ",
      length(tips), " tip(s) overlap adjusted taxa (need ≥2). ",
      "Refusing to drop or reconstruct the tree (method-decision: keep tax/sam/tree)."
    )
  }
  phy_tree(ps2) <- ape::keep.tip(tr, tips)
  ps2 <- prune_taxa(tips, ps2)
  # ensure OTU still matches pruned tips (MMUPHin may have extra taxa without tips)
  otu_keep <- intersect(taxa_names(ps2), tips)
  if (length(otu_keep) < 2L) {
    fail("After tree tip prune, fewer than 2 taxa remain")
  }
  ps2 <- prune_taxa(otu_keep, ps2)

  st <- assert_complete_phyloseq(ps2)

  out_rds <- file.path(outdir, "phyloseq_batchadj.rds")
  saveRDS(
    list(
      phyloseq = ps2,
      target = covariate,
      batch = batch_var,
      source_rds = rds_path,
      method = "MMUPHin::adjust_batch",
      abundances = "relative",
      generated = as.character(Sys.Date())
    ),
    out_rds
  )
  saveRDS(ps2, file.path(outdir, "phyloseq_batchadj_plain.rds"))

  report <- list(
    input_rds = rds_path,
    outdir = outdir,
    batch_var = batch_var,
    covariate = covariate,
    batch_levels = levels(meta[[batch_var]]),
    covariate_levels = levels(meta[[covariate]]),
    n_batch_levels = n_batch,
    n_features_in = nrow(rel),
    n_samples_in = ncol(rel),
    diagnostic_plot = if (file.exists(diagnostic_plot)) diagnostic_plot else NULL,
    phyloseq_rds = out_rds,
    phyloseq_structure = st,
    method = "MMUPHin::adjust_batch",
    reference = "ixodes/Metananalysis/ticks_metaanalyse.Rmd"
  )
  write_json(report, file.path(outdir, "removebatch-report.json"))
  message("Saved ", out_rds, " [", st$n_taxa, " taxa × ", st$n_samples,
          " samples; complete=", st$complete, "]")
  if (!is.null(report$diagnostic_plot)) message("Diagnostic: ", report$diagnostic_plot)
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  rds <- "test/code-review-phyloseq/grazing_phyloseq.rds"
  if (!file.exists(rds)) fail("Missing grazing RDS for self-test: ", rds)
  out <- "test/removebatch/grazing-self-test"
  rep <- run_removebatch(rds, out, batch_var = "batch", covariate = "grazing")
  if (!isTRUE(rep$phyloseq_structure$complete)) stop("adjusted phyloseq incomplete")
  if (rep$n_batch_levels < 2) stop("expected ≥2 batch levels")
  ps <- readRDS(rep$phyloseq_rds)$phyloseq
  # relative abundances: columns ~1
  mat <- as(otu_table(ps), "matrix")
  if (!taxa_are_rows(ps)) mat <- t(mat)
  cs <- colSums(mat)
  if (any(abs(cs - 1) > 1e-4)) stop("adjusted OTU columns are not relative abundances")
  # tree kept (tip-pruned original), never rebuilt
  tr0 <- phy_tree(readRDS(rds)$phyloseq, errorIfNULL = FALSE)
  tr1 <- phy_tree(ps, errorIfNULL = FALSE)
  if (is.null(tr1)) stop("adjusted phyloseq missing phy_tree")
  if (!all(tr1$tip.label %in% tr0$tip.label)) stop("tree tips not subset of original tree")
  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  rds <- args$rds %||% args$input
  if (is.null(rds) || !nzchar(rds)) fail("Required: --rds path/to/phyloseq.rds")
  outdir <- args$outdir %||% "test/removebatch/run"
  run_removebatch(
    rds_path = rds,
    outdir = outdir,
    batch_var = args$batch_var,
    covariate = args$covariate %||% args$target_var,
    diagnostic_plot = args$diagnostic_plot
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
