#!/usr/bin/env Rscript
# phyloseq2metacoder — phyloseq → metacoder::Taxmap via parse_phyloseq
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("metacoder", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  # parse_phyloseq looks up ranks_ref on the search path — must attach
  library(metacoder)
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

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path,
    target = NA_character_,
    batch = NA_character_,
    rarefaction_depth = NA_real_,
    abundances = NA_character_,
    rarefied = FALSE,
    batchadj = FALSE
  )
  if (inherits(obj, "phyloseq")) {
    meta$rarefied <- grepl("_rare(\\.rds)?$|phyloseq_rare_", basename(path))
    meta$batchadj <- grepl("_batchadj(\\.rds)?$", basename(path))
    return(list(ps = obj, meta = meta))
  }
  if (is.list(obj) && !is.null(obj$phyloseq) && inherits(obj$phyloseq, "phyloseq")) {
    meta$target <- obj$target %||% NA_character_
    meta$batch <- obj$batch %||% NA_character_
    meta$rarefaction_depth <- obj$rarefaction_depth %||% NA_real_
    meta$abundances <- obj$abundances %||% NA_character_
    meta$rarefied <- !is.null(obj$rarefaction_depth) ||
      grepl("_rare(\\.rds)?$", basename(path)) ||
      grepl("phyloseq_rare_", basename(path))
    meta$batchadj <- identical(obj$abundances, "relative") ||
      grepl("_batchadj(\\.rds)?$", basename(path)) ||
      isTRUE(obj$batch_adjusted)
    return(list(ps = obj$phyloseq, meta = meta))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

validate_ps <- function(ps, path) {
  if (is.null(otu_table(ps, errorIfNULL = FALSE))) {
    fail("phyloseq missing otu_table: ", path)
  }
  if (is.null(tax_table(ps, errorIfNULL = FALSE))) {
    fail("phyloseq missing tax_table (required for parse_phyloseq): ", path)
  }
  if (ntaxa(ps) < 1L) fail("phyloseq has zero taxa: ", path)
  if (nsamples(ps) < 1L) fail("phyloseq has zero samples: ", path)
  invisible(TRUE)
}

try_path <- function(p) {
  if (!file.exists(p)) return(NULL)
  loaded <- load_ps(p)
  validate_ps(loaded$ps, p)
  loaded
}

resolve_input_rds <- function(rds = NULL,
                              prefer_rare = TRUE,
                              prefer_batchadj = TRUE) {
  notes <- character(0)
  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  candidates_batch <- c(
    "test/code-review-phyloseq/grazing_phyloseq_batchadj.rds",
    "test/removebatch/grazing/phyloseq_batchadj.rds"
  )
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"

  if (!is.null(rds) && nzchar(rds)) {
    loaded <- try_path(rds)
    if (is.null(loaded)) fail("Unusable RDS: ", rds)
    loaded$notes <- notes
    return(loaded)
  }

  if (prefer_rare) {
    for (p in candidates_rare) {
      hit <- try_path(p)
      if (!is.null(hit)) {
        hit$meta$rarefied <- TRUE
        hit$notes <- c(notes, paste0("Using rarefied object: ", p))
        return(hit)
      }
    }
    rare_dir <- "test/rarefaction-analysis"
    if (dir.exists(rare_dir)) {
      hits <- sort(Sys.glob(file.path(rare_dir, "**", "phyloseq_rare_*.rds")), decreasing = TRUE)
      hits <- hits[!grepl("_plain\\.rds$", hits)]
      for (p in hits) {
        hit <- try_path(p)
        if (!is.null(hit)) {
          hit$meta$rarefied <- TRUE
          hit$notes <- c(notes, paste0("Using rarefied object: ", p))
          return(hit)
        }
      }
    }
    notes <- c(notes, "No rarefied object found")
  }

  if (prefer_batchadj) {
    for (p in candidates_batch) {
      hit <- try_path(p)
      if (!is.null(hit)) {
        hit$meta$batchadj <- TRUE
        hit$notes <- c(notes, paste0("Using batch-removed object: ", p))
        return(hit)
      }
    }
    batch_dir <- "test/removebatch"
    if (dir.exists(batch_dir)) {
      hits <- sort(Sys.glob(file.path(batch_dir, "**", "phyloseq_batchadj.rds")), decreasing = TRUE)
      for (p in hits) {
        hit <- try_path(p)
        if (!is.null(hit)) {
          hit$meta$batchadj <- TRUE
          hit$notes <- c(notes, paste0("Using batch-removed object: ", p))
          return(hit)
        }
      }
    }
    notes <- c(notes, "No batch-removed object found")
  }

  if (!file.exists(raw)) fail("No --rds and default grazing count RDS missing: ", raw)
  loaded <- try_path(raw)
  if (is.null(loaded)) fail("Raw RDS unusable: ", raw)
  loaded$notes <- c(notes, paste0("Falling back to raw counts: ", raw))
  loaded
}

parse_to_metacoder <- function(ps, calc_abund = FALSE) {
  obj <- tryCatch(
    metacoder::parse_phyloseq(ps),
    error = function(e) fail("parse_phyloseq failed: ", conditionMessage(e))
  )
  if (!inherits(obj, "Taxmap")) {
    fail("parse_phyloseq did not return Taxmap; got: ", paste(class(obj), collapse = ", "))
  }
  if (isTRUE(calc_abund)) {
    if (!"otu_table" %in% names(obj$data)) {
      fail("Taxmap missing data$otu_table; cannot calc_taxon_abund")
    }
    obj$data$taxon_counts <- metacoder::calc_taxon_abund(obj, data = "otu_table")
  }
  obj
}

run_convert <- function(rds = NULL, outdir,
                        prefer_rare = TRUE,
                        prefer_batchadj = TRUE,
                        calc_abund = FALSE) {
  ensure_dir(outdir)
  loaded <- resolve_input_rds(
    rds,
    prefer_rare = prefer_rare,
    prefer_batchadj = prefer_batchadj
  )
  ps <- loaded$ps
  meta <- loaded$meta
  notes <- loaded$notes %||% character(0)

  message("Input: ", meta$path,
          " (rarefied=", isTRUE(meta$rarefied),
          ", batchadj=", isTRUE(meta$batchadj),
          ", ntaxa=", ntaxa(ps),
          ", nsamples=", nsamples(ps), ")")

  obj <- parse_to_metacoder(ps, calc_abund = calc_abund)
  out_rds <- file.path(outdir, "metacoder.rds")
  saveRDS(obj, out_rds)

  n_tax <- length(metacoder::taxon_ids(obj))
  report <- list(
    skill = "phyloseq2metacoder",
    input_rds = meta$path,
    rarefied = isTRUE(meta$rarefied),
    batchadj = isTRUE(meta$batchadj),
    rarefaction_depth = meta$rarefaction_depth,
    abundances = meta$abundances,
    prefer_rare = prefer_rare,
    prefer_batchadj = prefer_batchadj,
    calc_abund = isTRUE(calc_abund),
    phyloseq_ntaxa = ntaxa(ps),
    phyloseq_nsamples = nsamples(ps),
    metacoder_ntaxa = n_tax,
    data_slots = names(obj$data),
    metacoder_rds = out_rds,
    notes = notes
  )
  write_json(report, file.path(outdir, "phyloseq2metacoder-report.json"))
  message("Wrote ", out_rds, " (", n_tax, " taxa)")
  report
}

self_test <- function() {
  setwd(project_root())
  out <- "test/phyloseq2metacoder/grazing-self-test"
  rep <- run_convert(
    rds = NULL,
    outdir = out,
    prefer_rare = TRUE,
    prefer_batchadj = TRUE,
    calc_abund = TRUE
  )
  if (!isTRUE(rep$rarefied)) stop("expected rarefied input when available")
  if (!file.exists(rep$metacoder_rds)) stop("missing metacoder.rds")
  obj <- readRDS(rep$metacoder_rds)
  if (!inherits(obj, "Taxmap")) stop("saved object is not Taxmap")
  if (!"taxon_counts" %in% names(obj$data)) stop("expected taxon_counts after calc_abund")
  if (rep$metacoder_ntaxa < 1L) stop("zero taxa in Taxmap")

  # batchadj path still converts when rare preference is off
  out2 <- "test/phyloseq2metacoder/grazing-batchadj-self-test"
  rep2 <- run_convert(
    rds = "test/code-review-phyloseq/grazing_phyloseq_batchadj.rds",
    outdir = out2,
    prefer_rare = FALSE,
    prefer_batchadj = TRUE,
    calc_abund = FALSE
  )
  if (!file.exists(rep2$metacoder_rds)) stop("missing batchadj metacoder.rds")
  if (!isTRUE(rep2$batchadj)) stop("expected batchadj flag for batchadj RDS")

  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/phyloseq2metacoder/run"
  prefer_rare <- !identical(tolower(as.character(args$prefer_rare %||% "true")), "false")
  prefer_batchadj <- !identical(tolower(as.character(args$prefer_batchadj %||% "true")), "false")
  calc_abund <- identical(tolower(as.character(args$calc_abund %||% "false")), "true")

  run_convert(
    rds = args$rds,
    outdir = outdir,
    prefer_rare = prefer_rare,
    prefer_batchadj = prefer_batchadj,
    calc_abund = calc_abund
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
