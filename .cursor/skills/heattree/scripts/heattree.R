#!/usr/bin/env Rscript
# heattree — metacoder heat_tree; default merge taxa at Family
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("metacoder", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  # parse_phyloseq / filter_taxa need metacoder on the search path
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

DEFAULT_RANK <- "Family"
DEFAULT_MIN_LEAF <- 10

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path, rarefied = FALSE, batchadj = FALSE,
    rarefaction_depth = NA_real_, abundances = NA_character_
  )
  if (inherits(obj, "phyloseq")) {
    meta$rarefied <- grepl("_rare(\\.rds)?$|phyloseq_rare_", basename(path))
    meta$batchadj <- grepl("_batchadj(\\.rds)?$", basename(path))
    return(list(ps = obj, meta = meta))
  }
  if (is.list(obj) && !is.null(obj$phyloseq) && inherits(obj$phyloseq, "phyloseq")) {
    meta$rarefaction_depth <- obj$rarefaction_depth %||% NA_real_
    meta$abundances <- obj$abundances %||% NA_character_
    meta$rarefied <- !is.null(obj$rarefaction_depth) ||
      grepl("_rare|phyloseq_rare_", basename(path))
    meta$batchadj <- identical(obj$abundances, "relative") ||
      grepl("_batchadj", basename(path))
    return(list(ps = obj$phyloseq, meta = meta))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

validate_ps <- function(ps, path) {
  if (is.null(otu_table(ps, errorIfNULL = FALSE))) fail("missing otu_table: ", path)
  if (is.null(tax_table(ps, errorIfNULL = FALSE))) fail("missing tax_table: ", path)
  if (ntaxa(ps) < 1L || nsamples(ps) < 1L) fail("empty phyloseq: ", path)
  invisible(TRUE)
}

load_taxmap <- function(path) {
  if (!file.exists(path)) fail("Taxmap RDS not found: ", path)
  obj <- readRDS(path)
  if (!inherits(obj, "Taxmap")) fail("Not a Taxmap: ", path)
  list(obj = obj, path = path, source = "metacoder")
}

ps_to_taxmap <- function(ps) {
  obj <- tryCatch(
    parse_phyloseq(ps),
    error = function(e) fail("parse_phyloseq failed: ", conditionMessage(e))
  )
  if (!inherits(obj, "Taxmap")) fail("parse_phyloseq did not return Taxmap")
  obj
}

resolve_taxmap <- function(metacoder = NULL, rds = NULL,
                           prefer_rare = TRUE, prefer_batchadj = TRUE) {
  notes <- character(0)

  if (!is.null(metacoder) && nzchar(metacoder)) {
    hit <- load_taxmap(metacoder)
    hit$notes <- notes
    hit$ps_meta <- list(rarefied = NA, batchadj = NA)
    return(hit)
  }

  candidates_mc <- c(
    "test/phyloseq2metacoder/grazing/metacoder.rds",
    "test/phyloseq2metacoder/grazing-self-test/metacoder.rds"
  )
  if (is.null(rds) || !nzchar(rds)) {
    for (p in candidates_mc) {
      if (file.exists(p)) {
        hit <- load_taxmap(p)
        hit$notes <- c(notes, paste0("Using Taxmap: ", p))
        hit$ps_meta <- list(rarefied = NA, batchadj = NA)
        return(hit)
      }
    }
    mc_dir <- "test/phyloseq2metacoder"
    if (dir.exists(mc_dir)) {
      hits <- sort(Sys.glob(file.path(mc_dir, "**", "metacoder.rds")), decreasing = TRUE)
      for (p in hits) {
        hit <- load_taxmap(p)
        hit$notes <- c(notes, paste0("Using Taxmap: ", p))
        hit$ps_meta <- list(rarefied = NA, batchadj = NA)
        return(hit)
      }
    }
    notes <- c(notes, "No phyloseq2metacoder Taxmap found — falling back to phyloseq")
  }

  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  candidates_batch <- c(
    "test/code-review-phyloseq/grazing_phyloseq_batchadj.rds",
    "test/removebatch/grazing/phyloseq_batchadj.rds"
  )
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"

  pick_ps <- function(path) {
    if (!file.exists(path)) return(NULL)
    loaded <- load_ps(path)
    validate_ps(loaded$ps, path)
    loaded
  }

  loaded <- NULL
  if (!is.null(rds) && nzchar(rds)) {
    loaded <- pick_ps(rds)
    if (is.null(loaded)) fail("Unusable phyloseq RDS: ", rds)
  } else {
    if (prefer_rare) {
      for (p in candidates_rare) {
        loaded <- pick_ps(p)
        if (!is.null(loaded)) {
          notes <- c(notes, paste0("Using rarefied phyloseq: ", p))
          break
        }
      }
    }
    if (is.null(loaded) && prefer_batchadj) {
      for (p in candidates_batch) {
        loaded <- pick_ps(p)
        if (!is.null(loaded)) {
          notes <- c(notes, paste0("Using batch-removed phyloseq: ", p))
          break
        }
      }
    }
    if (is.null(loaded)) {
      loaded <- pick_ps(raw)
      if (is.null(loaded)) fail("No Taxmap or phyloseq input found")
      notes <- c(notes, paste0("Using raw phyloseq: ", raw))
    }
  }

  obj <- ps_to_taxmap(loaded$ps)
  list(
    obj = obj,
    path = loaded$meta$path,
    source = "phyloseq",
    notes = notes,
    ps_meta = loaded$meta
  )
}

ensure_taxon_counts <- function(obj) {
  if (!"otu_table" %in% names(obj$data)) {
    fail("Taxmap missing data$otu_table")
  }
  if (!"taxon_counts" %in% names(obj$data)) {
    obj$data$taxon_counts <- calc_taxon_abund(obj, data = "otu_table")
  }
  tc <- obj$data$taxon_counts
  if (!"taxon_id" %in% names(tc)) {
    fail("taxon_counts missing taxon_id column")
  }
  num_cols <- names(tc)[vapply(tc, is.numeric, logical(1))]
  num_cols <- setdiff(num_cols, c("total", "leaf"))
  if (!length(num_cols)) fail("No numeric abundance columns in taxon_counts")
  mat <- as.matrix(tc[, num_cols, drop = FALSE])
  obj$data$taxon_counts$total <- round(rowMeans(mat), 1)
  obj$data$taxon_counts$leaf <- rowSums(mat)
  obj
}

match_rank <- function(obj, rank) {
  ranks <- unique(as.character(taxon_ranks(obj)))
  ranks <- ranks[!is.na(ranks) & nzchar(ranks)]
  hit <- ranks[tolower(ranks) == tolower(rank)]
  if (!length(hit)) {
    fail(
      "Rank '", rank, "' not found in Taxmap. Available: ",
      paste(ranks, collapse = ", ")
    )
  }
  hit[[1]]
}

merge_to_rank <- function(obj, rank, min_leaf = DEFAULT_MIN_LEAF, subset = NULL) {
  n_before <- length(taxon_ids(obj))
  rank_matched <- match_rank(obj, rank)

  if (!is.null(min_leaf) && is.finite(min_leaf) && min_leaf > 0) {
    obj <- filter_taxa(obj, leaf >= min_leaf)
  }

  if (!is.null(subset) && length(subset) && any(nzchar(subset))) {
    subset <- subset[nzchar(subset)]
    obj <- filter_taxa(obj, taxon_names %in% subset, subtaxa = TRUE)
  }

  # Merge deeper taxa into rank: keep rank + ancestors; reassign OTU obs
  obj <- filter_taxa(obj, taxon_ranks == rank_matched, supertaxa = TRUE)
  list(obj = obj, rank = rank_matched, n_before = n_before, n_after = length(taxon_ids(obj)))
}

save_heat_tree <- function(obj, out_prefix, width = 10, height = 10) {
  pdf_path <- paste0(out_prefix, ".pdf")
  png_path <- paste0(out_prefix, ".png")

  ht <- heat_tree(
    obj,
    node_label = taxon_names,
    node_size = n_obs,
    node_color = total,
    node_size_axis_label = "ASV count",
    node_color_axis_label = "Mean reads",
    layout = "davidson-harel",
    initial_layout = "reingold-tilford",
    output_file = pdf_path
  )

  if (isTRUE(capabilities("cairo"))) {
    grDevices::png(png_path, width = width, height = height, units = "in", res = 300, type = "cairo")
  } else if (requireNamespace("ragg", quietly = TRUE)) {
    ragg::agg_png(png_path, width = width, height = height, units = "in", res = 300)
  } else {
    grDevices::png(png_path, width = width * 300, height = height * 300, res = 300)
  }
  print(ht)
  grDevices::dev.off()

  list(pdf = pdf_path, png = png_path, plot = ht)
}

run_heattree <- function(metacoder = NULL, rds = NULL, outdir,
                         rank = DEFAULT_RANK,
                         min_leaf = DEFAULT_MIN_LEAF,
                         subset = NULL,
                         prefer_rare = TRUE,
                         prefer_batchadj = TRUE) {
  ensure_dir(outdir)
  resolved <- resolve_taxmap(
    metacoder = metacoder,
    rds = rds,
    prefer_rare = prefer_rare,
    prefer_batchadj = prefer_batchadj
  )
  obj <- ensure_taxon_counts(resolved$obj)
  merged <- merge_to_rank(obj, rank = rank, min_leaf = min_leaf, subset = subset)
  obj_f <- merged$obj

  if (merged$n_after < 2L) {
    fail("Fewer than 2 taxa after Family/rank merge; cannot draw heat tree")
  }

  message(
    "Heat tree: source=", resolved$source,
    " input=", resolved$path,
    " rank=", merged$rank,
    " taxa ", merged$n_before, " → ", merged$n_after
  )

  taxmap_path <- file.path(outdir, "heattree_taxmap.rds")
  saveRDS(obj_f, taxmap_path)

  figs <- save_heat_tree(obj_f, file.path(outdir, "heattree"))

  report <- list(
    skill = "heattree",
    input = resolved$path,
    input_source = resolved$source,
    rank = merged$rank,
    min_leaf = min_leaf,
    subset = subset %||% character(0),
    n_taxa_before = merged$n_before,
    n_taxa_after = merged$n_after,
    rarefied = resolved$ps_meta$rarefied %||% NA,
    batchadj = resolved$ps_meta$batchadj %||% NA,
    figures = list(pdf = figs$pdf, png = figs$png),
    taxmap_rds = taxmap_path,
    notes = resolved$notes %||% character(0)
  )
  write_json(report, file.path(outdir, "heattree-report.json"))
  message("Wrote ", figs$pdf, " / ", figs$png)
  report
}

self_test <- function() {
  setwd(project_root())
  out <- "test/heattree/grazing-self-test"
  rep <- run_heattree(
    metacoder = NULL,
    rds = NULL,
    outdir = out,
    rank = "Family",
    min_leaf = 10
  )
  if (!identical(rep$rank, "Family")) stop("expected Family merge")
  if (!file.exists(rep$figures$pdf)) stop("missing heattree.pdf")
  if (!file.exists(rep$figures$png)) stop("missing heattree.png")
  if (file.info(rep$figures$png)$size < 1000) stop("PNG too small")
  if (rep$n_taxa_after < 2L) stop("too few taxa after merge")

  # rank override still works
  out2 <- "test/heattree/grazing-class-self-test"
  rep2 <- run_heattree(
    metacoder = "test/phyloseq2metacoder/grazing/metacoder.rds",
    outdir = out2,
    rank = "Class",
    min_leaf = 10
  )
  if (!identical(rep2$rank, "Class")) stop("expected Class")
  if (!file.exists(rep2$figures$pdf)) stop("missing Class pdf")

  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/heattree/run"
  rank <- args$rank %||% DEFAULT_RANK
  min_leaf <- as.numeric(args$min_leaf %||% DEFAULT_MIN_LEAF)
  prefer_rare <- !identical(tolower(as.character(args$prefer_rare %||% "true")), "false")
  prefer_batchadj <- !identical(tolower(as.character(args$prefer_batchadj %||% "true")), "false")
  subset <- if (!is.null(args$subset) && nzchar(args$subset)) {
    trimws(strsplit(args$subset, ",", fixed = TRUE)[[1]])
  } else {
    NULL
  }

  run_heattree(
    metacoder = args$metacoder,
    rds = args$rds,
    outdir = outdir,
    rank = rank,
    min_leaf = min_leaf,
    subset = subset,
    prefer_rare = prefer_rare,
    prefer_batchadj = prefer_batchadj
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
