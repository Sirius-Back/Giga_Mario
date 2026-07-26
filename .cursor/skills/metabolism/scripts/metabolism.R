#!/usr/bin/env Rscript
# metabolism — import metabolic gene tables (no GO) + top-N pheatmap
suppressPackageStartupMessages({
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  stopifnot(requireNamespace("pheatmap", quietly = TRUE))
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

DEFAULT_TYPES <- c("product")
DEFAULT_TOP_N <- 200L
DEFAULT_SCALE <- "row"
LFC_LOW <- "#1B9E77"
LFC_MID <- "white"
LFC_HIGH <- "#D81B60"

METABOLISM_TYPES <- c("product", "ec", "ko")
GO_TYPES <- c("go", "GO", "ontology")

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

read_table_auto <- function(path) {
  if (!file.exists(path)) fail("File not found: ", path)
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("csv")) {
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else if (ext %in% c("tsv", "txt")) {
    utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    # try CSV then TSV
    tryCatch(
      utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
      error = function(e) {
        utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
      }
    )
  }
}

normalize_long <- function(df) {
  nms <- tolower(names(df))
  names(df) <- nms
  need <- c("sample", "function_type", "function_id", "count")
  if (!all(need %in% names(df))) {
    fail(
      "Long table missing columns ", paste(setdiff(need, names(df)), collapse = ", "),
      ". Expected: sample, function_type, function_id, count"
    )
  }
  df$sample <- as.character(df$sample)
  df$function_type <- tolower(as.character(df$function_type))
  df$function_id <- as.character(df$function_id)
  df$count <- as.numeric(df$count)
  if (any(is.na(df$count)) || any(df$count < 0, na.rm = TRUE)) {
    fail("Invalid count values in long table (NA or negative)")
  }
  if (!"function_label" %in% names(df)) {
    df$function_label <- paste0(df$function_type, "|", df$function_id)
  } else {
    df$function_label <- as.character(df$function_label)
  }
  df[, c("sample", "function_type", "function_id", "count", "function_label")]
}

wide_to_long <- function(mat, function_type = "product") {
  if (!is.matrix(mat) && !is.data.frame(mat)) fail("Matrix input must be matrix/data.frame")
  m <- as.matrix(mat)
  storage.mode(m) <- "numeric"
  if (is.null(rownames(m)) || any(!nzchar(rownames(m)))) {
    fail("Wide matrix must have gene rownames")
  }
  if (is.null(colnames(m)) || any(!nzchar(colnames(m)))) {
    fail("Wide matrix must have sample colnames")
  }
  genes <- rownames(m)
  samples <- colnames(m)
  rows <- vector("list", length(genes) * length(samples))
  k <- 1L
  for (i in seq_along(genes)) {
    for (j in seq_along(samples)) {
      v <- m[i, j]
      if (!is.na(v) && v != 0) {
        fid <- genes[[i]]
        if (!grepl(paste0("^", function_type, ":"), fid)) {
          fid <- paste0(function_type, ":", fid)
        }
        rows[[k]] <- data.frame(
          sample = samples[[j]],
          function_type = function_type,
          function_id = fid,
          count = as.numeric(v),
          function_label = paste0(function_type, "|", fid),
          stringsAsFactors = FALSE
        )
        k <- k + 1L
      }
    }
  }
  if (k == 1L) fail("Wide matrix has no non-zero entries")
  do.call(rbind, rows[seq_len(k - 1L)])
}

load_rds_matrix <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  if (is.matrix(obj) || is.data.frame(obj)) {
    return(list(counts = as.matrix(obj), source = "matrix_rds"))
  }
  if (is.list(obj) && !is.null(obj$counts)) {
    return(list(counts = as.matrix(obj$counts), source = "bakta_list_rds"))
  }
  fail("RDS must be a matrix or list with $counts: ", path)
}

# ---------------------------------------------------------------------------
# Bakta GFF3 (optional; sources Kristina bakta_gff3.R)
# ---------------------------------------------------------------------------

resolve_bakta_gff3_r <- function() {
  cands <- c(
    file.path(root, "..", "2026", "Kristina", "R", "bakta_gff3.R"),
    "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/R/bakta_gff3.R"
  )
  for (p in cands) {
    p <- normalizePath(p, mustWork = FALSE)
    if (file.exists(p)) return(p)
  }
  NULL
}

import_gff3_dir <- function(gff3_dir, sample_map = NULL) {
  bakta_r <- resolve_bakta_gff3_r()
  if (is.null(bakta_r)) {
    fail(
      "Cannot import GFF3: Kristina R/bakta_gff3.R not found. ",
      "Provide --long or --matrix instead."
    )
  }
  if (!dir.exists(gff3_dir)) fail("GFF3 directory not found: ", gff3_dir)
  source(bakta_r, local = TRUE)
  map_path <- sample_map
  if (is.null(map_path) || !nzchar(map_path)) {
    map_cands <- c(
      file.path(dirname(dirname(bakta_r)), "data/processed/bracken_sample_map.csv"),
      file.path(gff3_dir, "..", "..", "..", "processed", "bracken_sample_map.csv")
    )
    for (m in map_cands) {
      m <- normalizePath(m, mustWork = FALSE)
      if (file.exists(m)) {
        map_path <- m
        break
      }
    }
  }
  if (is.null(map_path) || !file.exists(map_path)) {
    # synthesize identity map from discovered ids
    files_tbl <- find_bakta_gff3_files(gff3_dir)
    if (!nrow(files_tbl)) fail("No GFF3 under ", gff3_dir)
    tmp <- tempfile(fileext = ".csv")
    utils::write.csv(
      data.frame(bracken_id = files_tbl$bakta_id, sample_id = files_tbl$bakta_id),
      tmp, row.names = FALSE
    )
    map_path <- tmp
  }
  res <- import_bakta_gff3_dir(
    bakta_root = gff3_dir,
    sample_map_path = map_path,
    prefer_rtracklayer = requireNamespace("rtracklayer", quietly = TRUE)
  )
  list(long = normalize_long(as.data.frame(res$long)), source = "gff3", bakta_r = bakta_r)
}

# ---------------------------------------------------------------------------
# Resolve input
# ---------------------------------------------------------------------------

default_long_candidates <- function() {
  c(
    file.path(root, ".cursor/skills/metabolism/fixtures/bakta_function_long.csv"),
    "test/metabolism/fixtures/bakta_function_long.csv",
    "test/metabolism/kristina/bakta_function_long.csv",
    file.path(root, "..", "2026", "Kristina", "data/processed/bakta_function_long.csv"),
    "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/bakta_function_long.csv"
  )
}

resolve_long <- function(long = NULL, matrix = NULL, rds = NULL, gff3_dir = NULL,
                         sample_map = NULL, matrix_type = "product") {
  notes <- character(0)

  if (!is.null(long) && nzchar(long)) {
    df <- normalize_long(read_table_auto(long))
    return(list(long = df, source = "long", path = long, notes = notes))
  }

  if (!is.null(rds) && nzchar(rds)) {
    hit <- load_rds_matrix(rds)
    # Bakta matrices use rownames like "product|product:..." — prefer long rebuild via labels
    rn <- rownames(hit$counts)
    if (all(grepl("\\|", rn))) {
      # convert labeled matrix → long
      m <- hit$counts
      rows <- list()
      k <- 1L
      for (i in seq_len(nrow(m))) {
        parts <- strsplit(rn[[i]], "|", fixed = TRUE)[[1]]
        ftype <- tolower(parts[[1]])
        fid <- if (length(parts) >= 2L) paste(parts[-1], collapse = "|") else rn[[i]]
        for (j in seq_len(ncol(m))) {
          v <- m[i, j]
          if (!is.na(v) && v != 0) {
            rows[[k]] <- data.frame(
              sample = colnames(m)[[j]],
              function_type = ftype,
              function_id = fid,
              count = as.numeric(v),
              function_label = rn[[i]],
              stringsAsFactors = FALSE
            )
            k <- k + 1L
          }
        }
      }
      df <- do.call(rbind, rows)
      return(list(long = df, source = hit$source, path = rds, notes = notes))
    }
    df <- wide_to_long(hit$counts, function_type = matrix_type)
    return(list(long = df, source = hit$source, path = rds, notes = notes))
  }

  if (!is.null(matrix) && nzchar(matrix)) {
    tab <- read_table_auto(matrix)
    if ("sample" %in% tolower(names(tab)) && "function_type" %in% tolower(names(tab))) {
      df <- normalize_long(tab)
      return(list(long = df, source = "long_via_matrix_arg", path = matrix, notes = notes))
    }
    rn <- tab[[1]]
    mat <- as.matrix(tab[, -1, drop = FALSE])
    rownames(mat) <- as.character(rn)
    df <- wide_to_long(mat, function_type = matrix_type)
    return(list(long = df, source = "wide_matrix", path = matrix, notes = notes))
  }

  if (!is.null(gff3_dir) && nzchar(gff3_dir)) {
    hit <- import_gff3_dir(gff3_dir, sample_map = sample_map)
    hit$path <- gff3_dir
    hit$notes <- notes
    return(hit)
  }

  for (p in default_long_candidates()) {
    p <- normalizePath(p, mustWork = FALSE)
    if (file.exists(p)) {
      df <- normalize_long(read_table_auto(p))
      notes <- c(notes, paste0("Auto-resolved long table: ", p))
      return(list(long = df, source = "auto_long", path = p, notes = notes))
    }
  }

  fail(
    "No metabolic table found. Provide --long, --matrix, --rds, or --gff3-dir ",
    "(or place fixtures under test/metabolism/)."
  )
}

# ---------------------------------------------------------------------------
# Filter + matrices
# ---------------------------------------------------------------------------

filter_metabolism <- function(long_df, types = DEFAULT_TYPES, drop_hypothetical = TRUE) {
  if (!nrow(long_df)) fail("Empty long table")
  # Always drop GO
  n_go <- sum(long_df$function_type %in% GO_TYPES)
  df <- long_df[!long_df$function_type %in% GO_TYPES, , drop = FALSE]
  types <- tolower(types)
  bad <- setdiff(types, METABOLISM_TYPES)
  if (length(bad)) {
    fail("Unsupported --types (metabolism skill excludes GO): ", paste(bad, collapse = ", "),
         ". Allowed: ", paste(METABOLISM_TYPES, collapse = ", "))
  }
  df <- df[df$function_type %in% types, , drop = FALSE]
  if (!nrow(df)) fail("No rows left after type filter: ", paste(types, collapse = ","))

  if (isTRUE(drop_hypothetical) && "product" %in% types) {
    hyp <- grepl("hypothetical protein", df$function_id, ignore.case = TRUE) |
      grepl("hypothetical protein", df$function_label, ignore.case = TRUE)
    n_hyp <- sum(hyp)
    df <- df[!hyp, , drop = FALSE]
  } else {
    n_hyp <- 0L
  }
  if (!nrow(df)) fail("No rows left after dropping hypothetical products")

  # Aggregate duplicate keys
  df <- stats::aggregate(
    count ~ sample + function_type + function_id + function_label,
    data = df, FUN = sum
  )
  list(long = df, n_go_dropped = as.integer(n_go), n_hypothetical_dropped = as.integer(n_hyp))
}

build_matrices <- function(long_df) {
  genes <- unique(long_df$function_label)
  samples <- unique(as.character(long_df$sample))
  mat <- matrix(0, nrow = length(genes), ncol = length(samples),
                dimnames = list(genes, samples))
  for (i in seq_len(nrow(long_df))) {
    mat[long_df$function_label[[i]], long_df$sample[[i]]] <-
      mat[long_df$function_label[[i]], long_df$sample[[i]]] + long_df$count[[i]]
  }
  cs <- colSums(mat)
  if (any(cs <= 0)) {
    fail("Some samples have zero counts after filtering: ",
         paste(names(cs)[cs <= 0], collapse = ", "))
  }
  rel <- sweep(mat, 2, cs, "/")
  list(counts = mat, rel = rel)
}

#' Load phyloseq and return named sample read depths.
load_ps_sample_depths <- function(ps_rds) {
  if (is.null(ps_rds) || !nzchar(ps_rds)) return(NULL)
  if (!file.exists(ps_rds)) fail("phyloseq RDS not found: ", ps_rds)
  if (!requireNamespace("phyloseq", quietly = TRUE)) {
    fail("phyloseq required for --ps-rds rarefaction/normalization")
  }
  obj <- readRDS(ps_rds)
  ps <- if (inherits(obj, "phyloseq")) {
    obj
  } else if (is.list(obj) && inherits(obj$phyloseq, "phyloseq")) {
    obj$phyloseq
  } else {
    fail("--ps-rds must be phyloseq or list with $phyloseq: ", ps_rds)
  }
  depths <- as.numeric(phyloseq::sample_sums(ps))
  names(depths) <- phyloseq::sample_names(ps)
  if (!length(depths) || any(!is.finite(depths)) || any(depths <= 0)) {
    fail("Invalid sample_sums in phyloseq: ", ps_rds)
  }
  # Relative-abundance phyloseq (sums ≈ 1) cannot drive rarefaction depth
  if (max(depths, na.rm = TRUE) < 10) {
    fail(
      "phyloseq sample_sums look like relative abundances (max=",
      max(depths, na.rm = TRUE),
      "), not read counts. Pass a counts phyloseq (e.g. phyloseq_counts.rds): ",
      ps_rds
    )
  }
  list(ps = ps, depths = depths, path = ps_rds)
}

#' Align metabolic sample IDs to phyloseq sample_sums (exact, ± `_smir`).
align_metab_ps_depths <- function(metab_ids, depths) {
  metab_ids <- as.character(metab_ids)
  dn <- names(depths)
  # Exact
  common <- intersect(metab_ids, dn)
  if (length(common) >= 2L) {
    return(list(
      depths = depths[common],
      metab_keep = common,
      map = stats::setNames(common, common)
    ))
  }
  # Strip / add `_smir`
  strip <- function(x) sub("_smir$", "", x)
  add <- function(x) ifelse(grepl("_smir$", x), x, paste0(x, "_smir"))
  # metab → strip match phyloseq
  m_strip <- strip(metab_ids)
  hit <- match(m_strip, dn)
  ok <- !is.na(hit)
  if (sum(ok) >= 2L) {
    keep_m <- metab_ids[ok]
    keep_p <- dn[hit[ok]]
    return(list(
      depths = stats::setNames(as.numeric(depths[keep_p]), keep_m),
      metab_keep = keep_m,
      map = stats::setNames(keep_p, keep_m)
    ))
  }
  # phyloseq strip match metab
  p_strip <- strip(dn)
  hit2 <- match(metab_ids, p_strip)
  ok2 <- !is.na(hit2)
  if (sum(ok2) >= 2L) {
    keep_m <- metab_ids[ok2]
    keep_p <- dn[hit2[ok2]]
    return(list(
      depths = stats::setNames(as.numeric(depths[keep_p]), keep_m),
      metab_keep = keep_m,
      map = stats::setNames(keep_p, keep_m)
    ))
  }
  # add _smir to phyloseq names
  p_add <- add(dn)
  hit3 <- match(metab_ids, p_add)
  ok3 <- !is.na(hit3)
  if (sum(ok3) >= 2L) {
    keep_m <- metab_ids[ok3]
    keep_p <- dn[hit3[ok3]]
    return(list(
      depths = stats::setNames(as.numeric(depths[keep_p]), keep_m),
      metab_keep = keep_m,
      map = stats::setNames(keep_p, keep_m)
    ))
  }
  list(depths = NULL, metab_keep = character(0), map = character(0))
}

resolve_ps_rds <- function(ps_rds = NULL) {
  cands <- c(
    ps_rds,
    "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/phyloseq_counts.rds",
    file.path(root, "..", "2026", "Kristina", "data/processed/phyloseq_counts.rds"),
    "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/phyloseq.rds",
    file.path(root, "..", "2026", "Kristina", "data/processed/phyloseq.rds"),
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds",
    "test/code-review-phyloseq/grazing_phyloseq.rds"
  )
  for (p in cands) {
    if (is.null(p) || !nzchar(p) || !file.exists(p)) next
    pack <- tryCatch(load_ps_sample_depths(p), error = function(e) NULL)
    # Prefer count-like phyloseq (sample_sums ≥ 10); skip relative-abundance objects
    if (!is.null(pack) && max(pack$depths, na.rm = TRUE) >= 10) {
      return(normalizePath(p, mustWork = FALSE))
    }
  }
  NULL
}

#' Rarefy metabolic counts to phyloseq-derived depth; normalize by phyloseq reads.
rarefy_normalize_to_ps <- function(counts, ps_pack, seed = 123L) {
  depths0 <- ps_pack$depths
  aligned <- align_metab_ps_depths(colnames(counts), depths0)
  if (is.null(aligned$depths) || length(aligned$metab_keep) < 2L) {
    fail(
      "Need ≥2 overlapping samples between metabolic matrix and phyloseq. ",
      "Metabolic: ", paste(utils::head(colnames(counts), 8), collapse = ","),
      "; phyloseq: ", paste(utils::head(names(depths0), 8), collapse = ",")
    )
  }
  depths <- aligned$depths
  common <- aligned$metab_keep
  dropped <- setdiff(colnames(counts), common)
  notes <- character(0)
  if (length(dropped)) {
    notes <- c(notes, paste0(
      "dropped ", length(dropped), " metabolic samples absent from phyloseq"
    ))
  }
  if (!identical(names(depths0), names(depths)) ||
      !all(common %in% names(depths0))) {
    notes <- c(notes, "aligned metabolic↔phyloseq sample IDs (exact or ±_smir)")
  }
  counts <- round(counts[, common, drop = FALSE])
  storage.mode(counts) <- "integer"
  ps_depth <- depths[common]
  metab_cs <- colSums(counts)
  # Drop near-empty metabolic libraries
  min_lib <- 50L
  keep_lib <- metab_cs >= min_lib
  if (sum(keep_lib) < 2L) {
    fail(
      "Fewer than 2 metabolic samples with library size ≥ ", min_lib,
      " after phyloseq overlap"
    )
  }
  if (any(!keep_lib)) {
    notes <- c(notes, paste0(
      "dropped ", sum(!keep_lib),
      " metabolic samples with library size < ", min_lib
    ))
  }
  counts <- counts[, keep_lib, drop = FALSE]
  ps_depth <- ps_depth[colnames(counts)]
  metab_cs <- colSums(counts)
  ps_min <- max(1L, as.integer(floor(min(ps_depth))))
  metab_min <- max(1L, as.integer(floor(min(metab_cs))))
  # Rarefy only when phyloseq depth is usable for a gene-rich table;
  # otherwise keep counts and normalize by phyloseq sample_sums.
  can_rarefy <- ps_min >= 1000L && ps_min <= metab_min
  if (can_rarefy) {
    target <- ps_min
    set.seed(as.integer(seed))
    if (requireNamespace("vegan", quietly = TRUE)) {
      rare <- t(vegan::rrarefy(t(counts), sample = target))
    } else if (requireNamespace("phyloseq", quietly = TRUE)) {
      tmp <- phyloseq::phyloseq(phyloseq::otu_table(counts, taxa_are_rows = TRUE))
      tmp <- phyloseq::rarefy_even_depth(
        tmp, sample.size = target, rngseed = as.integer(seed),
        replace = FALSE, trimOTUs = FALSE, verbose = FALSE
      )
      rare <- as(phyloseq::otu_table(tmp), "matrix")
      if (!phyloseq::taxa_are_rows(tmp)) rare <- t(rare)
      out <- matrix(0L, nrow = nrow(counts), ncol = ncol(counts),
                    dimnames = dimnames(counts))
      out[rownames(rare), colnames(rare)] <- rare
      rare <- out
    } else {
      fail("vegan or phyloseq required to rarefy metabolic counts")
    }
    notes <- c(notes, paste0("rarefied metabolic counts to phyloseq min depth=", target))
  } else {
    rare <- counts
    target <- NA_integer_
    notes <- c(notes, paste0(
      "skipped rarefy (phyloseq min=", ps_min, ", metabolic min=", metab_min,
      "); normalize by phyloseq sample_sums only"
    ))
  }
  # Normalize by related phyloseq sample read amounts
  rel <- sweep(rare, 2, as.numeric(ps_depth[colnames(rare)]), "/")
  list(
    counts = rare, rel = rel, depth = target,
    ps_depths = ps_depth, samples = colnames(rare), notes = notes,
    ps_rds = ps_pack$path
  )
}

select_top_genes <- function(rel, top_n = DEFAULT_TOP_N) {
  means <- rowMeans(rel)
  ord <- order(means, decreasing = TRUE)
  # top_n <= 0 → all genes (full-matrix heatmap)
  n <- if (is.null(top_n) || !is.finite(as.numeric(top_n)) || as.integer(top_n) <= 0L) {
    length(ord)
  } else {
    min(as.integer(top_n), length(ord))
  }
  if (n < 2L) fail("Need ≥2 genes for pheatmap; got ", n)
  keep <- names(means)[ord[seq_len(n)]]
  rank_df <- data.frame(
    rank = seq_len(n),
    function_label = keep,
    mean_rel = as.numeric(means[keep]),
    stringsAsFactors = FALSE
  )
  list(labels = keep, rank = rank_df, mat = rel[keep, , drop = FALSE])
}

short_labels <- function(labels) {
  vapply(labels, function(x) {
    # product|product:NAME → NAME; ko|ko:Kxxxxx → Kxxxxx
    parts <- strsplit(x, "|", fixed = TRUE)[[1]]
    id <- if (length(parts) >= 2L) paste(parts[-1], collapse = "|") else x
    id <- sub("^(product|ec|ko):", "", id)
    if (nchar(id) > 60L) paste0(substr(id, 1L, 57L), "...") else id
  }, character(1), USE.NAMES = FALSE)
}

plot_pheatmap <- function(mat, out_pdf, out_png, main, scale = DEFAULT_SCALE) {
  scale <- tolower(scale)
  if (!scale %in% c("none", "row", "column")) fail("Invalid --scale: ", scale)
  rn <- short_labels(rownames(mat))
  # uniquify display rownames
  if (anyDuplicated(rn)) {
    rn <- paste0(rn, " [", seq_along(rn), "]")
  }
  plot_mat <- mat
  rownames(plot_mat) <- rn

  cols <- grDevices::colorRampPalette(c(LFC_LOW, LFC_MID, LFC_HIGH))(50)
  ensure_dir(dirname(out_pdf))

  nr <- nrow(plot_mat)
  # Large gene sets: skip row clustering / names so full matrices stay tractable
  big <- nr > 400L
  show_rn <- nr <= 250L
  fs_row <- if (nr > 2000L) 1.5 else if (nr > 500L) 3 else 8
  h_in <- if (nr > 2000L) 24 else if (nr > 500L) 16 else max(6, 0.22 * nr + 3)
  h_px <- if (nr > 2000L) 4800 else if (nr > 500L) 3200 else max(1400, 50 * nr + 600)

  draw_hm <- function() {
    pheatmap::pheatmap(
      plot_mat,
      scale = scale,
      color = cols,
      border_color = NA,
      main = main,
      fontsize_row = fs_row,
      fontsize_col = 8,
      angle_col = 45,
      cluster_rows = !big,
      cluster_cols = ncol(plot_mat) <= 60L,
      show_rownames = show_rn,
      clustering_distance_rows = "euclidean",
      clustering_distance_cols = "euclidean"
    )
  }

  grDevices::pdf(out_pdf, width = 10, height = h_in)
  draw_hm()
  grDevices::dev.off()

  grDevices::png(
    out_png,
    width = 2400,
    height = h_px,
    res = 200,
    type = "cairo"
  )
  draw_hm()
  grDevices::dev.off()
  invisible(TRUE)
}

write_matrix_tsv <- function(mat, path) {
  ensure_dir(dirname(path))
  df <- data.frame(function_label = rownames(mat), mat, check.names = FALSE)
  utils::write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

run_metabolism <- function(long = NULL, matrix = NULL, rds = NULL, gff3_dir = NULL,
                           sample_map = NULL, outdir = "test/metabolism/run",
                           types = "product", top_n = DEFAULT_TOP_N,
                           scale = DEFAULT_SCALE, drop_hypothetical = TRUE,
                           matrix_type = "product", ps_rds = NULL,
                           use_ps_rds = TRUE, seed = 123L) {
  setwd(root)
  ensure_dir(outdir)

  type_vec <- trimws(unlist(strsplit(as.character(types), ",", fixed = TRUE)))
  type_vec <- type_vec[nzchar(type_vec)]

  resolved <- resolve_long(
    long = long, matrix = matrix, rds = rds, gff3_dir = gff3_dir,
    sample_map = sample_map, matrix_type = matrix_type
  )
  filt <- filter_metabolism(
    resolved$long, types = type_vec, drop_hypothetical = drop_hypothetical
  )
  mats <- build_matrices(filt$long)

  notes <- resolved$notes %||% character(0)
  rare_meta <- NULL
  ps_path <- NULL
  # Default: rarefy/normalize via phyloseq sample_sums (--ps-rds or auto-discover)
  if (isTRUE(use_ps_rds)) {
    if (!is.null(ps_rds) && nzchar(ps_rds)) {
      if (!file.exists(ps_rds)) fail("phyloseq RDS not found: ", ps_rds)
      ps_path <- normalizePath(ps_rds, mustWork = FALSE)
    } else {
      cand <- resolve_ps_rds(NULL)
      if (!is.null(cand)) {
        pack_try <- tryCatch(load_ps_sample_depths(cand), error = function(e) NULL)
        if (!is.null(pack_try)) {
          al <- align_metab_ps_depths(colnames(mats$counts), pack_try$depths)
          if (length(al$metab_keep) >= 2L) ps_path <- cand
        }
      }
    }
  }
  if (!is.null(ps_path)) {
    ps_pack <- load_ps_sample_depths(ps_path)
    rare_meta <- rarefy_normalize_to_ps(mats$counts, ps_pack, seed = seed)
    mats$counts <- rare_meta$counts
    mats$rel <- rare_meta$rel
    depth_note <- if (!is.null(rare_meta$depth) && is.finite(rare_meta$depth)) {
      paste0("rarefied to depth=", rare_meta$depth, "; ")
    } else {
      ""
    }
    notes <- c(notes, rare_meta$notes, paste0(
      depth_note, "normalized by phyloseq sample_sums (", basename(ps_path),
      "); relative abundance = counts / phyloseq reads"
    ))
  } else {
    notes <- c(notes, paste0(
      "no usable phyloseq --ps-rds (use_ps_rds=", isTRUE(use_ps_rds),
      "): relative abundance = counts / metabolic library size"
    ))
  }

  top <- select_top_genes(mats$rel, top_n = top_n)

  long_path <- file.path(outdir, "metabolism_long.csv")
  counts_path <- file.path(outdir, "metabolism_counts.tsv")
  rel_path <- file.path(outdir, "metabolism_rel.tsv")
  top_path <- file.path(outdir, "metabolism_top_genes.tsv")
  pdf_path <- file.path(outdir, "metabolism_heatmap.pdf")
  png_path <- file.path(outdir, "metabolism_heatmap.png")
  report_path <- file.path(outdir, "metabolism-report.json")

  utils::write.csv(filt$long, long_path, row.names = FALSE)
  write_matrix_tsv(mats$counts, counts_path)
  write_matrix_tsv(mats$rel, rel_path)
  utils::write.table(top$rank, top_path, sep = "\t", quote = FALSE, row.names = FALSE)

  main_title <- sprintf(
    "Top-%d of %d metabolic genes (%s; mean relative abundance)",
    nrow(top$rank), nrow(mats$rel), paste(type_vec, collapse = "+")
  )
  plot_pheatmap(top$mat, pdf_path, png_path, main = main_title, scale = scale)

  rep <- list(
    skill = "metabolism",
    input_source = resolved$source,
    input_path = resolved$path,
    types = type_vec,
    go_excluded = TRUE,
    n_go_dropped = filt$n_go_dropped,
    n_hypothetical_dropped = filt$n_hypothetical_dropped,
    drop_hypothetical = isTRUE(drop_hypothetical),
    n_samples = ncol(mats$counts),
    n_genes = nrow(mats$counts),
    top_n = nrow(top$rank),
    pheatmap_scale = scale,
    ps_rds = if (!is.null(rare_meta)) rare_meta$ps_rds else NULL,
    rarefaction_depth = if (!is.null(rare_meta)) rare_meta$depth else NULL,
    normalized_by = if (!is.null(rare_meta)) "phyloseq_sample_sums" else "metabolic_library_size",
    palette = c(LFC_LOW, LFC_MID, LFC_HIGH),
    package_versions = list(
      pheatmap = as.character(utils::packageVersion("pheatmap")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    figures = list(pdf = pdf_path, png = png_path),
    tables = list(
      long = long_path,
      counts = counts_path,
      rel = rel_path,
      top_genes = top_path
    ),
    notes = notes
  )
  write_json(rep, report_path)
  message(
    "metabolism OK: genes=", rep$n_genes, " samples=", rep$n_samples,
    " top_n=", rep$top_n, " pdf=", pdf_path
  )
  invisible(rep)
}

self_test <- function() {
  setwd(root)
  fixture <- file.path(root, ".cursor/skills/metabolism/fixtures/bakta_function_long.csv")
  if (!file.exists(fixture)) {
    fixture <- "test/metabolism/fixtures/bakta_function_long.csv"
  }
  if (!file.exists(fixture)) fail("Missing fixture: ", fixture)

  out <- "test/metabolism/self-test"
  if (dir.exists(out)) unlink(out, recursive = TRUE)
  # Fixture sample IDs do not overlap grazing/Kristina phyloseq — skip ps rarefy
  rep <- run_metabolism(
    long = fixture,
    outdir = out,
    types = "product,ko,ec",
    top_n = 10L,
    scale = "row",
    drop_hypothetical = TRUE,
    use_ps_rds = FALSE
  )
  if (!isTRUE(rep$go_excluded)) stop("GO must be excluded")
  if (!(rep$n_go_dropped >= 1L)) stop("fixture should drop ≥1 GO row")
  if (!file.exists(rep$figures$pdf) || !file.exists(rep$figures$png)) {
    stop("missing heatmap figures")
  }
  if (!file.exists(rep$tables$top_genes)) stop("missing top_genes")
  top <- utils::read.delim(rep$tables$top_genes, stringsAsFactors = FALSE)
  if (nrow(top) < 2L) stop("expected ≥2 top genes")
  if (any(grepl("^go\\|", top$function_label, ignore.case = TRUE))) {
    stop("GO labels must not appear in top genes")
  }
  long <- utils::read.csv(rep$tables$long, stringsAsFactors = FALSE)
  if (any(tolower(long$function_type) == "go")) stop("GO rows leaked into long CSV")

  # Full Kristina real long (product+ko+ec) + default phyloseq rarefy when available
  kristina <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/bakta_function_long.csv"
  kristina_ps <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/phyloseq_counts.rds"
  if (file.exists(kristina)) {
    out2 <- "test/metabolism/kristina"
    if (dir.exists(out2)) unlink(out2, recursive = TRUE)
    rep2 <- run_metabolism(
      long = kristina,
      outdir = out2,
      types = "product,ko,ec",
      top_n = 0L,  # full gene matrix on heatmap
      scale = "row",
      drop_hypothetical = TRUE,
      ps_rds = if (file.exists(kristina_ps)) kristina_ps else NULL,
      use_ps_rds = TRUE
    )
    if (!(rep2$n_genes >= 10000L)) {
      stop("Kristina product+ko+ec matrix too small: ", rep2$n_genes,
           " (expected ≥10000)")
    }
    if (!(rep2$top_n >= 10000L)) {
      stop("Kristina heatmap should include all genes; top_n=", rep2$top_n)
    }
    if (!(rep2$n_go_dropped > 0L)) stop("expected GO rows dropped from Kristina long")
    message(
      "SELF-TEST OK (fixture + Kristina full heatmap; genes=", rep2$n_genes,
      ", normalized_by=", rep2$normalized_by, ")"
    )
  } else {
    message("SELF-TEST OK (fixture only; Kristina long CSV not found)")
  }
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  drop_hyp <- !identical(tolower(as.character(args$drop_hypothetical %||% "true")), "false")
  no_ps <- identical(tolower(as.character(args$no_ps_rds %||% "false")), "true") ||
    identical(tolower(as.character(args$use_ps_rds %||% "true")), "false")
  run_metabolism(
    long = args$long,
    matrix = args$matrix,
    rds = args$rds,
    gff3_dir = args$gff3_dir,
    sample_map = args$sample_map,
    outdir = args$outdir %||% "test/metabolism/run",
    types = args$types %||% "product",
    top_n = as.integer(args$top_n %||% DEFAULT_TOP_N),
    scale = args$scale %||% DEFAULT_SCALE,
    drop_hypothetical = drop_hyp,
    matrix_type = args$matrix_type %||% "product",
    ps_rds = args$ps_rds %||% args$ps,
    use_ps_rds = !isTRUE(no_ps),
    seed = as.integer(args$seed %||% 123L)
  )
}

if (sys.nframe() == 0L) {
  tryCatch(
    { main(); quit(save = "no", status = 0) },
    error = function(e) {
      message("ERROR: ", conditionMessage(e))
      quit(save = "no", status = 1)
    }
  )
}
