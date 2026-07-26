#!/usr/bin/env Rscript
# rarefaction-analysis — curves + even-depth rarefaction (counts only)
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  library(ggplot2)
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

MIN_READS_FLOOR <- 1000L

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(path = path, target = NA_character_, batch = NA_character_, abundances = NA_character_)
  if (inherits(obj, "phyloseq")) return(list(ps = obj, meta = meta, wrapper = NULL))
  if (is.list(obj) && !is.null(obj$phyloseq) && inherits(obj$phyloseq, "phyloseq")) {
    meta$target <- obj$target %||% NA_character_
    meta$batch <- obj$batch %||% NA_character_
    meta$abundances <- obj$abundances %||% NA_character_
    return(list(ps = obj$phyloseq, meta = meta, wrapper = obj))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

is_count_like <- function(ps) {
  s <- sample_sums(ps)
  # relative / MMUPHin: all ~1; counts: typically >> 1
  med <- stats::median(as.numeric(s))
  med >= 10
}

companion_count_rds <- function(path) {
  # foo_batchadj.rds → foo.rds ; grazing_phyloseq_batchadj.rds → grazing_phyloseq.rds
  alt <- sub("_batchadj(\\.rds)$", "\\1", path)
  if (!identical(alt, path) && file.exists(alt)) return(alt)
  # also try plain beside
  plain <- file.path(dirname(path), "grazing_phyloseq.rds")
  if (file.exists(plain)) return(plain)
  NA_character_
}

resolve_input_rds <- function(rds = NULL, prefer_batchadj = TRUE) {
  notes <- character(0)
  if (!is.null(rds) && nzchar(rds)) {
    loaded <- load_ps(rds)
    if (!is_count_like(loaded$ps)) {
      alt <- companion_count_rds(rds)
      if (!is.na(alt)) {
        notes <- c(notes, paste0(
          "Input looks relative/non-count (median sample_sum=",
          round(stats::median(sample_sums(loaded$ps)), 4),
          "); falling back to count RDS: ", alt
        ))
        loaded <- load_ps(alt)
        loaded$meta$fallback_from <- rds
      } else {
        fail(
          "Rarefaction requires count abundances; input median sample_sum=",
          stats::median(sample_sums(loaded$ps)),
          ". Provide a count phyloseq RDS."
        )
      }
    }
    loaded$notes <- notes
    return(loaded)
  }
  # Auto-discover grazing test artifacts
  batchadj <- "test/code-review-phyloseq/grazing_phyloseq_batchadj.rds"
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"
  if (prefer_batchadj && file.exists(batchadj)) {
    loaded <- load_ps(batchadj)
    if (is_count_like(loaded$ps)) {
      loaded$notes <- "Using batch-adjusted count object"
      return(loaded)
    }
    notes <- c(notes, "Batch-adjusted object is relative — using count phyloseq instead")
  }
  if (!file.exists(raw)) fail("No --rds and default grazing count RDS missing: ", raw)
  loaded <- load_ps(raw)
  loaded$notes <- notes
  loaded
}

parse_depths_arg <- function(args, ps) {
  if (!is.null(args$depths) && nzchar(args$depths)) {
    ds <- as.integer(strsplit(args$depths, ",", fixed = TRUE)[[1]])
    ds <- ds[!is.na(ds) & ds > 0]
    if (!length(ds)) fail("Invalid --depths")
    return(sort(unique(ds)))
  }
  if (!is.null(args$depth) && nzchar(args$depth)) {
    d <- as.integer(args$depth)
    if (is.na(d) || d < 1) fail("Invalid --depth")
    return(d)
  }
  # Default: at least 1000 OR lowest sequencing depth
  sums <- sample_sums(ps)
  min_sum <- as.integer(min(sums))
  if (min_sum >= MIN_READS_FLOOR) {
    return(min_sum)
  }
  message(
    "Lowest depth ", min_sum, " < ", MIN_READS_FLOOR,
    " — will rarefy at ", MIN_READS_FLOOR, " after dropping shallower samples"
  )
  MIN_READS_FLOOR
}

# Main alpha measures for rarefaction curves (grazing Article: Observed + Shannon; + Simpson)
ALPHA_MEASURES <- c("Observed", "Shannon", "Simpson")

default_curve_depths <- function(ps) {
  max_sum <- as.integer(max(sample_sums(ps)))
  grid <- unique(as.integer(c(
    1L, 5L, 10L, 20L, 50L, 100L, 150L,
    seq(200L, min(2000L, max_sum), by = 100L),
    if (max_sum > 2000L) seq(2250L, max_sum, by = 250L) else integer(0)
  )))
  grid[grid <= max_sum & grid > 0]
}

#' Alpha rarefaction long table (grazing calculate.rarefaction pattern).
#' At each depth × replicate: rarefy_even_depth → estimate_richness(measures).
alpha_rarefaction_long <- function(ps, measures = ALPHA_MEASURES, depths = NULL,
                                   n_reps = 5L, seed = 123L) {
  if (is.null(depths)) depths <- default_curve_depths(ps)
  depths <- sort(unique(as.integer(depths)))
  depths <- depths[depths > 0]
  if (!length(depths)) fail("No valid curve depths")
  measures <- unique(as.character(measures))
  n_reps <- max(1L, as.integer(n_reps))

  rows <- list()
  k <- 0L
  for (d in depths) {
    keep <- sample_sums(ps) >= d
    if (sum(keep) < 1L) next
    ps_k <- prune_samples(keep, ps)
    ps_k <- prune_taxa(taxa_sums(ps_k) > 0, ps_k)
    for (r in seq_len(n_reps)) {
      set.seed(as.integer(seed) + as.integer(d) * 1000L + as.integer(r))
      ps_r <- rarefy_even_depth(
        ps_k,
        sample.size = as.integer(d),
        rngseed = as.integer(seed) + as.integer(d) * 1000L + as.integer(r),
        replace = FALSE,
        trimOTUs = TRUE,
        verbose = FALSE
      )
      adiv <- estimate_richness(ps_r, measures = measures)
      sn <- sample_names(ps_r)
      # estimate_richness may make.names() rownames — map back to phyloseq sample_names
      rn <- rownames(adiv)
      mapped <- sn[match(rn, make.names(sn))]
      if (any(is.na(mapped))) mapped <- sn[match(rn, sn)]
      if (any(is.na(mapped))) mapped <- rn
      adiv$Sample <- mapped
      for (m in measures) {
        if (!m %in% names(adiv)) next
        k <- k + 1L
        rows[[k]] <- data.frame(
          Sample = adiv$Sample,
          Measure = m,
          Depth = as.integer(d),
          Rep = as.integer(r),
          value = as.numeric(adiv[[m]]),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (!length(rows)) fail("Alpha rarefaction produced no rows")
  do.call(rbind, rows)
}

#' Per-sample mean across replicates (exact sample trajectories).
alpha_rarefaction_sample_means <- function(long_df) {
  agg <- stats::aggregate(
    value ~ Sample + Measure + Depth,
    data = long_df,
    FUN = mean
  )
  agg
}

plot_alpha_rarefaction <- function(sample_df, depths_vline, out_prefix,
                                   color_by = NULL, sam_df = NULL) {
  df <- sample_df
  if (!is.null(color_by) && !is.null(sam_df) && color_by %in% names(sam_df)) {
    df$target <- as.character(sam_df[df$Sample, color_by])
  } else {
    df$target <- df$Sample
    color_by <- "sample"
  }
  df$Measure <- factor(df$Measure, levels = intersect(ALPHA_MEASURES, unique(df$Measure)))

  # Exact samples: geom_line; target trend: geom_smooth (grazing Article pattern)
  p <- ggplot(df, aes(x = Depth, y = value, color = target, fill = target)) +
    geom_line(aes(group = Sample), alpha = 0.35, linewidth = 0.35) +
    geom_smooth(
      aes(group = target),
      method = "loess",
      se = TRUE,
      alpha = 0.12,
      linewidth = 0.9,
      formula = y ~ x
    ) +
    geom_vline(xintercept = depths_vline, linetype = "dashed", linewidth = 0.4, color = "grey30") +
    facet_wrap(~Measure, scales = "free_y") +
    labs(
      title = "Rarefaction curves (alpha diversity)",
      subtitle = paste0(
        "Exact samples (thin lines) + geom_smooth by ", color_by,
        "; depth(s): ", paste(depths_vline, collapse = ", ")
      ),
      x = "Sequencing depth",
      y = "Diversity index",
      color = color_by,
      fill = color_by
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "right")

  pdf_path <- paste0(out_prefix, ".pdf")
  png_path <- paste0(out_prefix, ".png")
  n_panels <- max(1L, length(unique(df$Measure)))
  ggsave(pdf_path, p, width = 4 * n_panels, height = 4.5)
  ggsave(png_path, p, width = 4 * n_panels, height = 4.5, dpi = 150)
  list(pdf = pdf_path, png = png_path, plot = p)
}

structure_slots <- function(ps) {
  list(
    tax_table = !is.null(tax_table(ps, errorIfNULL = FALSE)),
    otu_table = !is.null(otu_table(ps, errorIfNULL = FALSE)),
    sam_data = !is.null(sample_data(ps, errorIfNULL = FALSE)),
    tree_data = !is.null(phy_tree(ps, errorIfNULL = FALSE)),
    n_taxa = ntaxa(ps),
    n_samples = nsamples(ps),
    n_tree_tips = {
      tr <- phy_tree(ps, errorIfNULL = FALSE)
      if (is.null(tr)) 0L else length(tr$tip.label)
    },
    sample_sums_min = min(sample_sums(ps)),
    sample_sums_max = max(sample_sums(ps)),
    complete = !is.null(otu_table(ps, errorIfNULL = FALSE)) &&
      !is.null(tax_table(ps, errorIfNULL = FALSE)) &&
      !is.null(sample_data(ps, errorIfNULL = FALSE)) &&
      !is.null(phy_tree(ps, errorIfNULL = FALSE))
  )
}

rarefy_one <- function(ps, depth, seed) {
  keep <- sample_sums(ps) >= depth
  if (!any(keep)) fail("No samples with sample_sums >= ", depth)
  dropped <- sample_names(ps)[!keep]
  ps_k <- prune_samples(keep, ps)
  ps_k <- prune_taxa(taxa_sums(ps_k) > 0, ps_k)
  set.seed(as.integer(seed))
  ps_r <- rarefy_even_depth(
    ps_k,
    sample.size = as.integer(depth),
    rngseed = as.integer(seed),
    replace = FALSE,
    trimOTUs = TRUE,
    verbose = FALSE
  )
  # ensure tree tips match if present
  tr <- phy_tree(ps_r, errorIfNULL = FALSE)
  if (!is.null(tr)) {
    tips <- intersect(tr$tip.label, taxa_names(ps_r))
    if (length(tips) >= 2) {
      phy_tree(ps_r) <- ape::keep.tip(tr, tips)
      ps_r <- prune_taxa(tips, ps_r)
    }
  }
  list(ps = ps_r, dropped_samples = dropped)
}

run_rarefaction <- function(rds = NULL, outdir, depths = NULL, seed = 123L,
                            step = NULL, color_by = NULL, prefer_batchadj = TRUE,
                            n_reps = 5L) {
  ensure_dir(outdir)
  loaded <- resolve_input_rds(rds, prefer_batchadj = prefer_batchadj)
  ps <- loaded$ps
  if (any(sample_sums(ps) <= 0)) {
    fail("Samples with zero counts present: ", paste(sample_names(ps)[sample_sums(ps) <= 0], collapse = ","))
  }
  # round tiny floats if nearly integer
  otu <- as(otu_table(ps), "matrix")
  if (taxa_are_rows(ps)) {
    if (any(abs(otu - round(otu)) > 1e-6)) {
      fail("OTU table is not count-like (non-integer values). Rarefaction requires counts.")
    }
    otu_table(ps) <- otu_table(round(otu), taxa_are_rows = TRUE)
  } else {
    if (any(abs(otu - round(otu)) > 1e-6)) {
      fail("OTU table is not count-like (non-integer values). Rarefaction requires counts.")
    }
    otu_table(ps) <- otu_table(round(otu), taxa_are_rows = FALSE)
  }

  if (is.null(depths)) {
    # caller may pass via args later
    depths <- parse_depths_arg(list(), ps)
  }
  depths <- sort(unique(as.integer(depths)))

  message("Rarefaction depths: ", paste(depths, collapse = ", "))
  message("Input: ", loaded$meta$path, " (n=", nsamples(ps), ", min_sum=", min(sample_sums(ps)), ")")

  sam_df <- as(sample_data(ps), "data.frame")
  if (is.null(color_by) || !nzchar(color_by)) {
    color_by <- loaded$meta$target
    if (is.na(color_by) || !color_by %in% names(sam_df)) {
      color_by <- if ("grazing" %in% names(sam_df)) "grazing" else NULL
    }
  }

  n_reps <- max(1L, as.integer(n_reps))
  message("Computing alpha rarefaction curves (Observed, Shannon, Simpson; n_reps=", n_reps, ")...")
  alpha_long <- alpha_rarefaction_long(
    ps,
    measures = ALPHA_MEASURES,
    depths = default_curve_depths(ps),
    n_reps = n_reps,
    seed = seed
  )
  sample_means <- alpha_rarefaction_sample_means(alpha_long)
  utils::write.csv(alpha_long, file.path(outdir, "rarefaction_alpha_long.tsv"), row.names = FALSE)
  utils::write.csv(sample_means, file.path(outdir, "rarefaction_curves.tsv"), row.names = FALSE)
  figs <- plot_alpha_rarefaction(
    sample_means, depths,
    out_prefix = file.path(outdir, "rarefaction_curves"),
    color_by = color_by,
    sam_df = sam_df
  )

  objects <- list()
  for (d in depths) {
    one <- rarefy_one(ps, d, seed = seed)
    st <- structure_slots(one$ps)
    rds_path <- file.path(outdir, paste0("phyloseq_rare_", d, ".rds"))
    saveRDS(
      list(
        phyloseq = one$ps,
        target = loaded$meta$target %||% color_by,
        batch = loaded$meta$batch,
        rarefaction_depth = d,
        rarefaction_seed = as.integer(seed),
        source_rds = loaded$meta$path,
        dropped_samples = one$dropped_samples,
        generated = as.character(Sys.Date())
      ),
      rds_path
    )
    saveRDS(one$ps, file.path(outdir, paste0("phyloseq_rare_", d, "_plain.rds")))
    objects[[as.character(d)]] <- list(
      depth = d,
      rds = rds_path,
      dropped_samples = one$dropped_samples,
      n_dropped = length(one$dropped_samples),
      phyloseq_structure = st
    )
    message("Saved ", rds_path, " [", st$n_taxa, " × ", st$n_samples, "; complete=", st$complete, "]")
  }

  report <- list(
    input_rds = loaded$meta$path,
    notes = loaded$notes,
    prefer_batchadj = prefer_batchadj,
    depths = depths,
    depth_rule = paste0(
      "default: lowest sample_sum if >= ", MIN_READS_FLOOR,
      ", else ", MIN_READS_FLOOR, " after dropping shallower samples"
    ),
    seed = as.integer(seed),
    color_by = color_by,
    curves_tsv = file.path(outdir, "rarefaction_curves.tsv"),
    curves_alpha_long = file.path(outdir, "rarefaction_alpha_long.tsv"),
    curves_pdf = figs$pdf,
    curves_png = figs$png,
    alpha_measures = ALPHA_MEASURES,
    n_reps = as.integer(n_reps),
    objects = objects,
    input_sample_sums = list(
      min = min(sample_sums(ps)),
      median = stats::median(sample_sums(ps)),
      max = max(sample_sums(ps))
    )
  )
  write_json(report, file.path(outdir, "rarefaction-report.json"))
  message("Curves: ", figs$png)
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  # Prefer batchadj path in resolve — should fall back to counts
  out <- "test/rarefaction-analysis/grazing-self-test"
  rep <- run_rarefaction(
    rds = "test/code-review-phyloseq/grazing_phyloseq_batchadj.rds",
    outdir = out,
    seed = 123L,
    n_reps = 2L
  )
  if (!length(rep$depths)) stop("no depths")
  d <- rep$depths[[1]]
  if (d < MIN_READS_FLOOR) stop("default depth below floor")
  obj <- rep$objects[[as.character(d)]]
  if (!isTRUE(obj$phyloseq_structure$complete)) stop("rarefied phyloseq incomplete")
  if (!file.exists(rep$curves_png)) stop("missing curves png")
  if (!all(c("Observed", "Shannon", "Simpson") %in% rep$alpha_measures)) {
    stop("expected Observed/Shannon/Simpson measures")
  }
  # second depth → second object
  out2 <- "test/rarefaction-analysis/grazing-two-depths"
  rep2 <- run_rarefaction(
    rds = "test/code-review-phyloseq/grazing_phyloseq.rds",
    outdir = out2,
    depths = c(1000L, as.integer(rep$depths[[1]])),
    seed = 123L,
    n_reps = 2L
  )
  if (length(rep2$depths) != 2) stop("expected two depth objects")
  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/rarefaction-analysis/run"
  seed <- if (!is.null(args$seed)) as.integer(args$seed) else 123L
  step <- if (!is.null(args$step)) as.integer(args$step) else NULL
  n_reps <- if (!is.null(args$n_reps)) as.integer(args$n_reps) else 5L
  prefer <- !identical(tolower(as.character(args$prefer_batchadj %||% "true")), "false")

  # resolve ps first for default depths
  loaded <- resolve_input_rds(args$rds, prefer_batchadj = prefer)
  depths <- if (!is.null(args$depths) || !is.null(args$depth)) {
    parse_depths_arg(args, loaded$ps)
  } else {
    parse_depths_arg(list(), loaded$ps)
  }

  run_rarefaction(
    rds = loaded$meta$path,
    outdir = outdir,
    depths = depths,
    seed = seed,
    step = step,
    color_by = args$color_by,
    prefer_batchadj = prefer,
    n_reps = n_reps
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
