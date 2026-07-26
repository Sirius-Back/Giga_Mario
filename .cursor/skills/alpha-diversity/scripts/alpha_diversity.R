#!/usr/bin/env Rscript
# alpha-diversity — rarefied-preferred alpha metrics + raincloud/boxplot by targets
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
  stopifnot(requireNamespace("ggpubr", quietly = TRUE))
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

DEFAULT_MEASURES <- c("Observed", "Shannon", "Simpson", "InvSimpson")
SKIP_VARS <- c("seq", "ID", "SampleID", "sampleID", "sample_id", "Run", "run")

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path, target = NA_character_, batch = NA_character_,
    rarefaction_depth = NA_real_, abundances = NA_character_, rarefied = FALSE
  )
  if (inherits(obj, "phyloseq")) {
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
    return(list(ps = obj$phyloseq, meta = meta))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

is_count_like <- function(ps) {
  stats::median(as.numeric(sample_sums(ps))) >= 10
}

resolve_input_rds <- function(rds = NULL, prefer_rare = TRUE) {
  notes <- character(0)
  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"

  if (!is.null(rds) && nzchar(rds)) {
    loaded <- load_ps(rds)
    if (!is_count_like(loaded$ps)) {
      fail(
        "Alpha diversity requires count-like abundances; median sample_sum=",
        stats::median(sample_sums(loaded$ps)),
        ". Prefer rarefied or raw count RDS (not MMUPHin relative)."
      )
    }
    loaded$meta$rarefied <- loaded$meta$rarefied ||
      grepl("_rare|phyloseq_rare_", basename(rds))
    loaded$notes <- notes
    return(loaded)
  }

  if (prefer_rare) {
    for (p in candidates_rare) {
      if (file.exists(p)) {
        loaded <- load_ps(p)
        if (is_count_like(loaded$ps)) {
          loaded$meta$rarefied <- TRUE
          loaded$notes <- c(notes, paste0("Using rarefied object: ", p))
          return(loaded)
        }
      }
    }
    # scan rarefaction-analysis for any phyloseq_rare_*.rds
    rare_dir <- "test/rarefaction-analysis"
    if (dir.exists(rare_dir)) {
      hits <- sort(Sys.glob(file.path(rare_dir, "**", "phyloseq_rare_*.rds")), decreasing = TRUE)
      hits <- hits[!grepl("_plain\\.rds$", hits)]
      for (p in hits) {
        loaded <- load_ps(p)
        if (is_count_like(loaded$ps)) {
          loaded$meta$rarefied <- TRUE
          loaded$notes <- c(notes, paste0("Using rarefied object: ", p))
          return(loaded)
        }
      }
    }
    notes <- c(notes, "No rarefied object found — falling back to raw counts")
  }

  if (!file.exists(raw)) fail("No --rds and default grazing count RDS missing: ", raw)
  loaded <- load_ps(raw)
  loaded$notes <- notes
  loaded
}

resolve_targets <- function(ps, meta, targets_arg = NULL) {
  sam <- as(sample_data(ps), "data.frame")
  if (!is.null(targets_arg) && nzchar(targets_arg)) {
    tg <- trimws(strsplit(targets_arg, ",", fixed = TRUE)[[1]])
    tg <- tg[nzchar(tg)]
  } else if (!is.na(meta$target) && nzchar(meta$target)) {
    tg <- trimws(strsplit(as.character(meta$target), ",", fixed = TRUE)[[1]])
  } else {
    tg <- character(0)
  }

  # If a column literally stores the target *name* (constant string matching a column)
  if ("target" %in% names(sam) && !"target" %in% tg) {
    vals <- unique(as.character(sam$target))
    vals <- vals[!is.na(vals) & nzchar(vals)]
    if (length(vals) == 1L && vals %in% names(sam)) {
      tg <- unique(c(tg, vals))
    }
  }

  if (!length(tg)) {
    # discover grouping columns: 2–12 levels, not IDs / batch
    batch <- meta$batch
    for (nm in names(sam)) {
      if (nm %in% SKIP_VARS) next
      if (!is.na(batch) && identical(nm, batch)) next
      u <- unique(as.character(sam[[nm]]))
      u <- u[!is.na(u)]
      if (length(u) >= 2L && length(u) <= 12L) tg <- c(tg, nm)
    }
  }

  tg <- unique(tg)
  missing <- setdiff(tg, names(sam))
  if (length(missing)) fail("Target column(s) missing from sample_data: ", paste(missing, collapse = ", "))
  if (!length(tg)) fail("No target variables resolved; pass --targets")

  for (t in tg) {
    nlev <- length(unique(stats::na.omit(sam[[t]])))
    if (nlev < 2L) fail("Target '", t, "' has <2 levels")
  }
  tg
}

map_adiv_rownames <- function(adiv, sn) {
  rn <- rownames(adiv)
  mapped <- sn[match(rn, make.names(sn))]
  if (any(is.na(mapped))) mapped <- sn[match(rn, sn)]
  if (any(is.na(mapped))) mapped <- rn
  mapped
}

compute_alpha <- function(ps, measures = DEFAULT_MEASURES) {
  measures <- unique(as.character(measures))
  # Chao1 brings se.chao1 — allow but drop se later
  adiv <- estimate_richness(ps, measures = measures)
  sn <- sample_names(ps)
  adiv$Sample <- map_adiv_rownames(adiv, sn)

  tr <- phy_tree(ps, errorIfNULL = FALSE)
  if (!is.null(tr) && requireNamespace("picante", quietly = TRUE)) {
    otu <- as(otu_table(ps), "matrix")
    if (taxa_are_rows(ps)) otu <- t(otu)
    # align taxa
    common <- intersect(colnames(otu), tr$tip.label)
    if (length(common) >= 2L) {
      pd <- tryCatch(
        picante::pd(otu[, common, drop = FALSE], ape::keep.tip(tr, common), include.root = TRUE),
        error = function(e) NULL
      )
      if (!is.null(pd)) {
        adiv$FaithPD <- pd[map_adiv_rownames(adiv, sn), "PD"]
        # fallback index by rownames of pd
        if (all(is.na(adiv$FaithPD))) {
          adiv$FaithPD <- pd[match(make.names(adiv$Sample), rownames(pd)), "PD"]
        }
      }
    }
  }
  adiv
}

alpha_to_long <- function(adiv, sam_df, targets, measures) {
  drop_cols <- c("Sample", "se.chao1")
  metric_cols <- setdiff(names(adiv), drop_cols)
  # keep requested + FaithPD if present
  keep <- intersect(c(measures, "FaithPD"), metric_cols)
  if (!length(keep)) fail("No alpha measures computed")

  rows <- list()
  k <- 0L
  for (i in seq_len(nrow(adiv))) {
    sid <- adiv$Sample[[i]]
    for (m in keep) {
      k <- k + 1L
      row <- list(Sample = sid, Measure = m, value = as.numeric(adiv[[m]][[i]]))
      for (t in targets) {
        row[[t]] <- as.character(sam_df[sid, t])
      }
      rows[[k]] <- as.data.frame(row, stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}

pairwise_comps <- function(levels) {
  levels <- as.character(unique(levels))
  levels <- levels[!is.na(levels)]
  if (length(levels) < 2L) return(list())
  utils::combn(levels, 2L, simplify = FALSE)
}

compute_stats <- function(long_df, target) {
  out <- list()
  k <- 0L
  for (m in unique(long_df$Measure)) {
    d <- long_df[long_df$Measure == m & !is.na(long_df[[target]]), , drop = FALSE]
    if (nrow(d) < 3L) next
    d[[target]] <- as.factor(d[[target]])
    if (nlevels(d[[target]]) < 2L) next
    kw <- tryCatch(
      stats::kruskal.test(stats::as.formula(paste("value ~", target)), data = d),
      error = function(e) NULL
    )
    k <- k + 1L
    out[[k]] <- data.frame(
      target = target,
      Measure = m,
      test = "kruskal",
      group1 = NA_character_,
      group2 = NA_character_,
      n = nrow(d),
      p = if (is.null(kw)) NA_real_ else unname(kw$p.value),
      p_adj = if (is.null(kw)) NA_real_ else unname(kw$p.value),
      stringsAsFactors = FALSE
    )
    # pairwise Wilcoxon
    if (requireNamespace("rstatix", quietly = TRUE)) {
      pw <- tryCatch(
        rstatix::pairwise_wilcox_test(
          data = d,
          formula = stats::as.formula(paste("value ~", target)),
          p.adjust.method = "BH"
        ),
        error = function(e) NULL
      )
      if (!is.null(pw) && nrow(pw)) {
        for (i in seq_len(nrow(pw))) {
          k <- k + 1L
          out[[k]] <- data.frame(
            target = target,
            Measure = m,
            test = "wilcox_bh",
            group1 = as.character(pw$group1[[i]]),
            group2 = as.character(pw$group2[[i]]),
            n = nrow(d),
            p = as.numeric(pw$p[[i]]),
            p_adj = as.numeric(pw$p.adj[[i]]),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  if (!length(out)) {
    return(data.frame(
      target = character(), Measure = character(), test = character(),
      group1 = character(), group2 = character(), n = integer(),
      p = numeric(), p_adj = numeric(), stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, out)
}

plot_alpha <- function(long_df, target, style = c("raincloud", "boxplot"),
                       out_prefix, rare_depth = NA) {
  style <- match.arg(style)
  df <- long_df[!is.na(long_df[[target]]), , drop = FALSE]
  df[[target]] <- factor(df[[target]])
  comps <- pairwise_comps(levels(df[[target]]))

  aes_map <- aes(x = .data[[target]], y = value, fill = .data[[target]])
  p <- ggplot(df, aes_map)

  if (identical(style, "raincloud")) {
    if (!requireNamespace("ggviolinbox", quietly = TRUE)) {
      fail("ggviolinbox required for raincloud style (install_github('dsmutin/ggviolinbox'))")
    }
    p <- p +
      ggviolinbox::geom_halfviolin(
        panel = "right", width = 0.5, trim = TRUE,
        position = position_nudge(x = 0.05), show.legend = FALSE
      ) +
      geom_jitter(width = 0.05, height = 0, size = 1.4, alpha = 0.75, aes(color = .data[[target]])) +
      ggviolinbox::geom_halfboxplot(
        panel = "left", width = 0.25, outliers = FALSE,
        position = position_nudge(x = -0.05)
      )
  } else {
    p <- p +
      geom_boxplot(outlier.shape = NA, alpha = 0.7, width = 0.65) +
      geom_jitter(width = 0.12, height = 0, size = 1.4, alpha = 0.75, aes(color = .data[[target]]))
  }

  # Overall KW p-value labels (top-left); pairwise brackets separate
  kw_labs <- do.call(rbind, lapply(unique(as.character(df$Measure)), function(m) {
    d <- df[df$Measure == m & !is.na(df[[target]]), , drop = FALSE]
    if (nrow(d) < 3L) return(NULL)
    kw <- tryCatch(
      stats::kruskal.test(stats::as.formula(paste("value ~", target)), data = d),
      error = function(e) NULL
    )
    if (is.null(kw)) return(NULL)
    data.frame(
      Measure = m,
      label = paste0("p-value = ", formatC(unname(kw$p.value), format = "fg", digits = 3)),
      stringsAsFactors = FALSE
    )
  }))

  p <- p +
    facet_wrap(~Measure, scales = "free_y") +
    geom_text(
      data = kw_labs,
      aes(x = -Inf, y = Inf, label = label),
      inherit.aes = FALSE,
      hjust = -0.05, vjust = 1.4, size = 3.1, color = "grey15"
    )

  if (length(comps)) {
    p <- p + ggpubr::stat_compare_means(
      comparisons = comps,
      method = "wilcox.test",
      p.adjust.method = "BH",
      hide.ns = TRUE,
      tip.length = 0.01
    )
  }

  subtitle <- paste0(
    "target=", target, "; style=", style,
    if (!is.na(rare_depth)) paste0("; rarefied depth=", rare_depth) else "; input=raw/counts"
  )
  p <- p +
    labs(
      title = "Alpha diversity",
      subtitle = subtitle,
      x = target,
      y = "Diversity index",
      fill = target,
      color = target
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "right") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.25)))

  pdf_path <- paste0(out_prefix, ".pdf")
  png_path <- paste0(out_prefix, ".png")
  n_m <- length(unique(df$Measure))
  w <- max(7, 2.8 * min(n_m, 4))
  h <- if (n_m > 4) 7 else 4.8
  ggsave(pdf_path, p, width = w, height = h)
  ggsave(png_path, p, width = w, height = h, dpi = 150)
  list(pdf = pdf_path, png = png_path, plot = p)
}

run_alpha <- function(rds = NULL, outdir, targets = NULL, style = "raincloud",
                      measures = DEFAULT_MEASURES, prefer_rare = TRUE) {
  ensure_dir(outdir)
  style <- match.arg(style, c("raincloud", "boxplot"))
  loaded <- resolve_input_rds(rds, prefer_rare = prefer_rare)
  ps <- loaded$ps

  if (any(sample_sums(ps) <= 0)) {
    fail("Samples with zero counts: ", paste(sample_names(ps)[sample_sums(ps) <= 0], collapse = ","))
  }

  tg <- resolve_targets(ps, loaded$meta, targets_arg = targets)
  message("Input: ", loaded$meta$path, " (rarefied=", loaded$meta$rarefied, ")")
  message("Targets: ", paste(tg, collapse = ", "))
  message("Style: ", style)

  sam_df <- as(sample_data(ps), "data.frame")
  adiv <- compute_alpha(ps, measures = measures)
  long_df <- alpha_to_long(adiv, sam_df, tg, measures = measures)
  utils::write.csv(long_df, file.path(outdir, "alpha_long.tsv"), row.names = FALSE)

  stats_all <- lapply(tg, function(t) compute_stats(long_df, t))
  stats_df <- do.call(rbind, stats_all)
  utils::write.csv(stats_df, file.path(outdir, "alpha_stats.tsv"), row.names = FALSE)

  figs <- list()
  for (t in tg) {
    prefix <- file.path(outdir, paste0("alpha_", t, "_", style))
    figs[[t]] <- plot_alpha(
      long_df, t, style = style, out_prefix = prefix,
      rare_depth = loaded$meta$rarefaction_depth
    )
    message("Plot: ", figs[[t]]$png)
  }

  report <- list(
    input_rds = loaded$meta$path,
    rarefied = loaded$meta$rarefied,
    rarefaction_depth = loaded$meta$rarefaction_depth,
    notes = loaded$notes,
    targets = tg,
    measures = intersect(c(measures, "FaithPD"), unique(long_df$Measure)),
    style = style,
    n_samples = nsamples(ps),
    n_taxa = ntaxa(ps),
    alpha_long = file.path(outdir, "alpha_long.tsv"),
    alpha_stats = file.path(outdir, "alpha_stats.tsv"),
    figures = lapply(figs, function(f) list(pdf = f$pdf, png = f$png))
  )
  write_json(report, file.path(outdir, "alpha-diversity-report.json"))
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  out <- "test/alpha-diversity/grazing-self-test"
  rep <- run_alpha(
    rds = NULL,
    outdir = out,
    style = "raincloud",
    prefer_rare = TRUE
  )
  if (!isTRUE(rep$rarefied)) stop("expected rarefied input when available")
  if (!length(rep$targets)) stop("no targets")
  png <- rep$figures[[rep$targets[[1]]]]$png
  if (!file.exists(png)) stop("missing plot png")
  # boxplot style also works
  out2 <- "test/alpha-diversity/grazing-boxplot-self-test"
  rep2 <- run_alpha(
    rds = "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    outdir = out2,
    style = "boxplot",
    prefer_rare = TRUE
  )
  if (!file.exists(rep2$figures[[rep2$targets[[1]]]]$png)) stop("missing boxplot png")
  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/alpha-diversity/run"
  style <- args$style %||% "raincloud"
  prefer <- !identical(tolower(as.character(args$prefer_rare %||% "true")), "false")
  measures <- if (!is.null(args$measures) && nzchar(args$measures)) {
    trimws(strsplit(args$measures, ",", fixed = TRUE)[[1]])
  } else {
    DEFAULT_MEASURES
  }

  run_alpha(
    rds = args$rds,
    outdir = outdir,
    targets = args$targets,
    style = style,
    measures = measures,
    prefer_rare = prefer
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
