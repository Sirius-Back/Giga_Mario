#!/usr/bin/env Rscript
# ordination — sPLS-DA (default, rarefied) or NMDS + envfit
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
source(file.path(root, ".cursor/skills/_shared/import/taxon_format.R"))

SKIP_VARS <- c("seq", "ID", "SampleID", "sampleID", "sample_id", "Run", "run")

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path, target = NA_character_, batch = NA_character_,
    rarefaction_depth = NA_real_, abundances = NA_character_, rarefied = FALSE
  )
  if (inherits(obj, "phyloseq")) return(list(ps = obj, meta = meta))
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

resolve_input_rds <- function(rds = NULL, prefer_rare = TRUE, require_rare = FALSE) {
  notes <- character(0)
  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"

  try_path <- function(p, mark_rare = FALSE) {
    if (!file.exists(p)) return(NULL)
    loaded <- load_ps(p)
    if (!is_count_like(loaded$ps)) return(NULL)
    if (mark_rare) loaded$meta$rarefied <- TRUE
    loaded$meta$rarefied <- loaded$meta$rarefied ||
      grepl("_rare|phyloseq_rare_", basename(p))
    loaded
  }

  if (!is.null(rds) && nzchar(rds)) {
    loaded <- try_path(rds)
    if (is.null(loaded)) fail("Unusable RDS (missing or not count-like): ", rds)
    if (require_rare && !isTRUE(loaded$meta$rarefied)) {
      # if user passed raw but rare companion exists, prefer rare
      alt <- sub("\\.rds$", "_rare.rds", rds)
      alt2 <- sub("_batchadj\\.rds$", "_rare.rds", rds)
      for (a in unique(c(alt, alt2, candidates_rare))) {
        hit <- try_path(a, mark_rare = TRUE)
        if (!is.null(hit) && isTRUE(hit$meta$rarefied)) {
          hit$notes <- c(notes, paste0("sPLS-DA requires rarefied; switched to ", a))
          return(hit)
        }
      }
      fail("sPLS-DA requires a rarefied phyloseq object; none found beside ", rds)
    }
    loaded$notes <- notes
    return(loaded)
  }

  if (prefer_rare || require_rare) {
    for (p in candidates_rare) {
      hit <- try_path(p, mark_rare = TRUE)
      if (!is.null(hit)) {
        hit$notes <- c(notes, paste0("Using rarefied object: ", p))
        return(hit)
      }
    }
    rare_dir <- "test/rarefaction-analysis"
    if (dir.exists(rare_dir)) {
      hits <- sort(Sys.glob(file.path(rare_dir, "**", "phyloseq_rare_*.rds")), decreasing = TRUE)
      hits <- hits[!grepl("_plain\\.rds$", hits)]
      for (p in hits) {
        hit <- try_path(p, mark_rare = TRUE)
        if (!is.null(hit)) {
          hit$notes <- c(notes, paste0("Using rarefied object: ", p))
          return(hit)
        }
      }
    }
    if (require_rare) fail("sPLS-DA requires rarefied phyloseq; none found under test/")
    notes <- c(notes, "No rarefied object found — falling back to raw counts")
  }

  if (!file.exists(raw)) fail("No --rds and default grazing count RDS missing: ", raw)
  loaded <- try_path(raw)
  if (is.null(loaded)) fail("Raw RDS unusable: ", raw)
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
  if ("target" %in% names(sam) && !"target" %in% tg) {
    vals <- unique(as.character(sam$target))
    vals <- vals[!is.na(vals) & nzchar(vals)]
    if (length(vals) == 1L && vals %in% names(sam)) tg <- unique(c(tg, vals))
  }
  if (!length(tg)) {
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
  if (length(missing)) fail("Target column(s) missing: ", paste(missing, collapse = ", "))
  if (!length(tg)) fail("No target variables resolved; pass --targets")
  for (t in tg) {
    if (length(unique(stats::na.omit(sam[[t]]))) < 2L) fail("Target '", t, "' has <2 levels")
  }
  tg
}

resolve_batch <- function(ps, meta, batch_arg = NULL) {
  sam <- as(sample_data(ps), "data.frame")
  b <- if (!is.null(batch_arg) && nzchar(batch_arg)) batch_arg else meta$batch
  if (is.na(b) || !nzchar(as.character(b))) {
    if ("batch" %in% names(sam)) b <- "batch" else return(NA_character_)
  }
  if (!b %in% names(sam)) fail("Batch column missing: ", b)
  as.character(b)
}

otu_samples_by_taxa <- function(ps) {
  otu <- as(otu_table(ps), "matrix")
  if (taxa_are_rows(ps)) otu <- t(otu)
  storage.mode(otu) <- "double"
  # drop zero-variance taxa
  keep <- apply(otu, 2, function(z) stats::sd(z) > 0)
  if (!any(keep)) fail("No taxa with positive variance")
  otu[, keep, drop = FALSE]
}

taxon_label <- function(ps, otu_ids) {
  taxon_plot_labels_from_ps(ps, otu_ids)$display
}

taxon_label_markdown <- function(ps, otu_ids) {
  taxon_plot_labels_from_ps(ps, otu_ids)$markdown
}

save_plot <- function(p, prefix, width = 7, height = 5) {
  pdf <- paste0(prefix, ".pdf")
  png <- paste0(prefix, ".png")
  ggsave(pdf, p, width = width, height = height)
  ggsave(png, p, width = width, height = height, dpi = 150)
  list(pdf = pdf, png = png)
}

# ---- sPLS-DA ----

prediction_rects <- function(fit, xlim, ylim, resolution = 80L) {
  stopifnot(requireNamespace("mixOmics", quietly = TRUE))
  xs <- seq(xlim[1], xlim[2], length.out = resolution)
  ys <- seq(ylim[1], ylim[2], length.out = resolution)
  grid <- expand.grid(comp1 = xs, comp2 = ys, KEEP.OUT.ATTRS = FALSE)
  # mixOmics predict on dummy X is awkward; use background.predict polygons → rasterize via nearest class
  bg <- mixOmics::background.predict(
    fit, comp.predicted = 2, dist = "max.dist",
    xlim = xlim, ylim = ylim, resolution = resolution
  )
  # Build rect grid: assign each cell by nearest background point class
  dx <- diff(xs)[1]
  dy <- diff(ys)[1]
  classes <- names(bg)
  # For each grid cell center, find owning class from bg point clouds (point-in? nearest)
  pts <- do.call(rbind, lapply(classes, function(cl) {
    m <- bg[[cl]]
    if (is.null(m) || !nrow(m)) return(NULL)
    data.frame(x = m[, 1], y = m[, 2], class = cl, stringsAsFactors = FALSE)
  }))
  if (is.null(pts) || !nrow(pts)) fail("background.predict returned empty regions")

  # Map each cell center to nearest bg point's class
  centers <- expand.grid(x = xs, y = ys, KEEP.OUT.ATTRS = FALSE)
  # subsample if huge — resolution^2 is fine at 80
  cls <- character(nrow(centers))
  for (i in seq_len(nrow(centers))) {
    d2 <- (pts$x - centers$x[i])^2 + (pts$y - centers$y[i])^2
    cls[i] <- pts$class[which.min(d2)]
  }
  data.frame(
    xmin = centers$x - dx / 2,
    xmax = centers$x + dx / 2,
    ymin = centers$y - dy / 2,
    ymax = centers$y + dy / 2,
    class = cls,
    stringsAsFactors = FALSE
  )
}

run_splsda_target <- function(ps, target, outdir, keepX = c(10L, 10L), seed = 123L,
                              rare_depth = NA) {
  stopifnot(requireNamespace("mixOmics", quietly = TRUE))
  X <- otu_samples_by_taxa(ps)
  sam <- as(sample_data(ps), "data.frame")
  Y <- factor(as.character(sam[rownames(X), target]))
  names(Y) <- rownames(X)
  if (nlevels(Y) < 2L) fail("sPLS-DA target '", target, "' needs ≥2 levels")

  set.seed(as.integer(seed))
  fit <- mixOmics::splsda(X, Y, ncomp = 2, keepX = as.integer(keepX), scale = TRUE)

  scores <- data.frame(
    Sample = rownames(fit$variates$X),
    comp1 = fit$variates$X[, 1],
    comp2 = fit$variates$X[, 2],
    group = as.character(Y[rownames(fit$variates$X)]),
    stringsAsFactors = FALSE
  )
  scores$group <- factor(scores$group, levels = levels(Y))

  pad <- 0.08
  xr <- range(scores$comp1)
  yr <- range(scores$comp2)
  xlim <- xr + c(-1, 1) * diff(xr) * pad
  ylim <- yr + c(-1, 1) * diff(yr) * pad
  rects <- prediction_rects(fit, xlim, ylim, resolution = 70L)

  subtitle <- paste0(
    "target=", target, "; method=sPLS-DA",
    if (!is.na(rare_depth)) paste0("; rarefied depth=", rare_depth) else ""
  )

  p_main <- ggplot() +
    geom_rect(
      data = rects,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = class),
      alpha = 0.22, color = NA
    ) +
    geom_point(
      data = scores,
      aes(x = comp1, y = comp2, color = group),
      size = 2.8
    ) +
    labs(
      title = "sPLS-DA ordination",
      subtitle = subtitle,
      x = "sPLS-DA component 1",
      y = "sPLS-DA component 2",
      color = target,
      fill = "region"
    ) +
    theme_bw(base_size = 11) +
    coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE)

  figs <- list()
  figs$main <- save_plot(p_main, file.path(outdir, paste0("splsda_", target, "_main")), 7, 5.2)

  loadings <- as.data.frame(fit$loadings$X)
  loadings$OTU <- rownames(loadings)
  lab_df <- taxon_plot_labels_from_ps(ps, loadings$OTU)
  loadings$label <- lab_df$display[match(loadings$OTU, lab_df$otu)]
  loadings$label_md <- lab_df$markdown[match(loadings$OTU, lab_df$otu)]
  loadings$fontface <- lab_df$fontface[match(loadings$OTU, lab_df$otu)]

  for (comp in 1:2) {
    col <- colnames(fit$loadings$X)[comp]
    d <- data.frame(
      OTU = loadings$OTU,
      label = loadings$label,
      label_md = loadings$label_md,
      fontface = loadings$fontface,
      loading = loadings[[col]],
      stringsAsFactors = FALSE
    )
    d <- d[abs(d$loading) > 0, , drop = FALSE]
    if (!nrow(d)) {
      d <- head(data.frame(
        OTU = loadings$OTU, label = loadings$label, label_md = loadings$label_md,
        fontface = loadings$fontface, loading = loadings[[col]],
        stringsAsFactors = FALSE
      )[order(-abs(loadings[[col]])), ], keepX[comp])
    }
    d$OTU <- factor(d$OTU, levels = d$OTU[order(d$loading)])
    p_ld <- ggplot(d, aes(x = OTU, y = loading, fill = loading > 0)) +
      geom_col(show.legend = FALSE) +
      coord_flip() +
      labs(
        title = paste0("sPLS-DA loadings - component ", comp),
        subtitle = subtitle,
        x = NULL,
        y = "Loading"
      ) +
      theme_bw(base_size = 11)
    if (requireNamespace("ggtext", quietly = TRUE)) {
      p_ld <- p_ld +
        scale_x_discrete(labels = setNames(d$label_md, as.character(d$OTU))) +
        theme(axis.text.y = ggtext::element_markdown(size = 8))
    } else {
      p_ld <- p_ld +
        scale_x_discrete(labels = setNames(d$label, as.character(d$OTU))) +
        theme(axis.text.y = element_text(size = 8))
    }
    figs[[paste0("loading", comp)]] <- save_plot(
      p_ld,
      file.path(outdir, paste0("splsda_", target, "_loading", comp)),
      width = 7, height = max(4, 0.28 * nrow(d) + 2)
    )
  }

  list(fit = fit, scores = scores, figures = figs, keepX = keepX)
}

# ---- NMDS ----

run_nmds_target <- function(ps, target, batch_var, outdir, top_features = 5L,
                            permutations = 999L, seed = 123L, rare_depth = NA) {
  stopifnot(requireNamespace("vegan", quietly = TRUE))
  X <- otu_samples_by_taxa(ps)
  # relative
  Xrel <- X / rowSums(X)
  sam <- as(sample_data(ps), "data.frame")
  sam <- sam[rownames(Xrel), , drop = FALSE]

  set.seed(as.integer(seed))
  nmds <- vegan::metaMDS(Xrel, distance = "bray", k = 2, trymax = 100, autotransform = FALSE)

  scores <- as.data.frame(vegan::scores(nmds, display = "sites"))
  colnames(scores) <- c("NMDS1", "NMDS2")
  scores$Sample <- rownames(scores)
  scores$group <- as.character(sam[scores$Sample, target])

  env <- data.frame(row.names = rownames(Xrel))
  env[[target]] <- factor(as.character(sam[[target]]))
  if (!is.na(batch_var) && nzchar(batch_var) && batch_var %in% names(sam)) {
    env[[batch_var]] <- factor(as.character(sam[[batch_var]]))
  }

  ef_meta <- vegan::envfit(nmds, env, permutations = as.integer(permutations), na.rm = TRUE)

  # Feature arrows: envfit on most variable taxa, then top_features by r²
  vars <- apply(Xrel, 2, stats::var)
  n_cand <- min(ncol(Xrel), max(30L, as.integer(top_features) * 6L))
  cand <- names(sort(vars, decreasing = TRUE))[seq_len(n_cand)]
  ef_taxa <- vegan::envfit(
    nmds, Xrel[, cand, drop = FALSE],
    permutations = as.integer(permutations), na.rm = TRUE
  )

  env_rows <- list()
  k <- 0L
  if (!is.null(ef_meta$factors) && length(ef_meta$factors$r)) {
    for (nm in names(ef_meta$factors$r)) {
      k <- k + 1L
      env_rows[[k]] <- data.frame(
        type = "factor",
        variable = nm,
        NMDS1 = NA_real_, NMDS2 = NA_real_,
        r2 = unname(ef_meta$factors$r[[nm]]),
        p = unname(ef_meta$factors$pvals[[nm]]),
        stringsAsFactors = FALSE
      )
    }
  }
  if (!is.null(ef_taxa$vectors) && length(ef_taxa$vectors$r)) {
    ord <- order(ef_taxa$vectors$r, decreasing = TRUE)
    top_i <- ord[seq_len(min(as.integer(top_features), length(ord)))]
    arrows <- as.data.frame(ef_taxa$vectors$arrows[top_i, , drop = FALSE])
    arrows$variable <- rownames(arrows)
    arrows$r2 <- unname(ef_taxa$vectors$r[top_i])
    arrows$p <- unname(ef_taxa$vectors$pvals[top_i])
    lab_df <- taxon_plot_labels_from_ps(ps, arrows$variable)
    arrows$label <- lab_df$display[match(arrows$variable, lab_df$otu)]
    # scale arrows to plot
    mul <- 0.45 * max(abs(c(scores$NMDS1, scores$NMDS2))) /
      max(1e-8, max(abs(c(arrows$NMDS1, arrows$NMDS2))))
    arrows$xend <- arrows$NMDS1 * mul * sqrt(arrows$r2)
    arrows$yend <- arrows$NMDS2 * mul * sqrt(arrows$r2)
    for (i in seq_len(nrow(arrows))) {
      k <- k + 1L
      env_rows[[k]] <- data.frame(
        type = "feature",
        variable = arrows$variable[[i]],
        NMDS1 = arrows$NMDS1[[i]],
        NMDS2 = arrows$NMDS2[[i]],
        r2 = arrows$r2[[i]],
        p = arrows$p[[i]],
        stringsAsFactors = FALSE
      )
    }
  } else {
    arrows <- data.frame()
  }

  env_df <- if (length(env_rows)) do.call(rbind, env_rows) else data.frame()
  utils::write.csv(env_df, file.path(outdir, paste0("nmds_", target, "_envfit.tsv")), row.names = FALSE)

  subtitle <- paste0(
    "target=", target,
    if (!is.na(batch_var)) paste0("; batch=", batch_var) else "",
    "; stress=", signif(nmds$stress, 3),
    if (!is.na(rare_depth)) paste0("; rarefied depth=", rare_depth) else "; input=raw/counts"
  )

  # factor centroids for batch/target annotation in caption via envfit table
  p <- ggplot(scores, aes(x = NMDS1, y = NMDS2, color = group)) +
    geom_point(size = 2.8) +
    stat_ellipse(aes(group = group), level = 0.95, linewidth = 0.45, show.legend = FALSE)

  if (nrow(arrows)) {
    p <- p +
      geom_segment(
        data = arrows,
        aes(x = 0, y = 0, xend = xend, yend = yend),
        inherit.aes = FALSE,
        arrow = arrow(length = grid::unit(0.18, "cm")),
        color = "grey25", linewidth = 0.45
      )
    if (requireNamespace("ggrepel", quietly = TRUE)) {
      p <- p + ggrepel::geom_text_repel(
        data = arrows,
        aes(x = xend, y = yend, label = label),
        inherit.aes = FALSE,
        size = 3, max.overlaps = 20, color = "grey10"
      )
    } else {
      p <- p + geom_text(
        data = arrows,
        aes(x = xend, y = yend, label = label),
        inherit.aes = FALSE, size = 3, color = "grey10", vjust = -0.4
      )
    }
  }

  meta_lab <- env_df[env_df$type == "factor", , drop = FALSE]
  if (nrow(meta_lab)) {
    txt <- paste0(
      meta_lab$variable, ": R²=", signif(meta_lab$r2, 3),
      ", p=", signif(meta_lab$p, 3),
      collapse = "\n"
    )
    p <- p + annotate(
      "text", x = -Inf, y = Inf, label = paste0("envfit\n", txt),
      hjust = -0.05, vjust = 1.1, size = 3, color = "grey20"
    )
  }

  p <- p +
    labs(
      title = "NMDS (Bray-Curtis)",
      subtitle = subtitle,
      color = target
    ) +
    theme_bw(base_size = 11)

  fig <- save_plot(p, file.path(outdir, paste0("nmds_", target)), 7.5, 5.5)
  list(
    nmds = nmds,
    envfit_meta = ef_meta,
    envfit_features = ef_taxa,
    envfit_table = env_df,
    figures = list(main = fig),
    stress = nmds$stress
  )
}

run_ordination <- function(rds = NULL, outdir, method = c("splsda", "nmds"),
                           targets = NULL, batch_var = NULL,
                           keepX = c(10L, 10L), top_features = 5L,
                           permutations = 999L, seed = 123L,
                           prefer_rare = TRUE) {
  ensure_dir(outdir)
  method <- match.arg(method)
  require_rare <- identical(method, "splsda")
  loaded <- resolve_input_rds(rds, prefer_rare = prefer_rare, require_rare = require_rare)
  ps <- loaded$ps
  if (require_rare && !isTRUE(loaded$meta$rarefied)) {
    fail("sPLS-DA requires rarefied object; input is not rarefied: ", loaded$meta$path)
  }
  if (any(sample_sums(ps) <= 0)) {
    fail("Zero-count samples: ", paste(sample_names(ps)[sample_sums(ps) <= 0], collapse = ","))
  }

  tg <- resolve_targets(ps, loaded$meta, targets_arg = targets)
  batch <- resolve_batch(ps, loaded$meta, batch_arg = batch_var)

  message("Method: ", method)
  message("Input: ", loaded$meta$path, " (rarefied=", loaded$meta$rarefied, ")")
  message("Targets: ", paste(tg, collapse = ", "))

  results <- list()
  figures <- list()

  for (t in tg) {
    if (identical(method, "splsda")) {
      one <- run_splsda_target(
        ps, t, outdir, keepX = keepX, seed = seed,
        rare_depth = loaded$meta$rarefaction_depth
      )
      results[[t]] <- list(keepX = keepX)
      figures[[t]] <- one$figures
      message("sPLS-DA plots: ", one$figures$main$png)
    } else {
      one <- run_nmds_target(
        ps, t, batch, outdir,
        top_features = top_features,
        permutations = permutations,
        seed = seed,
        rare_depth = loaded$meta$rarefaction_depth
      )
      results[[t]] <- list(stress = one$stress, envfit = one$envfit_table)
      figures[[t]] <- one$figures
      message("NMDS plot: ", one$figures$main$png)
    }
  }

  report <- list(
    method = method,
    input_rds = loaded$meta$path,
    rarefied = loaded$meta$rarefied,
    rarefaction_depth = loaded$meta$rarefaction_depth,
    notes = loaded$notes,
    targets = tg,
    batch = batch,
    keepX = if (identical(method, "splsda")) as.integer(keepX) else NULL,
    top_features = if (identical(method, "nmds")) as.integer(top_features) else NULL,
    permutations = as.integer(permutations),
    seed = as.integer(seed),
    n_samples = nsamples(ps),
    n_taxa = ntaxa(ps),
    figures = lapply(figures, function(f) {
      lapply(f, function(x) list(pdf = x$pdf, png = x$png))
    }),
    results = results
  )
  write_json(report, file.path(outdir, "ordination-report.json"))
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  out1 <- "test/ordination/grazing-self-test-splsda"
  rep1 <- run_ordination(
    rds = NULL, outdir = out1, method = "splsda",
    keepX = c(8L, 8L), seed = 123L
  )
  if (!isTRUE(rep1$rarefied)) stop("sPLS-DA expected rarefied")
  if (!file.exists(rep1$figures[[rep1$targets[[1]]]]$main$png)) stop("missing sPLS-DA main")
  if (!file.exists(rep1$figures[[rep1$targets[[1]]]]$loading1$png)) stop("missing loading1")
  if (!file.exists(rep1$figures[[rep1$targets[[1]]]]$loading2$png)) stop("missing loading2")

  out2 <- "test/ordination/grazing-self-test-nmds"
  rep2 <- run_ordination(
    rds = "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    outdir = out2, method = "nmds",
    top_features = 5L, permutations = 99L, seed = 123L
  )
  if (!file.exists(rep2$figures[[rep2$targets[[1]]]]$main$png)) stop("missing NMDS plot")
  message("SELF-TEST OK")
  invisible(list(splsda = rep1, nmds = rep2))
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/ordination/run"
  method <- tolower(args$method %||% "splsda")
  if (method %in% c("spls-da", "spls_da", "plsda")) method <- "splsda"
  prefer <- !identical(tolower(as.character(args$prefer_rare %||% "true")), "false")
  seed <- if (!is.null(args$seed)) as.integer(args$seed) else 123L
  top_features <- if (!is.null(args$top_features)) as.integer(args$top_features) else 5L
  permutations <- if (!is.null(args$permutations)) as.integer(args$permutations) else 999L
  keepX <- c(10L, 10L)
  if (!is.null(args$keepX) && nzchar(args$keepX)) {
    keepX <- as.integer(strsplit(args$keepX, ",", fixed = TRUE)[[1]])
    if (length(keepX) == 1L) keepX <- c(keepX, keepX)
  }

  run_ordination(
    rds = args$rds,
    outdir = outdir,
    method = method,
    targets = args$targets,
    batch_var = args$batch_var,
    keepX = keepX,
    top_features = top_features,
    permutations = permutations,
    seed = seed,
    prefer_rare = prefer
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
