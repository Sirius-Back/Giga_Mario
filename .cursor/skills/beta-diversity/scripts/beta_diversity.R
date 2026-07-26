#!/usr/bin/env Rscript
# beta-diversity — Aitchison + wUniFrac PCoA + PERMANOVA on plot (rarefied-preferred)
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("vegan", quietly = TRUE))
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

DEFAULT_DISTANCES <- c("aitchison", "wunifrac")
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
        "Beta diversity requires count-like abundances; median sample_sum=",
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

  if ("target" %in% names(sam) && !"target" %in% tg) {
    vals <- unique(as.character(sam$target))
    vals <- vals[!is.na(vals) & nzchar(vals)]
    if (length(vals) == 1L && vals %in% names(sam)) {
      tg <- unique(c(tg, vals))
    }
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
  if (length(missing)) fail("Target column(s) missing from sample_data: ", paste(missing, collapse = ", "))
  if (!length(tg)) fail("No target variables resolved; pass --targets")
  for (t in tg) {
    if (length(unique(stats::na.omit(sam[[t]]))) < 2L) fail("Target '", t, "' has <2 levels")
  }
  tg
}

ps_relative <- function(ps) {
  transform_sample_counts(ps, function(x) {
    s <- sum(x)
    if (s <= 0) fail("Zero-sum sample in relative transform")
    x / s
  })
}

ps_clr <- function(ps_rel) {
  if (requireNamespace("microbiome", quietly = TRUE)) {
    return(microbiome::transform(ps_rel, "clr"))
  }
  # manual CLR with half-min pseudocount for zeros
  otu <- as(otu_table(ps_rel), "matrix")
  tar <- taxa_are_rows(ps_rel)
  if (!tar) otu <- t(otu)
  # features × samples
  pos <- otu[otu > 0]
  pc <- if (length(pos)) min(pos) / 2 else 1e-6
  otu2 <- otu + pc
  clr <- apply(otu2, 2, function(x) log(x) - mean(log(x)))
  otu_table(ps_rel) <- otu_table(clr, taxa_are_rows = TRUE)
  ps_rel
}

normalize_distance_name <- function(d) {
  d <- tolower(trimws(d))
  aliases <- c(
    wunifrac = "wunifrac", `weighted-unifrac` = "wunifrac",
    `weighted_unifrac` = "wunifrac", `weighted unifrac` = "wunifrac",
    aitchison = "aitchison", clr = "aitchison",
    bray = "bray", `bray-curtis` = "bray",
    unifrac = "unifrac", jaccard = "jaccard"
  )
  if (!d %in% names(aliases)) fail("Unknown distance: ", d)
  unname(aliases[[d]])
}

compute_distance <- function(ps, dist_name, ps_rel = NULL, ps_clr_obj = NULL) {
  dn <- normalize_distance_name(dist_name)
  if (is.null(ps_rel)) ps_rel <- ps_relative(ps)

  if (identical(dn, "aitchison")) {
    if (is.null(ps_clr_obj)) ps_clr_obj <- ps_clr(ps_rel)
    return(list(
      name = "aitchison",
      label = "Aitchison (CLR + Euclidean)",
      dist = phyloseq::distance(ps_clr_obj, method = "euclidean"),
      ps_ord = ps_clr_obj
    ))
  }
  if (identical(dn, "wunifrac")) {
    tr <- phy_tree(ps_rel, errorIfNULL = FALSE)
    if (is.null(tr)) fail("wunifrac requires a phylogenetic tree in the phyloseq object")
    return(list(
      name = "wunifrac",
      label = "Weighted UniFrac",
      dist = phyloseq::distance(ps_rel, method = "wunifrac"),
      ps_ord = ps_rel
    ))
  }
  if (identical(dn, "unifrac")) {
    tr <- phy_tree(ps_rel, errorIfNULL = FALSE)
    if (is.null(tr)) fail("unifrac requires a phylogenetic tree")
    return(list(
      name = "unifrac",
      label = "Unweighted UniFrac",
      dist = phyloseq::distance(ps_rel, method = "unifrac"),
      ps_ord = ps_rel
    ))
  }
  if (identical(dn, "bray")) {
    return(list(
      name = "bray",
      label = "Bray–Curtis",
      dist = phyloseq::distance(ps_rel, method = "bray"),
      ps_ord = ps_rel
    ))
  }
  if (identical(dn, "jaccard")) {
    return(list(
      name = "jaccard",
      label = "Jaccard",
      dist = phyloseq::distance(ps_rel, method = "jaccard"),
      ps_ord = ps_rel
    ))
  }
  fail("Unhandled distance: ", dn)
}

run_permanova <- function(dist_obj, grouping, permutations = 999L) {
  df <- data.frame(group = as.factor(grouping), stringsAsFactors = TRUE)
  rownames(df) <- labels(dist_obj)
  # align order
  df <- df[labels(dist_obj), , drop = FALSE]
  if (nlevels(df$group) < 2L) {
    return(list(p = NA_real_, r2 = NA_real_, table = NULL))
  }
  fit <- vegan::adonis2(dist_obj ~ group, data = df, permutations = as.integer(permutations))
  list(
    p = unname(fit$`Pr(>F)`[[1]]),
    r2 = unname(fit$R2[[1]]),
    F = unname(fit$F[[1]]),
    table = fit
  )
}

pcoa_df <- function(ps_ord, dist_obj, target) {
  ord <- ordinate(ps_ord, method = "PCoA", distance = dist_obj)
  eig <- ord$values$Eigenvalues
  # use Relative_eig if present
  if (!is.null(ord$values$Relative_eig)) {
    var_exp <- ord$values$Relative_eig * 100
  } else {
    var_exp <- eig / sum(pmax(eig, 0)) * 100
  }
  coords <- as.data.frame(ord$vectors[, 1:2, drop = FALSE])
  colnames(coords) <- c("Axis.1", "Axis.2")
  coords$Sample <- rownames(coords)
  sam <- as(sample_data(ps_ord), "data.frame")
  coords[[target]] <- as.character(sam[coords$Sample, target])
  list(
    coords = coords,
    var1 = var_exp[[1]],
    var2 = var_exp[[2]],
    ord = ord
  )
}

plot_beta_target <- function(panels, target, out_prefix, rare_depth = NA) {
  rows <- lapply(panels, function(pn) {
    d <- pn$coords
    d$Distance <- pn$label
    d$perm_plain <- sprintf(
      "PERMANOVA\np-value = %s\nR² = %s",
      formatC(pn$permanova$p, format = "fg", digits = 3),
      formatC(pn$permanova$r2, format = "fg", digits = 3)
    )
    d$perm_rich <- sprintf(
      "PERMANOVA<br>p-value = %s<br>R<sup>2</sup> = %s",
      formatC(pn$permanova$p, format = "fg", digits = 3),
      formatC(pn$permanova$r2, format = "fg", digits = 3)
    )
    d$facet <- sprintf("%s\nPCo1 %.1f%% · PCo2 %.1f%%", pn$label, pn$var1, pn$var2)
    d
  })
  df <- do.call(rbind, rows)
  df[[target]] <- factor(df[[target]])
  facet_levels <- unique(df$facet)
  df$facet <- factor(df$facet, levels = facet_levels)

  lab_pos <- unique(df[, c("facet", "perm_plain", "perm_rich"), drop = FALSE])
  lab_pos$facet <- factor(lab_pos$facet, levels = facet_levels)
  lab_pos$Axis.1 <- -Inf
  lab_pos$Axis.2 <- Inf

  subtitle <- paste0(
    "target=", target,
    if (!is.na(rare_depth)) paste0("; rarefied depth=", rare_depth) else "; input=raw/counts"
  )

  p <- ggplot(df, aes(x = Axis.1, y = Axis.2, color = .data[[target]])) +
    geom_point(size = 2.8, alpha = 0.9) +
    stat_ellipse(aes(group = .data[[target]]), geom = "path", level = 0.95, linewidth = 0.5, show.legend = FALSE) +
    facet_wrap(~facet, scales = "free") +
    labs(
      title = "Beta diversity (PCoA)",
      subtitle = subtitle,
      x = NULL, y = NULL, color = target
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "right", strip.text = element_text(size = 9))

  if (requireNamespace("ggtext", quietly = TRUE)) {
    p <- p + ggtext::geom_richtext(
      data = lab_pos,
      aes(x = Axis.1, y = Axis.2, label = perm_rich),
      inherit.aes = FALSE,
      hjust = -0.02, vjust = 1.05, size = 3.2,
      fill = NA, label.color = NA, color = "grey15"
    )
  } else {
    p <- p + geom_text(
      data = lab_pos,
      aes(x = Axis.1, y = Axis.2, label = perm_plain),
      inherit.aes = FALSE,
      hjust = -0.02, vjust = 1.05, size = 3, lineheight = 0.95, color = "grey15"
    )
  }

  pdf_path <- paste0(out_prefix, ".pdf")
  png_path <- paste0(out_prefix, ".png")
  n_d <- length(panels)
  ggsave(pdf_path, p, width = max(8, 4.5 * n_d), height = 5)
  ggsave(png_path, p, width = max(8, 4.5 * n_d), height = 5, dpi = 150)
  list(pdf = pdf_path, png = png_path, plot = p)
}

run_beta <- function(rds = NULL, outdir, targets = NULL,
                     distances = DEFAULT_DISTANCES,
                     permutations = 999L, prefer_rare = TRUE) {
  ensure_dir(outdir)
  loaded <- resolve_input_rds(rds, prefer_rare = prefer_rare)
  ps <- loaded$ps
  if (any(sample_sums(ps) <= 0)) {
    fail("Samples with zero counts: ", paste(sample_names(ps)[sample_sums(ps) <= 0], collapse = ","))
  }

  tg <- resolve_targets(ps, loaded$meta, targets_arg = targets)
  distances <- vapply(distances, normalize_distance_name, character(1))
  distances <- unique(distances)

  message("Input: ", loaded$meta$path, " (rarefied=", loaded$meta$rarefied, ")")
  message("Targets: ", paste(tg, collapse = ", "))
  message("Distances: ", paste(distances, collapse = ", "))

  ps_rel <- ps_relative(ps)
  ps_clr_obj <- if ("aitchison" %in% distances) ps_clr(ps_rel) else NULL

  dist_cache <- list()
  for (d in distances) {
    dist_cache[[d]] <- compute_distance(ps, d, ps_rel = ps_rel, ps_clr_obj = ps_clr_obj)
  }

  sam_df <- as(sample_data(ps), "data.frame")
  perm_rows <- list()
  figs <- list()
  k <- 0L

  for (t in tg) {
    panels <- list()
    for (d in distances) {
      info <- dist_cache[[d]]
      # sample order for grouping
      labs_d <- labels(info$dist)
      grouping <- as.character(sam_df[labs_d, t])
      perm <- run_permanova(info$dist, grouping, permutations = permutations)
      pc <- pcoa_df(info$ps_ord, info$dist, t)
      panels[[d]] <- list(
        coords = pc$coords,
        var1 = pc$var1,
        var2 = pc$var2,
        label = info$label,
        distance = info$name,
        permanova = perm
      )
      k <- k + 1L
      perm_rows[[k]] <- data.frame(
        target = t,
        distance = info$name,
        distance_label = info$label,
        permutations = as.integer(permutations),
        F = perm$F %||% NA_real_,
        R2 = perm$r2,
        p = perm$p,
        stringsAsFactors = FALSE
      )
    }
    prefix <- file.path(outdir, paste0("beta_", t, "_pcoa"))
    figs[[t]] <- plot_beta_target(
      panels, t, out_prefix = prefix,
      rare_depth = loaded$meta$rarefaction_depth
    )
    message("Plot: ", figs[[t]]$png)
  }

  perm_df <- do.call(rbind, perm_rows)
  utils::write.csv(perm_df, file.path(outdir, "beta_permanova.tsv"), row.names = FALSE)

  report <- list(
    input_rds = loaded$meta$path,
    rarefied = loaded$meta$rarefied,
    rarefaction_depth = loaded$meta$rarefaction_depth,
    notes = loaded$notes,
    targets = tg,
    distances = distances,
    permutations = as.integer(permutations),
    n_samples = nsamples(ps),
    n_taxa = ntaxa(ps),
    has_tree = !is.null(phy_tree(ps, errorIfNULL = FALSE)),
    permanova_tsv = file.path(outdir, "beta_permanova.tsv"),
    figures = lapply(figs, function(f) list(pdf = f$pdf, png = f$png)),
    permanova = perm_df
  )
  write_json(report, file.path(outdir, "beta-diversity-report.json"))
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  out <- "test/beta-diversity/grazing-self-test"
  rep <- run_beta(
    rds = NULL,
    outdir = out,
    distances = DEFAULT_DISTANCES,
    permutations = 199L,
    prefer_rare = TRUE
  )
  if (!isTRUE(rep$rarefied)) stop("expected rarefied input when available")
  if (!all(c("aitchison", "wunifrac") %in% rep$distances)) stop("default distances missing")
  png <- rep$figures[[rep$targets[[1]]]]$png
  if (!file.exists(png)) stop("missing plot png")
  if (!file.exists(rep$permanova_tsv)) stop("missing permanova tsv")
  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/beta-diversity/run"
  prefer <- !identical(tolower(as.character(args$prefer_rare %||% "true")), "false")
  permutations <- if (!is.null(args$permutations)) as.integer(args$permutations) else 999L
  distances <- if (!is.null(args$distances) && nzchar(args$distances)) {
    trimws(strsplit(args$distances, ",", fixed = TRUE)[[1]])
  } else {
    DEFAULT_DISTANCES
  }

  run_beta(
    rds = args$rds,
    outdir = outdir,
    targets = args$targets,
    distances = distances,
    permutations = permutations,
    prefer_rare = prefer
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
