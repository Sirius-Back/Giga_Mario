#!/usr/bin/env Rscript
# difftree-ggtree — rarefied phy_tree + required ancombc LFC via ggtreeExtra::geom_fruit
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ape", quietly = TRUE))
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
  stopifnot(requireNamespace("ggtree", quietly = TRUE))
  stopifnot(requireNamespace("ggtreeExtra", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  library(ggplot2)
  library(ggtree)
  library(ggtreeExtra)
})
# ggplot2 ≥4 renamed is.waive → is_waiver; ggtree still calls is.waive
is.waive <- ggplot2::is_waiver

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

LFC_LOW <- "#1B9E77"
LFC_MID <- "gray80"
LFC_HIGH <- "#D81B60"
DEFAULT_MAX_TIPS <- 80L
DEFAULT_LFC_CUT <- 0.5
DEFAULT_Q_CUT <- 0.05

#' Plain data.frame from phyloseq sample_data (avoids sample_data `[` deparse bugs).
plain_sample_df <- function(ps) {
  sd <- as(sample_data(ps), "data.frame")
  out <- as.data.frame(sd, stringsAsFactors = FALSE, check.names = FALSE)
  class(out) <- "data.frame"
  rownames(out) <- rownames(sd)
  out
}

#' Safe character vector of a sample_data column, keyed by sample name.
sample_group_chr <- function(ps, group_col) {
  sd <- plain_sample_df(ps)
  if (!group_col %in% names(sd)) {
    fail("sample_data missing column: ", group_col)
  }
  v <- as.character(sd[[group_col]])
  names(v) <- rownames(sd)
  v
}

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path, target = NA_character_, batch = NA_character_,
    rarefaction_depth = NA_real_, rarefied = FALSE
  )
  if (inherits(obj, "phyloseq")) {
    meta$rarefied <- grepl("_rare(\\.rds)?$|phyloseq_rare_", basename(path))
    return(list(ps = obj, meta = meta))
  }
  if (is.list(obj) && !is.null(obj$phyloseq) && inherits(obj$phyloseq, "phyloseq")) {
    meta$target <- obj$target %||% NA_character_
    meta$batch <- obj$batch %||% NA_character_
    meta$rarefaction_depth <- obj$rarefaction_depth %||% NA_real_
    meta$rarefied <- !is.null(obj$rarefaction_depth) ||
      grepl("_rare|phyloseq_rare_", basename(path))
    return(list(ps = obj$phyloseq, meta = meta))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

is_count_like <- function(ps) {
  stats::median(as.numeric(sample_sums(ps))) >= 10
}

resolve_rare_ps <- function(rds = NULL, allow_non_rare = FALSE) {
  notes <- character(0)
  candidates <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  try_one <- function(p) {
    if (!file.exists(p)) return(NULL)
    loaded <- load_ps(p)
    if (!is_count_like(loaded$ps)) return(NULL)
    loaded$meta$rarefied <- isTRUE(loaded$meta$rarefied) ||
      grepl("_rare|phyloseq_rare_", basename(p))
    loaded
  }
  if (!is.null(rds) && nzchar(rds)) {
    loaded <- try_one(rds)
    if (is.null(loaded)) fail("Unusable phyloseq RDS: ", rds)
    if (!isTRUE(loaded$meta$rarefied) && !allow_non_rare) {
      for (a in unique(c(sub("\\.rds$", "_rare.rds", rds), candidates))) {
        hit <- try_one(a)
        if (!is.null(hit) && isTRUE(hit$meta$rarefied)) {
          hit$notes <- c(notes, paste0("Switched to rarefied: ", a))
          return(hit)
        }
      }
      fail("difftree-ggtree requires rarefied phyloseq; got: ", rds)
    }
    loaded$notes <- notes
    return(loaded)
  }
  for (p in candidates) {
    hit <- try_one(p)
    if (!is.null(hit) && isTRUE(hit$meta$rarefied)) {
      hit$notes <- c(notes, paste0("Using rarefied: ", p))
      return(hit)
    }
  }
  rare_dir <- "test/rarefaction-analysis"
  if (dir.exists(rare_dir)) {
    hits <- sort(Sys.glob(file.path(rare_dir, "**", "phyloseq_rare_*.rds")), decreasing = TRUE)
    hits <- hits[!grepl("_plain\\.rds$", hits)]
    for (p in hits) {
      hit <- try_one(p)
      if (!is.null(hit)) {
        hit$notes <- c(notes, paste0("Using rarefied: ", p))
        return(hit)
      }
    }
  }
  fail("No rarefied phyloseq found; run rarefaction-analysis or pass --rds")
}

ancombc_report_ok <- function(dir) {
  rep <- file.path(dir, "ancombc-report.json")
  if (!file.exists(rep)) return(list(ok = NA, path = rep, detail = "no report"))
  j <- tryCatch(jsonlite::fromJSON(rep, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(j)) return(list(ok = FALSE, path = rep, detail = "unreadable report"))
  summ <- j$level_summary
  if (is.null(summ) || !length(summ)) {
    return(list(ok = TRUE, path = rep, detail = "report present, no level_summary"))
  }
  oks <- vapply(summ, function(x) isTRUE(x$ok), logical(1))
  list(
    ok = all(oks),
    path = rep,
    detail = if (all(oks)) "all levels ok" else paste(
      "failed:",
      paste(vapply(summ[!oks], function(x) paste0(x$target, "/", x$level), character(1)),
            collapse = ",")
    )
  )
}

require_ancombc <- function(ancombc_csv = NULL, ancombc_dir = NULL) {
  notes <- character(0)
  if (!is.null(ancombc_csv) && nzchar(ancombc_csv)) {
    if (!file.exists(ancombc_csv)) fail("ANCOM-BC CSV not found: ", ancombc_csv)
    dir <- dirname(ancombc_csv)
    chk <- ancombc_report_ok(dir)
    if (isFALSE(chk$ok)) {
      fail("ANCOM-BC report not OK — will not re-run. ", chk$detail, " (", chk$path, ")")
    }
    return(list(path = ancombc_csv, dir = dir, report_ok = chk$ok, notes = notes))
  }

  dirs <- character(0)
  if (!is.null(ancombc_dir) && nzchar(ancombc_dir)) dirs <- c(dirs, ancombc_dir)
  dirs <- c(dirs, "test/ancombc/grazing", "test/ancombc/grazing-self-test")
  if (dir.exists("test/ancombc")) {
    dirs <- c(dirs, dirname(Sys.glob("test/ancombc/**/ancombc_results.csv")))
    dirs <- c(dirs, dirname(Sys.glob("test/ancombc/**/ancombc_results.tsv")))
  }
  dirs <- unique(dirs[nzchar(dirs) & !is.na(dirs) & dir.exists(dirs)])

  for (d in dirs) {
    candidates <- c(
      file.path(d, "ancombc_results.csv"),
      file.path(d, "ancombc2_results_all_levels.csv"),
      file.path(d, "ancombc_results.tsv")
    )
    hit <- candidates[file.exists(candidates)]
    if (!length(hit)) next
    chk <- ancombc_report_ok(d)
    if (isFALSE(chk$ok)) {
      fail("ANCOM-BC found but not OK — will not re-run. ", chk$detail, " (", chk$path, ")")
    }
    notes <- c(notes, paste0("Importing ANCOM-BC: ", hit[[1]]), chk$detail %||% "")
    return(list(path = hit[[1]], dir = d, report_ok = chk$ok, notes = notes))
  }
  fail(
    "difftree-ggtree requires previous ancombc results. ",
    "None found under test/ancombc/ (or --ancombc-dir). ",
    "Run @ancombc first; this skill will not re-run it."
  )
}

read_table_auto <- function(path) {
  if (!file.exists(path)) fail("Table not found: ", path)
  ext <- tolower(tools::file_ext(path))
  if (ext == "csv") {
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    utils::read.table(path, sep = "\t", header = TRUE, stringsAsFactors = FALSE,
                      check.names = FALSE)
  }
}

normalize_ancombc_long <- function(df) {
  names(df) <- tolower(names(df))
  need <- c("taxon", "level", "term", "lfc")
  if (!all(need %in% names(df))) {
    fail("ANCOM table missing columns: ", paste(setdiff(need, names(df)), collapse = ", "))
  }
  if (!"p" %in% names(df)) df$p <- NA_real_
  if (!"q" %in% names(df)) df$q <- NA_real_
  if (!"target" %in% names(df)) df$target <- "target"
  df[, c("target", "level", "taxon", "term", "lfc", "p", "q")]
}

resolve_tree <- function(ps) {
  tr <- phy_tree(ps, errorIfNULL = FALSE)
  if (!is.null(tr) && inherits(tr, "phylo") && length(tr$tip.label) >= 2L) {
    keep <- intersect(tr$tip.label, taxa_names(ps))
    if (length(keep) < 2L) fail("phy_tree tip overlap with OTUs < 2")
    tr <- ape::keep.tip(ape::as.phylo(tr), keep)
    return(list(tree = tr, source = "phy_tree"))
  }
  # Fallback from the same phyloseq object only: taxonomy formula tree (never invented distances)
  tax <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
  ranks <- setdiff(rank_names(ps), "tip_rank")
  if (length(ranks) < 2L) {
    fail(
      "No phy_tree in phyloseq and insufficient taxonomy ranks for formula tree. ",
      "Difftrees require a tree from the phyloseq object."
    )
  }
  tax$taxa_id <- rownames(tax)
  tr <- tryCatch(
    build_formula_or_hclust_tree(tax, tip_col = "taxa_id", abundance = NULL),
    error = function(e) NULL
  )
  if (is.null(tr) || !inherits(tr, "phylo") || length(tr$tip.label) < 2L) {
    # Last resort: ape formula on ranks (still from phyloseq tax_table)
    for (nm in c(ranks, "taxa_id")) {
      x <- as.character(tax[[nm]])
      x[is.na(x) | !nzchar(x)] <- "Unclassified"
      tax[[nm]] <- factor(x, levels = unique(x))
    }
    fml <- stats::as.formula(paste("~", paste(c(ranks, "taxa_id"), collapse = " / ")))
    tr <- ape::as.phylo(data = tax, formula = fml)
    # tip labels from formula are factor levels of taxa_id
    map <- setNames(as.character(tax$taxa_id), make.names(as.character(tax$taxa_id), unique = TRUE))
    tr$tip.label <- unname(ifelse(
      tr$tip.label %in% names(map), map[tr$tip.label],
      ifelse(tr$tip.label %in% as.character(tax$taxa_id), tr$tip.label, tr$tip.label)
    ))
  }
  keep <- intersect(tr$tip.label, taxa_names(ps))
  if (length(keep) < 2L) fail("Taxonomy formula tree tip overlap with OTUs < 2")
  tr <- ape::keep.tip(ape::as.phylo(tr), keep)
  list(tree = tr, source = "tax_formula")
}

build_tip_table <- function(ps, an_asv, tree, max_tips, lfc_cut, q_cut) {
  tips <- tree$tip.label
  lab <- taxon_plot_labels_from_ps(ps, tips)
  lab$id <- tips

  an_asv$log2_lfc <- as.numeric(an_asv$lfc) / log(2)
  an_asv$q <- as.numeric(an_asv$q)
  an_asv$p <- as.numeric(an_asv$p)

  m <- match(tips, an_asv$taxon)
  tip_df <- data.frame(
    id = tips,
    display = lab$display,
    fontface = lab$fontface,
    log2_lfc = ifelse(is.na(m), 0, an_asv$log2_lfc[m]),
    p = ifelse(is.na(m), NA_real_, an_asv$p[m]),
    q = ifelse(is.na(m), NA_real_, an_asv$q[m]),
    stringsAsFactors = FALSE
  )
  tip_df$abs_lfc <- abs(tip_df$log2_lfc)
  tip_df$sig_q <- !is.na(tip_df$q) & tip_df$q < q_cut
  tip_df$sig_lfc <- tip_df$abs_lfc > lfc_cut
  tip_df$keep <- tip_df$sig_q | tip_df$sig_lfc

  sel <- tip_df[tip_df$keep, , drop = FALSE]
  if (!nrow(sel)) {
    sel <- tip_df[order(-tip_df$abs_lfc), , drop = FALSE]
    sel <- utils::head(sel, max_tips)
    sel$keep <- TRUE
    message("No tips passed q/lfc cuts — using top ", nrow(sel), " by |log2_lfc|")
  } else if (nrow(sel) > max_tips) {
    sel <- sel[order(-sel$abs_lfc), , drop = FALSE]
    sel <- utils::head(sel, max_tips)
    message("Capped to max_tips=", max_tips, " by |log2_lfc|")
  }
  list(all = tip_df, selected = sel)
}

#' Tip table for a taxonomic rank from multilevel ANCOM-BC (Genus/Family/…).
build_tip_table_rank <- function(ps, an_rank, level, max_tips, lfc_cut, q_cut) {
  level_col <- colnames(tax_table(ps))[tolower(colnames(tax_table(ps))) == tolower(level)]
  if (!length(level_col)) {
    fail("tax_table missing rank column for level=", level)
  }
  level_col <- level_col[[1]]
  tt <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
  an_rank <- as.data.frame(an_rank, stringsAsFactors = FALSE)
  an_rank$log2_lfc <- as.numeric(an_rank$lfc) / log(2)
  an_rank$q <- as.numeric(an_rank$q)
  an_rank$p <- as.numeric(an_rank$p)
  an_rank$taxon <- as.character(an_rank$taxon)

  # Aggregate OTUs → rank tip; pick representative OTU per rank name
  tt$otu <- rownames(tt)
  tt$rank_name <- as.character(tt[[level_col]])
  tt <- tt[!is.na(tt$rank_name) & nzchar(tt$rank_name) &
             !grepl("^unclassified$", tt$rank_name, ignore.case = TRUE), , drop = FALSE]
  if (!nrow(tt)) fail("No classified taxa at level ", level)

  # Match ANCOM taxon to rank_name (exact / case-insensitive)
  an_rank$match_key <- tolower(an_rank$taxon)
  tt$match_key <- tolower(tt$rank_name)
  merged <- merge(
    unique(tt[, c("rank_name", "match_key", "otu")]),
    an_rank,
    by = "match_key",
    all.x = FALSE, all.y = FALSE
  )
  if (!nrow(merged)) {
    message("No ANCOM taxa matched tax_table at level=", level)
    return(list(all = merged, selected = merged))
  }
  # One row per rank tip
  merged <- merged[order(-abs(merged$log2_lfc)), , drop = FALSE]
  merged <- merged[!duplicated(merged$rank_name), , drop = FALSE]

  tip_df <- data.frame(
    id = merged$otu,
    display = merged$rank_name,
    fontface = ifelse(tolower(level) %in% c("genus", "species"), "italic", "plain"),
    log2_lfc = merged$log2_lfc,
    p = merged$p,
    q = merged$q,
    abs_lfc = abs(merged$log2_lfc),
    stringsAsFactors = FALSE
  )
  tip_df$sig_q <- !is.na(tip_df$q) & tip_df$q < q_cut
  tip_df$sig_lfc <- tip_df$abs_lfc > lfc_cut
  tip_df$keep <- tip_df$sig_q | tip_df$sig_lfc
  sel <- tip_df[tip_df$keep, , drop = FALSE]
  if (!nrow(sel)) {
    sel <- utils::head(tip_df[order(-tip_df$abs_lfc), , drop = FALSE], max_tips)
    sel$keep <- TRUE
  } else if (nrow(sel) > max_tips) {
    sel <- utils::head(sel[order(-sel$abs_lfc), , drop = FALSE], max_tips)
  }
  list(all = tip_df, selected = sel)
}

plot_diff_ggtree <- function(tree, tip_sel, layout = "circular",
                             title = NULL,
                             tip_offset = NULL,
                             fruit_offset = NULL,
                             fruit_pwidth = NULL) {
  keep <- as.character(tip_sel$id)
  if (length(keep) < 2L) fail("Need ≥2 tips for ggtree")
  tr <- ape::keep.tip(ape::as.phylo(tree), keep)

  disp <- as.character(tip_sel$display)
  disp[is.na(disp) | !nzchar(disp)] <- keep
  face <- as.character(tip_sel$fontface %||% "plain")
  face[is.na(face) | !nzchar(face)] <- "plain"
  # uniquify only colliding display names
  dup <- duplicated(disp) | duplicated(disp, fromLast = TRUE)
  if (any(dup)) {
    disp[dup] <- paste0(disp[dup], " [", substr(keep[dup], 1L, 8L), "]")
  }

  tip_meta <- data.frame(
    label = keep,
    display = disp,
    fontface = face,
    stringsAsFactors = FALSE
  )
  fruit <- data.frame(
    id = keep,
    log2_lfc = as.numeric(tip_sel$log2_lfc),
    stringsAsFactors = FALSE
  )
  stopifnot(all(fruit$id %in% tr$tip.label))

  lim <- max(abs(fruit$log2_lfc), na.rm = TRUE)
  if (!is.finite(lim) || lim <= 0) lim <- 1

  # Grazing Article.Rmd order: tree → tip labels → fruit bars (fruit offset=0 after tips).
  is_circ <- identical(layout, "circular")
  tip_offset <- if (!is.null(tip_offset)) {
    as.numeric(tip_offset)
  } else if (is_circ) {
    0.5
  } else {
    0.2
  }
  fruit_offset <- if (!is.null(fruit_offset)) {
    as.numeric(fruit_offset)
  } else if (is_circ) {
    0
  } else {
    0.05
  }
  fruit_pwidth <- if (!is.null(fruit_pwidth)) {
    as.numeric(fruit_pwidth)
  } else if (is_circ) {
    0.35
  } else {
    0.40
  }

  # tip_rank-aware faces (not all italic).
  # Use boolean subset: ggtree aes_string strips quotes from fontface == "italic".
  tip_meta$is_italic <- tip_meta$fontface == "italic"
  italic_lab <- tip_meta$is_italic
  if (is_circ) {
    p <- ggtree(
      tr, layout = layout, open.angle = 10, branch.length = "none"
    ) %<+% tip_meta
  } else {
    p <- ggtree(tr, layout = layout, branch.length = "none") %<+% tip_meta
  }
  if (any(italic_lab)) {
    p <- p + geom_tiplab(
      aes(label = display, subset = is_italic),
      size = 2.2, offset = tip_offset, align = TRUE, linesize = 0.1,
      fontface = "italic"
    )
  }
  if (any(!italic_lab)) {
    p <- p + geom_tiplab(
      aes(label = display, subset = !is_italic),
      size = 2.2, offset = tip_offset, align = TRUE, linesize = 0.1,
      fontface = "plain"
    )
  }
  p <- p +
    geom_fruit(
      data = fruit,
      geom = geom_col,
      mapping = aes(y = id, x = log2_lfc, fill = log2_lfc),
      offset = fruit_offset,
      pwidth = fruit_pwidth,
      size = 0.2,
      axis.params = list(
        axis = "x",
        text.size = 1.8,
        hjust = 1,
        vjust = 0.5,
        nbreak = 3
      ),
      grid.params = list()
    ) +
    scale_fill_gradient2(
      low = LFC_LOW, mid = LFC_MID, high = LFC_HIGH,
      midpoint = 0,
      limits = c(-lim, lim),
      name = "log2 LFC"
    ) +
    theme(legend.position = "right")

  if (is_circ) {
    p <- p + ggplot2::expand_limits(x = c(0, 3))
  } else {
    p <- p + hexpand(0.85)
  }
  if (!is.null(title) && nzchar(title)) {
    p <- p + ggtitle(title) +
      theme(plot.title = element_text(face = "bold", size = 11))
  }
  p
}

#' PacBio Article-style default: circular/radial tree with highlighted DA taxa
#' + abundance boxplots (ggdiffclade + ggdiffbox layout), driven by multilevel ANCOM-BC.
#' Tree MUST come from the phyloseq object (phy_tree or taxonomy formula) — never invented.
plot_diff_cladobox <- function(ps, tip_sel, tree, group_col = NULL, title = NULL,
                               colors = NULL, tree_source = NA_character_) {
  if (!requireNamespace("ggpubr", quietly = TRUE)) {
    fail("ggpubr required for --layout cladobox")
  }
  if (!requireNamespace("ggtext", quietly = TRUE)) {
    fail("ggtext required for cladobox markdown tip labels")
  }
  if (is.null(tree) || !inherits(tree, "phylo")) {
    fail("cladobox requires a phylo tree from the phyloseq object")
  }
  keep <- as.character(tip_sel$id)
  if (length(keep) < 2L) fail("Need ≥2 tips for cladobox plot")

  keep_tree <- intersect(keep, tree$tip.label)
  if (length(keep_tree) < 2L) {
    fail(
      "Difftree cladobox: need ≥2 selected tips on the phyloseq tree (",
      tree_source %||% "unknown", "). Found ", length(keep_tree),
      " of ", length(keep),
      ". Trees must come from phy_tree / tax_formula on the phyloseq object — ",
      "synthetic hclust trees are not allowed."
    )
  }
  if (length(keep_tree) < length(keep)) {
    message(
      "cladobox: dropping ", length(keep) - length(keep_tree),
      " tips absent from phyloseq tree"
    )
  }
  tr <- ape::keep.tip(ape::as.phylo(tree), keep_tree)
  tip_sel <- tip_sel[match(keep_tree, tip_sel$id), , drop = FALSE]
  keep <- keep_tree

  disp <- as.character(tip_sel$display)
  disp[is.na(disp) | !nzchar(disp)] <- keep
  face <- as.character(tip_sel$fontface %||% "plain")
  face[is.na(face) | !nzchar(face)] <- "plain"
  pval <- as.numeric(tip_sel$p)
  qval <- as.numeric(tip_sel$q)
  lfc <- as.numeric(tip_sel$log2_lfc)
  neglogp <- -log10(pmax(pval, .Machine$double.xmin))
  neglogp[!is.finite(neglogp)] <- 0

  tip_meta <- data.frame(
    label = keep,
    display = disp,
    fontface = face,
    is_italic = face == "italic",
    log2_lfc = lfc,
    neglog10_p = neglogp,
    sig = (!is.na(qval) & qval < DEFAULT_Q_CUT) | (abs(lfc) > DEFAULT_LFC_CUT),
    stringsAsFactors = FALSE
  )
  tip_meta$markdown <- ifelse(
    tip_meta$is_italic,
    paste0("*", tip_meta$display, "*"),
    tip_meta$display
  )
  lim <- max(abs(tip_meta$log2_lfc), na.rm = TRUE)
  if (!is.finite(lim) || lim <= 0) lim <- 1

  # Circular cladogram (PacBio ggdiffclade layout="radial" analogue)
  # branch.length="none" → cladogram; legends stacked by row at bottom
  p_tree <- ggtree(
    tr, layout = "circular", open.angle = 10, size = 0.15,
    branch.length = "none"
  ) %<+% tip_meta +
    geom_tippoint(
      aes(size = neglog10_p, color = log2_lfc, alpha = sig),
      stroke = 0.2
    ) +
    scale_color_gradient2(
      low = LFC_LOW, mid = LFC_MID, high = LFC_HIGH,
      midpoint = 0, limits = c(-lim, lim), name = "log2 LFC"
    ) +
    scale_size_binned("p-value, -lg", range = c(1, 3)) +
    scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.25), guide = "none") +
    geom_tiplab(
      aes(label = display, subset = is_italic),
      size = 2, offset = 0.08, align = TRUE, linesize = 0.1, fontface = "italic"
    ) +
    geom_tiplab(
      aes(label = display, subset = !is_italic),
      size = 2, offset = 0.08, align = TRUE, linesize = 0.1, fontface = "plain"
    ) +
    theme(
      panel.background = element_rect(fill = NA),
      legend.position = "bottom",
      legend.box = "vertical",
      legend.box.just = "left",
      legend.spacing.y = unit(0.15, "cm"),
      # Balance: shrink left empty band; give tip labels room on the right
      plot.margin = margin(4, 28, 4, 2, unit = "pt")
    ) +
    guides(
      color = guide_colorbar(
        order = 1, direction = "horizontal", title.position = "top",
        barwidth = unit(4, "cm"), barheight = unit(0.35, "cm")
      ),
      size = guide_legend(
        order = 2, direction = "horizontal", title.position = "top", nrow = 1
      )
    )

  # Abundance boxplots for significant tips (ggdiffbox analogue)
  sd <- plain_sample_df(ps)
  if (is.null(group_col) || !nzchar(group_col) || !group_col %in% names(sd)) {
    for (cand in c("Condition", "grazing", "group", "Group", "treatment", "genus")) {
      if (cand %in% names(sd)) {
        group_col <- cand
        break
      }
    }
  }
  if (is.null(group_col) || !group_col %in% names(sd)) {
    fail("cladobox boxplots need a sample grouping column")
  }
  grp_all <- sample_group_chr(ps, group_col)
  keep_sam <- !is.na(grp_all) & nzchar(grp_all)
  if (!all(keep_sam)) {
    ps_box <- prune_samples(keep_sam, ps)
    sd <- plain_sample_df(ps_box)
    grp_all <- sample_group_chr(ps_box, group_col)
  } else {
    ps_box <- ps
  }

  sig_ids <- tip_meta$label[tip_meta$sig]
  if (!length(sig_ids)) sig_ids <- tip_meta$label
  sig_ids <- intersect(sig_ids, taxa_names(ps_box))
  if (!length(sig_ids)) {
    # Rank tips may not be OTU ids — try matching tax_table column
    tt <- as.data.frame(tax_table(ps_box), stringsAsFactors = FALSE)
    for (rk in c("Genus", "Family", "Species", "Order", "Class")) {
      if (rk %in% names(tt)) {
        hit <- rownames(tt)[as.character(tt[[rk]]) %in% tip_meta$display[tip_meta$sig]]
        if (length(hit)) {
          sig_ids <- hit
          break
        }
      }
    }
  }
  if (!length(sig_ids)) fail("No taxa available for cladobox abundance boxes")

  otu <- as(otu_table(ps_box), "matrix")
  if (!taxa_are_rows(ps_box)) otu <- t(otu)
  otu <- otu[intersect(sig_ids, rownames(otu)), , drop = FALSE]
  ss <- as.numeric(sample_sums(ps_box))
  names(ss) <- sample_names(ps_box)
  rel <- sweep(otu, 2, pmax(ss[colnames(otu)], 1e-12), "/")

  long <- as.data.frame(as.table(rel), stringsAsFactors = FALSE)
  names(long) <- c("id", "Sample", "rel_abd")
  # Map to display
  if (all(long$id %in% tip_meta$label)) {
    long$feature <- tip_meta$display[match(long$id, tip_meta$label)]
    long$LDAmean <- abs(tip_meta$log2_lfc[match(long$id, tip_meta$label)])
  } else {
    tt <- as.data.frame(tax_table(ps_box), stringsAsFactors = FALSE)
    long$feature <- long$id
    for (rk in c("Genus", "Family", "Species")) {
      if (rk %in% names(tt)) {
        long$feature <- as.character(tt[as.character(long$id), rk])
        break
      }
    }
    long$LDAmean <- 0
  }
  long$group <- unname(grp_all[as.character(long$Sample)])
  long <- long[!is.na(long$group) & nzchar(long$group), , drop = FALSE]
  if (!nrow(long)) fail("cladobox boxes: no rows after group join (check sample_data)")
  # Guard against sample_data deparse artifacts (single concatenated level)
  if (length(unique(long$group)) == 1L && grepl("^c\\(", unique(long$group)[[1]])) {
    fail(
      "cladobox boxes: group levels look deparsed (sample_data extraction bug). ",
      "Got: ", unique(long$group)[[1]]
    )
  }
  long$feature <- factor(
    long$feature,
    levels = unique(long$feature[order(-long$LDAmean)])
  )
  long$group <- factor(long$group, levels = unique(long$group))

  grp_lv <- levels(long$group)
  if (is.null(colors) || is.null(names(colors)) ||
      !all(grp_lv %in% names(colors))) {
    colors <- if (length(grp_lv) <= 2L) {
      c("#1B9E77", "#D81B60")[seq_along(grp_lv)]
    } else {
      OKABE <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
      OKABE[seq_along(grp_lv)]
    }
    names(colors) <- grp_lv
  }

  p_box <- ggplot(long, aes(x = feature, y = rel_abd, fill = group)) +
    geom_boxplot(outlier.size = 0.4, width = 0.7, notch = FALSE, alpha = 0.9) +
    coord_flip() +
    scale_y_continuous(name = "abundance", n.breaks = 3, trans = "sqrt") +
    scale_fill_manual(values = colors, drop = FALSE) +
    labs(x = NULL, fill = group_col) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.y = ggtext::element_markdown(size = 8),
      axis.text.x = element_text(size = 8),
      legend.position = "bottom",
      legend.box = "vertical",
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 8, 4, 4, unit = "pt")
    )

  combined <- ggpubr::ggarrange(
    p_tree, p_box,
    align = "hv",
    widths = c(1, 1.05)
  )
  if (!is.null(title) && nzchar(title)) {
    combined <- ggpubr::annotate_figure(
      combined, top = ggpubr::text_grob(title, face = "bold", size = 11)
    )
  }
  combined
}

#' PacBio-style two-sided: tree+LFC (left) | abundance boxes (right), ggarrange.
plot_diff_twosided <- function(ps, tree, tip_sel, group_col = NULL, title = NULL) {
  if (!requireNamespace("ggpubr", quietly = TRUE)) {
    fail("ggpubr required for --layout twosided")
  }
  keep <- as.character(tip_sel$id)
  if (length(keep) < 2L) fail("Need ≥2 tips for twosided plot")
  tr <- ape::keep.tip(ape::as.phylo(tree), keep)

  disp <- as.character(tip_sel$display)
  disp[is.na(disp) | !nzchar(disp)] <- keep
  face <- as.character(tip_sel$fontface %||% "plain")
  face[is.na(face) | !nzchar(face)] <- "plain"
  dup <- duplicated(disp) | duplicated(disp, fromLast = TRUE)
  if (any(dup)) {
    disp[dup] <- paste0(disp[dup], " [", substr(keep[dup], 1L, 8L), "]")
  }
  tip_meta <- data.frame(
    label = keep,
    display = disp,
    fontface = face,
    log2_lfc = as.numeric(tip_sel$log2_lfc),
    stringsAsFactors = FALSE
  )
  lim <- max(abs(tip_meta$log2_lfc), na.rm = TRUE)
  if (!is.finite(lim) || lim <= 0) lim <- 1

  tip_meta$is_italic <- tip_meta$fontface == "italic"
  p_tree <- ggtree(tr, layout = "rectangular", branch.length = "none") %<+% tip_meta
  italic_lab <- tip_meta$is_italic
  if (any(italic_lab)) {
    p_tree <- p_tree + geom_tiplab(
      aes(label = display, subset = is_italic),
      size = 2.4, offset = 0.02, fontface = "italic"
    )
  }
  if (any(!italic_lab)) {
    p_tree <- p_tree + geom_tiplab(
      aes(label = display, subset = !is_italic),
      size = 2.4, offset = 0.02, fontface = "plain"
    )
  }
  p_tree <- p_tree +
    geom_tippoint(aes(color = log2_lfc), size = 2.2) +
    scale_color_gradient2(
      low = LFC_LOW, mid = LFC_MID, high = LFC_HIGH,
      midpoint = 0, limits = c(-lim, lim), name = "log2 LFC"
    ) +
    hexpand(0.35) +
    theme(legend.position = "bottom")

  sd <- plain_sample_df(ps)
  if (is.null(group_col) || !nzchar(group_col) || !group_col %in% names(sd)) {
    for (cand in c("Condition", "grazing", "group", "Group", "treatment")) {
      if (cand %in% names(sd)) {
        group_col <- cand
        break
      }
    }
  }
  if (is.null(group_col) || !group_col %in% names(sd)) {
    fail("twosided abundance boxes need a sample grouping column; pass via target or sample_data")
  }
  grp_all <- sample_group_chr(ps, group_col)

  otu <- as(otu_table(ps), "matrix")
  if (!taxa_are_rows(ps)) otu <- t(otu)
  otu <- otu[keep, , drop = FALSE]
  ss <- as.numeric(sample_sums(ps))
  names(ss) <- sample_names(ps)
  rel <- sweep(otu, 2, pmax(ss[colnames(otu)], 1e-12), "/")

  long <- as.data.frame(as.table(rel), stringsAsFactors = FALSE)
  names(long) <- c("id", "Sample", "rel_abd")
  long$display <- tip_meta$display[match(long$id, tip_meta$label)]
  long$group <- unname(grp_all[as.character(long$Sample)])
  long <- long[!is.na(long$group) & nzchar(long$group), , drop = FALSE]
  long$group <- factor(long$group, levels = unique(long$group))
  # order y by tree tip order (top = first tip after ladderize)
  tip_ord <- rev(tr$tip.label)
  long$display <- factor(
    long$display,
    levels = tip_meta$display[match(tip_ord, tip_meta$label)]
  )

  p_box <- ggplot(long, aes(x = rel_abd, y = display, fill = group)) +
    geom_boxplot(outlier.size = 0.4, width = 0.7, alpha = 0.85) +
    scale_x_continuous(trans = "sqrt", name = "Relative abundance") +
    labs(y = NULL, fill = group_col) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.y = element_text(face = "italic", size = 7),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )

  # LFC bar strip between tree and boxes (PacBio-like two panel + effect)
  p_lfc <- ggplot(tip_meta, aes(x = log2_lfc, y = factor(display, levels = levels(long$display)), fill = log2_lfc)) +
    geom_col(width = 0.7) +
    scale_fill_gradient2(
      low = LFC_LOW, mid = LFC_MID, high = LFC_HIGH,
      midpoint = 0, limits = c(-lim, lim), guide = "none"
    ) +
    labs(x = "log2 LFC", y = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.minor = element_blank()
    )

  combined <- ggpubr::ggarrange(
    p_tree, p_lfc, p_box,
    nrow = 1, widths = c(0.9, 0.45, 1.1),
    align = "h", labels = if (!is.null(title) && nzchar(title)) NULL else NULL
  )
  if (!is.null(title) && nzchar(title)) {
    combined <- ggpubr::annotate_figure(combined, top = ggpubr::text_grob(title, face = "bold", size = 11))
  }
  combined
}

safe_name <- function(x) gsub("[^A-Za-z0-9._-]+", "_", x)

save_plot <- function(p, prefix, width = 10, height = 10) {
  pdf_path <- paste0(prefix, ".pdf")
  png_path <- paste0(prefix, ".png")
  ggsave(pdf_path, p, width = width, height = height, device = grDevices::pdf)
  if (isTRUE(capabilities("cairo"))) {
    ggsave(png_path, p, width = width, height = height, dpi = 300, device = "png")
  } else {
    ggsave(png_path, p, width = width, height = height, dpi = 300)
  }
  list(pdf = pdf_path, png = png_path)
}

run_difftree_ggtree <- function(rds = NULL, outdir,
                                ancombc_csv = NULL, ancombc_dir = NULL,
                                layout = c("cladobox", "twosided", "circular", "rectangular", "fruit"),
                                max_tips = DEFAULT_MAX_TIPS,
                                lfc_cut = DEFAULT_LFC_CUT,
                                q_cut = DEFAULT_Q_CUT,
                                targets = NULL, terms = NULL,
                                levels = NULL,
                                allow_non_rare = FALSE,
                                tip_offset = NULL,
                                fruit_offset = NULL,
                                fruit_pwidth = NULL) {
  layout <- match.arg(layout)
  if (identical(layout, "fruit")) layout <- "circular"
  ensure_dir(outdir)

  loaded <- resolve_rare_ps(rds, allow_non_rare = allow_non_rare)
  ps <- ensure_finalized_taxonomy(loaded$ps)
  meta <- loaded$meta
  if (!isTRUE(meta$rarefied) && !allow_non_rare) {
    fail("Input not rarefied: ", meta$path)
  }

  ancom <- require_ancombc(ancombc_csv, ancombc_dir)
  an_long <- normalize_ancombc_long(read_table_auto(ancom$path))
  if (!nrow(an_long)) fail("Empty ANCOM-BC table: ", ancom$path)

  avail_levels <- unique(as.character(an_long$level))
  if (!is.null(levels) && nzchar(levels)) {
    lev_use <- trimws(strsplit(levels, ",", fixed = TRUE)[[1]])
  } else if (identical(layout, "cladobox")) {
    pref <- c("Genus", "Family", "Order", "Class", "Phylum", "ASV")
    lev_use <- intersect(pref, avail_levels)
    if (!length(lev_use)) lev_use <- avail_levels
  } else {
    lev_use <- if ("ASV" %in% avail_levels) "ASV" else avail_levels[[1]]
  }
  if (!length(lev_use)) fail("No usable ANCOM-BC levels in ", ancom$path)

  tree_res <- resolve_tree(ps)
  tree <- tree_res$tree

  tg <- if (!is.null(targets) && nzchar(targets)) {
    trimws(strsplit(targets, ",", fixed = TRUE)[[1]])
  } else {
    unique(as.character(an_long$target))
  }
  tm <- if (!is.null(terms) && nzchar(terms)) {
    trimws(strsplit(terms, ",", fixed = TRUE)[[1]])
  } else {
    unique(as.character(an_long$term))
  }
  tm <- tm[!grepl("Intercept", tm, ignore.case = TRUE)]

  message(
    "difftree-ggtree: rarefied=", meta$rarefied,
    " tree=", tree_res$source, " tips=", length(tree$tip.label),
    " ancombc=", ancom$path,
    " levels=", paste(lev_use, collapse = ","),
    " layout=", layout,
    " ancombc_rerun=FALSE"
  )

  figures <- list()
  tips_out <- list()

  for (t in tg) {
    for (term in tm) {
      for (lev in lev_use) {
        an_sub <- an_long[
          an_long$target == t & an_long$term == term &
            toupper(an_long$level) == toupper(lev),
          , drop = FALSE
        ]
        if (!nrow(an_sub)) {
          message("  skip ", t, "/", term, "/", lev, " (no rows)")
          next
        }
        if (toupper(lev) == "ASV") {
          tip_pack <- build_tip_table(ps, an_sub, tree, max_tips, lfc_cut, q_cut)
        } else {
          tip_pack <- build_tip_table_rank(ps, an_sub, lev, max_tips, lfc_cut, q_cut)
        }
        sel <- tip_pack$selected
        if (is.null(sel) || !nrow(sel) || nrow(sel) < 2L) {
          message("  skip ", t, "/", term, "/", lev, " (need ≥2 tips)")
          next
        }
        message("  ", t, "/", term, "/", lev, ": selected_tips=", nrow(sel))

        title <- paste0("ANCOM-BC log2 LFC — ", t, " / ", term, " / ", lev)
        if (identical(layout, "cladobox")) {
          p <- plot_diff_cladobox(
            ps, sel, tree = tree, group_col = t, title = title,
            tree_source = tree_res$source
          )
          wh <- c(14, 10)
        } else if (identical(layout, "twosided")) {
          p <- plot_diff_twosided(ps, tree, sel, group_col = t, title = title)
          wh <- c(14, max(8, min(16, 0.22 * nrow(sel) + 4)))
        } else {
          # fruit layouts also require tips on the phyloseq tree
          keep_fruit <- intersect(as.character(sel$id), tree$tip.label)
          if (length(keep_fruit) < 2L) {
            fail(
              "Difftree fruit layout: need ≥2 tips on phyloseq tree; found ",
              length(keep_fruit)
            )
          }
          sel <- sel[match(keep_fruit, sel$id), , drop = FALSE]
          p <- plot_diff_ggtree(
            tree, sel, layout = layout, title = title,
            tip_offset = tip_offset, fruit_offset = fruit_offset,
            fruit_pwidth = fruit_pwidth
          )
          wh <- if (identical(layout, "circular")) c(12, 12) else c(11, 10)
        }
        prefix <- file.path(
          outdir,
          paste0("difftree_ggtree_", safe_name(t), "_", safe_name(term), "_", safe_name(lev))
        )
        figs <- save_plot(p, prefix, width = wh[[1]], height = wh[[2]])
        figures[[length(figures) + 1L]] <- list(
          name = paste0(t, "_", term, "_", lev),
          target = t, term = term, level = lev,
          n_tips = nrow(sel), pdf = figs$pdf, png = figs$png
        )
        sel$target <- t
        sel$term <- term
        sel$level <- lev
        tips_out[[length(tips_out) + 1L]] <- sel
      }
    }
  }
  if (!length(figures)) fail("No ggtree figures produced")

  tips_df <- do.call(rbind, tips_out)
  tips_path <- file.path(outdir, "difftree_ggtree_tips.tsv")
  utils::write.table(tips_df, tips_path, sep = "\t", quote = FALSE, row.names = FALSE)

  report <- list(
    skill = "difftree-ggtree",
    input_rds = meta$path,
    rarefied = isTRUE(meta$rarefied),
    rarefaction_depth = meta$rarefaction_depth,
    tree_source = tree_res$source,
    ancombc_path = ancom$path,
    ancombc_report_ok = ancom$report_ok,
    ancombc_rerun = FALSE,
    layout = layout,
    levels = lev_use,
    max_tips = max_tips,
    lfc_cut = lfc_cut,
    q_cut = q_cut,
    targets = tg,
    terms = tm,
    figures = figures,
    tips_tsv = tips_path,
    notes = c(loaded$notes %||% character(0), ancom$notes %||% character(0))
  )
  write_json(report, file.path(outdir, "difftree-ggtree-report.json"))
  message("Wrote ", length(figures), " figure set(s)")
  report
}

self_test <- function() {
  setwd(project_root())
  if (!file.exists("test/ancombc/grazing/ancombc_results.csv") &&
      !file.exists("test/ancombc/grazing/ancombc_results.tsv")) {
    stop("self-test requires existing ancombc grazing results (do not regenerate)")
  }
  out <- "test/difftree-ggtree/grazing-self-test"
  rep <- run_difftree_ggtree(
    rds = NULL,
    outdir = out,
    ancombc_dir = "test/ancombc/grazing",
    layout = "cladobox",
    levels = "Genus,ASV",
    max_tips = 40L,
    lfc_cut = 0.5,
    q_cut = 0.05
  )
  if (!isTRUE(rep$rarefied)) stop("expected rarefied")
  if (is.null(rep$ancombc_path) || !nzchar(rep$ancombc_path)) stop("expected ancombc import")
  if (isTRUE(rep$ancombc_rerun)) stop("must not rerun ancombc")
  if (!length(rep$figures)) stop("no figures")
  for (fg in rep$figures) {
    if (!file.exists(fg$png) || file.info(fg$png)$size < 2000) {
      stop("png missing/small: ", fg$png)
    }
  }
  if (!file.exists(rep$tips_tsv)) stop("missing tips tsv")
  message("SELF-TEST OK (n_figs=", length(rep$figures), ", ancombc_rerun=FALSE)")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/difftree-ggtree/run"
  layout <- args$layout %||% "cladobox"
  max_tips <- as.integer(args$max_tips %||% DEFAULT_MAX_TIPS)
  lfc_cut <- as.numeric(args$lfc_cut %||% DEFAULT_LFC_CUT)
  q_cut <- as.numeric(args$q_cut %||% DEFAULT_Q_CUT)
  allow_non_rare <- identical(tolower(as.character(args$allow_non_rare %||% "false")), "true")
  tip_offset <- if (!is.null(args$tip_offset)) as.numeric(args$tip_offset) else NULL
  fruit_offset <- if (!is.null(args$fruit_offset)) as.numeric(args$fruit_offset) else NULL
  fruit_pwidth <- if (!is.null(args$fruit_pwidth)) as.numeric(args$fruit_pwidth) else NULL

  run_difftree_ggtree(
    rds = args$rds,
    outdir = outdir,
    ancombc_csv = args$ancombc_csv,
    ancombc_dir = args$ancombc_dir,
    layout = layout,
    max_tips = max_tips,
    lfc_cut = lfc_cut,
    q_cut = q_cut,
    targets = args$targets,
    terms = args$terms,
    levels = args$levels,
    allow_non_rare = allow_non_rare,
    tip_offset = tip_offset,
    fruit_offset = fruit_offset,
    fruit_pwidth = fruit_pwidth
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
