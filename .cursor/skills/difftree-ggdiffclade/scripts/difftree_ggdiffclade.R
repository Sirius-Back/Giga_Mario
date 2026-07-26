#!/usr/bin/env Rscript
# difftree-ggdiffclade — MicrobiotaProcess diff_analysis + ggdiffclade + ggdiffbox
# Source layout: PacBio Article.Rmd (primary) / Kristina codebase.Rmd (copy)
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("MicrobiotaProcess", quietly = TRUE))
  stopifnot(requireNamespace("coin", quietly = TRUE)) # kruskal_test / wilcox_test
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
  stopifnot(requireNamespace("ggpubr", quietly = TRUE))
  stopifnot(requireNamespace("ggtext", quietly = TRUE))
  stopifnot(requireNamespace("dplyr", quietly = TRUE))
  stopifnot(requireNamespace("stringr", quietly = TRUE))
  stopifnot(requireNamespace("forcats", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  library(MicrobiotaProcess)
  library(coin)
  library(ggplot2)
  library(dplyr)
  library(stringr)
  library(forcats)
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

OKABE <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GRAZING_COLS <- c("#4E9E23", "#FFA600", "#BE475A")
PACBIO_2 <- c("lightgreen", "pink")
STANDARD_RANKS <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

# ---------------------------------------------------------------------------
# Taxon label cleanup (MicrobiotaProcess long s__…_g__…_o__… strings)
# ---------------------------------------------------------------------------

is_placeholder_mp_name <- function(x) {
  x <- trimws(as.character(x))
  x <- sub("\\s+ASV[0-9]+$", "", x, ignore.case = TRUE)
  x <- trimws(x)
  !nzchar(x) ||
    grepl("^(uncultured|unknown|unclassified|unassigned)(\\b|_)", x, ignore.case = TRUE) ||
    grepl("^unclassified_", x, ignore.case = TRUE)
}

#' Split MP feature id into ranked segments (preserves underscores inside names).
parse_mp_feature_segments <- function(feat) {
  feat <- as.character(feat)
  feat <- sub("^[a-z]:\\s*", "", feat) # drop ggdiffclade letter index "a: "
  feat <- trimws(feat)
  if (!nzchar(feat)) {
    return(data.frame(rank = character(0), name = character(0), stringsAsFactors = FALSE))
  }
  parts <- strsplit(feat, "_(?=[kpcofgs]__)", perl = TRUE)[[1]]
  ranks <- character(0)
  names_ <- character(0)
  for (p in parts) {
    m <- regmatches(p, regexec("^([kpcofgs])__(.*)$", p))[[1]]
    if (length(m) >= 3L) {
      ranks <- c(ranks, m[[2]])
      names_ <- c(names_, m[[3]])
    } else if (nzchar(p)) {
      ranks <- c(ranks, "s")
      names_ <- c(names_, p)
    }
  }
  data.frame(rank = ranks, name = names_, stringsAsFactors = FALSE)
}

extract_asv_tag <- function(name) {
  m <- regmatches(name, regexpr("ASV[0-9]+", name, ignore.case = TRUE))
  if (length(m) && nzchar(m[[1]])) toupper(m[[1]]) else NA_character_
}

strip_asv_tag <- function(name) {
  trimws(sub("\\s*ASV[0-9]+\\s*$", "", as.character(name), ignore.case = TRUE))
}

#' Latin-like display name (for italics); numeric/code tokens stay plain.
looks_latin_taxon <- function(x) {
  x <- trimws(as.character(x))
  nzchar(x) &&
    grepl("^[A-Za-z]", x) &&
    !grepl("[0-9]", x) &&
    !is_placeholder_mp_name(x)
}

#' Compact display label from an MP feature / .LABEL string.
#' Examples:
#'   s__uncultured ASV5_…_o__Acidobacteriales → "s - uncultured Acidobacteriales ASV5"
#'   s__uncultured ASV4_…_o__uncultured       → "s - unknown ASV4"
#'   s__1921-2 ASV5                           → "s - 1921-2 ASV5"
trim_mp_taxon_label <- function(feat, markdown = TRUE) {
  feat <- as.character(feat)
  if (length(feat) != 1L) {
    return(vapply(feat, trim_mp_taxon_label, character(1), markdown = markdown, USE.NAMES = FALSE))
  }
  if (is.na(feat) || !nzchar(feat)) return(feat)

  letter_idx <- NA_character_
  if (grepl("^[a-z]:\\s+", feat)) {
    letter_idx <- substr(feat, 1L, 1L)
    feat <- sub("^[a-z]:\\s*", "", feat)
  }

  seg <- parse_mp_feature_segments(feat)
  if (!nrow(seg)) return(if (markdown && !is.na(letter_idx)) paste0("**", letter_idx, ":** ") else "")

  asv <- NA_character_
  for (nm in seg$name) {
    a <- extract_asv_tag(nm)
    if (!is.na(a)) {
      asv <- a
      break
    }
  }
  seg$core <- vapply(seg$name, strip_asv_tag, character(1))

  # Prefer deepest non-placeholder; fall back tip → root
  non_ph <- which(!vapply(seg$core, is_placeholder_mp_name, logical(1)))
  tip_ph <- is_placeholder_mp_name(seg$core[[1]])

  if (!length(non_ph)) {
    # all uncultured / unknown
    body <- "unknown"
    rk <- seg$rank[[1]]
    italic <- FALSE
  } else {
    i <- non_ph[[length(non_ph)]]
    body <- seg$core[[i]]
    rk <- seg$rank[[i]]
    italic <- looks_latin_taxon(body)
    if (tip_ph && i > 1L) {
      body <- paste("uncultured", body)
      # keep italics only on the classified token
      italic <- looks_latin_taxon(seg$core[[i]])
    }
  }

  # Leading rank letter from tip segment (usually s)
  lead <- seg$rank[[1]]
  if (markdown) {
    lead_fmt <- paste0("**", lead, "**")
    if (tip_ph && length(non_ph) && non_ph[[length(non_ph)]] > 1L && italic) {
      # "uncultured *Acidobacteriales*"
      classified <- seg$core[[non_ph[[length(non_ph)]]]]
      body_fmt <- paste0("uncultured *", classified, "*")
    } else if (identical(body, "unknown")) {
      body_fmt <- body
    } else if (italic) {
      body_fmt <- paste0("*", body, "*")
    } else {
      body_fmt <- body # numeric / code tokens (1921-2, AD3, LWQ8, …)
    }
    lab <- paste0(lead_fmt, " - ", body_fmt)
    if (!is.na(asv)) lab <- paste(lab, asv)
    if (!is.na(letter_idx)) lab <- paste0("**", letter_idx, ":** ", lab)
  } else {
    lab <- paste0(lead, " - ", body)
    if (!is.na(asv)) lab <- paste(lab, asv)
  }
  lab
}

#' Match key used to join clade labels ↔ ggdiffbox feature (no letter index).
mp_feature_key <- function(x) {
  x <- as.character(x)
  x <- sub("^[a-z]:\\s*", "", x)
  trimws(x)
}

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(path = path, target = NA_character_, rarefied = FALSE,
               rarefaction_depth = NA_real_)
  if (inherits(obj, "phyloseq")) {
    meta$rarefied <- grepl("_rare(\\.rds)?$|phyloseq_rare_", basename(path))
    return(list(ps = obj, meta = meta))
  }
  if (is.list(obj) && inherits(obj$phyloseq, "phyloseq")) {
    meta$target <- obj$target %||% NA_character_
    meta$rarefaction_depth <- obj$rarefaction_depth %||% NA_real_
    meta$rarefied <- !is.null(obj$rarefaction_depth) ||
      grepl("_rare|phyloseq_rare_", basename(path))
    return(list(ps = obj$phyloseq, meta = meta))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

resolve_rare_ps <- function(rds = NULL, allow_non_rare = FALSE) {
  cands <- c(
    rds,
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  for (p in cands) {
    if (is.null(p) || !nzchar(p) || !file.exists(p)) next
    loaded <- load_ps(p)
    if (!isTRUE(loaded$meta$rarefied) && !allow_non_rare) {
      if (!is.null(rds) && nzchar(rds) && file.exists(rds) &&
          identical(normalizePath(p), normalizePath(rds))) {
        fail("Input not rarefied (pass --allow-non-rare true): ", p)
      }
      next
    }
    return(loaded)
  }
  fail("No rarefied phyloseq found; pass --rds")
}

resolve_target <- function(ps, target = NULL, meta_target = NA_character_) {
  sd <- as(phyloseq::sample_data(ps), "data.frame")
  if (!is.null(target) && nzchar(target)) {
    if (!target %in% names(sd)) fail("Target column missing: ", target)
    return(target)
  }
  if (!is.na(meta_target) && nzchar(meta_target) && meta_target %in% names(sd)) {
    return(meta_target)
  }
  for (cand in c("grazing", "Condition", "genus", "Group", "group", "DIAGNOSIS")) {
    if (cand %in% names(sd)) return(cand)
  }
  fail("No target column; pass --target")
}

palette_for_n <- function(n) {
  if (n == 2L) return(PACBIO_2)
  if (n == 3L) return(GRAZING_COLS)
  rep_len(OKABE, n)
}

#' Prepare phyloseq for MicrobiotaProcess (PacBio-compatible tax ranks).
prepare_ps_for_mp <- function(ps, target) {
  # Drop phy_tree — MicrobiotaProcess builds taxonomy tree; ape/tidytree dual
  # phylo registration crashes reorderRcpp on many grazing-style trees.
  tt <- as.data.frame(phyloseq::tax_table(ps), stringsAsFactors = FALSE)
  keep_ranks <- intersect(STANDARD_RANKS, names(tt))
  if (length(keep_ranks) < 3L) {
    fail("Need ≥3 standard taxonomy ranks; have: ", paste(names(tt), collapse = ","))
  }
  tt <- tt[, keep_ranks, drop = FALSE]
  for (j in keep_ranks) {
    v <- as.character(tt[[j]])
    v[is.na(v)] <- ""
    v <- sub("^[a-z]__", "", v)
    v[!nzchar(v)] <- paste0("unclassified_", j)
    tt[[j]] <- v
  }
  sd <- as(phyloseq::sample_data(ps), "data.frame")
  if (!target %in% names(sd)) fail("sample_data missing target: ", target)
  raw <- as.character(sd[[target]])
  keep_s <- !is.na(raw) & nzchar(raw)
  if (!all(keep_s)) {
    ps <- phyloseq::prune_samples(keep_s, ps)
    sd <- as(phyloseq::sample_data(ps), "data.frame")
    raw <- as.character(sd[[target]])
  }
  sd[[target]] <- factor(raw)
  if (nlevels(sd[[target]]) < 2L) fail("Target needs ≥2 levels: ", target)
  tax_mat <- as.matrix(tt[phyloseq::taxa_names(ps), , drop = FALSE])
  phyloseq::phyloseq(
    phyloseq::otu_table(ps),
    phyloseq::tax_table(tax_mat),
    phyloseq::sample_data(sd)
  )
}

# ---------------------------------------------------------------------------
# Core plot (PacBio Article.Rmd structure; bottom legend from cladogram)
# ---------------------------------------------------------------------------

#' Build PacBio-style two-panel figure.
#' Left: ggdiffclade. Right: ggdiffbox (abundance + LDA; LDA hides shared y-text).
#' Bottom legend from ggdiffclade only.
plot_ggdiffclade_box <- function(deres, classgroup, colors, taxlevel = 3L) {
  dc0 <- ggdiffclade(
    obj = deres,
    alpha = 0.3,
    linewd = 0.15,
    skpointsize = 0.6,
    layout = "radial",
    taxlevel = as.integer(taxlevel),
    removeUnknown = TRUE,
    reduce = TRUE
  )
  if (length(dc0$layers) < 3L) {
    fail("ggdiffclade returned unexpected layers (need ≥3 for tip labels)")
  }

  # Trim long uncultured chains on the cladogram tip layer
  lab_raw <- dc0$layers[[3]]$data$.LABEL
  if (is.null(lab_raw) || all(is.na(lab_raw))) {
    lab_raw <- dc0$layers[[3]]$data$label
  }
  dc0$layers[[3]]$data$.LABEL <- vapply(
    lab_raw, trim_mp_taxon_label, character(1), markdown = TRUE
  )
  # Join key (no letter index) → trimmed markdown label
  tip_keys <- mp_feature_key(lab_raw)
  tip_labs <- dc0$layers[[3]]$data$.LABEL
  tip_map <- tip_labs
  names(tip_map) <- tip_keys
  # Prefer first occurrence for duplicate keys
  tip_map <- tip_map[!duplicated(names(tip_map)) & nzchar(names(tip_map))]

  # Legend-bearing cladogram (fill + size); panel itself drawn without legend
  dc_leg <- dc0 +
    scale_fill_diff_cladogram(values = colors) +
    theme(
      panel.background = element_rect(fill = NA),
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.margin = margin(0, 0, 0, 0),
      legend.title.position = "top"
    ) +
    scale_size_binned("p-value, -lg", range = c(1, 3)) +
    guides(
      color = guide_none(),
      fill = guide_legend(position = "bottom", nrow = 1),
      size = guide_legend(position = "bottom", nrow = 1)
    )

  dc <- dc0 +
    scale_fill_diff_cladogram(values = colors) +
    theme(
      panel.background = element_rect(fill = NA),
      legend.position = "none",
      plot.margin = margin(0, 0, 0, 0)
    ) +
    scale_size_binned("p-value, -lg", range = c(1, 3)) +
    guides(color = guide_none(), fill = guide_none(), size = guide_none())

  # --- right boxplots: label ALL features (incl. numeric codes) ---
  diffbox <- ggdiffbox(
    obj = deres,
    box_notch = FALSE,
    colorlist = colors,
    l_xlabtext = "abundance"
  )
  if (length(diffbox) < 2L) fail("ggdiffbox expected 2 panels; got ", length(diffbox))

  rhs <- diffbox[[2]]$data %>% dplyr::rename(feature = f)
  drop_cols <- intersect(c(classgroup), names(rhs))
  if (length(drop_cols)) {
    rhs <- rhs[, setdiff(names(rhs), drop_cols), drop = FALSE]
  }
  if (!"LDAmean" %in% names(rhs)) {
    fail("ggdiffbox panel 2 missing LDAmean column")
  }

  label_feature <- function(feat) {
    feat <- as.character(feat)
    key <- mp_feature_key(feat)
    if (key %in% names(tip_map)) {
      unname(tip_map[[key]])
    } else {
      trim_mp_taxon_label(feat, markdown = TRUE)
    }
  }

  box_dat <- diffbox[[1]][["layers"]][[1]][["data"]] %>%
    full_join(rhs, by = "feature")
  # Ensure numeric / code taxa keep a display label (never drop to blank)
  box_dat$feature_raw <- as.character(box_dat$feature)
  box_dat$feature <- vapply(box_dat$feature_raw, label_feature, character(1))
  # Empty markdown after trim → fall back to plain trimmed / raw key
  blank <- is.na(box_dat$feature) | !nzchar(box_dat$feature)
  if (any(blank)) {
    box_dat$feature[blank] <- vapply(
      box_dat$feature_raw[blank],
      function(z) trim_mp_taxon_label(z, markdown = FALSE),
      character(1)
    )
  }
  still_blank <- is.na(box_dat$feature) | !nzchar(box_dat$feature)
  if (any(still_blank)) {
    box_dat$feature[still_blank] <- box_dat$feature_raw[still_blank]
  }
  # LDA reorder (NA LDAmean → end)
  lda_ord <- box_dat$LDAmean
  lda_ord[is.na(lda_ord)] <- -Inf
  box_dat$feature <- factor(
    box_dat$feature,
    levels = unique(box_dat$feature[order(-lda_ord)])
  )
  diffbox[[1]][["layers"]][[1]][["data"]] <- box_dat

  # Also relabel LDA strip (panel 2) so names stay aligned
  if ("f" %in% names(diffbox[[2]]$data)) {
    f2 <- as.character(diffbox[[2]]$data$f)
    diffbox[[2]]$data$f <- factor(
      vapply(f2, label_feature, character(1)),
      levels = levels(box_dat$feature)
    )
  }

  # Right panels share the abundance y-order: keep y-text on abundance only;
  # suppress legends on both (ggdiffclade bottom legend is enough).
  diffbox[[1]] <- diffbox[[1]] +
    theme(
      legend.position = "none",
      axis.text.y = ggtext::element_markdown(size = 8),
      axis.text.x = element_text(size = 8)
    ) +
    scale_y_continuous(n.breaks = 3)

  diffbox[[2]] <- diffbox[[2]] +
    theme(
      axis.text.x = element_text(size = 8),
      axis.text.y = element_blank(),
      axis.title.y = element_blank(),
      axis.ticks.y = element_blank(),
      legend.position = "none"
    )

  # Two panels without legends + ggdiffclade legend only at bottom
  panels <- ggpubr::ggarrange(
    dc, diffbox,
    align = "hv",
    widths = c(0.7, 1),
    legend = "none"
  )
  leg <- tryCatch(ggpubr::get_legend(dc_leg), error = function(e) NULL)
  if (is.null(leg)) {
    return(panels)
  }
  ggpubr::ggarrange(
    panels, leg,
    ncol = 1,
    heights = c(1, 0.14)
  )
}

run_diff_analysis <- function(
    ps,
    classgroup,
    firstalpha = 0.05,
    secondalpha = 0.01,
    subclmin = 3L,
    lda = 3,
    strictmod = TRUE,
    seed = 42L
) {
  set.seed(as.integer(seed))
  MicrobiotaProcess::diff_analysis(
    obj = ps,
    classgroup = classgroup,
    mlfun = "lda",
    filtermod = "pvalue",
    firstcomfun = "kruskal_test",
    firstalpha = as.numeric(firstalpha),
    strictmod = isTRUE(strictmod),
    secondcomfun = "wilcox_test",
    subclmin = as.integer(subclmin),
    subclwilc = TRUE,
    secondalpha = as.numeric(secondalpha),
    lda = as.numeric(lda)
  )
}

save_combined <- function(plot_obj, prefix, width = 14, height = 10) {
  pdf <- paste0(prefix, ".pdf")
  png <- paste0(prefix, ".png")
  ggplot2::ggsave(pdf, plot_obj, width = width, height = height)
  png_ok <- FALSE
  tryCatch({
    ggplot2::ggsave(png, plot_obj, width = width, height = height, dpi = 300)
    png_ok <- TRUE
  }, error = function(e) message("PNG skip: ", conditionMessage(e)))
  list(pdf = pdf, png = if (png_ok && file.exists(png)) png else NULL)
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

run_difftree_ggdiffclade <- function(
    rds = NULL,
    outdir = "test/difftree-ggdiffclade/run",
    target = NULL,
    taxlevel = 3L,
    firstalpha = 0.05,
    secondalpha = 0.01,
    subclmin = 3L,
    lda = 3,
    strictmod = TRUE,
    seed = 42L,
    allow_non_rare = FALSE,
    min_mean_rel = 1e-5,
    min_sample_sum = 1000L
) {
  setwd(root)
  ensure_dir(outdir)

  loaded <- resolve_rare_ps(rds, allow_non_rare = allow_non_rare)
  ps0 <- loaded$ps
  meta <- loaded$meta
  tg <- resolve_target(ps0, target, meta$target)

  # Optional PacBio-style abundance filter before rarefy check
  notes <- character(0)
  ss <- as.numeric(phyloseq::sample_sums(ps0))
  if (any(ss < as.numeric(min_sample_sum))) {
    keep <- ss >= as.numeric(min_sample_sum)
    notes <- c(notes, paste0("dropped ", sum(!keep), " samples below sum ", min_sample_sum))
    ps0 <- phyloseq::prune_samples(keep, ps0)
  }
  if (is.finite(min_mean_rel) && min_mean_rel > 0) {
    rel <- phyloseq::transform_sample_counts(ps0, function(x) x / sum(x))
    otu <- as(phyloseq::otu_table(rel), "matrix")
    if (!phyloseq::taxa_are_rows(rel)) otu <- t(otu)
    keep_t <- rowMeans(otu) > as.numeric(min_mean_rel)
    if (sum(keep_t) >= 10L) {
      notes <- c(notes, paste0("mean-rel filter >", min_mean_rel, " kept ", sum(keep_t), " taxa"))
      ps0 <- phyloseq::prune_taxa(keep_t, ps0)
    }
  }

  ps <- prepare_ps_for_mp(ps0, tg)
  lev <- levels(factor(as(phyloseq::sample_data(ps), "data.frame")[[tg]]))
  colors <- palette_for_n(length(lev))
  names(colors) <- lev

  message(
    "difftree-ggdiffclade: target=", tg,
    " levels=", paste(lev, collapse = ","),
    " ntaxa=", phyloseq::ntaxa(ps), " nsamples=", phyloseq::nsamples(ps),
    " engine=MicrobiotaProcess::diff_analysis"
  )

  deres <- run_diff_analysis(
    ps, classgroup = tg,
    firstalpha = firstalpha, secondalpha = secondalpha,
    subclmin = subclmin, lda = lda, strictmod = strictmod, seed = seed
  )
  if (!inherits(deres, "diffAnalysisClass")) {
    fail("diff_analysis did not return diffAnalysisClass")
  }
  n_sig <- nrow(deres@result)
  if (!n_sig || n_sig < 1L) {
    fail("No significantly discriminative features from diff_analysis")
  }

  res_path <- file.path(outdir, "diff_analysis_result.tsv")
  utils::write.table(
    deres@result, res_path, sep = "\t", quote = FALSE, row.names = TRUE
  )

  combined <- plot_ggdiffclade_box(
    deres, classgroup = tg, colors = unname(colors), taxlevel = taxlevel
  )
  figs <- save_combined(
    combined,
    file.path(outdir, paste0("difftree_ggdiffclade_", tg)),
    width = 14, height = 10
  )

  report <- list(
    skill = "difftree-ggdiffclade",
    engine = "MicrobiotaProcess",
    source_code = list(
      primary = "/mnt/tank/scratch/dsmutin/ixodes/PacBio/Article.Rmd",
      primary_chunks = c("DE genus ~L1876–1970", "DE sex ~L2099–2160+"),
      copy = "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/codebase.Rmd",
      copy_chunks = c("DE genus ~L1709–1803", "DE sex ~L1932–2026")
    ),
    input_rds = meta$path,
    rarefied = meta$rarefied,
    target = tg,
    levels = lev,
    colors = as.list(colors),
    n_significant = as.integer(n_sig),
    taxlevel = as.integer(taxlevel),
    diff_analysis = list(
      mlfun = "lda",
      firstcomfun = "kruskal_test",
      firstalpha = firstalpha,
      secondcomfun = "wilcox_test",
      secondalpha = secondalpha,
      subclmin = as.integer(subclmin),
      subclwilc = TRUE,
      strictmod = isTRUE(strictmod),
      lda = lda
    ),
    layout = list(
      left = "ggdiffclade radial (legend none)",
      right = "ggdiffbox abundance + LDA (legend none; LDA y-text blank)",
      bottom = "ggdiffclade fill/size legend only",
      widths = c(0.7, 1)
    ),
    package_versions = list(
      MicrobiotaProcess = as.character(utils::packageVersion("MicrobiotaProcess")),
      phyloseq = as.character(utils::packageVersion("phyloseq")),
      coin = as.character(utils::packageVersion("coin")),
      ggtree = as.character(utils::packageVersion("ggtree"))
    ),
    figures = figs,
    tables = list(diff_analysis_result = res_path),
    notes = notes
  )
  write_json(report, file.path(outdir, "difftree-ggdiffclade-report.json"))
  message(
    "difftree-ggdiffclade OK: n_sig=", n_sig,
    " figure=", figs$pdf
  )
  invisible(report)
}

self_test <- function() {
  setwd(root)
  out <- "test/difftree-ggdiffclade/self-test"
  if (dir.exists(out)) unlink(out, recursive = TRUE)
  # Grazing self-test: relax alphas slightly for small rarefied subset stability
  rep <- run_difftree_ggdiffclade(
    rds = "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    outdir = out,
    target = "grazing",
    taxlevel = 3L,
    firstalpha = 0.1,
    secondalpha = 0.05,
    subclmin = 2L,
    lda = 2,
    strictmod = FALSE,
    seed = 1L,
    min_mean_rel = 0,
    min_sample_sum = 0L
  )
  if (!file.exists(rep$figures$pdf)) stop("missing combined PDF")
  if (!file.exists(rep$tables$diff_analysis_result)) stop("missing result TSV")
  if (rep$n_significant < 1L) stop("expected ≥1 significant feature")
  message("SELF-TEST OK: n_sig=", rep$n_significant, " pdf=", rep$figures$pdf)
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  run_difftree_ggdiffclade(
    rds = args$rds,
    outdir = args$outdir %||% "test/difftree-ggdiffclade/run",
    target = args$target %||% args$targets,
    taxlevel = as.integer(args$taxlevel %||% 3L),
    firstalpha = as.numeric(args$firstalpha %||% 0.05),
    secondalpha = as.numeric(args$secondalpha %||% 0.01),
    subclmin = as.integer(args$subclmin %||% 3L),
    lda = as.numeric(args$lda %||% 3),
    strictmod = !identical(tolower(as.character(args$strictmod %||% "true")), "false"),
    seed = as.integer(args$seed %||% 42L),
    allow_non_rare = identical(tolower(as.character(args$allow_non_rare %||% "false")), "true"),
    min_mean_rel = as.numeric(args$min_mean_rel %||% 1e-5),
    min_sample_sum = as.integer(args$min_sample_sum %||% 1000L)
  )
}

if (sys.nframe() == 0L) {
  tryCatch(
    {
      main()
      quit(save = "no", status = 0)
    },
    error = function(e) {
      message("ERROR: ", conditionMessage(e))
      quit(save = "no", status = 1)
    }
  )
}
