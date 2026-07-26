#!/usr/bin/env Rscript
# isa — Indicator Species Analysis + grazing Figure 3 layout
suppressPackageStartupMessages({
  for (p in c(
    "phyloseq", "ggplot2", "ggpubr", "ggtext", "ggforce", "dplyr", "tidyr",
    "forcats", "stringr", "indicspecies", "eulerr", "Rtsne", "mixOmics",
    "permute", "jsonlite", "grid"
  )) {
    if (!requireNamespace(p, quietly = TRUE)) {
      stop("Missing package: ", p, call. = FALSE)
    }
  }
  library(phyloseq)
  library(ggplot2)
  library(dplyr)
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

SKIP_VARS <- c(
  "seq", "ID", "SampleID", "sampleID", "sample_id", "Run", "run",
  "Condition2", "condition2"
)

# Grazing Article palette (3-level default); Okabe–Ito otherwise
GRAZING_COLS <- c("#4E9E23", "#FFA600", "#BE475A")
OKABE <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
COLORS_NA <- "gray"

`%||%` <- function(a, b) if (!is.null(a) && length(a) && !all(is.na(a))) a else b

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

attach_tip_labels <- function(ps) {
  ps <- ensure_finalized_taxonomy(ps)
  labs <- taxon_plot_labels_from_ps(ps, taxa_names(ps))
  tt <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
  tt$taxa <- labs$display[match(rownames(tt), labs$otu)]
  tt$taxa[is.na(tt$taxa) | !nzchar(tt$taxa)] <- rownames(tt)[is.na(tt$taxa) | !nzchar(tt$taxa)]
  tt$fontface <- labs$fontface[match(rownames(tt), labs$otu)]
  tax <- as.matrix(tt)
  rownames(tax) <- rownames(tt)
  tax_table(ps) <- tax_table(tax)
  ps
}

#' Map multipatt s.* membership columns onto original target levels.
map_multipatt_mem_cols <- function(mem_cols, lev) {
  stripped <- sub("^s\\.", "", mem_cols)
  vapply(stripped, function(nm) {
    if (nm %in% lev) return(nm)
    nm_sp <- gsub("\\.", " ", nm)
    hit <- lev[tolower(lev) == tolower(nm_sp)]
    if (length(hit)) return(hit[[1]])
    hit <- lev[tolower(lev) == tolower(nm)]
    if (length(hit)) return(hit[[1]])
    # numeric codes: s.1 → "1"
    hit <- lev[as.character(lev) == as.character(nm)]
    if (length(hit)) return(hit[[1]])
    nm_sp
  }, character(1), USE.NAMES = FALSE)
}

map_to_short <- function(cond, lev, short) {
  c <- as.character(cond)
  out <- unname(short[c])
  miss <- which(is.na(out) & !is.na(c) & nzchar(c))
  for (i in miss) {
    hit <- lev[tolower(lev) == tolower(c[[i]])]
    if (length(hit)) out[[i]] <- unname(short[hit[[1]]])
  }
  out
}

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

resolve_input_rds <- function(rds = NULL, allow_relative = FALSE) {
  notes <- character(0)
  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"

  if (!is.null(rds) && nzchar(rds)) {
    loaded <- load_ps(rds)
    if (!is_count_like(loaded$ps) && !allow_relative) {
      fail(
        "ISA requires count-like abundances; median sample_sum=",
        stats::median(sample_sums(loaded$ps)),
        ". Prefer rarefied/raw (not MMUPHin relative) or pass --allow-relative true."
      )
    }
    loaded$meta$rarefied <- loaded$meta$rarefied ||
      grepl("_rare|phyloseq_rare_", basename(rds))
    loaded$notes <- notes
    return(loaded)
  }

  for (p in candidates_rare) {
    if (file.exists(p)) {
      loaded <- load_ps(p)
      if (is_count_like(loaded$ps)) {
        loaded$meta$rarefied <- TRUE
        loaded$notes <- c(notes, paste0("auto-resolved rarefied: ", p))
        return(loaded)
      }
    }
  }
  if (file.exists(raw)) {
    loaded <- load_ps(raw)
    loaded$notes <- c(notes, paste0("auto-resolved raw (will rarefy): ", raw))
    return(loaded)
  }
  fail("No phyloseq RDS found; pass --rds")
}

discover_target <- function(ps, target_arg = NULL, meta_target = NA_character_) {
  sd <- as.data.frame(sample_data(ps), stringsAsFactors = FALSE)
  if (!is.null(target_arg) && nzchar(target_arg)) {
    tg <- trimws(strsplit(target_arg, ",", fixed = TRUE)[[1]][[1]])
    if (!tg %in% names(sd)) fail("Target column not in sample_data: ", tg)
    return(tg)
  }
  if (!is.na(meta_target) && nzchar(meta_target) && meta_target %in% names(sd)) {
    return(meta_target)
  }
  for (cand in c("Condition", "grazing", "group", "Group", "treatment")) {
    if (cand %in% names(sd)) return(cand)
  }
  cats <- names(sd)[vapply(sd, function(x) {
    ux <- unique(as.character(x[!is.na(x)]))
    length(ux) >= 2L && length(ux) <= 12L && !is.numeric(x)
  }, logical(1))]
  cats <- setdiff(cats, SKIP_VARS)
  if (!length(cats)) fail("No categorical target found; pass --target")
  cats[[1]]
}

palette_for_levels <- function(levels_chr) {
  n <- length(levels_chr)
  if (n == 3L) {
    cols <- GRAZING_COLS
  } else {
    cols <- rep_len(OKABE, n)
  }
  names(cols) <- levels_chr
  cols
}

#' Document theme: prefer Article theme_main() when available.
isa_base_theme <- function() {
  if (exists("theme_main", mode = "function", inherits = TRUE)) {
    return(theme_main())
  }
  if (exists("theme_main", envir = .GlobalEnv, inherits = FALSE)) {
    tm <- get("theme_main", envir = .GlobalEnv)
    if (is.function(tm)) return(tm())
  }
  theme_minimal(base_size = 11)
}

save_panel <- function(plot_obj, prefix, width = 5, height = 4) {
  pdf <- paste0(prefix, ".pdf")
  png <- paste0(prefix, ".png")
  png_ok <- FALSE
  is_gg <- inherits(plot_obj, c("ggplot", "gg", "gtable", "arrange", "patchwork")) ||
    inherits(plot_obj, "ggassemble")
  if (is_gg || inherits(plot_obj, "ggplot")) {
    ggplot2::ggsave(pdf, plot_obj, width = width, height = height)
    tryCatch({
      grDevices::png(
        png, width = width * 300, height = height * 300, res = 300,
        type = "cairo"
      )
      print(plot_obj)
      grDevices::dev.off()
      png_ok <- TRUE
    }, error = function(e) {
      if (grDevices::dev.cur() > 1) grDevices::dev.off()
      message("PNG skip: ", conditionMessage(e))
    })
  } else {
    grDevices::pdf(pdf, width = width, height = height)
    print(plot_obj)
    grDevices::dev.off()
    tryCatch({
      grDevices::png(
        png, width = width * 300, height = height * 300, res = 300,
        type = "cairo"
      )
      print(plot_obj)
      grDevices::dev.off()
      png_ok <- TRUE
    }, error = function(e) {
      if (grDevices::dev.cur() > 1) grDevices::dev.off()
      message("PNG skip: ", conditionMessage(e))
    })
  }
  list(pdf = pdf, png = if (png_ok && file.exists(png)) png else NULL)
}

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

run_isa <- function(
    rds = NULL,
    outdir = "test/isa/run",
    target = NULL,
    top_n = 30L,
    nperm = 9999L,
    depth = NULL,
    seed = 123L,
    keepX = c(10L, 10L),
    allow_relative = FALSE,
    perplexity = 15,
    theta = 0.95
) {
  setwd(project_root())
  ensure_dir(outdir)
  loaded <- resolve_input_rds(rds, allow_relative = allow_relative)
  ps0 <- attach_tip_labels(loaded$ps)
  tg <- discover_target(ps0, target, loaded$meta$target)

  sd <- as.data.frame(sample_data(ps0), stringsAsFactors = FALSE)
  raw_tg <- sd[[tg]]
  # Drop NA / empty target samples BEFORE levels, Euler, or multipatt
  na_tg <- is.na(raw_tg) |
    (is.character(raw_tg) & !nzchar(trimws(as.character(raw_tg)))) |
    (is.factor(raw_tg) & is.na(raw_tg))
  if (any(na_tg)) {
    message(
      "Dropping ", sum(na_tg), " / ", length(na_tg),
      " samples with NA/empty target '", tg, "' (prevents NA on Euler)"
    )
    ps0 <- prune_samples(!na_tg, ps0)
    sd <- as.data.frame(sample_data(ps0), stringsAsFactors = FALSE)
    raw_tg <- sd[[tg]]
  }
  if (!length(raw_tg) || all(is.na(raw_tg))) {
    fail("No non-NA target values left in column: ", tg)
  }
  # Prefer numeric order when target is integer-coded (grazing 1/2/3 → S1/S2/S3)
  if (is.numeric(raw_tg) || all(grepl("^[0-9]+$", as.character(raw_tg[!is.na(raw_tg)])))) {
    lev <- as.character(sort(unique(as.integer(as.character(raw_tg)))))
  } else {
    lev <- levels(factor(raw_tg))
    if (is.null(lev) || !length(lev)) lev <- unique(as.character(raw_tg))
    lev <- as.character(lev[!is.na(lev) & nzchar(lev)])
  }
  if (length(lev) < 2L || length(lev) > 3L) {
    fail(
      "ISA grazing layout requires 2–3 target levels; found ",
      length(lev), " in ", tg, ": ", paste(lev, collapse = ", ")
    )
  }

  short <- paste0("S", seq_along(lev))
  names(short) <- lev
  sd$Condition <- factor(as.character(raw_tg), levels = lev)
  c2 <- map_to_short(sd$Condition, lev, short)
  n_na_c2 <- sum(is.na(c2))
  if (n_na_c2 > 0L) {
    message(
      "WARNING: ", n_na_c2, " samples have NA Condition2 after mapping target '",
      tg, "' (levels: ", paste(lev, collapse = ", "), "). Dropping those samples."
    )
    keep_s <- !is.na(c2)
    ps0 <- prune_samples(keep_s, ps0)
    sd <- as.data.frame(sample_data(ps0), stringsAsFactors = FALSE)
    sd$Condition <- factor(as.character(sd[[tg]]), levels = lev)
    c2 <- map_to_short(sd$Condition, lev, short)
  }
  if (any(is.na(c2))) fail("Condition2 still NA after drop; check target column: ", tg)
  sd$Condition2 <- factor(as.character(c2), levels = short)
  sample_data(ps0) <- sample_data(sd)

  colours.condition <- palette_for_levels(lev)
  colours.condition2 <- colours.condition
  names(colours.condition2) <- short

  # Rarefy if needed
  if (!isTRUE(loaded$meta$rarefied) || !is.null(depth)) {
    ss <- as.numeric(sample_sums(ps0))
    d <- if (!is.null(depth)) as.integer(depth) else {
      m <- min(ss)
      if (m < 1000L) 1000L else m
    }
    if (any(ss < d)) {
      keep <- ss >= d
      message("Dropping ", sum(!keep), " samples below depth ", d)
      ps0 <- prune_samples(keep, ps0)
    }
    set.seed(as.integer(seed))
    ps_rf <- rarefy_even_depth(ps0, sample.size = d, rngseed = as.integer(seed), verbose = FALSE)
    loaded$meta$rarefied <- TRUE
    loaded$meta$rarefaction_depth <- d
  } else {
    ps_rf <- ps0
    if (is.na(loaded$meta$rarefaction_depth)) {
      loaded$meta$rarefaction_depth <- round(stats::median(sample_sums(ps_rf)))
    }
  }

  ps_norm <- transform_sample_counts(ps_rf, function(x) x / sum(x))

  # Long abundance for euler + boxplots
  otu_df <- as.data.frame(otu_table(ps_rf))
  if (!taxa_are_rows(ps_rf)) otu_df <- as.data.frame(t(otu_df))
  data.abundance.rf <- otu_df %>%
    tibble::rownames_to_column("OTU") %>%
    tidyr::pivot_longer(-OTU, names_to = "Group", values_to = "Abundance") %>%
    left_join(
      as.data.frame(sample_data(ps_rf)) %>% tibble::rownames_to_column("Group"),
      by = "Group"
    ) %>%
    left_join(
      as.data.frame(tax_table(ps_rf)) %>% tibble::rownames_to_column("OTU"),
      by = "OTU"
    ) %>%
    dplyr::filter(!is.na(Condition2) & !is.na(Condition))

  if (any(is.na(data.abundance.rf$Condition2))) {
    fail("Internal error: Condition2 still NA in abundance table after filter")
  }

  data.euler <- data.abundance.rf %>%
    dplyr::summarize(Abundance = mean(Abundance), .by = c(OTU, Condition2)) %>%
    tidyr::pivot_wider(
      values_from = Abundance, names_from = Condition2, id_cols = OTU
    ) %>%
    tibble::column_to_rownames("OTU")
  # Keep only known short labels — never pass NA colnames to eulerr
  keep_e <- intersect(colnames(data.euler), as.character(short))
  if (!length(keep_e)) fail("Euler matrix has no valid Condition2 columns")
  data.euler <- data.euler[, keep_e, drop = FALSE]
  data.euler[is.na(data.euler)] <- 0
  data.euler[data.euler > 0] <- 1
  if (any(is.na(colnames(data.euler)))) fail("Euler colnames contain NA")

  fill_ord <- short[c(length(short), seq_len(length(short) - 1L))]
  if (length(short) == 3L) fill_ord <- short[c(3, 1, 2)]
  fill_ord <- intersect(as.character(fill_ord), colnames(data.euler))
  # Keep names so fig3b can index by colnames (unname+char-index → NA fills)
  fill_named <- colours.condition2[fill_ord]
  fill_cols <- unname(fill_named)

  fig3a <- data.euler[, fill_ord, drop = FALSE] %>%
    eulerr::venn() %>%
    plot(fill = fill_cols)

  # multipatt
  X <- as(otu_table(ps_norm), "matrix")
  if (!taxa_are_rows(ps_norm)) X <- t(X)
  X <- t(X) # samples × taxa
  cluster <- as.character(sample_data(ps_norm)$Condition)
  if (any(is.na(cluster) | !nzchar(cluster))) {
    fail("multipatt cluster has NA/empty Condition after target cleanup")
  }
  set.seed(as.integer(seed))
  isa.results <- indicspecies::multipatt(
    x = X,
    cluster = cluster,
    func = "r.g",
    control = permute::how(nperm = as.integer(nperm))
  )

  sign_df <- isa.results$sign %>%
    as.data.frame() %>%
    tibble::rownames_to_column("OTU") %>%
    arrange(dplyr::desc(stat)) %>%
    subset(p.value < 0.05)

  if (!nrow(sign_df)) fail("No significant indicator taxa at p < 0.05")

  # membership columns: s.<level>
  mem_cols <- grep("^s\\.", names(sign_df), value = TRUE)
  if (!length(mem_cols)) fail("multipatt sign table missing s.* membership columns")

  # Rename membership columns to original levels (robust to s.1 / s.grazing2)
  new_names <- map_multipatt_mem_cols(mem_cols, lev)
  names(sign_df)[match(mem_cols, names(sign_df))] <- new_names
  conditions <- intersect(lev, new_names)
  if (!length(conditions)) {
    # fall back to renamed columns that look like single-group membership
    conditions <- new_names[!grepl("\\s|\\+", new_names)]
    message("WARNING: multipatt columns did not match levels; using: ",
            paste(conditions, collapse = ", "))
  }
  if (!length(conditions)) fail("Could not map multipatt membership columns to target levels")

  data.isa <- sign_df %>%
    left_join(
      as.data.frame(tax_table(ps_norm)) %>% tibble::rownames_to_column("OTU"),
      by = "OTU"
    )

  utils::write.csv(data.isa, file.path(outdir, "ISA_sp.csv"), row.names = FALSE)

  data.isa.long <- data.isa %>%
    tidyr::pivot_longer(cols = all_of(conditions), names_to = "Condition", values_to = "value") %>%
    mutate(
      Condition2 = factor(map_to_short(Condition, lev, short), levels = short)
    )
  n_na_isa <- sum(is.na(data.isa.long$Condition2) & data.isa.long$value > 0)
  if (n_na_isa > 0L) {
    message("WARNING: ", n_na_isa, " ISA membership rows have NA Condition2; dropping")
  }
  data.isa.long <- data.isa.long %>%
    subset(value > 0 & !is.na(Condition2)) %>%
    arrange(dplyr::desc(stat))

  if (!nrow(data.isa.long)) fail("No ISA membership rows after Condition2 mapping")

  utils::write.table(
    data.isa.long, file.path(outdir, "isa_long.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  # Euler b/c — colnames must be short labels with no NA
  euler_isa_mat <- as.data.frame(data.isa[, conditions, drop = FALSE])
  if (length(conditions) == 3L && all(conditions %in% lev)) {
    ord_c <- conditions[match(lev[c(3, 1, 2)], conditions)]
    ord_c <- ord_c[!is.na(ord_c)]
    euler_isa_mat <- euler_isa_mat[, ord_c, drop = FALSE]
    colnames(euler_isa_mat) <- unname(short[ord_c])
  } else {
    colnames(euler_isa_mat) <- unname(short[conditions])
  }
  if (any(is.na(colnames(euler_isa_mat)))) {
    fail(
      "ISA Euler colnames NA after mapping; conditions=",
      paste(conditions, collapse = ","), " lev=", paste(lev, collapse = ",")
    )
  }
  fig3b_fills <- unname(colours.condition2[colnames(euler_isa_mat)])
  if (any(is.na(fig3b_fills))) {
    fail(
      "ISA fig3b fill colours NA for colnames: ",
      paste(colnames(euler_isa_mat), collapse = ","),
      " palette names: ", paste(names(colours.condition2), collapse = ",")
    )
  }
  fig3b <- euler_isa_mat %>%
    eulerr::venn() %>%
    plot(fill = fig3b_fills)

  fig3c_mat <- data.euler[intersect(data.isa$OTU, rownames(data.euler)), fill_ord, drop = FALSE]
  fig3c <- fig3c_mat %>%
    eulerr::venn() %>%
    plot(fill = fill_cols)

  n.OTUs <- min(as.integer(top_n), nrow(data.isa.long), length(unique(data.isa.long$OTU)))
  # Top N unique OTUs by stat
  top_otus <- data.isa.long %>%
    distinct(OTU, .keep_all = TRUE) %>%
    arrange(dplyr::desc(stat)) %>%
    head(n.OTUs) %>%
    pull(OTU)

  data.isa.long.prep <- data.isa.long %>%
    subset(OTU %in% top_otus) %>%
    mutate(OTU = forcats::fct_reorder(OTU, -stat))

  taxa_lab <- data.isa.long.prep %>%
    distinct(OTU, .keep_all = TRUE)
  taxa_map <- setNames(as.character(taxa_lab$taxa), as.character(taxa_lab$OTU))

  fig3d1 <- data.isa.long.prep %>%
    ggplot(aes(x = OTU, y = stat, fill = Condition2)) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    labs(x = "", y = "Indicator Value") +
    isa_base_theme() +
    scale_fill_manual(NULL, values = colours.condition2, breaks = names(colours.condition2)) +
    scale_x_discrete(labels = function(x) unname(taxa_map[x])) +
    theme(
      axis.text.y = element_text(face = "italic"),
      panel.grid.major.y = element_blank(),
      legend.position = "bottom"
    )

  fig3d2_df <- data.isa.long.prep %>%
    distinct(OTU, .keep_all = TRUE) %>%
    mutate(
      OTU = forcats::fct_reorder(OTU, -stat),
      neglog10_p = -log10(pmax(as.numeric(p.value), .Machine$double.xmin))
    )
  # Audit: identical p-values often = permutation floor 1/(nperm+1)
  p_vals <- as.numeric(fig3d2_df$p.value)
  p_unique <- length(unique(round(p_vals, 10)))
  p_floor <- 1 / (as.integer(nperm) + 1)
  p_audit <- data.frame(
    OTU = as.character(fig3d2_df$OTU),
    taxa = unname(taxa_map[as.character(fig3d2_df$OTU)]),
    stat = as.numeric(fig3d2_df$stat),
    p.value = p_vals,
    neglog10_p = as.numeric(fig3d2_df$neglog10_p),
    at_perm_floor = abs(p_vals - p_floor) < 1e-12,
    stringsAsFactors = FALSE
  )
  utils::write.table(
    p_audit, file.path(outdir, "isa_pvalue_audit.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )
  if (p_unique <= 1L && nrow(fig3d2_df) > 1L) {
    message(
      "WARNING: all plotted p-values identical (",
      unique(p_vals)[[1]],
      "). With nperm=", nperm, " the floor is ~",
      signif(p_floor, 3),
      ". Use --nperm 9999 for finer resolution; y-axis is -log10(p)."
    )
  } else if (p_unique < nrow(fig3d2_df) / 2) {
    message(
      "NOTE: only ", p_unique, " unique p-values among ", nrow(fig3d2_df),
      " taxa (nperm=", nperm, "); many may sit at the permutation floor ",
      signif(p_floor, 3), "."
    )
  }

  fig3d2 <- fig3d2_df %>%
    ggplot(aes(x = OTU, y = neglog10_p, fill = neglog10_p)) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    labs(x = "ASVs", y = "-log<sub>10</sub>(p)") +
    isa_base_theme() +
    scale_fill_gradient("-log10(p)", low = "lightgreen", high = "purple") +
    scale_x_discrete(labels = function(x) unname(taxa_map[x])) +
    theme(
      axis.text.y = element_blank(),
      axis.title.x = ggtext::element_markdown(),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(vjust = 0.8)
    )

  data.isa.boxplot <- data.abundance.rf %>%
    group_by(Group) %>%
    mutate(value = Abundance / sum(Abundance)) %>%
    ungroup() %>%
    left_join(data.isa, by = "OTU") %>%
    subset(!is.na(p.value)) %>%
    arrange(dplyr::desc(stat)) %>%
    mutate(OTU = forcats::fct_inorder(OTU)) %>%
    subset(OTU %in% top_otus) %>%
    mutate(OTU = forcats::fct_reorder(OTU, -stat))

  fig3d3 <- data.isa.boxplot %>%
    ggplot(aes(x = OTU, y = value, fill = Condition2)) +
    geom_boxplot(width = 0.7, outlier.shape = NA, alpha = 0.5) +
    ggpubr::stat_compare_means(
      label.x.npc = 0, label.y = 0.55,
      vjust = 0.8, hjust = 0, label = "p.signif"
    ) +
    labs(y = "Abundance", x = "") +
    scale_fill_manual(NULL, values = colours.condition2, breaks = names(colours.condition2)) +
    isa_base_theme() +
    coord_flip() +
    scale_y_sqrt() +
    theme(
      axis.text.y = element_blank(),
      panel.grid.minor.x = element_blank(),
      legend.position = "none"
    )

  # t-SNE on taxa (rows)
  otu_mat <- as(otu_table(ps_rf), "matrix")
  if (!taxa_are_rows(ps_rf)) otu_mat <- t(otu_mat)
  n_taxa <- nrow(otu_mat)
  perp <- min(as.numeric(perplexity), max(2, floor((n_taxa - 1) / 3)))
  set.seed(as.integer(seed))
  tsne.result <- Rtsne::Rtsne(
    otu_mat, check_duplicates = FALSE,
    perplexity = perp, theta = as.numeric(theta)
  )

  data.tsne <- data.frame(
    TSNE1 = tsne.result$Y[, 1],
    TSNE2 = tsne.result$Y[, 2],
    OTU = rownames(otu_mat),
    stringsAsFactors = FALSE
  ) %>%
    left_join(data.isa, by = "OTU")

  # Non-indicators: one gray disc per OTU
  data.tsne.na <- data.tsne %>%
    subset(is.na(index)) %>%
    distinct(OTU, TSNE1, TSNE2)

  # Indicators: pie slices only for positive memberships (amount > 0)
  data.tsne.ind <- data.tsne %>%
    subset(!is.na(index)) %>%
    tidyr::pivot_longer(cols = all_of(conditions), names_to = "name", values_to = "value") %>%
    mutate(
      value = as.numeric(value),
      value = ifelse(is.na(value), 0, value),
      name = factor(as.character(name), levels = lev)
    ) %>%
    subset(value > 0)

  fig3e <- ggplot()
  if (nrow(data.tsne.na) > 0L) {
    fig3e <- fig3e +
      ggforce::geom_arc_bar(
        data = data.tsne.na,
        aes(x0 = TSNE1, y0 = TSNE2, r0 = 0, r = 0.5, amount = 1),
        stat = "pie", alpha = 0.2, n = 100, fill = COLORS_NA, color = "transparent"
      )
  }
  if (nrow(data.tsne.ind) > 0L) {
    fig3e <- fig3e +
      ggforce::geom_arc_bar(
        data = data.tsne.ind,
        aes(x0 = TSNE1, y0 = TSNE2, r0 = 0, r = 0.5, amount = value, fill = name),
        color = "transparent",
        stat = "pie", alpha = 0.5, n = 100
      )
  }
  fig3e <- fig3e +
    coord_fixed() +
    labs(x = "t-SNE 1", y = "t-SNE 2") +
    scale_fill_manual(values = colours.condition, breaks = names(colours.condition), drop = FALSE) +
    isa_base_theme() +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      legend.position = "none"
    )

  # sPLS-DA
  X_spls <- t(as(otu_table(ps_rf), "matrix"))
  if (!taxa_are_rows(ps_rf)) {
    X_spls <- as(otu_table(ps_rf), "matrix")
  } else {
    X_spls <- t(as(otu_table(ps_rf), "matrix"))
  }
  Y <- sample_data(ps_rf)$Condition
  set.seed(as.integer(seed))
  splsda.result <- mixOmics::splsda(
    X = X_spls, Y = Y, ncomp = 2, keepX = as.integer(keepX)
  )

  data.splsda <- data.frame(
    Sample = rownames(X_spls),
    sPLSDA1 = splsda.result$variates$X[, 1],
    sPLSDA2 = splsda.result$variates$X[, 2],
    Condition = Y,
    stringsAsFactors = FALSE
  )

  fig3f <- ggplot(data.splsda, aes(x = sPLSDA1, y = sPLSDA2, color = Condition, fill = Condition)) +
    geom_point(size = 3) +
    stat_ellipse(level = 0.95, geom = "polygon", alpha = 0.1) +
    scale_color_manual(values = colours.condition, breaks = names(colours.condition)) +
    scale_fill_manual(values = colours.condition, breaks = names(colours.condition)) +
    labs(x = "sPLS-DA comp 1", y = "sPLS-DA comp 2") +
    isa_base_theme() +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      legend.position = "none"
    )

  data.splsda.loadings <- data.frame(
    OTU = rownames(splsda.result$loadings$X),
    Loading1 = splsda.result$loadings$X[, 1],
    Loading2 = splsda.result$loadings$X[, 2],
    stringsAsFactors = FALSE
  ) %>%
    tidyr::pivot_longer(cols = -1, names_to = "loading", values_to = "vals") %>%
    mutate(loading = stringr::str_replace_all(loading, "Loading", "sPLS-DA loading ")) %>%
    arrange(dplyr::desc(abs(vals))) %>%
    left_join(data.isa, by = "OTU") %>%
    tidyr::pivot_longer(cols = all_of(conditions), names_to = "name", values_to = "value") %>%
    subset(abs(vals) > 0) %>%
    mutate(name = ifelse(abs(value) != 0, name, NA_character_)) %>%
    group_by(OTU) %>%
    mutate(has_non_na_ISA = any(!is.na(name))) %>%
    ungroup() %>%
    subset(!is.na(name) | !has_non_na_ISA) %>%
    mutate(OTU = forcats::fct_reorder(OTU, vals))

  if ("taxa" %in% names(data.splsda.loadings)) {
    data.splsda.loadings$taxa[is.na(data.splsda.loadings$taxa)] <- "unclassified**"
  } else {
    data.splsda.loadings$taxa <- as.character(data.splsda.loadings$OTU)
  }

  lab_load <- data.splsda.loadings %>%
    distinct(OTU, .keep_all = TRUE)
  lab_map <- setNames(as.character(lab_load$taxa), as.character(lab_load$OTU))

  fig3g <- data.splsda.loadings %>%
    ggplot(aes(x = OTU, y = vals, fill = name)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~loading, scales = "free") +
    labs(x = "ASVs", y = "Loading Value") +
    scale_x_discrete(labels = function(x) unname(lab_map[x])) +
    coord_flip() +
    scale_fill_manual(values = colours.condition, breaks = names(colours.condition), na.value = COLORS_NA) +
    scale_y_continuous(breaks = seq(-1, 1, by = 0.2)) +
    isa_base_theme() +
    theme(axis.text.y = element_text(face = "italic"), legend.position = "none")

  # Save individual panels
  panels <- list()
  panels$a <- save_panel(fig3a, file.path(outdir, "isa_fig3a_euler_all"), 4, 4)
  panels$b <- save_panel(fig3b, file.path(outdir, "isa_fig3b_euler_isa"), 4, 4)
  panels$c <- save_panel(fig3c, file.path(outdir, "isa_fig3c_euler_occ"), 4, 4)
  panels$d1 <- save_panel(fig3d1, file.path(outdir, "isa_fig3d1_indval"), 6, 8)
  panels$d2 <- save_panel(fig3d2, file.path(outdir, "isa_fig3d2_pvalue"), 3, 8)
  panels$d3 <- save_panel(fig3d3, file.path(outdir, "isa_fig3d3_abundance"), 6, 8)
  panels$e <- save_panel(fig3e, file.path(outdir, "isa_fig3e_tsne"), 5, 5)
  panels$f <- save_panel(fig3f, file.path(outdir, "isa_fig3f_splsda"), 5, 5)
  panels$g <- save_panel(fig3g, file.path(outdir, "isa_fig3g_loadings"), 8, 5)

  # Combined grazing layout (legend row may be empty if guides unavailable)
  legend_row <- tryCatch(
    ggpubr::ggarrange(
      ggpubr::get_legend(fig3d2), ggpubr::get_legend(fig3d1),
      align = "h"
    ),
    error = function(e) ggplot() + theme_void()
  )

  combined <- ggpubr::ggarrange(
    ggpubr::ggarrange(
      fig3a, fig3b, fig3c,
      labels = c("(a)", "(b)", "(c)"),
      nrow = 1, align = "h"
    ),
    ggpubr::ggarrange(
      fig3d2, fig3d1, fig3d3,
      labels = c("(d)", "", ""),
      widths = c(1, 5, 5), legend = "none",
      align = "h", nrow = 1
    ),
    ggpubr::ggarrange(
      fig3e, fig3f, fig3g,
      widths = c(1, 1, 2), nrow = 1,
      labels = c("(e)", "(f)", "(g)"), legend = "none"
    ),
    legend_row,
    align = "hv",
    nrow = 4, heights = c(1, 2.2, 1, 0.2)
  )

  fig_pdf <- file.path(outdir, "isa_figure3.pdf")
  fig_png <- file.path(outdir, "isa_figure3.png")
  ggplot2::ggsave(fig_pdf, combined, width = 12, height = 14)
  fig_png_out <- NULL
  tryCatch({
    grDevices::png(
      fig_png, width = 12 * 300, height = 14 * 300, res = 300,
      type = "cairo"
    )
    print(combined)
    grDevices::dev.off()
    fig_png_out <- fig_png
  }, error = function(e) {
    if (grDevices::dev.cur() > 1) grDevices::dev.off()
    message("Combined PNG skip: ", conditionMessage(e))
  })

  report <- list(
    input_rds = loaded$meta$path,
    rarefied = loaded$meta$rarefied,
    rarefaction_depth = loaded$meta$rarefaction_depth,
    notes = loaded$notes,
    target = tg,
    levels = lev,
    short_labels = as.list(short),
    nperm = as.integer(nperm),
    func = "r.g",
    p_cutoff = 0.05,
    n_unique_p_plotted = p_unique,
    p_perm_floor = p_floor,
    n_at_perm_floor = sum(p_audit$at_perm_floor),
    pvalue_audit = file.path(outdir, "isa_pvalue_audit.tsv"),
    top_n = as.integer(n.OTUs),
    n_significant = nrow(data.isa),
    keepX = as.integer(keepX),
    seed = as.integer(seed),
    isa_table = file.path(outdir, "ISA_sp.csv"),
    isa_long = file.path(outdir, "isa_long.tsv"),
    figure_combined = list(pdf = fig_pdf, png = fig_png_out),
    panels = panels
  )
  write_json(report, file.path(outdir, "isa-report.json"))
  message(
    "ISA OK: n_sig=", report$n_significant,
    " figure=", fig_pdf
  )
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  out <- "test/isa/grazing-self-test"
  rep <- run_isa(
    rds = "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    outdir = out,
    target = "grazing",
    top_n = 20L,
    nperm = 999L,
    seed = 123L
  )
  if (!file.exists(rep$figure_combined$pdf)) stop("missing combined PDF")
  if (!file.exists(rep$isa_table)) stop("missing ISA_sp.csv")
  if (rep$n_significant < 1L) stop("expected significant indicators")
  if (!file.exists(file.path(out, "isa_pvalue_audit.tsv"))) stop("missing p-value audit")
  long <- utils::read.table(rep$isa_long, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  if (any(is.na(long$Condition2))) stop("isa_long has NA Condition2")
  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  keepX <- c(10L, 10L)
  if (!is.null(args$keepX) && nzchar(args$keepX)) {
    keepX <- as.integer(trimws(strsplit(args$keepX, ",", fixed = TRUE)[[1]]))
  }
  run_isa(
    rds = args$rds,
    outdir = args$outdir %||% "test/isa/run",
    target = args$target %||% args$targets,
    top_n = as.integer(args$top_n %||% 30L),
    nperm = as.integer(args$nperm %||% 9999L),
    depth = if (!is.null(args$depth)) as.integer(args$depth) else NULL,
    seed = as.integer(args$seed %||% 123L),
    keepX = keepX,
    allow_relative = identical(tolower(as.character(args$allow_relative %||% "false")), "true"),
    perplexity = as.numeric(args$perplexity %||% 15),
    theta = as.numeric(args$theta %||% 0.95)
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
