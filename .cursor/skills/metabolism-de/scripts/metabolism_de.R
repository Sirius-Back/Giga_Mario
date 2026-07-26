#!/usr/bin/env Rscript
# metabolism-de — ANCOM-BC2 on Bakta metabolic tables (no GO) + volcano / LFC±SE
suppressPackageStartupMessages({
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ANCOMBC", quietly = TRUE))
  library(phyloseq)
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

COL_UP <- "#D81B60"
COL_DOWN <- "#1B9E77"
COL_NS <- "grey70"
METABOLISM_TYPES <- c("product", "ec", "ko")
GO_TYPES <- c("go", "GO", "ontology")

read_table_auto <- function(path) {
  if (!file.exists(path)) fail("File not found: ", path)
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("csv")) {
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  }
}

normalize_long <- function(df) {
  names(df) <- tolower(names(df))
  need <- c("sample", "function_type", "function_id", "count")
  if (!all(need %in% names(df))) {
    fail("Long table missing: ", paste(setdiff(need, names(df)), collapse = ", "))
  }
  df$sample <- as.character(df$sample)
  df$function_type <- tolower(as.character(df$function_type))
  df$function_id <- as.character(df$function_id)
  df$count <- as.numeric(df$count)
  if (any(is.na(df$count)) || any(df$count < 0)) fail("Invalid counts")
  if (!"function_label" %in% names(df)) {
    df$function_label <- paste0(df$function_type, "|", df$function_id)
  }
  df[, c("sample", "function_type", "function_id", "count", "function_label")]
}

filter_metab <- function(long_df, types = "product", drop_hypothetical = TRUE) {
  n_go <- sum(long_df$function_type %in% GO_TYPES)
  df <- long_df[!long_df$function_type %in% GO_TYPES, , drop = FALSE]
  types <- tolower(trimws(unlist(strsplit(as.character(types), ",", fixed = TRUE))))
  bad <- setdiff(types, METABOLISM_TYPES)
  if (length(bad)) fail("Unsupported types (GO excluded): ", paste(bad, collapse = ","))
  df <- df[df$function_type %in% types, , drop = FALSE]
  if (!nrow(df)) fail("No rows after type filter")
  n_hyp <- 0L
  if (isTRUE(drop_hypothetical) && "product" %in% types) {
    hyp <- grepl("hypothetical protein", df$function_id, ignore.case = TRUE)
    n_hyp <- sum(hyp)
    df <- df[!hyp, , drop = FALSE]
  }
  if (!nrow(df)) fail("No rows after hypothetical drop")
  df <- stats::aggregate(
    count ~ sample + function_type + function_id + function_label,
    data = df, FUN = sum
  )
  list(long = df, n_go_dropped = as.integer(n_go), n_hyp = as.integer(n_hyp), types = types)
}

build_count_matrix <- function(long_df) {
  genes <- sort(unique(long_df$function_label))
  samples <- sort(unique(long_df$sample))
  mat <- matrix(0L, nrow = length(genes), ncol = length(samples),
                dimnames = list(genes, samples))
  for (i in seq_len(nrow(long_df))) {
    mat[long_df$function_label[[i]], long_df$sample[[i]]] <-
      as.integer(round(long_df$count[[i]]))
  }
  mat
}

#' Rarefy metabolic counts using sample_sums from a related phyloseq object.
rarefy_mat_to_ps <- function(mat, ps, seed = 123L) {
  depths0 <- as.numeric(sample_sums(ps))
  names(depths0) <- sample_names(ps)
  notes <- character(0)
  if (!length(depths0) || any(!is.finite(depths0)) || any(depths0 <= 0)) {
    notes <- c(notes, "ps rarefy skipped: invalid sample_sums")
    return(list(mat = mat, notes = notes, depth = NA_real_))
  }
  if (max(depths0, na.rm = TRUE) < 10) {
    notes <- c(notes, paste0(
      "ps rarefy skipped: sample_sums look like relative abundances (max=",
      max(depths0, na.rm = TRUE), ")"
    ))
    return(list(mat = mat, notes = notes, depth = NA_real_))
  }
  # Align IDs (exact or ±_smir) — same rules as metabolism skill
  strip <- function(x) sub("_smir$", "", x)
  common <- intersect(colnames(mat), names(depths0))
  depths <- depths0
  if (length(common) < 2L) {
    m_strip <- strip(colnames(mat))
    hit <- match(m_strip, names(depths0))
    ok <- !is.na(hit)
    if (sum(ok) >= 2L) {
      keep_m <- colnames(mat)[ok]
      keep_p <- names(depths0)[hit[ok]]
      depths <- stats::setNames(as.numeric(depths0[keep_p]), keep_m)
      common <- keep_m
      notes <- c(notes, "aligned metabolic↔phyloseq sample IDs (±_smir)")
    }
  }
  if (length(common) < 2L) {
    p_strip <- strip(names(depths0))
    hit2 <- match(colnames(mat), p_strip)
    ok2 <- !is.na(hit2)
    if (sum(ok2) >= 2L) {
      keep_m <- colnames(mat)[ok2]
      keep_p <- names(depths0)[hit2[ok2]]
      depths <- stats::setNames(as.numeric(depths0[keep_p]), keep_m)
      common <- keep_m
      notes <- c(notes, "aligned metabolic↔phyloseq sample IDs (±_smir)")
    }
  }
  if (length(common) < 2L) {
    notes <- c(notes, "ps rarefy skipped: <2 overlapping samples")
    return(list(mat = mat, notes = notes, depth = NA_real_))
  }
  mat <- mat[, common, drop = FALSE]
  depths <- depths[common]
  cs <- colSums(mat)
  min_lib <- 50L
  keep_lib <- cs >= min_lib
  if (sum(keep_lib) < 2L) {
    notes <- c(notes, paste0("ps rarefy skipped: <2 samples with library ≥ ", min_lib))
    return(list(mat = mat, notes = notes, depth = NA_real_))
  }
  if (any(!keep_lib)) {
    notes <- c(notes, paste0(
      "dropped ", sum(!keep_lib), " metabolic samples with library size < ", min_lib
    ))
  }
  mat <- mat[, keep_lib, drop = FALSE]
  depths <- depths[colnames(mat)]
  cs <- colSums(mat)
  ps_min <- max(1L, as.integer(floor(min(depths))))
  metab_min <- max(1L, as.integer(floor(min(cs))))
  # ANCOM-BC needs intact gene counts — skip rarefy when depth would destroy sparsity
  can_rarefy <- ps_min >= 1000L && ps_min <= metab_min
  if (can_rarefy) {
    target <- ps_min
    set.seed(as.integer(seed))
    if (requireNamespace("vegan", quietly = TRUE)) {
      rare <- t(vegan::rrarefy(t(mat), sample = target))
    } else {
      tmp <- phyloseq(otu_table(mat, taxa_are_rows = TRUE))
      tmp <- rarefy_even_depth(
        tmp, sample.size = target, rngseed = as.integer(seed),
        replace = FALSE, trimOTUs = FALSE, verbose = FALSE
      )
      rare <- as(otu_table(tmp), "matrix")
      if (!taxa_are_rows(tmp)) rare <- t(rare)
      out <- matrix(0L, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
      out[rownames(rare), colnames(rare)] <- rare
      rare <- out
    }
    notes <- c(notes, paste0("rarefied metabolic counts to phyloseq min depth=", target))
    list(mat = rare, notes = notes, depth = target)
  } else {
    notes <- c(notes, paste0(
      "skipped rarefy for DE (phyloseq min=", ps_min, ", metabolic min=", metab_min,
      "); using raw counts after dropping low-library samples"
    ))
    list(mat = mat, notes = notes, depth = NA_real_)
  }
}

load_metadata <- function(metadata = NULL, ps_rds = NULL, group_col = "group") {
  if (!is.null(ps_rds) && nzchar(ps_rds)) {
    if (!file.exists(ps_rds)) fail("phyloseq RDS missing: ", ps_rds)
    ps <- readRDS(ps_rds)
    if (is.list(ps) && inherits(ps$phyloseq, "phyloseq")) ps <- ps$phyloseq
    if (!inherits(ps, "phyloseq")) fail("--ps-rds must be phyloseq")
    sd <- as(sample_data(ps), "data.frame")
    sd$sample <- rownames(sd)
    if (!group_col %in% names(sd) && "Группа" %in% names(sd) &&
        identical(group_col, "group")) {
      sd$group <- sd[["Группа"]]
    }
    meta <- sd
    path <- ps_rds
  } else if (!is.null(metadata) && nzchar(metadata)) {
    meta <- read_table_auto(metadata)
    nms <- tolower(names(meta))
    if ("sample_id" %in% nms && !"sample" %in% nms) {
      names(meta)[match("sample_id", nms)] <- "sample"
    }
    names(meta)[match("sample", tolower(names(meta)))] <- "sample"
    meta$sample <- as.character(meta$sample)
    path <- metadata
  } else {
    fail("Provide --metadata or --ps-rds")
  }
  if (!group_col %in% names(meta)) {
    fail("Missing group column '", group_col, "'. Have: ", paste(names(meta), collapse = ", "))
  }
  meta$group <- factor(as.character(meta[[group_col]]))
  if (nlevels(meta$group) < 2L) fail("Need ≥2 groups")
  list(meta = meta, path = path, group_col = group_col)
}

matrix_to_phyloseq <- function(mat, meta) {
  samples <- intersect(colnames(mat), meta$sample)
  if (length(samples) < 4L) fail("Need ≥4 overlapping samples; got ", length(samples))
  mat <- mat[, samples, drop = FALSE]
  meta <- meta[match(samples, meta$sample), , drop = FALSE]
  rownames(meta) <- meta$sample
  # sanitize group for ANCOMBC
  raw <- levels(factor(as.character(meta$group)))
  safe <- make.names(raw, unique = TRUE)
  map <- setNames(raw, safe)
  meta$group <- factor(safe[match(as.character(meta$group), raw)], levels = safe)
  mat <- mat[rowSums(mat) > 0, , drop = FALSE]
  # Phyloseq/ANCOMBC require unique taxa_names (make.names of Bakta labels can collide)
  feat_raw <- rownames(mat)
  feat_id <- make.names(feat_raw, unique = TRUE)
  if (any(feat_id != feat_raw) || anyDuplicated(feat_raw)) {
    rownames(mat) <- feat_id
  }
  tax <- matrix(
    cbind(
      feature = feat_raw,
      function_type = sub("\\|.*$", "", feat_raw),
      function_id = sub("^[^|]+\\|", "", feat_raw)
    ),
    ncol = 3,
    dimnames = list(rownames(mat), c("feature", "function_type", "function_id"))
  )
  ps <- phyloseq(
    otu_table(mat, taxa_are_rows = TRUE),
    sample_data(meta[, "group", drop = FALSE]),
    tax_table(tax)
  )
  list(ps = ps, level_map = map, samples = samples, feature_map = setNames(feat_raw, rownames(mat)))
}

tidy_ancombc2 <- function(out, level_map = NULL) {
  if (is.null(out) || is.null(out$res)) return(NULL)
  res <- out$res
  # ANCOMBC 2.x: res may be data.frame with lfc_/se_ columns or list with matrices
  if (is.data.frame(res)) {
    res_df <- res
    if (!"taxon" %in% names(res_df) && !is.null(rownames(res_df))) {
      res_df$taxon <- rownames(res_df)
    }
  } else if (is.list(res) && !is.null(res$lfc)) {
    lfc <- as.data.frame(res$lfc)
    se <- as.data.frame(res$se)
    p <- as.data.frame(res$p_val)
    q <- as.data.frame(res$q_val)
    taxa <- rownames(lfc)
    res_df <- data.frame(taxon = taxa, stringsAsFactors = FALSE)
    for (nm in names(lfc)) {
      res_df[[paste0("lfc_", nm)]] <- lfc[[nm]]
      res_df[[paste0("se_", nm)]] <- se[[nm]]
      res_df[[paste0("p_", nm)]] <- p[[nm]]
      res_df[[paste0("q_", nm)]] <- q[[nm]]
    }
  } else {
    return(NULL)
  }
  lfc_cols <- grep("^lfc_", names(res_df), value = TRUE)
  lfc_cols <- lfc_cols[!grepl("Intercept", lfc_cols, ignore.case = TRUE)]
  if (!length(lfc_cols)) return(NULL)
  rows <- list()
  k <- 1L
  for (lc in lfc_cols) {
    term <- sub("^lfc_", "", lc)
    se_c <- paste0("se_", term)
    p_c <- paste0("p_", term)
    q_c <- paste0("q_", term)
    # alternate naming in some ANCOMBC versions
    if (!p_c %in% names(res_df)) p_c <- paste0("p_val_", term)
    if (!q_c %in% names(res_df)) q_c <- paste0("q_val_", term)
    if (!se_c %in% names(res_df)) se_c <- paste0("se_", term)
    for (i in seq_len(nrow(res_df))) {
      lfc_v <- as.numeric(res_df[[lc]][[i]])
      se_v <- if (se_c %in% names(res_df)) as.numeric(res_df[[se_c]][[i]]) else NA_real_
      p_v <- if (p_c %in% names(res_df)) as.numeric(res_df[[p_c]][[i]]) else NA_real_
      q_v <- if (q_c %in% names(res_df)) as.numeric(res_df[[q_c]][[i]]) else NA_real_
      rows[[k]] <- data.frame(
        feature = as.character(res_df$taxon[[i]]),
        term = term,
        lfc = lfc_v,
        se = se_v,
        p = p_v,
        q = q_v,
        log2_lfc = lfc_v / log(2),
        lfcSE = se_v / log(2),
        stringsAsFactors = FALSE
      )
      k <- k + 1L
    }
  }
  out_df <- do.call(rbind, rows)
  # restore readable group labels in term if make.names was used
  if (!is.null(level_map)) {
    for (nm in names(level_map)) {
      out_df$term <- gsub(nm, level_map[[nm]], out_df$term, fixed = TRUE)
    }
  }
  out_df
}

run_ancombc_metab <- function(ps, prv_cut = 0.1, seed = 123L) {
  set.seed(as.integer(seed))
  pairwise <- nlevels(sample_data(ps)$group) > 2L
  message(
    "ANCOM-BC2 metabolism-DE: ntaxa=", ntaxa(ps),
    " nsamples=", nsamples(ps), " pairwise=", pairwise
  )
  out <- tryCatch(
    ANCOMBC::ancombc2(
      data = ps,
      assay_name = "counts",
      tax_level = NULL,
      fix_formula = "group",
      rand_formula = NULL,
      p_adj_method = "fdr",
      pseudo = 0,
      pseudo_sens = FALSE,
      prv_cut = prv_cut,
      group = "group",
      global = FALSE,
      pairwise = pairwise,
      struc_zero = FALSE,
      neg_lb = FALSE,
      verbose = FALSE,
      n_cl = 1L
    ),
    error = function(e) {
      message("ancombc2 failed: ", conditionMessage(e))
      NULL
    }
  )
  out
}

classify_deg <- function(df, lfc_cut = 1, padj_cut = 0.05) {
  lab <- rep("NS", nrow(df))
  ok <- !is.na(df$q) & !is.na(df$log2_lfc)
  lab[ok & df$q < padj_cut & df$log2_lfc >= lfc_cut] <- "Up"
  lab[ok & df$q < padj_cut & df$log2_lfc <= -lfc_cut] <- "Down"
  df$diffexpressed <- lab
  df
}

short_lab <- function(x) {
  x <- sub("^[^|]+\\|", "", x)
  x <- sub("^(product|ec|ko):", "", x)
  ifelse(nchar(x) > 50, paste0(substr(x, 1, 47), "..."), x)
}

plot_volcano <- function(df, out_pdf, out_png, lfc_cut, padj_cut, title) {
  suppressPackageStartupMessages(library(ggplot2))
  d <- classify_deg(df, lfc_cut, padj_cut)
  d <- d[!is.na(d$log2_lfc) & !is.na(d$q), , drop = FALSE]
  d$neglog10q <- -log10(pmax(d$q, .Machine$double.xmin))
  d$label <- ifelse(d$diffexpressed != "NS", short_lab(d$feature), NA_character_)
  cols <- c(Up = COL_UP, Down = COL_DOWN, NS = COL_NS)
  p <- ggplot(d, aes(.data$log2_lfc, .data$neglog10q, color = .data$diffexpressed)) +
    geom_point(alpha = 0.75, size = 1.5) +
    geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = 2, linewidth = 0.3) +
    geom_hline(yintercept = -log10(padj_cut), linetype = 2, linewidth = 0.3) +
    scale_color_manual(values = cols, name = NULL) +
    labs(title = title, x = expression(log[2]~fold~change),
         y = expression(-log[10]~adjusted~italic(q))) +
    theme_bw(base_size = 11) + theme(legend.position = "top")
  if (requireNamespace("ggrepel", quietly = TRUE)) {
    p <- p + ggrepel::geom_text_repel(
      aes(label = .data$label), size = 2.4, max.overlaps = 15,
      show.legend = FALSE, na.rm = TRUE
    )
  }
  ggplot2::ggsave(out_pdf, p, width = 7, height = 6)
  ggplot2::ggsave(out_png, p, width = 7, height = 6, dpi = 300)
}

plot_top_lfc <- function(df, out_pdf, out_png, top_n = 20L, title = NULL) {
  suppressPackageStartupMessages(library(ggplot2))
  d <- df[!is.na(df$log2_lfc) & !is.na(df$lfcSE), , drop = FALSE]
  d <- d[order(-abs(d$log2_lfc)), , drop = FALSE]
  d <- utils::head(d, as.integer(top_n))
  d$label <- short_lab(d$feature)
  if (anyDuplicated(d$label)) d$label <- paste0(d$label, " [", seq_len(nrow(d)), "]")
  d$direction <- ifelse(d$log2_lfc >= 0, "Up", "Down")
  d$label <- factor(d$label, levels = rev(d$label))
  p <- ggplot(d, aes(.data$label, .data$log2_lfc, fill = .data$direction)) +
    geom_col(width = 0.7) +
    geom_errorbar(
      aes(ymin = .data$log2_lfc - .data$lfcSE, ymax = .data$log2_lfc + .data$lfcSE),
      width = 0.25, linewidth = 0.4
    ) +
    coord_flip() +
    scale_fill_manual(values = c(Up = COL_UP, Down = COL_DOWN), name = NULL) +
    labs(title = title %||% paste0("Top-", nrow(d), " metabolic |log2FC| ± SE"),
         x = NULL, y = expression(log[2]~fold~change)) +
    theme_bw(base_size = 11) + theme(legend.position = "top")
  h <- max(5, 0.28 * nrow(d) + 2)
  ggplot2::ggsave(out_pdf, p, width = 9, height = h)
  ggplot2::ggsave(out_png, p, width = 9, height = h, dpi = 300)
}

default_long <- function() {
  c(
    file.path(root, ".cursor/skills/metabolism-de/fixtures/bakta_function_long.csv"),
    file.path(root, ".cursor/skills/metabolism/fixtures/bakta_function_long.csv"),
    "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/bakta_function_long.csv"
  )
}

run_metabolism_de <- function(long = NULL, metadata = NULL, ps_rds = NULL,
                              outdir = "test/metabolism-de/run",
                              types = "product", group_col = "group",
                              prv_cut = 0.05, lfc_cut = 0.5, padj_cut = 0.05,
                              top_n = 20L, drop_hypothetical = TRUE,
                              use_ps_rds = TRUE, seed = 123L) {
  setwd(root)
  ensure_dir(outdir)
  if (is.null(long) || !nzchar(long)) {
    for (p in default_long()) {
      if (file.exists(p)) { long <- p; break }
    }
    if (is.null(long)) fail("No metabolic long table; pass --long")
  }
  filt <- filter_metab(normalize_long(read_table_auto(long)), types = types,
                       drop_hypothetical = drop_hypothetical)
  mat <- build_count_matrix(filt$long)
  n_genes_imported <- nrow(mat)
  meta_pack <- load_metadata(metadata, ps_rds, group_col)
  notes <- character(0)
  dropped <- setdiff(unique(normalize_long(read_table_auto(long))$sample), colnames(mat))
  if (length(dropped)) notes <- c(notes, paste0("samples absent after filter: ", paste(dropped, collapse = ",")))
  notes <- c(notes, paste0(
    "import funnel: long_rows=", nrow(filt$long),
    " unique_genes=", n_genes_imported,
    " go_dropped=", filt$n_go_dropped,
    " hyp_dropped=", filt$n_hyp,
    " types=", paste(filt$types, collapse = "+")
  ))

  # Rarefy using phyloseq library sizes by default (--ps-rds or Kristina auto)
  if (isTRUE(use_ps_rds)) {
    if (is.null(ps_rds) || !nzchar(as.character(ps_rds))) {
      cand <- c(
        "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/phyloseq_counts.rds",
        file.path(root, "..", "2026", "Kristina", "data/processed/phyloseq_counts.rds"),
        "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/phyloseq.rds",
        file.path(root, "..", "2026", "Kristina", "data/processed/phyloseq.rds")
      )
      for (p in cand) {
        if (!file.exists(p)) next
        ps_try <- tryCatch({
          obj <- readRDS(p)
          if (is.list(obj) && inherits(obj$phyloseq, "phyloseq")) obj <- obj$phyloseq
          obj
        }, error = function(e) NULL)
        if (!inherits(ps_try, "phyloseq")) next
        ss <- tryCatch(as.numeric(sample_sums(ps_try)), error = function(e) 0)
        if (length(ss) && is.finite(max(ss)) && max(ss) >= 10) {
          ps_rds <- p
          break
        }
      }
    }
    if (!is.null(ps_rds) && nzchar(as.character(ps_rds)) && file.exists(ps_rds)) {
      ps_obj <- readRDS(ps_rds)
      if (is.list(ps_obj) && inherits(ps_obj$phyloseq, "phyloseq")) ps_obj <- ps_obj$phyloseq
      if (inherits(ps_obj, "phyloseq")) {
        rr <- rarefy_mat_to_ps(mat, ps_obj, seed = seed)
        mat <- rr$mat
        notes <- c(notes, rr$notes)
      }
    }
  } else {
    notes <- c(notes, "use_ps_rds=FALSE: skipped phyloseq rarefy/normalize")
  }

  pack <- matrix_to_phyloseq(mat, meta_pack$meta)
  # Soft prevalence prefilter (ANCOM-BC also applies prv_cut)
  prv <- rowMeans(as(otu_table(pack$ps), "matrix") > 0)
  n_pre_prv <- length(prv)
  n_keep <- sum(prv >= prv_cut)
  if (n_keep < 2L) {
    # Relax once so volcano is not empty on sparse fixtures / real tables
    soft <- min(prv_cut, 0.02)
    message("WARNING: prv_cut=", prv_cut, " leaves <2 features; relaxing to ", soft)
    prv_cut <- soft
    n_keep <- sum(prv >= prv_cut)
  }
  if (n_keep < 2L) fail("Fewer than 2 features pass prv_cut=", prv_cut)
  pack$ps <- prune_taxa(prv >= prv_cut, pack$ps)
  notes <- c(notes, paste0(
    "prevalence filter: ", n_pre_prv, " -> ", n_keep,
    " genes (prv_cut=", prv_cut, ")"
  ))
  if (n_keep < 20L) {
    message(
      "NOTE: only ", n_keep,
      " genes after prevalence filter — volcano will look sparse. ",
      "Tiny fixtures or high --prv_cut are the usual cause; Kristina product+ko+ec is ~14k genes."
    )
  }

  out <- run_ancombc_metab(pack$ps, prv_cut = min(prv_cut, 0.1), seed = seed)
  deg <- tidy_ancombc2(out, level_map = pack$level_map)
  if (is.null(deg) || !nrow(deg)) fail("ANCOM-BC2 returned no tidy results")
  if (!is.null(pack$feature_map)) {
    mapped <- unname(pack$feature_map[as.character(deg$feature)])
    ok <- !is.na(mapped) & nzchar(mapped)
    deg$feature[ok] <- mapped[ok]
  }

  deg$function_type <- sub("\\|.*$", "", deg$feature)
  deg <- classify_deg(deg, lfc_cut, padj_cut)

  res_path <- file.path(outdir, "metabolism_de_results.tsv")
  utils::write.table(deg, res_path, sep = "\t", quote = FALSE, row.names = FALSE)

  # primary term for plots: first non-intercept
  terms <- unique(deg$term)
  term1 <- terms[[1]]
  d1 <- deg[deg$term == term1, , drop = FALSE]
  if (nrow(d1) < 5L) {
    message(
      "WARNING: volcano has only ", nrow(d1),
      " features for term ", term1,
      " (prv_cut=", prv_cut, "). Lower --prv_cut to retain more genes."
    )
  }
  vol_pdf <- file.path(outdir, "metabolism_de_volcano.pdf")
  vol_png <- file.path(outdir, "metabolism_de_volcano.png")
  bar_pdf <- file.path(outdir, "metabolism_de_top20_lfc.pdf")
  bar_png <- file.path(outdir, "metabolism_de_top20_lfc.png")
  plot_volcano(d1, vol_pdf, vol_png, lfc_cut, padj_cut,
               title = paste0(
                 "Metabolism ANCOM-BC2 volcano (", term1, "; n=", nrow(d1), ")"
               ))
  plot_top_lfc(d1, bar_pdf, bar_png, top_n = top_n,
               title = paste0("Top-", top_n, " metabolic |log2FC| ± SE (", term1, ")"))

  n_sig <- sum(d1$diffexpressed != "NS", na.rm = TRUE)
  rep <- list(
    skill = "metabolism-de",
    method = "ancombc2",
    input_long = long,
    metadata = meta_pack$path,
    types = filt$types,
    go_excluded = TRUE,
    n_go_dropped = filt$n_go_dropped,
    n_hypothetical_dropped = filt$n_hyp,
    n_genes_imported = n_genes_imported,
    n_features_tested = length(unique(deg$feature)),
    n_significant = n_sig,
    plot_term = term1,
    prv_cut = prv_cut,
    lfc_cut = lfc_cut,
    padj_cut = padj_cut,
    notes = notes,
    package_versions = list(
      ANCOMBC = as.character(utils::packageVersion("ANCOMBC")),
      phyloseq = as.character(utils::packageVersion("phyloseq"))
    ),
    figures = list(volcano_pdf = vol_pdf, volcano_png = vol_png,
                   top_lfc_pdf = bar_pdf, top_lfc_png = bar_png),
    tables = list(results = res_path)
  )
  write_json(rep, file.path(outdir, "metabolism-de-report.json"))
  message("metabolism-de OK: features=", rep$n_features_tested, " sig=", n_sig)
  invisible(rep)
}

self_test <- function() {
  setwd(root)
  fix_long <- file.path(root, ".cursor/skills/metabolism-de/fixtures/bakta_function_long.csv")
  fix_meta <- file.path(root, ".cursor/skills/metabolism-de/fixtures/metadata.csv")
  if (!file.exists(fix_long)) fail("Missing fixture long")
  out <- "test/metabolism-de/self-test"
  if (dir.exists(out)) unlink(out, recursive = TRUE)
  # Fixture: do not auto-pull Kristina phyloseq (sample IDs won't match)
  rep <- run_metabolism_de(
    long = fix_long, metadata = fix_meta,
    outdir = out,
    types = "product,ko,ec", prv_cut = 0.05, lfc_cut = 0.5, padj_cut = 0.2,
    top_n = 10L, seed = 1L, use_ps_rds = FALSE
  )
  if (!isTRUE(rep$go_excluded)) stop("GO must be excluded")
  if (!file.exists(rep$tables$results)) stop("missing results")
  if (!file.exists(rep$figures$volcano_pdf)) stop("missing volcano")
  message("SELF-TEST fixture OK (features=", rep$n_features_tested,
          ", sig=", rep$n_significant, ")")

  kristina <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/bakta_function_long.csv"
  kristina_ps <- "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/phyloseq_counts.rds"
  if (file.exists(kristina) && file.exists(kristina_ps)) {
    out2 <- "test/metabolism-de/kristina"
    if (dir.exists(out2)) unlink(out2, recursive = TRUE)
    rep2 <- run_metabolism_de(
      long = kristina,
      ps_rds = kristina_ps,
      outdir = out2,
      types = "product,ko,ec",
      group_col = "Группа",
      # 0.05 → ~7k genes (hours); 0.2 keeps ~1.7k for a tractable full-data self-test
      prv_cut = 0.2,
      lfc_cut = 0.5,
      padj_cut = 0.05,
      top_n = 30L,
      seed = 1L
    )
    if (!(rep2$n_genes_imported >= 10000L)) {
      stop("Kristina DE import too small: ", rep2$n_genes_imported)
    }
    if (!(rep2$n_features_tested >= 1000L)) {
      stop("Kristina volcano too sparse after prv: ", rep2$n_features_tested)
    }
    if (!file.exists(rep2$figures$volcano_pdf)) stop("missing Kristina volcano")
    message(
      "SELF-TEST Kristina OK (imported=", rep2$n_genes_imported,
      ", tested=", rep2$n_features_tested, ", sig=", rep2$n_significant, ")"
    )
  } else {
    message("SELF-TEST: Kristina long/phyloseq not both found — fixture only")
  }
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) { self_test(); return(invisible(0)) }
  drop_hyp <- !identical(tolower(as.character(args$drop_hypothetical %||% "true")), "false")
  no_ps <- identical(tolower(as.character(args$no_ps_rds %||% "false")), "true") ||
    identical(tolower(as.character(args$use_ps_rds %||% "true")), "false")
  run_metabolism_de(
    long = args$long,
    metadata = args$metadata,
    ps_rds = args$ps_rds,
    outdir = args$outdir %||% "test/metabolism-de/run",
    types = args$types %||% "product",
    group_col = args$group_col %||% "group",
    prv_cut = as.numeric(args$prv_cut %||% 0.05),
    lfc_cut = as.numeric(args$lfc_cut %||% 0.5),
    padj_cut = as.numeric(args$padj_cut %||% 0.05),
    top_n = as.integer(args$top_n %||% 20L),
    drop_hypothetical = drop_hyp,
    use_ps_rds = !isTRUE(no_ps),
    seed = as.integer(args$seed %||% 123L)
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
