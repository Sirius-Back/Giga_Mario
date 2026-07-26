#!/usr/bin/env Rscript
# difftree-metacoder — rarefied Taxmap heat trees; import ancombc if OK, else default
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("metacoder", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(phyloseq)
  library(metacoder)
})
# phyloseq also exports filter_taxa — always use metacoder's
filter_taxa <- metacoder::filter_taxa
heat_tree <- metacoder::heat_tree
calc_taxon_abund <- metacoder::calc_taxon_abund
parse_phyloseq <- metacoder::parse_phyloseq
taxon_ids <- metacoder::taxon_ids
taxon_names <- metacoder::taxon_names
taxon_ranks <- metacoder::taxon_ranks

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
DEFAULT_MIN_LEAF_DEFAULT <- 10
DEFAULT_MIN_LEAF_ANCOM <- 1
DEFAULT_LFC_CUT <- 0.5
DEFAULT_P_CUT <- 1
LFC_COLORS <- c("#1B9E77", "gray", "#D81B60")

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
      fail("difftree-metacoder requires rarefied phyloseq; got: ", rds)
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

ps_to_taxmap <- function(ps) {
  obj <- tryCatch(
    parse_phyloseq(ps),
    error = function(e) fail("parse_phyloseq failed: ", conditionMessage(e))
  )
  if (!inherits(obj, "Taxmap")) fail("parse_phyloseq did not return Taxmap")
  obj
}

resolve_taxmap <- function(metacoder = NULL, rds = NULL, allow_non_rare = FALSE) {
  notes <- character(0)
  if (!is.null(metacoder) && nzchar(metacoder)) {
    if (!file.exists(metacoder)) fail("Taxmap RDS missing: ", metacoder)
    obj <- readRDS(metacoder)
    if (!inherits(obj, "Taxmap")) fail("Not a Taxmap: ", metacoder)
    loaded <- resolve_rare_ps(rds, allow_non_rare = allow_non_rare)
    return(list(
      obj = obj, path = metacoder, source = "metacoder",
      ps_meta = loaded$meta, notes = c(notes, loaded$notes %||% character(0),
                                       paste0("Taxmap from ", metacoder))
    ))
  }
  loaded <- resolve_rare_ps(rds, allow_non_rare = allow_non_rare)
  list(
    obj = ps_to_taxmap(loaded$ps),
    path = loaded$meta$path,
    source = "phyloseq",
    ps_meta = loaded$meta,
    notes = loaded$notes %||% character(0)
  )
}

ensure_taxon_counts <- function(obj) {
  if (!"otu_table" %in% names(obj$data)) fail("Taxmap missing data$otu_table")
  if (!"taxon_counts" %in% names(obj$data)) {
    obj$data$taxon_counts <- calc_taxon_abund(obj, data = "otu_table")
  }
  tc <- obj$data$taxon_counts
  if (!"taxon_id" %in% names(tc)) fail("taxon_counts missing taxon_id")
  num_cols <- names(tc)[vapply(tc, is.numeric, logical(1))]
  num_cols <- setdiff(num_cols, c("total", "leaf", "log2_lfc", "lfc", "pval"))
  # drop prior lfc columns from numeric pool
  num_cols <- num_cols[!grepl("^(lfc_|log2_lfc)", num_cols)]
  if (!length(num_cols)) fail("No numeric abundance columns in taxon_counts")
  mat <- as.matrix(tc[, num_cols, drop = FALSE])
  obj$data$taxon_counts$total <- round(rowMeans(mat), 1)
  obj$data$taxon_counts$leaf <- rowSums(mat)
  obj
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

ancombc_report_ok <- function(dir) {
  rep <- file.path(dir, "ancombc-report.json")
  if (!file.exists(rep)) return(list(ok = NA, path = rep, detail = "no report"))
  j <- tryCatch(jsonlite::fromJSON(rep, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(j)) return(list(ok = FALSE, path = rep, detail = "unreadable report"))
  summ <- j$level_summary
  if (is.null(summ) || !length(summ)) {
    return(list(ok = isTRUE(j$rarefied), path = rep, detail = "no level_summary"))
  }
  oks <- vapply(summ, function(x) isTRUE(x$ok), logical(1))
  list(
    ok = all(oks),
    path = rep,
    detail = if (all(oks)) "all levels ok" else paste("failed:", paste(
      vapply(summ[!oks], function(x) paste0(x$target, "/", x$level), character(1)),
      collapse = ","
    ))
  )
}

find_ancombc <- function(ancombc_csv = NULL, ancombc_dir = NULL) {
  notes <- character(0)
  if (!is.null(ancombc_csv) && nzchar(ancombc_csv)) {
    if (!file.exists(ancombc_csv)) fail("ANCOM-BC CSV not found: ", ancombc_csv)
    dir <- dirname(ancombc_csv)
    chk <- ancombc_report_ok(dir)
    if (isFALSE(chk$ok)) {
      fail("ANCOM-BC report not OK — do not re-run from difftree; fix ancombc first. ",
           chk$detail, " (", chk$path, ")")
    }
    return(list(path = ancombc_csv, dir = dir, report_ok = chk$ok, notes = notes))
  }

  dirs <- character(0)
  if (!is.null(ancombc_dir) && nzchar(ancombc_dir)) dirs <- c(dirs, ancombc_dir)
  dirs <- c(
    dirs,
    "test/ancombc/grazing",
    "test/ancombc/grazing-self-test"
  )
  if (dir.exists("test/ancombc")) {
    dirs <- c(dirs, dirname(Sys.glob("test/ancombc/**/ancombc_results.csv")))
    dirs <- c(dirs, dirname(Sys.glob("test/ancombc/**/ancombc_results.tsv")))
  }
  dirs <- unique(dirs[nzchar(dirs) & dir.exists(dirs)])

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
      fail(
        "Found ANCOM-BC dir but report not OK — will not re-run ancombc. ",
        chk$detail, " (", chk$path, "). Fix ancombc or pass --mode default."
      )
    }
    notes <- c(notes, paste0("Importing ANCOM-BC: ", hit[[1]]), chk$detail)
    return(list(path = hit[[1]], dir = d, report_ok = chk$ok, notes = notes))
  }
  NULL
}

normalize_ancombc_long <- function(df) {
  nms <- names(df)
  # already long
  if (all(c("taxon", "level", "term", "lfc") %in% nms) ||
      all(c("taxon", "level", "term", "lfc") %in% tolower(nms))) {
    names(df) <- tolower(names(df))
    need <- c("taxon", "level", "term", "lfc")
    if (!all(need %in% names(df))) fail("ANCOM long table missing columns: ", paste(need, collapse = ", "))
    if (!"p" %in% names(df)) df$p <- NA_real_
    if (!"target" %in% names(df)) df$target <- "target"
    return(df[, c("target", "level", "taxon", "term", "lfc", "p")])
  }
  fail("Unrecognized ANCOM-BC table format; need long columns target,level,taxon,term,lfc[,p]")
}

map_ancom_to_taxon_ids <- function(obj, an_long) {
  ti <- data.frame(
    taxon_id = as.character(taxon_ids(obj)),
    taxon_name = as.character(taxon_names(obj)),
    taxon_rank = as.character(taxon_ranks(obj)),
    stringsAsFactors = FALSE
  )
  otu <- obj$data$otu_table
  if (!all(c("taxon_id", "otu_id") %in% names(otu))) {
    fail("Taxmap otu_table needs taxon_id and otu_id for ASV mapping")
  }
  otu_map <- data.frame(
    taxon_name = as.character(otu$otu_id),
    taxon_id = as.character(otu$taxon_id),
    stringsAsFactors = FALSE
  )

  asv <- an_long[toupper(an_long$level) == "ASV", , drop = FALSE]
  ranks <- an_long[toupper(an_long$level) != "ASV", , drop = FALSE]

  asv_m <- data.frame()
  if (nrow(asv)) {
    tmp <- data.frame(
      taxon_name = as.character(asv$taxon),
      target = asv$target,
      term = asv$term,
      lfc = as.numeric(asv$lfc),
      p = as.numeric(asv$p),
      level = "ASV",
      stringsAsFactors = FALSE
    )
    asv_m <- merge(tmp, otu_map, by = "taxon_name")
    asv_m$taxon_rank <- "ASV"
  }

  rank_m <- data.frame()
  if (nrow(ranks)) {
    tmp <- data.frame(
      taxon_name = as.character(ranks$taxon),
      taxon_rank = as.character(ranks$level),
      target = ranks$target,
      term = ranks$term,
      lfc = as.numeric(ranks$lfc),
      p = as.numeric(ranks$p),
      level = as.character(ranks$level),
      stringsAsFactors = FALSE
    )
    rank_m <- merge(tmp, ti, by = c("taxon_name", "taxon_rank"))
  }

  out <- rbind(
    if (nrow(asv_m)) asv_m[, c("taxon_id", "target", "term", "level", "lfc", "p")] else NULL,
    if (nrow(rank_m)) rank_m[, c("taxon_id", "target", "term", "level", "lfc", "p")] else NULL
  )
  if (is.null(out) || !nrow(out)) fail("No ANCOM-BC rows mapped to Taxmap taxon_id")
  out
}

# Per taxon_id: min-p effect → log2_lfc (0 if p > p_cut)
collapse_diff <- function(mapped, target, term, p_cut = DEFAULT_P_CUT) {
  sub <- mapped[mapped$target == target & mapped$term == term, , drop = FALSE]
  if (!nrow(sub)) {
    return(data.frame(taxon_id = character(), log2_lfc = numeric(), pval = numeric(),
                      stringsAsFactors = FALSE))
  }
  parts <- lapply(split(sub, sub$taxon_id), function(d) {
    if (all(is.na(d$p))) {
      j <- which.max(abs(d$lfc))
    } else {
      j <- which.min(d$p)
    }
    pv <- d$p[[j]]
    lfc <- d$lfc[[j]]
    log2_lfc <- if (is.na(pv) || pv > p_cut) 0 else lfc / log(2)
    data.frame(
      taxon_id = d$taxon_id[[1]],
      log2_lfc = log2_lfc,
      pval = pv,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, parts)
}

attach_lfc_column <- function(obj, diff_df, col = "log2_lfc") {
  obj <- ensure_taxon_counts(obj)
  tc <- obj$data$taxon_counts
  tc[[col]] <- 0
  if (nrow(diff_df)) {
    m <- match(as.character(tc$taxon_id), as.character(diff_df$taxon_id))
    tc[[col]][!is.na(m)] <- diff_df$log2_lfc[m[!is.na(m)]]
    tc[[col]][is.na(tc[[col]])] <- 0
  }
  obj$data$taxon_counts <- tc
  obj
}

match_rank <- function(obj, rank) {
  ranks <- unique(as.character(taxon_ranks(obj)))
  ranks <- ranks[!is.na(ranks) & nzchar(ranks)]
  hit <- ranks[tolower(ranks) == tolower(rank)]
  if (!length(hit)) fail("Rank not found: ", rank, "; have ", paste(ranks, collapse = ", "))
  hit[[1]]
}

save_heat_tree_generic <- function(obj, out_prefix, color_var, color_label,
                                   color_range = NULL, width = 10, height = 10) {
  pdf_path <- paste0(out_prefix, ".pdf")
  png_path <- paste0(out_prefix, ".png")

  # Round leaf for layout backends that cast sizes to integer
  if ("taxon_counts" %in% names(obj$data) && "leaf" %in% names(obj$data$taxon_counts)) {
    obj$data$taxon_counts$leaf <- as.integer(round(obj$data$taxon_counts$leaf))
  }

  layouts <- list(
    list(layout = "davidson-harel", initial_layout = "reingold-tilford"),
    list(layout = "fr", initial_layout = NULL),
    list(layout = "da", initial_layout = "re")
  )

  ht <- NULL
  last_err <- NULL
  for (lay in layouts) {
    ht <- tryCatch({
      if (identical(color_var, "log2_lfc")) {
        if (is.null(lay$initial_layout)) {
          heat_tree(
            obj,
            node_label = taxon_names,
            node_size = n_obs,
            node_color = log2_lfc,
            node_size_axis_label = "ASV count",
            node_color_axis_label = color_label,
            node_color_range = color_range %||% LFC_COLORS,
            layout = lay$layout,
            output_file = pdf_path
          )
        } else {
          heat_tree(
            obj,
            node_label = taxon_names,
            node_size = n_obs,
            node_color = log2_lfc,
            node_size_axis_label = "ASV count",
            node_color_axis_label = color_label,
            node_color_range = color_range %||% LFC_COLORS,
            layout = lay$layout,
            initial_layout = lay$initial_layout,
            output_file = pdf_path
          )
        }
      } else if (identical(color_var, "total")) {
        if (is.null(lay$initial_layout)) {
          heat_tree(
            obj,
            node_label = taxon_names,
            node_size = n_obs,
            node_color = total,
            node_size_axis_label = "ASV count",
            node_color_axis_label = color_label,
            layout = lay$layout,
            output_file = pdf_path
          )
        } else {
          heat_tree(
            obj,
            node_label = taxon_names,
            node_size = n_obs,
            node_color = total,
            node_size_axis_label = "ASV count",
            node_color_axis_label = color_label,
            layout = lay$layout,
            initial_layout = lay$initial_layout,
            output_file = pdf_path
          )
        }
      } else {
        fail("Unsupported heat_tree color_var: ", color_var)
      }
    }, error = function(e) {
      last_err <<- conditionMessage(e)
      message("heat_tree layout=", lay$layout, " failed: ", last_err)
      NULL
    })
    if (!is.null(ht)) break
  }
  if (is.null(ht)) fail("heat_tree failed for all layouts. Last error: ", last_err %||% "unknown")

  if (isTRUE(capabilities("cairo"))) {
    grDevices::png(png_path, width = width, height = height, units = "in", res = 300, type = "cairo")
  } else if (requireNamespace("ragg", quietly = TRUE)) {
    ragg::agg_png(png_path, width = width, height = height, units = "in", res = 300)
  } else {
    grDevices::png(png_path, width = width * 300, height = height * 300, res = 300)
  }
  print(ht)
  grDevices::dev.off()
  list(pdf = pdf_path, png = png_path)
}

safe_name <- function(x) gsub("[^A-Za-z0-9._-]+", "_", x)

run_default <- function(obj, outdir, rank = DEFAULT_RANK,
                        min_leaf = DEFAULT_MIN_LEAF_DEFAULT) {
  obj <- ensure_taxon_counts(obj)
  rank_m <- match_rank(obj, rank)
  if (min_leaf > 0) obj <- filter_taxa(obj, leaf >= min_leaf)
  obj <- filter_taxa(obj, taxon_ranks == rank_m, supertaxa = TRUE)
  if (length(taxon_ids(obj)) < 2L) fail("Too few taxa after default merge")

  taxmap_path <- file.path(outdir, "difftree_taxmap.rds")
  saveRDS(obj, taxmap_path)
  figs <- save_heat_tree_generic(
    obj, file.path(outdir, "difftree_default"),
    color_var = "total",
    color_label = "Mean reads"
  )
  list(
    mode = "default",
    figures = list(list(name = "default", pdf = figs$pdf, png = figs$png)),
    taxmap_rds = taxmap_path,
    n_taxa = length(taxon_ids(obj))
  )
}

run_ancombc_mode <- function(obj, an_long, outdir,
                             min_leaf = DEFAULT_MIN_LEAF_ANCOM,
                             lfc_cut = DEFAULT_LFC_CUT,
                             p_cut = DEFAULT_P_CUT,
                             targets = NULL, terms = NULL) {
  mapped <- map_ancom_to_taxon_ids(obj, an_long)
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
  # drop intercept-like terms
  tm <- tm[!grepl("Intercept", tm, ignore.case = TRUE)]

  figures <- list()
  last_obj <- obj
  for (t in tg) {
    for (term in tm) {
      diff_df <- collapse_diff(mapped, target = t, term = term, p_cut = p_cut)
      n_sig <- sum(abs(diff_df$log2_lfc) > lfc_cut, na.rm = TRUE)
      message("ANCOM tree: target=", t, " term=", term,
              " mapped_taxa=", nrow(diff_df), " |log2_lfc|>", lfc_cut, "=", n_sig)

      obj_t <- attach_lfc_column(obj, diff_df, col = "log2_lfc")
      obj_f <- obj_t
      if (min_leaf > 0 || lfc_cut > 0) {
        obj_f <- filter_taxa(
          obj_t,
          leaf >= min_leaf & abs(log2_lfc) > lfc_cut,
          subtaxa = TRUE,
          supertaxa = TRUE
        )
      }
      if (length(taxon_ids(obj_f)) < 2L) {
        message("  skip plot (too few taxa after LFC filter); relaxing to leaf-only")
        obj_f <- filter_taxa(obj_t, leaf >= min_leaf, subtaxa = TRUE, supertaxa = TRUE)
      }
      if (length(taxon_ids(obj_f)) < 2L) {
        message("  still too few taxa — skip ", t, "/", term)
        next
      }

      prefix <- file.path(outdir, paste0("difftree_", safe_name(t), "_", safe_name(term)))
      figs <- save_heat_tree_generic(
        obj_f, prefix,
        color_var = "log2_lfc",
        color_label = paste0("log2 LFC (", t, ": ", term, ")"),
        color_range = LFC_COLORS
      )
      figures[[length(figures) + 1L]] <- list(
        name = paste0(t, "_", term),
        target = t,
        term = term,
        pdf = figs$pdf,
        png = figs$png,
        n_taxa = length(taxon_ids(obj_f)),
        n_lfc_pass = n_sig
      )
      last_obj <- obj_f
    }
  }
  if (!length(figures)) fail("No differential heat trees produced from ANCOM-BC data")

  taxmap_path <- file.path(outdir, "difftree_taxmap.rds")
  saveRDS(last_obj, taxmap_path)
  list(mode = "ancombc", figures = figures, taxmap_rds = taxmap_path,
       n_taxa = length(taxon_ids(last_obj)), targets = tg, terms = tm)
}

run_difftree <- function(rds = NULL, metacoder = NULL, outdir,
                         mode = c("auto", "ancombc", "default"),
                         ancombc_csv = NULL, ancombc_dir = NULL,
                         rank = DEFAULT_RANK,
                         min_leaf = NULL,
                         lfc_cut = DEFAULT_LFC_CUT,
                         p_cut = DEFAULT_P_CUT,
                         targets = NULL, terms = NULL,
                         allow_non_rare = FALSE) {
  mode <- match.arg(mode)
  ensure_dir(outdir)

  resolved <- resolve_taxmap(metacoder, rds, allow_non_rare = allow_non_rare)
  if (!isTRUE(resolved$ps_meta$rarefied) && !allow_non_rare) {
    fail("Input not rarefied: ", resolved$ps_meta$path)
  }
  obj <- resolved$obj
  notes <- resolved$notes %||% character(0)

  ancom <- NULL
  if (mode %in% c("auto", "ancombc")) {
    ancom <- find_ancombc(ancombc_csv, ancombc_dir)
    if (mode == "ancombc" && is.null(ancom)) {
      fail("mode=ancombc but no OK ANCOM-BC results found (will not re-run ancombc)")
    }
  }
  if (mode == "auto") {
    mode <- if (!is.null(ancom)) "ancombc" else "default"
  }
  if (!is.null(ancom)) notes <- c(notes, ancom$notes)

  message(
    "difftree-metacoder: mode=", mode,
    " rarefied=", resolved$ps_meta$rarefied,
    " input=", resolved$path,
    if (!is.null(ancom)) paste0(" ancombc=", ancom$path) else ""
  )

  if (identical(mode, "ancombc")) {
    an_long <- normalize_ancombc_long(read_table_auto(ancom$path))
    ml <- min_leaf %||% DEFAULT_MIN_LEAF_ANCOM
    result <- run_ancombc_mode(
      obj, an_long, outdir,
      min_leaf = ml, lfc_cut = lfc_cut, p_cut = p_cut,
      targets = targets, terms = terms
    )
  } else {
    ml <- min_leaf %||% DEFAULT_MIN_LEAF_DEFAULT
    result <- run_default(obj, outdir, rank = rank, min_leaf = ml)
  }

  report <- list(
    skill = "difftree-metacoder",
    mode = result$mode,
    input = resolved$path,
    input_source = resolved$source,
    rarefied = isTRUE(resolved$ps_meta$rarefied),
    rarefaction_depth = resolved$ps_meta$rarefaction_depth,
    ancombc_imported = !is.null(ancom),
    ancombc_path = if (!is.null(ancom)) ancom$path else NA_character_,
    ancombc_report_ok = if (!is.null(ancom)) ancom$report_ok else NA,
    ancombc_rerun = FALSE,
    figures = result$figures,
    taxmap_rds = result$taxmap_rds,
    n_taxa = result$n_taxa,
    notes = notes
  )
  write_json(report, file.path(outdir, "difftree-metacoder-report.json"))
  message("Wrote ", length(result$figures), " figure set(s); mode=", result$mode)
  report
}

self_test <- function() {
  setwd(project_root())

  # --- default mode (ignore ancombc) ---
  out_def <- "test/difftree-metacoder/grazing-default"
  rep_def <- run_difftree(
    rds = NULL,
    outdir = out_def,
    mode = "default"
  )
  if (!identical(rep_def$mode, "default")) stop("expected default mode")
  if (isTRUE(rep_def$ancombc_imported)) stop("default must not import ancombc")
  if (isTRUE(rep_def$ancombc_rerun)) stop("must never rerun ancombc")
  png_def <- rep_def$figures[[1]]$png
  if (!file.exists(png_def) || file.info(png_def)$size < 1000) stop("default png missing/small")

  # --- ancombc mode (import existing; do not rerun) ---
  if (!file.exists("test/ancombc/grazing/ancombc_results.csv") &&
      !file.exists("test/ancombc/grazing/ancombc_results.tsv")) {
    stop("self-test requires existing ancombc grazing results (do not regenerate here)")
  }
  out_ac <- "test/difftree-metacoder/grazing-ancombc"
  rep_ac <- run_difftree(
    rds = NULL,
    outdir = out_ac,
    mode = "ancombc",
    ancombc_dir = "test/ancombc/grazing"
  )
  if (!identical(rep_ac$mode, "ancombc")) stop("expected ancombc mode")
  if (!isTRUE(rep_ac$ancombc_imported)) stop("expected ancombc import")
  if (isTRUE(rep_ac$ancombc_rerun)) stop("must not rerun ancombc")
  if (!length(rep_ac$figures)) stop("no ancombc figures")
  for (fg in rep_ac$figures) {
    if (!file.exists(fg$png) || file.info(fg$png)$size < 1000) {
      stop("ancombc png missing/small: ", fg$png)
    }
  }

  message("SELF-TEST OK (default + ancombc; ancombc_rerun=FALSE)")
  invisible(list(default = rep_def, ancombc = rep_ac))
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/difftree-metacoder/run"
  mode <- args$mode %||% "auto"
  min_leaf <- if (is.null(args$min_leaf)) NULL else as.numeric(args$min_leaf)
  lfc_cut <- as.numeric(args$lfc_cut %||% DEFAULT_LFC_CUT)
  p_cut <- as.numeric(args$p_cut %||% DEFAULT_P_CUT)
  allow_non_rare <- identical(tolower(as.character(args$allow_non_rare %||% "false")), "true")

  run_difftree(
    rds = args$rds,
    metacoder = args$metacoder,
    outdir = outdir,
    mode = mode,
    ancombc_csv = args$ancombc_csv,
    ancombc_dir = args$ancombc_dir,
    rank = args$rank %||% DEFAULT_RANK,
    min_leaf = min_leaf,
    lfc_cut = lfc_cut,
    p_cut = p_cut,
    targets = args$targets,
    terms = args$terms,
    allow_non_rare = allow_non_rare
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
