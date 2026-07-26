#!/usr/bin/env Rscript
# go — metagenome GO DEG (DESeq2) + enricher + volcano / top-20 LFC±SE
suppressPackageStartupMessages({
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
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
GO_BASIC_URL <- "https://purl.obolibrary.org/obo/go/go-basic.obo"

# ---------------------------------------------------------------------------
# GO resources
# ---------------------------------------------------------------------------

parse_obo_term_names <- function(obo_path) {
  if (!file.exists(obo_path)) fail("OBO not found: ", obo_path)
  lines <- readLines(obo_path, warn = FALSE)
  starts <- grep("^\\[Term\\]", lines)
  if (!length(starts)) fail("No [Term] sections in ", obo_path)
  ids <- character(0)
  names_ <- character(0)
  for (i in seq_along(starts)) {
    start <- starts[[i]]
    end <- if (i < length(starts)) starts[[i + 1L]] - 1L else length(lines)
    block <- lines[start:end]
    id_line <- grep("^id:", block, value = TRUE)
    name_line <- grep("^name:", block, value = TRUE)
    if (length(id_line) && length(name_line)) {
      ids <- c(ids, sub("^id:\\s*", "", id_line[[1]]))
      names_ <- c(names_, sub("^name:\\s*", "", name_line[[1]]))
    }
  }
  data.frame(go_id = ids, go_name = names_, stringsAsFactors = FALSE)
}

resolve_obo_candidates <- function(cache_dir) {
  unique(c(
    file.path(cache_dir, "go-basic.obo"),
    file.path(cache_dir, "go.obo"),
    file.path(root, ".cursor/skills/go/cache/go-basic.obo"),
    file.path(root, "refs/go/go-basic.obo"),
    file.path(root, "refs/go/go.obo"),
    Sys.glob(file.path(getwd(), "*.obo")),
    Sys.glob(file.path(root, "*.obo"))
  ))
}

ensure_go_resources <- function(cache_dir = file.path(root, ".cursor/skills/go/cache"),
                                fetch_if_missing = TRUE) {
  ensure_dir(cache_dir)
  notes <- character(0)
  has_godb <- requireNamespace("GO.db", quietly = TRUE) &&
    requireNamespace("AnnotationDbi", quietly = TRUE)
  n_godb <- if (has_godb) {
    length(AnnotationDbi::keys(GO.db::GOTERM))
  } else {
    NA_integer_
  }
  if (has_godb) notes <- c(notes, paste0("Using GO.db (", n_godb, " terms)"))

  obo_path <- NA_character_
  fetched <- FALSE
  for (p in resolve_obo_candidates(cache_dir)) {
    if (nzchar(p) && file.exists(p)) {
      obo_path <- normalizePath(p)
      notes <- c(notes, paste0("Found local OBO: ", obo_path))
      break
    }
  }

  if (is.na(obo_path) && isTRUE(fetch_if_missing) && !has_godb) {
    dest <- file.path(cache_dir, "go-basic.obo")
    message("Downloading go-basic.obo → ", dest)
    ok <- tryCatch({
      utils::download.file(GO_BASIC_URL, destfile = dest, mode = "wb", quiet = TRUE)
      TRUE
    }, error = function(e) {
      notes <<- c(notes, paste0("OBO download failed: ", conditionMessage(e)))
      FALSE
    })
    if (ok && file.exists(dest) && file.info(dest)$size > 1000) {
      obo_path <- normalizePath(dest)
      fetched <- TRUE
      notes <- c(notes, "Downloaded go-basic.obo")
    }
  } else if (!is.na(obo_path)) {
    notes <- c(notes, paste0("Using local OBO: ", obo_path))
  } else if (has_godb) {
    notes <- c(notes, "OBO not required (GO.db available)")
  }

  if (!has_godb && (is.na(obo_path) || !file.exists(obo_path))) {
    fail(
      "No GO ontology available. Install Bioconductor GO.db or provide go-basic.obo ",
      "under ", cache_dir, " (or allow --fetch-go true)."
    )
  }

  obo_names <- NULL
  if (!is.na(obo_path) && file.exists(obo_path) && !has_godb) {
    obo_names <- parse_obo_term_names(obo_path)
  }

  list(
    has_godb = has_godb,
    n_godb = n_godb,
    obo_path = if (is.na(obo_path)) NULL else obo_path,
    obo_names = obo_names,
    fetched = fetched,
    cache_dir = cache_dir,
    notes = notes
  )
}

annotate_go_ids <- function(go_ids, go_res) {
  norm <- vapply(as.character(go_ids), function(x) {
    m <- regmatches(x, regexpr("[Gg][Oo]:[0-9]+", x))
    if (length(m) && nzchar(m[[1]])) {
      return(paste0("GO:", sub("^[Gg][Oo]:", "", m[[1]])))
    }
    x
  }, character(1), USE.NAMES = FALSE)

  names_out <- rep(NA_character_, length(norm))
  ont_out <- rep(NA_character_, length(norm))
  if (isTRUE(go_res$has_godb)) {
    for (i in seq_along(norm)) {
      id <- norm[[i]]
      if (!nzchar(id)) next
      trm <- tryCatch(GO.db::GOTERM[[id]], error = function(e) NULL)
      if (!is.null(trm)) {
        names_out[[i]] <- AnnotationDbi::Term(trm)
        ont_out[[i]] <- AnnotationDbi::Ontology(trm)
      }
    }
  } else if (!is.null(go_res$obo_names)) {
    mp <- go_res$obo_names
    idx <- match(norm, mp$go_id)
    names_out <- mp$go_name[idx]
  }
  data.frame(go_id = norm, go_name = names_out, ontology = ont_out, stringsAsFactors = FALSE)
}

# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------

read_table_auto <- function(path) {
  if (!file.exists(path)) fail("File not found: ", path)
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("csv")) {
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  }
}

normalize_go_id <- function(x) {
  vapply(as.character(x), function(z) {
    m <- regmatches(z, regexpr("[Gg][Oo]:[0-9]+", z))
    if (length(m) && nzchar(m[[1]])) paste0("GO:", sub("^[Gg][Oo]:", "", m[[1]])) else z
  }, character(1), USE.NAMES = FALSE)
}

load_go_long <- function(path) {
  df <- read_table_auto(path)
  names(df) <- tolower(names(df))
  need <- c("sample", "function_type", "function_id", "count")
  if (!all(need %in% names(df))) {
    fail("Long table missing: ", paste(setdiff(need, names(df)), collapse = ", "))
  }
  df <- df[tolower(df$function_type) == "go", , drop = FALSE]
  if (!nrow(df)) fail("No function_type=go rows in ", path)
  df$sample <- as.character(df$sample)
  df$go_id <- normalize_go_id(df$function_id)
  df$count <- as.numeric(df$count)
  if (any(is.na(df$count)) || any(df$count < 0)) fail("Invalid GO counts")
  df
}

build_go_matrix <- function(go_long) {
  agg <- stats::aggregate(count ~ go_id + sample, data = go_long, FUN = sum)
  genes <- sort(unique(agg$go_id))
  samples <- sort(unique(agg$sample))
  mat <- matrix(0L, nrow = length(genes), ncol = length(samples),
                dimnames = list(genes, samples))
  for (i in seq_len(nrow(agg))) {
    mat[agg$go_id[[i]], agg$sample[[i]]] <- as.integer(round(agg$count[[i]]))
  }
  mat
}

load_metadata <- function(metadata = NULL, ps_rds = NULL, group_col = "group") {
  if (!is.null(ps_rds) && nzchar(ps_rds)) {
    if (!requireNamespace("phyloseq", quietly = TRUE)) fail("phyloseq required for --ps-rds")
    if (!file.exists(ps_rds)) fail("phyloseq RDS not found: ", ps_rds)
    suppressPackageStartupMessages(library(phyloseq))
    ps <- readRDS(ps_rds)
    if (!inherits(ps, "phyloseq")) {
      if (is.list(ps) && inherits(ps$phyloseq, "phyloseq")) ps <- ps$phyloseq
      else fail("--ps-rds must be phyloseq")
    }
    sd <- as(phyloseq::sample_data(ps), "data.frame")
    sd$sample <- rownames(sd)
    # map Russian Kristina columns
    if (!group_col %in% names(sd)) {
      aliases <- c(
        group = "Группа", Group = "Группа", age_group = "Группа",
        visit = "Визит", Visit = "Визит"
      )
      if (group_col %in% names(aliases) && aliases[[group_col]] %in% names(sd)) {
        sd[[group_col]] <- sd[[aliases[[group_col]]]]
      } else if ("Группа" %in% names(sd) && identical(group_col, "group")) {
        sd$group <- sd[["Группа"]]
      }
    }
    meta <- sd
    meta_path <- ps_rds
  } else if (!is.null(metadata) && nzchar(metadata)) {
    meta <- read_table_auto(metadata)
    names(meta) <- sub("^SampleID$", "sample", names(meta), ignore.case = TRUE)
    if (!"sample" %in% tolower(names(meta))) {
      # first column as sample if named sample_id
      nms <- tolower(names(meta))
      if ("sample_id" %in% nms) {
        names(meta)[match("sample_id", nms)] <- "sample"
      } else {
        fail("Metadata must contain sample column")
      }
    } else {
      names(meta)[match("sample", tolower(names(meta)))] <- "sample"
    }
    meta$sample <- as.character(meta$sample)
    meta_path <- metadata
  } else {
    fail("Provide --metadata or --ps-rds (unless --deg)")
  }
  if (!group_col %in% names(meta)) {
    fail("Group column '", group_col, "' not in metadata. Columns: ",
         paste(names(meta), collapse = ", "))
  }
  meta$group <- as.factor(as.character(meta[[group_col]]))
  if (nlevels(meta$group) < 2L) fail("Need ≥2 groups in ", group_col)
  list(meta = meta, group_col = group_col, path = meta_path)
}

filter_prevalence <- function(mat, prv_cut = 0.1) {
  prv <- rowMeans(mat > 0)
  keep <- prv >= prv_cut
  if (sum(keep) < 2L) fail("Fewer than 2 GO terms pass prv_cut=", prv_cut)
  mat[keep, , drop = FALSE]
}

# ---------------------------------------------------------------------------
# ANCOM-BC2 DEG (default) — Kristina Bakta pathway pattern
# ---------------------------------------------------------------------------

run_ancombc_go <- function(mat, meta, contrast = NULL, prv_cut = 0.1, seed = 123L) {
  if (!requireNamespace("ANCOMBC", quietly = TRUE)) fail("ANCOMBC is required")
  if (!requireNamespace("phyloseq", quietly = TRUE)) fail("phyloseq is required")
  suppressPackageStartupMessages(library(phyloseq))
  samples <- intersect(colnames(mat), meta$sample)
  if (length(samples) < 4L) fail("Need ≥4 overlapping samples for ANCOM-BC2; got ", length(samples))
  mat <- mat[, samples, drop = FALSE]
  meta <- meta[match(samples, meta$sample), , drop = FALSE]
  rownames(meta) <- meta$sample
  meta$group <- factor(as.character(meta$group))
  if (nlevels(meta$group) < 2L) fail("After sample overlap, <2 groups remain")
  keep_s <- colSums(mat) > 0
  if (!all(keep_s)) {
    mat <- mat[, keep_s, drop = FALSE]
    meta <- meta[colnames(mat), , drop = FALSE]
    meta$group <- factor(as.character(meta$group))
  }
  raw_levels <- levels(meta$group)
  safe_levels <- make.names(raw_levels, unique = TRUE)
  level_map <- setNames(raw_levels, safe_levels)
  meta$group <- factor(safe_levels[match(as.character(meta$group), raw_levels)], levels = safe_levels)

  tax <- matrix(
    cbind(go_id = rownames(mat)),
    ncol = 1,
    dimnames = list(rownames(mat), "go_id")
  )
  ps <- phyloseq(
    otu_table(mat, taxa_are_rows = TRUE),
    sample_data(meta[, "group", drop = FALSE]),
    tax_table(tax)
  )
  pairwise <- nlevels(meta$group) > 2L
  set.seed(as.integer(seed))
  message("ANCOM-BC2 GO: ntaxa=", ntaxa(ps), " nsamples=", nsamples(ps),
          " pairwise=", pairwise)
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
  if (is.null(out) || is.null(out$res)) fail("ANCOM-BC2 returned no results for GO")

  res <- out$res
  if (is.data.frame(res)) {
    res_df <- res
    if (!"taxon" %in% names(res_df)) res_df$taxon <- rownames(res_df)
  } else if (is.list(res) && !is.null(res$lfc)) {
    lfc <- as.data.frame(res$lfc)
    se <- as.data.frame(res$se)
    p <- as.data.frame(res$p_val)
    q <- as.data.frame(res$q_val)
    res_df <- data.frame(taxon = rownames(lfc), stringsAsFactors = FALSE)
    for (nm in names(lfc)) {
      res_df[[paste0("lfc_", nm)]] <- lfc[[nm]]
      res_df[[paste0("se_", nm)]] <- se[[nm]]
      res_df[[paste0("p_", nm)]] <- p[[nm]]
      res_df[[paste0("q_", nm)]] <- q[[nm]]
    }
  } else {
    fail("Unrecognized ANCOM-BC2 result structure")
  }

  lfc_cols <- grep("^lfc_", names(res_df), value = TRUE)
  lfc_cols <- lfc_cols[!grepl("Intercept", lfc_cols, ignore.case = TRUE)]
  if (!length(lfc_cols)) fail("No non-intercept LFC columns in ANCOM-BC2 output")

  # pick contrast term
  pick_col <- lfc_cols[[1]]
  if (!is.null(contrast) && nzchar(contrast)) {
    parts <- trimws(unlist(strsplit(contrast, ",", fixed = TRUE)))
    # prefer column matching numerator level
    hit <- grep(make.names(parts[[1]]), lfc_cols, value = TRUE)
    if (length(hit)) pick_col <- hit[[1]]
  }
  term <- sub("^lfc_", "", pick_col)
  se_c <- paste0("se_", term)
  p_c <- paste0("p_", term)
  q_c <- paste0("q_", term)
  if (!p_c %in% names(res_df)) p_c <- paste0("p_val_", term)
  if (!q_c %in% names(res_df)) q_c <- paste0("q_val_", term)

  deg <- data.frame(
    go_id = as.character(res_df$taxon),
    log2FoldChange = as.numeric(res_df[[pick_col]]) / log(2),
    lfcSE = if (se_c %in% names(res_df)) as.numeric(res_df[[se_c]]) / log(2) else NA_real_,
    pvalue = if (p_c %in% names(res_df)) as.numeric(res_df[[p_c]]) else NA_real_,
    padj = if (q_c %in% names(res_df)) as.numeric(res_df[[q_c]]) else NA_real_,
    stringsAsFactors = FALSE
  )
  # readable contrast label
  num_lab <- raw_levels[[length(raw_levels)]]
  den_lab <- raw_levels[[1]]
  for (nm in names(level_map)) {
    if (grepl(nm, term, fixed = TRUE)) {
      num_lab <- level_map[[nm]]
      den_lab <- setdiff(raw_levels, num_lab)[[1]]
      break
    }
  }
  deg$contrast <- paste0(num_lab, "_vs_", den_lab)
  list(
    deg = deg,
    contrast = c(num = num_lab, den = den_lab),
    n_samples = ncol(mat),
    groups = raw_levels,
    method = "ancombc2"
  )
}

# ---------------------------------------------------------------------------
# DESeq2 DEG (optional --method deseq2)
# ---------------------------------------------------------------------------

run_deseq_go <- function(mat, meta, contrast = NULL, seed = 123L) {
  if (!requireNamespace("DESeq2", quietly = TRUE)) fail("DESeq2 is required")
  samples <- intersect(colnames(mat), meta$sample)
  if (length(samples) < 4L) fail("Need ≥4 overlapping samples for DESeq2; got ", length(samples))
  mat <- mat[, samples, drop = FALSE]
  meta <- meta[match(samples, meta$sample), , drop = FALSE]
  rownames(meta) <- meta$sample
  # drop unused group levels
  meta$group <- factor(as.character(meta$group))
  if (nlevels(meta$group) < 2L) fail("After sample overlap, <2 groups remain")

  # remove zero-total samples / all-zero already filtered
  keep_s <- colSums(mat) > 0
  if (!all(keep_s)) {
    mat <- mat[, keep_s, drop = FALSE]
    meta <- meta[colnames(mat), , drop = FALSE]
    meta$group <- factor(as.character(meta$group))
  }
  if (nlevels(meta$group) < 2L) fail("Groups collapsed after dropping empty samples")

  set.seed(as.integer(seed))
  # Sanitize level labels for DESeq2 (keep map for reporting)
  raw_levels <- levels(meta$group)
  safe_levels <- make.names(raw_levels, unique = TRUE)
  level_map <- setNames(raw_levels, safe_levels)
  meta$group <- factor(safe_levels[match(as.character(meta$group), raw_levels)], levels = safe_levels)

  dds <- DESeq2::DESeqDataSetFromMatrix(
    countData = mat,
    colData = meta,
    design = ~ group
  )
  # Sparse Bakta GO matrices often have zeros in every row — use poscounts size factors
  dds <- DESeq2::estimateSizeFactors(dds, type = "poscounts")
  dds <- DESeq2::DESeq(dds, quiet = TRUE)

  levels_g <- levels(meta$group)
  raw_by_safe <- level_map
  if (!is.null(contrast) && nzchar(contrast)) {
    parts <- trimws(unlist(strsplit(contrast, ",", fixed = TRUE)))
    if (length(parts) != 2L) fail("--contrast must be numerator,denominator")
    # allow raw or safe names
    to_safe <- function(x) {
      if (x %in% levels_g) return(x)
      hit <- safe_levels[match(x, raw_levels)]
      if (length(hit) == 1L && !is.na(hit)) return(hit)
      fail("Contrast level not in data: ", x,
           " (have ", paste(raw_levels, collapse = ", "), ")")
    }
    num <- to_safe(parts[[1]])
    den <- to_safe(parts[[2]])
  } else {
    num <- levels_g[[length(levels_g)]]
    den <- levels_g[[1]]
  }
  res <- DESeq2::results(dds, contrast = c("group", num, den))
  res_df <- as.data.frame(res)
  res_df$go_id <- rownames(res_df)
  res_df$log2FoldChange <- as.numeric(res_df$log2FoldChange)
  res_df$lfcSE <- as.numeric(res_df$lfcSE)
  res_df$pvalue <- as.numeric(res_df$pvalue)
  res_df$padj <- as.numeric(res_df$padj)
  num_lab <- unname(raw_by_safe[[num]] %||% num)
  den_lab <- unname(raw_by_safe[[den]] %||% den)
  res_df$contrast <- paste0(num_lab, "_vs_", den_lab)
  list(
    deg = res_df,
    contrast = c(num = num_lab, den = den_lab),
    n_samples = ncol(mat),
    groups = unname(raw_by_safe[levels_g])
  )
}

load_deg_table <- function(path) {
  df <- read_table_auto(path)
  nms <- tolower(names(df))
  names(df) <- nms
  id_col <- if ("go_id" %in% nms) "go_id" else if ("gene" %in% nms) "gene" else if ("pathway" %in% nms) "pathway" else NA
  if (is.na(id_col)) fail("--deg must have go_id/gene column")
  lfc <- if ("log2foldchange" %in% nms) "log2foldchange" else if ("log2_lfc" %in% nms) "log2_lfc" else if ("lfc" %in% nms) "lfc" else NA
  se <- if ("lfcse" %in% nms) "lfcse" else if ("se" %in% nms) "se" else if ("lfc_se" %in% nms) "lfc_se" else NA
  p <- if ("pvalue" %in% nms) "pvalue" else if ("p" %in% nms) "p" else if ("p_value" %in% nms) "p_value" else NA
  padj <- if ("padj" %in% nms) "padj" else if ("q" %in% nms) "q" else if ("q_value" %in% nms) "q_value" else NA
  if (is.na(lfc) || is.na(se) || is.na(p) || is.na(padj)) {
    fail("--deg needs log2FoldChange, lfcSE, pvalue, padj (or aliases)")
  }
  out <- data.frame(
    go_id = normalize_go_id(df[[id_col]]),
    log2FoldChange = as.numeric(df[[lfc]]),
    lfcSE = as.numeric(df[[se]]),
    pvalue = as.numeric(df[[p]]),
    padj = as.numeric(df[[padj]]),
    stringsAsFactors = FALSE
  )
  out
}

# ---------------------------------------------------------------------------
# Enricher (significant GO IDs vs ontology parents)
# ---------------------------------------------------------------------------

build_term2gene_ancestors <- function(universe_go, go_res) {
  if (!isTRUE(go_res$has_godb)) {
    return(NULL)
  }
  universe_go <- unique(normalize_go_id(universe_go))
  universe_go <- universe_go[nzchar(universe_go)]
  if (!length(universe_go)) return(NULL)

  pull_anc <- function(ids, env) {
    got <- AnnotationDbi::mget(ids, envir = env, ifnotfound = NA)
    lapply(got, function(a) {
      if (length(a) == 1L && is.na(a[[1]])) return(character(0))
      setdiff(as.character(unlist(a, use.names = FALSE)), c("all", NA_character_))
    })
  }
  bp <- pull_anc(universe_go, GO.db::GOBPANCESTOR)
  mf <- pull_anc(universe_go, GO.db::GOMFANCESTOR)
  cc <- pull_anc(universe_go, GO.db::GOCCANCESTOR)

  term_list <- character(0)
  gene_list <- character(0)
  for (i in seq_along(universe_go)) {
    g <- universe_go[[i]]
    anc <- unique(c(g, bp[[i]], mf[[i]], cc[[i]]))
    term_list <- c(term_list, anc)
    gene_list <- c(gene_list, rep(g, length(anc)))
  }
  data.frame(term = term_list, gene = gene_list, stringsAsFactors = FALSE)
}

run_enricher <- function(deg, go_res, lfc_cut = 1, padj_cut = 0.05) {
  if (!requireNamespace("clusterProfiler", quietly = TRUE)) {
    return(list(table = NULL, note = "clusterProfiler not installed; skipped enricher"))
  }
  if (!isTRUE(go_res$has_godb)) {
    return(list(table = NULL, note = "enricher requires GO.db for ancestor TERM2GENE; skipped"))
  }
  sig <- deg[!is.na(deg$padj) & deg$padj < padj_cut &
               !is.na(deg$log2FoldChange) & abs(deg$log2FoldChange) >= lfc_cut, , drop = FALSE]
  universe <- unique(deg$go_id[!is.na(deg$go_id)])
  if (nrow(sig) < 3L) {
    return(list(table = NULL, note = paste0("Fewer than 3 significant GO terms for enricher (n=", nrow(sig), ")")))
  }
  t2g <- build_term2gene_ancestors(universe, go_res)
  if (is.null(t2g) || !nrow(t2g)) {
    return(list(table = NULL, note = "Failed to build TERM2GENE"))
  }
  ego <- tryCatch(
    clusterProfiler::enricher(
      gene = unique(sig$go_id),
      universe = universe,
      TERM2GENE = t2g,
      pAdjustMethod = "BH",
      pvalueCutoff = 1,
      qvalueCutoff = 1,
      minGSSize = 2,
      maxGSSize = 5000
    ),
    error = function(e) e
  )
  if (inherits(ego, "error")) {
    return(list(table = NULL, note = paste0("enricher failed: ", conditionMessage(ego))))
  }
  tab <- as.data.frame(ego)
  if (!nrow(tab)) {
    return(list(table = tab, note = "enricher returned 0 terms"))
  }
  # annotate descriptions
  ann <- annotate_go_ids(tab$ID, go_res)
  tab$Description <- ifelse(!is.na(ann$go_name), ann$go_name, tab$Description)
  tab$ontology <- ann$ontology
  list(table = tab, note = paste0("enricher OK: ", nrow(tab), " terms; sig genes=", nrow(sig)))
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

classify_deg <- function(deg, lfc_cut = 1, padj_cut = 0.05) {
  lab <- rep("NS", nrow(deg))
  ok <- !is.na(deg$padj) & !is.na(deg$log2FoldChange)
  lab[ok & deg$padj < padj_cut & deg$log2FoldChange >= lfc_cut] <- "Up"
  lab[ok & deg$padj < padj_cut & deg$log2FoldChange <= -lfc_cut] <- "Down"
  deg$diffexpressed <- lab
  deg
}

plot_volcano <- function(deg, out_pdf, out_png, lfc_cut = 1, padj_cut = 0.05, title = NULL) {
  suppressPackageStartupMessages(library(ggplot2))
  df <- classify_deg(deg, lfc_cut, padj_cut)
  df <- df[!is.na(df$log2FoldChange) & !is.na(df$padj), , drop = FALSE]
  df$neglog10p <- -log10(pmax(df$padj, .Machine$double.xmin))
  df$label <- ifelse(
    df$diffexpressed != "NS",
    ifelse(!is.na(df$go_name) & nzchar(df$go_name), df$go_name, df$go_id),
    NA_character_
  )
  # label top by |LFC| among sig
  sig <- which(df$diffexpressed != "NS")
  if (length(sig) > 15L) {
    keep_lab <- sig[order(-abs(df$log2FoldChange[sig]))[seq_len(15)]]
    df$label[-keep_lab] <- NA_character_
  }
  cols <- c(Up = COL_UP, Down = COL_DOWN, NS = COL_NS)
  p <- ggplot(df, aes(.data$log2FoldChange, .data$neglog10p, color = .data$diffexpressed)) +
    geom_point(alpha = 0.75, size = 1.6) +
    geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = 2, linewidth = 0.3) +
    geom_hline(yintercept = -log10(padj_cut), linetype = 2, linewidth = 0.3) +
    scale_color_manual(values = cols, name = NULL) +
    labs(
      title = title %||% "GO term DEG volcano",
      x = expression(log[2]~fold~change),
      y = expression(-log[10]~adjusted~italic(p))
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "top")
  if (requireNamespace("ggrepel", quietly = TRUE) && any(!is.na(df$label))) {
    p <- p + ggrepel::geom_text_repel(
      aes(label = .data$label), size = 2.5, max.overlaps = 20,
      show.legend = FALSE, na.rm = TRUE
    )
  }
  ensure_dir(dirname(out_pdf))
  ggplot2::ggsave(out_pdf, p, width = 7, height = 6)
  ggplot2::ggsave(out_png, p, width = 7, height = 6, dpi = 300)
  invisible(p)
}

plot_top_lfc_bars <- function(deg, out_pdf, out_png, top_n = 20L, title = NULL) {
  suppressPackageStartupMessages(library(ggplot2))
  df <- deg[!is.na(deg$log2FoldChange) & !is.na(deg$lfcSE), , drop = FALSE]
  if (!nrow(df)) fail("No DEG rows with LFC and SE for barplot")
  df <- df[order(-abs(df$log2FoldChange)), , drop = FALSE]
  df <- utils::head(df, as.integer(top_n))
  df$label <- ifelse(
    !is.na(df$go_name) & nzchar(df$go_name),
    paste0(df$go_id, " - ", df$go_name),
    df$go_id
  )
  if (anyDuplicated(df$label)) df$label <- paste0(df$label, " [", seq_len(nrow(df)), "]")
  df$direction <- ifelse(df$log2FoldChange >= 0, "Up", "Down")
  df$label <- factor(df$label, levels = rev(df$label))
  p <- ggplot(df, aes(.data$label, .data$log2FoldChange, fill = .data$direction)) +
    geom_col(width = 0.7) +
    geom_errorbar(
      aes(ymin = .data$log2FoldChange - .data$lfcSE, ymax = .data$log2FoldChange + .data$lfcSE),
      width = 0.25, linewidth = 0.4
    ) +
    coord_flip() +
    scale_fill_manual(values = c(Up = COL_UP, Down = COL_DOWN), name = NULL) +
    labs(
      title = title %||% paste0("Top-", nrow(df), " GO terms by |log2FC| (± SE)"),
      x = NULL,
      y = expression(log[2]~fold~change)
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "top")
  h <- max(5, 0.28 * nrow(df) + 2)
  ggplot2::ggsave(out_pdf, p, width = 9, height = h)
  ggplot2::ggsave(out_png, p, width = 9, height = h, dpi = 300)
  invisible(p)
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

default_long_candidates <- function() {
  c(
    file.path(root, ".cursor/skills/go/fixtures/go_function_long.csv"),
    "/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/data/processed/bakta_function_long.csv",
    file.path(root, "..", "2026", "Kristina", "data/processed/bakta_function_long.csv")
  )
}

run_go <- function(long = NULL, metadata = NULL, ps_rds = NULL, deg_path = NULL,
                   outdir = "test/go/run", group_col = "group", contrast = NULL,
                   prv_cut = 0.1, lfc_cut = 1, padj_cut = 0.05, top_n = 20L,
                   go_cache = file.path(root, ".cursor/skills/go/cache"),
                   fetch_go = TRUE, seed = 123L,
                   method = c("ancombc", "deseq2")) {
  method <- match.arg(method)
  setwd(root)
  ensure_dir(outdir)

  go_res <- ensure_go_resources(cache_dir = go_cache, fetch_if_missing = fetch_go)

  counts_path <- NULL
  de_info <- NULL

  if (!is.null(deg_path) && nzchar(deg_path)) {
    deg <- load_deg_table(deg_path)
    input_source <- "deg"
    input_path <- deg_path
  } else {
    if (is.null(long) || !nzchar(long)) {
      long <- NULL
      for (p in default_long_candidates()) {
        if (file.exists(p)) {
          long <- p
          break
        }
      }
      if (is.null(long)) fail("No GO long table found; pass --long")
    }
    go_long <- load_go_long(long)
    mat <- build_go_matrix(go_long)
    mat <- filter_prevalence(mat, prv_cut = prv_cut)
    meta_pack <- load_metadata(metadata = metadata, ps_rds = ps_rds, group_col = group_col)
    if (identical(method, "ancombc")) {
      de_info <- run_ancombc_go(mat, meta_pack$meta, contrast = contrast,
                                prv_cut = prv_cut, seed = seed)
      input_source <- "long+ancombc2"
    } else {
      de_info <- run_deseq_go(mat, meta_pack$meta, contrast = contrast, seed = seed)
      input_source <- "long+deseq2"
    }
    deg <- de_info$deg
    counts_path <- file.path(outdir, "go_counts.tsv")
    utils::write.table(
      data.frame(go_id = rownames(mat), mat, check.names = FALSE),
      counts_path, sep = "\t", quote = FALSE, row.names = FALSE
    )
    input_path <- long
  }

  ann <- annotate_go_ids(deg$go_id, go_res)
  deg$go_id <- ann$go_id
  deg$go_name <- ann$go_name
  deg$ontology <- ann$ontology
  deg <- classify_deg(deg, lfc_cut = lfc_cut, padj_cut = padj_cut)

  deg_path_out <- file.path(outdir, "go_deg.tsv")
  utils::write.table(deg, deg_path_out, sep = "\t", quote = FALSE, row.names = FALSE)

  enr <- run_enricher(deg, go_res, lfc_cut = lfc_cut, padj_cut = padj_cut)
  enr_path <- file.path(outdir, "go_enrichment.tsv")
  if (!is.null(enr$table) && nrow(enr$table)) {
    utils::write.table(enr$table, enr_path, sep = "\t", quote = FALSE, row.names = FALSE)
  } else {
    utils::write.table(
      data.frame(note = enr$note %||% "empty"),
      enr_path, sep = "\t", quote = FALSE, row.names = FALSE
    )
  }

  contrast_lab <- if (!is.null(de_info)) {
    paste0(de_info$contrast[["num"]], " vs ", de_info$contrast[["den"]])
  } else {
    "precomputed DEG"
  }

  vol_pdf <- file.path(outdir, "go_volcano.pdf")
  vol_png <- file.path(outdir, "go_volcano.png")
  bar_pdf <- file.path(outdir, "go_top20_lfc.pdf")
  bar_png <- file.path(outdir, "go_top20_lfc.png")
  plot_volcano(
    deg, vol_pdf, vol_png, lfc_cut = lfc_cut, padj_cut = padj_cut,
    title = paste0("GO ", toupper(method), " volcano (", contrast_lab, ")")
  )
  plot_top_lfc_bars(
    deg, bar_pdf, bar_png, top_n = top_n,
    title = paste0("Top-", top_n, " GO |log2FC| ± SE (", contrast_lab, ")")
  )

  n_sig <- sum(deg$diffexpressed != "NS", na.rm = TRUE)
  rep <- list(
    skill = "go",
    method = if (!is.null(de_info$method)) de_info$method else method,
    input_source = input_source,
    input_path = input_path,
    go_db = go_res$has_godb,
    go_db_n_terms = go_res$n_godb,
    obo_path = go_res$obo_path,
    go_fetched = go_res$fetched,
    go_notes = go_res$notes,
    n_go_tested = nrow(deg),
    n_significant = n_sig,
    lfc_cut = lfc_cut,
    padj_cut = padj_cut,
    contrast = contrast_lab,
    enricher_note = enr$note,
    n_enrichment_terms = if (!is.null(enr$table)) nrow(enr$table) else 0L,
    package_versions = list(
      GO.db = if (go_res$has_godb) as.character(utils::packageVersion("GO.db")) else NA,
      ANCOMBC = if (requireNamespace("ANCOMBC", quietly = TRUE)) as.character(utils::packageVersion("ANCOMBC")) else NA,
      DESeq2 = if (requireNamespace("DESeq2", quietly = TRUE)) as.character(utils::packageVersion("DESeq2")) else NA,
      clusterProfiler = if (requireNamespace("clusterProfiler", quietly = TRUE)) as.character(utils::packageVersion("clusterProfiler")) else NA,
      ggplot2 = as.character(utils::packageVersion("ggplot2"))
    ),
    figures = list(
      volcano_pdf = vol_pdf, volcano_png = vol_png,
      top_lfc_pdf = bar_pdf, top_lfc_png = bar_png
    ),
    tables = list(
      deg = deg_path_out,
      enrichment = enr_path,
      counts = counts_path
    )
  )
  write_json(rep, file.path(outdir, "go-report.json"))
  message(
    "go OK: method=", rep$method, " tested=", rep$n_go_tested, " sig=", rep$n_significant,
    " enricher=", rep$enricher_note, " volcano=", vol_pdf
  )
  invisible(rep)
}

self_test <- function() {
  setwd(root)
  fix_long <- file.path(root, ".cursor/skills/go/fixtures/go_function_long.csv")
  fix_meta <- file.path(root, ".cursor/skills/go/fixtures/metadata.csv")
  if (!file.exists(fix_long) || !file.exists(fix_meta)) {
    fail("Missing fixtures under .cursor/skills/go/fixtures/")
  }
  out <- "test/go/self-test"
  if (dir.exists(out)) unlink(out, recursive = TRUE)
  # Prefer GO.db; do not require network if GO.db present
  rep <- run_go(
    long = fix_long,
    metadata = fix_meta,
    group_col = "group",
    outdir = out,
    prv_cut = 0.2,
    lfc_cut = 0.5,
    padj_cut = 0.1,
    top_n = 10L,
    fetch_go = TRUE,
    seed = 1L,
    method = "ancombc"
  )
  if (!identical(rep$method, "ancombc2") && !identical(rep$method, "ancombc")) {
    stop("expected default method ancombc2; got ", rep$method)
  }
  if (!isTRUE(rep$go_db) && is.null(rep$obo_path)) stop("GO resources missing after ensure")
  if (!file.exists(rep$figures$volcano_pdf)) stop("missing volcano pdf")
  if (!file.exists(rep$figures$top_lfc_pdf)) stop("missing top LFC pdf")
  deg <- utils::read.delim(rep$tables$deg, stringsAsFactors = FALSE)
  if (!all(c("log2FoldChange", "lfcSE", "padj", "go_id") %in% names(deg))) {
    stop("DEG columns incomplete")
  }
  if (nrow(deg) < 5L) stop("expected ≥5 tested GO terms in fixture")
  message("SELF-TEST OK (fixture DEG n=", nrow(deg), ", sig=", rep$n_significant, ")")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  fetch <- !identical(tolower(as.character(args$fetch_go %||% "true")), "false")
  method <- tolower(as.character(args$method %||% "ancombc"))
  run_go(
    long = args$long,
    metadata = args$metadata,
    ps_rds = args$ps_rds,
    deg_path = args$deg,
    outdir = args$outdir %||% "test/go/run",
    group_col = args$group_col %||% "group",
    contrast = args$contrast,
    prv_cut = as.numeric(args$prv_cut %||% 0.1),
    lfc_cut = as.numeric(args$lfc_cut %||% 1),
    padj_cut = as.numeric(args$padj_cut %||% 0.05),
    top_n = as.integer(args$top_n %||% 20L),
    go_cache = args$go_cache %||% file.path(root, ".cursor/skills/go/cache"),
    fetch_go = fetch,
    seed = as.integer(args$seed %||% 123L),
    method = method
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
