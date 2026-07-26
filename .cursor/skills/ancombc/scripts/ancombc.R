#!/usr/bin/env Rscript
# ancombc — ANCOM-BC2 on rarefied phyloseq; all/specified targets × aggregation levels
suppressPackageStartupMessages({
  stopifnot(requireNamespace("phyloseq", quietly = TRUE))
  stopifnot(requireNamespace("ANCOMBC", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
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

SKIP_VARS <- c("seq", "ID", "SampleID", "sampleID", "sample_id", "Run", "run")
SKIP_RANKS <- c("tip_rank")

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path, target = NA_character_, batch = NA_character_,
    rarefaction_depth = NA_real_, abundances = NA_character_, rarefied = FALSE
  )
  if (inherits(obj, "phyloseq")) {
    meta$rarefied <- grepl("_rare(\\.rds)?$|phyloseq_rare_", basename(path))
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

resolve_input_rds <- function(rds = NULL, require_rare = TRUE, allow_non_rare = FALSE) {
  notes <- character(0)
  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )

  try_path <- function(p, mark_rare = FALSE) {
    if (!file.exists(p)) return(NULL)
    loaded <- load_ps(p)
    if (!is_count_like(loaded$ps)) return(NULL)
    if (mark_rare) loaded$meta$rarefied <- TRUE
    loaded$meta$rarefied <- isTRUE(loaded$meta$rarefied) ||
      grepl("_rare|phyloseq_rare_", basename(p))
    loaded
  }

  if (!is.null(rds) && nzchar(rds)) {
    loaded <- try_path(rds)
    if (is.null(loaded)) {
      fail("Unusable RDS (missing or not count-like): ", rds)
    }
    if (require_rare && !isTRUE(loaded$meta$rarefied) && !allow_non_rare) {
      alt <- sub("\\.rds$", "_rare.rds", rds)
      alt2 <- sub("_batchadj\\.rds$", "_rare.rds", rds)
      for (a in unique(c(alt, alt2, candidates_rare))) {
        hit <- try_path(a, mark_rare = TRUE)
        if (!is.null(hit) && isTRUE(hit$meta$rarefied)) {
          hit$notes <- c(notes, paste0("ANCOM-BC requires rarefied; switched to ", a))
          return(hit)
        }
      }
      fail(
        "ANCOM-BC skill requires a rarefied phyloseq object. ",
        "Pass a *_rare.rds or run rarefaction-analysis first. Got: ", rds
      )
    }
    loaded$notes <- notes
    return(loaded)
  }

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
  fail("No rarefied phyloseq found under test/; run rarefaction-analysis or pass --rds")
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
  if (length(missing)) fail("Target column(s) missing: ", paste(missing, collapse = ", "))
  if (!length(tg)) fail("No target variables resolved; pass --targets")

  for (t in tg) {
    nlev <- length(unique(stats::na.omit(sam[[t]])))
    if (nlev < 2L) fail("Target '", t, "' has <2 levels")
  }
  tg
}

resolve_levels <- function(ps, levels_arg = NULL) {
  ranks <- phyloseq::rank_names(ps)
  ranks <- ranks[!ranks %in% SKIP_RANKS]
  default <- c("ASV", ranks)

  if (is.null(levels_arg) || !nzchar(levels_arg)) {
    return(default)
  }
  lev <- trimws(strsplit(levels_arg, ",", fixed = TRUE)[[1]])
  lev <- lev[nzchar(lev)]
  # map case-insensitive to actual rank names; ASV stays ASV
  out <- character(0)
  for (L in lev) {
    if (toupper(L) == "ASV") {
      out <- c(out, "ASV")
      next
    }
    hit <- ranks[tolower(ranks) == tolower(L)]
    if (!length(hit)) {
      fail("Level '", L, "' not in rank_names: ", paste(ranks, collapse = ", "))
    }
    out <- c(out, hit[[1]])
  }
  unique(out)
}

prepare_ps_for_target <- function(ps, target, covariates = NULL) {
  sam <- as(sample_data(ps), "data.frame")
  keep <- !is.na(sam[[target]])
  if (!is.null(covariates) && length(covariates)) {
    for (cv in covariates) {
      if (!cv %in% names(sam)) fail("Covariate missing from sample_data: ", cv)
      keep <- keep & !is.na(sam[[cv]])
    }
  }
  if (sum(keep) < 4L) fail("Too few non-NA samples for target '", target, "': ", sum(keep))
  ps2 <- prune_samples(keep, ps)
  ps2 <- prune_taxa(taxa_sums(ps2) > 0, ps2)
  sam2 <- as(sample_data(ps2), "data.frame")
  sam2[[target]] <- factor(sam2[[target]])
  if (nlevels(sam2[[target]]) < 2L) {
    fail("Target '", target, "' has <2 levels after NA drop")
  }
  for (cv in covariates %||% character(0)) {
    if (is.character(sam2[[cv]]) || is.logical(sam2[[cv]])) {
      sam2[[cv]] <- factor(sam2[[cv]])
    }
  }
  sample_data(ps2) <- sample_data(sam2)
  list(ps = ps2, n_levels = nlevels(sam2[[target]]))
}

build_formula <- function(target, covariates = NULL) {
  parts <- c(target, covariates %||% character(0))
  paste(parts, collapse = " + ")
}

extract_res_df <- function(out) {
  if (is.null(out)) return(NULL)
  if (!is.null(out$res) && is.data.frame(out$res)) return(out$res)
  if (!is.null(out$res_df) && is.data.frame(out$res_df)) return(out$res_df)
  # older list-of-matrices style
  if (!is.null(out$res) && is.list(out$res) && !is.null(out$res$lfc)) {
    lfc <- as.data.frame(out$res$lfc)
    se <- as.data.frame(out$res$se)
    p <- as.data.frame(out$res$p_val)
    q <- as.data.frame(out$res$q_val)
    taxa <- rownames(lfc)
    df <- data.frame(taxon = taxa, stringsAsFactors = FALSE)
    for (nm in names(lfc)) {
      df[[paste0("lfc_", nm)]] <- lfc[[nm]]
      df[[paste0("se_", nm)]] <- se[[nm]]
      df[[paste0("p_", nm)]] <- p[[nm]]
      df[[paste0("q_", nm)]] <- q[[nm]]
    }
    return(df)
  }
  NULL
}

tidy_res_long <- function(res_df, target, level) {
  if (is.null(res_df) || !nrow(res_df)) {
    return(data.frame(
      target = character(), level = character(), taxon = character(),
      term = character(), lfc = numeric(), se = numeric(),
      W = numeric(), p = numeric(), q = numeric(),
      stringsAsFactors = FALSE
    ))
  }
  taxon_col <- if ("taxon" %in% names(res_df)) "taxon" else names(res_df)[[1]]
  lfc_cols <- grep("^lfc_", names(res_df), value = TRUE)
  # drop intercept-only terms from primary long table but keep all contrast terms
  lfc_cols <- lfc_cols[!grepl("lfc_\\(Intercept\\)$", lfc_cols)]
  if (!length(lfc_cols)) {
    lfc_cols <- grep("^lfc_", names(res_df), value = TRUE)
  }
  rows <- list()
  for (lc in lfc_cols) {
    term <- sub("^lfc_", "", lc)
    se_c <- paste0("se_", term)
    p_c <- paste0("p_", term)
    q_c <- paste0("q_", term)
    w_c <- paste0("W_", term)
    rows[[length(rows) + 1L]] <- data.frame(
      target = target,
      level = level,
      taxon = as.character(res_df[[taxon_col]]),
      term = term,
      lfc = as.numeric(res_df[[lc]]),
      se = if (se_c %in% names(res_df)) as.numeric(res_df[[se_c]]) else NA_real_,
      W = if (w_c %in% names(res_df)) as.numeric(res_df[[w_c]]) else NA_real_,
      p = if (p_c %in% names(res_df)) as.numeric(res_df[[p_c]]) else NA_real_,
      q = if (q_c %in% names(res_df)) as.numeric(res_df[[q_c]]) else NA_real_,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

run_ancombc2_once <- function(ps, target, level, covariates = NULL,
                              prv_cut = 0.1, p_adj_method = "fdr",
                              pseudo_sens = FALSE, pairwise = NULL,
                              seed = 123L) {
  prep <- prepare_ps_for_target(ps, target, covariates)
  ps2 <- prep$ps
  if (is.null(pairwise)) pairwise <- prep$n_levels > 2L

  fix_f <- build_formula(target, covariates)
  tax_level <- if (identical(level, "ASV")) NULL else level

  set.seed(seed)
  message("ANCOM-BC2: target=", target, " level=", level,
          " ntaxa=", ntaxa(ps2), " nsamples=", nsamples(ps2),
          " formula=", fix_f, " pairwise=", pairwise)

  out <- tryCatch(
    ANCOMBC::ancombc2(
      data = ps2,
      assay_name = "counts",
      tax_level = tax_level,
      fix_formula = fix_f,
      rand_formula = NULL,
      p_adj_method = p_adj_method,
      pseudo = 0,
      pseudo_sens = pseudo_sens,
      prv_cut = prv_cut,
      group = target,
      global = FALSE,
      pairwise = pairwise,
      struc_zero = FALSE,
      neg_lb = FALSE,
      verbose = FALSE,
      n_cl = 1L
    ),
    error = function(e) {
      message("ERROR ancombc2 (", target, "/", level, "): ", conditionMessage(e))
      NULL
    }
  )
  res_df <- extract_res_df(out)
  list(
    res = res_df,
    ok = !is.null(res_df) && nrow(res_df) > 0,
    n_taxa = if (!is.null(res_df)) nrow(res_df) else 0L,
    pairwise = pairwise,
    formula = fix_f,
    error = if (is.null(out)) "ancombc2 failed" else NA_character_
  )
}

safe_name <- function(x) {
  gsub("[^A-Za-z0-9._-]+", "_", x)
}

run_ancombc <- function(rds = NULL, outdir,
                        targets = NULL, levels = NULL,
                        covariates = NULL,
                        prv_cut = 0.1,
                        p_adj_method = "fdr",
                        pseudo_sens = FALSE,
                        pairwise = NULL,
                        require_rare = TRUE,
                        allow_non_rare = FALSE,
                        seed = 123L) {
  ensure_dir(outdir)
  loaded <- resolve_input_rds(
    rds,
    require_rare = require_rare,
    allow_non_rare = allow_non_rare
  )
  ps <- loaded$ps
  meta <- loaded$meta
  if (!isTRUE(meta$rarefied) && !allow_non_rare) {
    fail("Input is not marked rarefied: ", meta$path)
  }

  tg <- resolve_targets(ps, meta, targets_arg = targets)
  lev <- resolve_levels(ps, levels_arg = levels)

  cov <- if (is.null(covariates) || !nzchar(covariates)) {
    character(0)
  } else {
    trimws(strsplit(covariates, ",", fixed = TRUE)[[1]])
  }
  cov <- cov[nzchar(cov)]

  message(
    "Input: ", meta$path,
    " rarefied=", meta$rarefied,
    " depth=", meta$rarefaction_depth %||% NA,
    " targets=", paste(tg, collapse = ","),
    " levels=", paste(lev, collapse = ",")
  )

  nested <- list()
  long_list <- list()
  level_summary <- list()

  for (t in tg) {
    nested[[t]] <- list()
    for (L in lev) {
      one <- run_ancombc2_once(
        ps, target = t, level = L, covariates = cov,
        prv_cut = prv_cut, p_adj_method = p_adj_method,
        pseudo_sens = pseudo_sens, pairwise = pairwise, seed = seed
      )
      nested[[t]][[L]] <- one$res
      wide_path <- file.path(
        outdir,
        paste0("ancombc_", safe_name(t), "_", safe_name(L), ".tsv")
      )
      if (one$ok) {
        utils::write.table(
          one$res, wide_path, sep = "\t", quote = FALSE,
          row.names = FALSE
        )
        long_list[[length(long_list) + 1L]] <- tidy_res_long(one$res, t, L)
      }
      n_sig <- if (one$ok) {
        qcols <- grep("^q_", names(one$res), value = TRUE)
        qcols <- qcols[!grepl("q_\\(Intercept\\)$", qcols)]
        if (!length(qcols)) 0L else sum(vapply(qcols, function(qc) {
          sum(one$res[[qc]] < 0.05, na.rm = TRUE)
        }, integer(1)))
      } else {
        0L
      }
      level_summary[[length(level_summary) + 1L]] <- list(
        target = t,
        level = L,
        ok = one$ok,
        n_taxa = one$n_taxa,
        n_sig_q05 = n_sig,
        pairwise = one$pairwise,
        formula = one$formula,
        wide_tsv = if (one$ok) wide_path else NA_character_,
        error = one$error
      )
      message(
        "  → ", t, "/", L, ": ok=", one$ok,
        " n_taxa=", one$n_taxa, " n_sig_q<0.05=", n_sig
      )
    }
  }

  long_df <- if (length(long_list)) {
    do.call(rbind, long_list)
  } else {
    data.frame(
      target = character(), level = character(), taxon = character(),
      term = character(), lfc = numeric(), se = numeric(),
      W = numeric(), p = numeric(), q = numeric(),
      stringsAsFactors = FALSE
    )
  }
  long_path <- file.path(outdir, "ancombc_results.tsv")
  utils::write.table(long_df, long_path, sep = "\t", quote = FALSE, row.names = FALSE)

  rds_path <- file.path(outdir, "ancombc_results.rds")
  saveRDS(nested, rds_path)

  # Verify every requested level produced a usable table (needed next stage)
  failed <- Filter(function(x) !isTRUE(x$ok), level_summary)
  if (length(failed)) {
    msg <- paste(vapply(failed, function(x) {
      paste0(x$target, "/", x$level, if (!is.na(x$error)) paste0(" (", x$error, ")") else "")
    }, character(1)), collapse = "; ")
    fail("ANCOM-BC failed for level(s) — needed for next stage: ", msg)
  }

  levels_ok <- unique(vapply(level_summary, `[[`, character(1), "level"))
  if (!all(lev %in% levels_ok)) {
    fail("Missing results for levels: ", paste(setdiff(lev, levels_ok), collapse = ", "))
  }

  report <- list(
    skill = "ancombc",
    method = "ancombc2",
    input_rds = meta$path,
    rarefied = isTRUE(meta$rarefied),
    rarefaction_depth = meta$rarefaction_depth,
    targets = tg,
    levels = lev,
    covariates = cov,
    prv_cut = prv_cut,
    p_adj_method = p_adj_method,
    pseudo_sens = pseudo_sens,
    n_rows_long = nrow(long_df),
    n_sig_q05 = if (nrow(long_df)) sum(long_df$q < 0.05, na.rm = TRUE) else 0L,
    level_summary = level_summary,
    results_tsv = long_path,
    results_rds = rds_path,
    notes = loaded$notes %||% character(0)
  )
  write_json(report, file.path(outdir, "ancombc-report.json"))
  message(
    "Wrote ", long_path, " (", nrow(long_df), " rows); ",
    "levels OK: ", paste(lev, collapse = ", ")
  )
  report
}

self_test <- function() {
  setwd(project_root())
  out <- "test/ancombc/grazing-self-test"
  # Multi-level check (ASV + Family + Genus) — required for next-stage confidence
  rep <- run_ancombc(
    rds = NULL,
    outdir = out,
    targets = "grazing",
    levels = "ASV,Family,Genus",
    prv_cut = 0.1,
    pseudo_sens = FALSE,
    pairwise = FALSE,
    require_rare = TRUE,
    seed = 123L
  )
  if (!isTRUE(rep$rarefied)) stop("expected rarefied input")
  if (!all(c("ASV", "Family", "Genus") %in% rep$levels)) stop("levels mismatch")
  if (!file.exists(rep$results_tsv)) stop("missing ancombc_results.tsv")
  if (!file.exists(rep$results_rds)) stop("missing ancombc_results.rds")
  long <- utils::read.table(rep$results_tsv, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  if (!nrow(long)) stop("empty long results")
  for (L in c("ASV", "Family", "Genus")) {
    n <- sum(long$level == L)
    if (n < 1L) stop("no long rows for level ", L)
    wide <- file.path(out, paste0("ancombc_grazing_", L, ".tsv"))
    if (!file.exists(wide)) stop("missing wide TSV for ", L)
  }
  # Family should have fewer taxa than ASV
  n_asv <- Filter(function(x) x$level == "ASV", rep$level_summary)[[1]]$n_taxa
  n_fam <- Filter(function(x) x$level == "Family", rep$level_summary)[[1]]$n_taxa
  if (!(n_fam < n_asv)) stop("expected Family n_taxa < ASV n_taxa; got ", n_fam, " vs ", n_asv)
  nested <- readRDS(rep$results_rds)
  if (is.null(nested$grazing$Family)) stop("nested Family missing")
  if (is.null(nested$grazing$ASV)) stop("nested ASV missing")
  message("SELF-TEST OK (ASV=", n_asv, " Family=", n_fam,
          " Genus=", Filter(function(x) x$level == "Genus", rep$level_summary)[[1]]$n_taxa, ")")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  outdir <- args$outdir %||% "test/ancombc/run"
  prv_cut <- as.numeric(args$prv_cut %||% 0.1)
  p_adj <- args$p_adj_method %||% "fdr"
  pseudo_sens <- identical(tolower(as.character(args$pseudo_sens %||% "false")), "true")
  allow_non_rare <- identical(tolower(as.character(args$allow_non_rare %||% "false")), "true")
  pairwise <- if (is.null(args$pairwise)) {
    NULL
  } else {
    identical(tolower(as.character(args$pairwise)), "true")
  }
  seed <- as.integer(args$seed %||% 123L)

  run_ancombc(
    rds = args$rds,
    outdir = outdir,
    targets = args$targets,
    levels = args$levels,
    covariates = args$covariates,
    prv_cut = prv_cut,
    p_adj_method = p_adj,
    pseudo_sens = pseudo_sens,
    pairwise = pairwise,
    require_rare = TRUE,
    allow_non_rare = allow_non_rare,
    seed = seed
  )
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
