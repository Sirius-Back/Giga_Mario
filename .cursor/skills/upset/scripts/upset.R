#!/usr/bin/env Rscript
# upset — ComplexUpset of taxon presence; default combine samples by target only
suppressPackageStartupMessages({
  for (p in c(
    "phyloseq", "ggplot2", "ComplexUpset", "dplyr", "tidyr", "jsonlite",
    "RColorBrewer", "ggpubr"
  )) {
    if (!requireNamespace(p, quietly = TRUE)) stop("Missing package: ", p, call. = FALSE)
  }
  library(phyloseq)
  library(ggplot2)
  library(ComplexUpset)
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

SKIP_VARS <- c(
  "seq", "ID", "SampleID", "sampleID", "sample_id", "Run", "run",
  "Condition2", "condition2"
)
GRAZING_COLS <- c("#4E9E23", "#FFA600", "#BE475A")
OKABE <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

`%||%` <- function(a, b) if (!is.null(a) && length(a) && !all(is.na(a))) a else b

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(
    path = path, target = NA_character_, batch = NA_character_,
    rarefaction_depth = NA_real_, rarefied = FALSE
  )
  if (inherits(obj, "phyloseq")) return(list(ps = obj, meta = meta))
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

resolve_input_rds <- function(rds = NULL) {
  candidates_rare <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds"
  )
  raw <- "test/code-review-phyloseq/grazing_phyloseq.rds"
  if (!is.null(rds) && nzchar(rds)) {
    loaded <- load_ps(rds)
    loaded$notes <- character(0)
    return(loaded)
  }
  for (p in candidates_rare) {
    if (file.exists(p)) {
      loaded <- load_ps(p)
      loaded$notes <- paste0("auto-resolved rarefied: ", p)
      return(loaded)
    }
  }
  if (file.exists(raw)) {
    loaded <- load_ps(raw)
    loaded$notes <- paste0("auto-resolved raw: ", raw)
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
    length(ux) >= 2L && length(ux) <= 12L
  }, logical(1))]
  cats <- setdiff(cats, SKIP_VARS)
  if (!length(cats)) fail("No categorical target found; pass --target")
  cats[[1]]
}

palette_for_sets <- function(n) {
  if (n == 3L) return(GRAZING_COLS)
  if (n <= length(OKABE)) return(OKABE[seq_len(n)])
  if (n <= 12L) return(RColorBrewer::brewer.pal(max(3L, n), "Set3")[seq_len(n)])
  grDevices::colorRampPalette(RColorBrewer::brewer.pal(12, "Set3"))(n)
}

#' Presence matrix: taxa × set (0/1). Prefer MicrobiotaProcess::get_upset.
build_upset_matrix <- function(ps, group_col) {
  sd <- as.data.frame(sample_data(ps), stringsAsFactors = FALSE)
  if (!group_col %in% names(sd)) fail("Grouping column missing: ", group_col)
  # Ensure factor with stable level order
  raw <- sd[[group_col]]
  if (is.numeric(raw) || all(grepl("^[0-9]+$", as.character(raw[!is.na(raw)])))) {
    lev <- as.character(sort(unique(as.integer(as.character(raw)))))
  } else {
    lev <- levels(factor(raw))
    if (is.null(lev) || !length(lev)) lev <- unique(as.character(raw[!is.na(raw)]))
    lev <- as.character(lev)
  }
  sd[[group_col]] <- factor(as.character(raw), levels = lev)
  sample_data(ps) <- sample_data(sd)

  if (requireNamespace("MicrobiotaProcess", quietly = TRUE)) {
    ud <- tryCatch(
      MicrobiotaProcess::get_upset(obj = ps, factorNames = group_col),
      error = function(e) NULL
    )
    if (!is.null(ud) && ncol(ud) >= 2L) {
      # Drop non-set annotation columns if any slipped in
      set_cols <- intersect(lev, colnames(ud))
      if (!length(set_cols)) {
        # get_upset may name columns by level labels as-is
        set_cols <- colnames(ud)[vapply(ud, function(x) {
          all(x %in% c(0, 1, TRUE, FALSE, NA))
        }, logical(1))]
      }
      if (length(set_cols) >= 2L) {
        out <- as.data.frame(ud[, set_cols, drop = FALSE], stringsAsFactors = FALSE)
        out[] <- lapply(out, function(x) {
          x <- as.numeric(as.character(x))
          x[is.na(x)] <- 0
          as.integer(x > 0)
        })
        return(list(matrix = out, sets = colnames(out), engine = "MicrobiotaProcess::get_upset"))
      }
    }
  }

  # Fallback: any-sample presence within each group level
  otu <- as(otu_table(ps), "matrix")
  if (!taxa_are_rows(ps)) otu <- t(otu)
  mat <- matrix(0, nrow = nrow(otu), ncol = length(lev),
                dimnames = list(rownames(otu), lev))
  for (lv in lev) {
    samps <- rownames(sd)[as.character(sd[[group_col]]) == lv]
    samps <- intersect(samps, colnames(otu))
    if (!length(samps)) next
    mat[, lv] <- as.integer(rowSums(otu[, samps, drop = FALSE] > 0) > 0)
  }
  out <- as.data.frame(mat)
  list(matrix = out, sets = colnames(out), engine = "fallback_any_presence")
}

save_plot <- function(plot_obj, prefix, width = 10, height = 7) {
  pdf <- paste0(prefix, ".pdf")
  png <- paste0(prefix, ".png")
  ggplot2::ggsave(pdf, plot_obj, width = width, height = height)
  png_out <- NULL
  tryCatch({
    grDevices::png(png, width = width * 300, height = height * 300, res = 300, type = "cairo")
    print(plot_obj)
    grDevices::dev.off()
    png_out <- png
  }, error = function(e) {
    if (grDevices::dev.cur() > 1) grDevices::dev.off()
    message("PNG skip: ", conditionMessage(e))
  })
  list(pdf = pdf, png = png_out)
}

run_upset <- function(
    rds = NULL,
    outdir = "test/upset/run",
    target = NULL,
    factors = NULL,
    min_size = 3L,
    annotate = NULL,
    width = 10,
    height = 7,
    label_mode = "count"
) {
  setwd(project_root())
  ensure_dir(outdir)
  loaded <- resolve_input_rds(rds)
  ps <- loaded$ps
  sd <- as.data.frame(sample_data(ps), stringsAsFactors = FALSE)

  mode <- "target"
  group_col <- NULL
  factor_cols <- NULL

  if (!is.null(factors) && nzchar(factors)) {
    mode <- "factors"
    factor_cols <- trimws(strsplit(factors, ",", fixed = TRUE)[[1]])
    miss <- setdiff(factor_cols, names(sd))
    if (length(miss)) fail("Unknown --factors columns: ", paste(miss, collapse = ", "))
    if (length(factor_cols) == 1L) {
      group_col <- factor_cols[[1]]
    } else {
      # PacBio-style combined label (opt-in only)
      group_col <- ".upset_group"
      sd[[group_col]] <- do.call(paste, c(sd[factor_cols], sep = " | "))
      sample_data(ps) <- sample_data(sd)
    }
  } else {
    group_col <- discover_target(ps, target, loaded$meta$target)
  }

  built <- build_upset_matrix(ps, group_col)
  upset_data <- built$matrix
  sets <- built$sets
  if (length(sets) < 2L) fail("Need ≥2 non-empty sets; found: ", paste(sets, collapse = ", "))

  # Drop taxa absent from all sets
  keep <- rowSums(upset_data[, sets, drop = FALSE]) > 0
  upset_data <- upset_data[keep, , drop = FALSE]
  if (!nrow(upset_data)) fail("No taxa with presence in any set")

  # Optional annotation column for intersection_size fill (must be on rows = taxa)
  if (!is.null(annotate) && nzchar(annotate)) {
    # Not sample annotation — skip unless column already on matrix; document as unused for taxon rows
    message("Note: --annotate applies only if a taxon-level column is pre-joined; sample metadata annotate is skipped for presence UpSet.")
  }

  n_sets <- length(sets)
  set_colors <- palette_for_sets(n_sets)
  names(set_colors) <- sets

  uq_list <- lapply(seq_along(sets), function(i) {
    ComplexUpset::upset_query(
      set = sets[[i]],
      fill = set_colors[[i]],
      only_components = "overall_sizes"
    )
  })

  # Theme baseline (PacBio note: avoid fragile legend.text merges in annotations)
  old_theme <- theme_get()
  on.exit(theme_set(old_theme), add = TRUE)
  theme_set(theme_minimal() + theme(axis.title.x = element_text(), axis.title.y = element_text()))

  n_taxa <- nrow(upset_data)
  # Intersection labels: counts by default (not %); override with --label-mode percent
  if (identical(tolower(as.character(label_mode)), "percent")) {
    size_label_aes <- aes(
      label = paste0(
        round(!!ComplexUpset::get_size_mode("exclusive_intersection") / n_taxa * 100, 1),
        "%"
      )
    )
  } else {
    size_label_aes <- aes(
      label = !!ComplexUpset::get_size_mode("exclusive_intersection")
    )
  }
  gg <- ComplexUpset::upset(
    upset_data,
    intersect = sets,
    name = if (identical(mode, "target")) paste0("Target: ", group_col) else "Groups",
    width_ratio = 0.2,
    stripes = "white",
    min_size = as.integer(min_size),
    base_annotations = list(
      "Taxa in sets" = (
        ComplexUpset::intersection_size(
          bar_number_threshold = 2000,
          text_mapping = size_label_aes,
          text = list(check_overlap = TRUE, size = 3)
        ) +
          theme(panel.grid = element_blank())
      )
    ),
    set_sizes = (
      ComplexUpset::upset_set_size() +
        ggplot2::geom_text(
          aes(label = after_stat(count)),
          hjust = 1.1, stat = "count", size = 3
        ) +
        scale_y_reverse(n.breaks = 3) +
        ylab("Taxa sets") +
        # left margin ×1.5 vs prior (5.5 + 12 pt)
        theme(plot.margin = margin(5.5, 5.5, 5.5, (5.5 + 1.2 * 10) * 1.5, unit = "pt"))
    ),
    themes = ComplexUpset::upset_modify_themes(
      list(
        "intersections_matrix" = theme(
          axis.text.y = element_text(face = "italic"),
          axis.title.x = element_text(),
          axis.title.y = element_blank()
        ),
        "overall_sizes" = theme(
          plot.margin = margin(5.5, 5.5, 5.5, (5.5 + 12) * 1.5, unit = "pt")
        )
      )
    ),
    sort_intersections_by = c("degree", "cardinality"),
    sort_intersections = "descending",
    sort_sets = FALSE,
    queries = uq_list
  )

  mat_path <- file.path(outdir, "upset_matrix.tsv")
  utils::write.table(
    cbind(taxon = rownames(upset_data), upset_data),
    mat_path, sep = "\t", quote = FALSE, row.names = FALSE
  )

  figs <- save_plot(gg, file.path(outdir, "upset"), width = width, height = height)

  report <- list(
    input_rds = loaded$meta$path,
    notes = loaded$notes,
    mode = mode,
    group_column = group_col,
    factor_columns = factor_cols,
    sets = sets,
    n_sets = n_sets,
    n_taxa = n_taxa,
    min_size = as.integer(min_size),
    label_mode = as.character(label_mode),
    matrix_engine = built$engine,
    matrix = mat_path,
    figures = figs,
    package = "ComplexUpset"
  )
  write_json(report, file.path(outdir, "upset-report.json"))
  message(
    "UpSet OK: mode=", mode, " group=", group_col,
    " n_sets=", n_sets, " n_taxa=", n_taxa,
    " figure=", figs$pdf
  )
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  out <- "test/upset/grazing-self-test"
  rep <- run_upset(
    rds = "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    outdir = out,
    target = "grazing",
    min_size = 3L
  )
  if (!identical(rep$mode, "target")) stop("expected mode=target")
  if (rep$n_sets != 3L) stop("expected 3 grazing levels")
  if (!file.exists(rep$figures$pdf)) stop("missing upset.pdf")
  if (!file.exists(rep$matrix)) stop("missing matrix")
  message("SELF-TEST OK")
  invisible(rep)
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  run_upset(
    rds = args$rds,
    outdir = args$outdir %||% "test/upset/run",
    target = args$target %||% args$targets,
    factors = args$factors,
    min_size = as.integer(args$min_size %||% 3L),
    annotate = args$annotate,
    width = as.numeric(args$width %||% 10),
    height = as.numeric(args$height %||% 7),
    label_mode = args$label_mode %||% "count"
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
