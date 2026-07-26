#!/usr/bin/env Rscript
# Shared helpers for metabarcoding-import / metagenomic-import / bracken-parse

`%||%` <- function(a, b) if (!is.null(a) && length(a) && !all(is.na(a))) a else b

project_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  start <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = FALSE)
  } else {
    getwd()
  }
  cur <- if (file.exists(start) && !dir.exists(start)) dirname(start) else start
  for (i in seq_len(8)) {
    if (file.exists(file.path(cur, "artifact-registry.md")) ||
        dir.exists(file.path(cur, ".cursor", "hooks"))) {
      return(cur)
    }
    parent <- dirname(cur)
    if (identical(parent, cur)) break
    cur <- parent
  }
  getwd()
}

shared_import_dir <- function() {
  file.path(project_root(), ".cursor/skills/_shared/import")
}

fail_missing_metadata <- function(extra = NULL) {
  msg <- paste0(
    "Missing metadata. Hooks MUST use: metadata; data (all available); then reconstruct the tree. ",
    "Exit and run skill @fix-metadata to locate or repair metadata",
    if (!is.null(extra)) paste0(" (", extra, ")") else "",
    "."
  )
  fail(msg)
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

parse_kv_args <- function(argv = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    key <- sub("^--", "", key)
    if (key %in% c("self-test", "help")) {
      out[[gsub("-", "_", key)]] <- TRUE
      i <- i + 1L
      next
    }
    if (i == length(argv)) stop("Missing value for --", key)
    out[[gsub("-", "_", key)]] <- argv[[i + 1L]]
    i <- i + 2L
  }
  out
}

fail <- function(...) {
  msg <- paste0(...)
  message("ERROR: ", msg)
  quit(save = "no", status = 2)
}

write_json <- function(obj, path) {
  ensure_dir(dirname(path))
  writeLines(jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE, null = "null"), path)
}

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

read_sample_metadata <- function(path) {
  if (!file.exists(path)) fail("Sample metadata not found: ", path)
  ext <- tolower(tools::file_ext(path))
  df <- if (ext %in% c("csv")) {
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  }
  id_col <- intersect(c("sampleID", "SampleID", "sample-id", "sample_id", "Run", "sample"), names(df))
  if (!length(id_col)) {
    fail("Metadata must contain sampleID (or SampleID/sample-id/Run): ", path)
  }
  id_col <- id_col[[1]]
  df$sampleID <- as.character(df[[id_col]])
  if (any(!nzchar(df$sampleID) | is.na(df$sampleID))) {
    fail("Metadata has empty sampleID values: ", path)
  }
  if (anyDuplicated(df$sampleID)) fail("Duplicate sampleID in metadata: ", path)
  rownames(df) <- df$sampleID
  df
}

# ---------------------------------------------------------------------------
# Taxonomy cleanup (codebase-aligned)
# ---------------------------------------------------------------------------

HOST_NAME_PATTERNS <- c(
  "^Homo(\\s|$)",
  "^Homo sapiens",
  "Chordata",
  "Metazoa",
  "Mammalia",
  "Hominidae",
  "Primates"
)

is_host_taxon_name <- function(names_vec) {
  x <- as.character(names_vec)
  Reduce(`|`, lapply(HOST_NAME_PATTERNS, function(p) grepl(p, x, ignore.case = TRUE)))
}

normalize_unclassified_vec <- function(x) {
  x <- as.character(x)
  bad <- is.na(x) | !nzchar(trimws(x)) |
    grepl("^(Unclassified|Unassigned|Unknown|NA)([_ ].*)?$", x, ignore.case = TRUE) |
    grepl("^unclassified", x, ignore.case = TRUE) |
    grepl("^unassigned", x, ignore.case = TRUE) |
    grepl("^uncultured", x, ignore.case = TRUE) |
    grepl("^unknown", x, ignore.case = TRUE)
  x[bad] <- "Unclassified"
  x
}

is_placeholder_taxon <- function(x) {
  x <- as.character(x)
  is.na(x) | !nzchar(trimws(x)) |
    grepl("^(Unclassified|Unassigned|Unknown|NA)([_* ].*)?$", trimws(x), ignore.case = TRUE) |
    grepl("^unclassified", x, ignore.case = TRUE) |
    grepl("^unassigned", x, ignore.case = TRUE) |
    grepl("^uncultured", trimws(x), ignore.case = TRUE) |
    grepl("^unknown", trimws(x), ignore.case = TRUE)
}

#' Clearify-style display label + source rank from one taxonomy row (placeholders NA'd).
display_tip_name <- function(tax_row, ranks = NULL) {
  tax_row <- as.list(tax_row)
  if (is.null(ranks)) {
    ranks <- intersect(RANK_COLS, names(tax_row))
    if (!length(ranks)) ranks <- intersect(RANK_COLS_LC, names(tax_row))
  }
  get_val <- function(rk) {
    if (!rk %in% names(tax_row)) return(NA_character_)
    v <- as.character(tax_row[[rk]])
    if (is_placeholder_taxon(v)) NA_character_ else v
  }
  gcol <- intersect(c("Genus", "genus"), names(tax_row))
  gcol <- if (length(gcol)) gcol[[1]] else NA_character_
  scol <- intersect(c("Species", "species"), names(tax_row))
  scol <- if (length(scol)) scol[[1]] else NA_character_
  fcol <- intersect(c("Family", "family"), names(tax_row))
  fcol <- if (length(fcol)) fcol[[1]] else NA_character_
  ccol <- intersect(c("Class", "class"), names(tax_row))
  ccol <- if (length(ccol)) ccol[[1]] else NA_character_

  genus <- if (!is.na(gcol)) get_val(gcol) else NA_character_
  family <- if (!is.na(fcol)) get_val(fcol) else NA_character_
  class <- if (!is.na(ccol)) get_val(ccol) else NA_character_
  species <- if (!is.na(scol)) get_val(scol) else NA_character_

  if (!is.na(species) && nzchar(species) &&
      grepl("\\s", species) && (is.na(genus) || species != genus)) {
    return(list(name = species, rank = scol))
  }
  if (!is.na(genus) && nzchar(genus)) {
    return(list(name = genus, rank = gcol))
  }
  if (!is.na(family) && nzchar(family)) {
    return(list(name = family, rank = fcol))
  }
  if (!is.na(class) && nzchar(class)) {
    return(list(name = class, rank = ccol))
  }
  for (rk in rev(ranks)) {
    v <- get_val(rk)
    if (!is.na(v) && nzchar(v)) return(list(name = v, rank = rk))
  }
  deepest <- if (length(ranks)) ranks[[length(ranks)]] else "Species"
  list(name = "Unknown", rank = deepest)
}

#' Strip Candidatus_ / Candidatus prefixes (grazing Article pattern).
strip_candidatus <- function(x) {
  x <- as.character(x)
  x <- gsub("^Candidatus[_ ]+", "", x, ignore.case = TRUE)
  x <- gsub("\\bCandidatus[_ ]+", "", x, ignore.case = TRUE)
  x
}

#' Clean rank columns: placeholders, Candidatus_, fill-forward last classified.
#' Ensures deepest rank is never Unclassified/Unassigned — promotes last known name.
#' Duplicate full lineages get " ASV1", " ASV2", … ordered by mean abundance (high→low).
finalize_taxonomy_for_phyloseq <- function(tax_df, otu_mat = NULL) {
  tax_df <- as.data.frame(tax_df, stringsAsFactors = FALSE)
  ranks <- intersect(c(RANK_COLS, RANK_COLS_LC), names(tax_df))
  if (!length(ranks)) {
    nm <- names(tax_df)
    for (std in RANK_COLS) {
      hit <- nm[tolower(nm) == tolower(std)]
      if (length(hit)) names(tax_df)[names(tax_df) == hit[[1]]] <- std
    }
    ranks <- intersect(RANK_COLS, names(tax_df))
  }
  if (!length(ranks)) fail("finalize_taxonomy_for_phyloseq: no rank columns found")

  for (rk in ranks) {
    tax_df[[rk]] <- strip_candidatus(normalize_unclassified_vec(tax_df[[rk]]))
    # treat placeholders (incl. uncultured/unknown) as missing before fill-forward
    tax_df[[rk]][is_placeholder_taxon(tax_df[[rk]])] <- NA_character_
  }

  deepest <- ranks[[length(ranks)]]
  tip_rank <- character(nrow(tax_df))
  tip_names <- character(nrow(tax_df))
  for (i in seq_len(nrow(tax_df))) {
    tip <- display_tip_name(tax_df[i, , drop = FALSE], ranks = ranks)
    tip_names[[i]] <- tip$name
    tip_rank[[i]] <- tip$rank
  }

  tax_df <- fill_na_last_classified(tax_df, ranks)
  for (rk in ranks) {
    tax_df[[rk]] <- normalize_unclassified_vec(tax_df[[rk]])
  }
  tax_df[[deepest]] <- tip_names
  tax_df$tip_rank <- tip_rank

  # duplicate disambiguation: " ASV1", " ASV2", … (never make.names / make.unique dots)
  lineage_base <- vapply(seq_len(nrow(tax_df)), function(i) {
    vals <- vapply(ranks, function(rk) {
      v <- as.character(tax_df[[rk]][[i]])
      if (identical(rk, deepest)) v <- sub(" ASV[0-9]+$", "", tip_names[[i]])
      v
    }, character(1))
    paste(vals, collapse = "|")
  }, character(1))

  mean_abd <- rep(0, nrow(tax_df))
  if (!is.null(otu_mat)) {
    om <- as.matrix(otu_mat)
    if (ncol(om) == nrow(tax_df) && nrow(om) != nrow(tax_df)) om <- t(om)
    if (!is.null(rownames(tax_df))) {
      if (!is.null(rownames(om))) {
        om <- om[match(rownames(tax_df), rownames(om)), , drop = FALSE]
      }
    }
    if (nrow(om) == nrow(tax_df)) mean_abd <- rowMeans(om, na.rm = TRUE)
  }
  mean_abd[is.na(mean_abd)] <- 0

  for (key in unique(lineage_base)) {
    idx <- which(lineage_base == key)
    if (length(idx) < 2L) next
    ord <- idx[order(mean_abd[idx], decreasing = TRUE)]
    for (k in seq_along(ord)) {
      i <- ord[[k]]
      base <- sub(" ASV[0-9]+$", "", as.character(tax_df[[deepest]][[i]]))
      tax_df[[deepest]][[i]] <- paste0(base, " ASV", k)
    }
  }

  # Unique lineages must NOT carry an ASV postfix (key on lineage_base, not tip string)
  for (key in unique(lineage_base)) {
    idx <- which(lineage_base == key)
    if (length(idx) == 1L) {
      i <- idx[[1]]
      tax_df[[deepest]][[i]] <- sub(" ASV[0-9]+$", "", as.character(tax_df[[deepest]][[i]]))
    }
  }

  tax_df
}

sanitize_tax_df_for_ape <- function(tax_df) {
  # Honey-style sanitize: plain data.frame of factors (avoids ape SET_STRING_ELT)
  tax_df <- as.data.frame(tax_df, stringsAsFactors = FALSE)
  as.data.frame(
    lapply(tax_df, function(x) {
      x <- normalize_unclassified_vec(x)
      x <- gsub("/", "_", x)
      factor(x, levels = unique(x))
    }),
    stringsAsFactors = FALSE
  )
}

#' Safe unique tip labels for ape::as.phylo (never make.names / make.unique dots).
unique_tip_labels_asv <- function(tips) {
  tips <- as.character(tips)
  out <- tips
  dup <- duplicated(out) | duplicated(out, fromLast = TRUE)
  if (!any(dup)) return(out)
  for (d in unique(out[dup])) {
    idx <- which(out == d)
    for (k in seq_along(idx)) {
      base <- sub(" ASV[0-9]+$", "", out[idx[[k]]])
      out[idx[[k]]] <- paste0(base, " ASV", k)
    }
  }
  out
}

#' Re-apply finalize_taxonomy when RDS still has placeholders / dotted make.unique tips.
ensure_finalized_taxonomy <- function(ps) {
  if (is.null(tax_table(ps, errorIfNULL = FALSE))) return(ps)
  tt <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
  deepest <- intersect(c("Species", "species", "Genus", "genus"), names(tt))
  deepest <- if (length(deepest)) deepest[[1]] else {
    rc <- intersect(c(RANK_COLS, RANK_COLS_LC), names(tt))
    if (length(rc)) rc[[length(rc)]] else NA_character_
  }
  needs <- !"tip_rank" %in% names(tt)
  if (!needs && !is.na(deepest)) {
    sp <- as.character(tt[[deepest]])
    needs <- any(grepl("\\.[0-9]+$", sp), na.rm = TRUE) ||
      any(is_placeholder_taxon(sp), na.rm = TRUE) ||
      any(grepl("^uncultured", sp, ignore.case = TRUE), na.rm = TRUE) ||
      # stale singleton ASV postfix (must re-finalize to strip)
      any(grepl(" ASV[0-9]+$", sp), na.rm = TRUE)
  }
  if (!needs) return(ps)
  message("Re-finalizing taxonomy (placeholders / dotted tip labels detected)")
  otu <- as(otu_table(ps), "matrix")
  if (!taxa_are_rows(ps)) otu <- t(otu)
  tax_f <- finalize_taxonomy_for_phyloseq(tt, otu_mat = otu)
  tax_mat <- as.matrix(tax_f)
  rownames(tax_mat) <- taxa_names(ps)
  tax_table(ps) <- tax_table(tax_mat)
  ps
}

build_formula_or_hclust_tree <- function(tax_df, tip_col = "taxa_id", abundance = NULL) {
  tax_df <- as.data.frame(tax_df, stringsAsFactors = FALSE)
  if (!tip_col %in% names(tax_df)) fail("build_formula_or_hclust_tree: missing tip column ", tip_col)
  tips <- as.character(tax_df[[tip_col]])
  tax_df$tip_label <- unique_tip_labels_asv(tips)
  # ape formula needs syntactically valid factor levels; map without changing display tips
  safe_lab <- paste0("t", seq_along(tax_df$tip_label))
  names(safe_lab) <- tax_df$tip_label
  tax_df$tip_label_safe <- unname(safe_lab)
  rank_cols <- setdiff(names(tax_df), c(tip_col, "tip_label", "tip_label_safe", "taxonomy_id", "tip_rank"))
  rank_cols <- intersect(c(RANK_COLS, RANK_COLS_LC, rank_cols), rank_cols)
  keep <- c(rank_cols, "tip_label_safe")
  phy_df <- sanitize_tax_df_for_ape(tax_df[, keep, drop = FALSE])
  form <- as.formula(paste("tip_label_safe ~", paste(c(rank_cols, "tip_label_safe"), collapse = " / ")))
  tr <- tryCatch(
    ape::as.phylo(form, data = phy_df, collapse = FALSE),
    error = function(e) NULL
  )
  if (!is.null(tr)) {
    # map safe labels back to original tip ids
    map <- setNames(tips, tax_df$tip_label_safe)
    tr$tip.label <- unname(ifelse(tr$tip.label %in% names(map), map[tr$tip.label], tr$tip.label))
    if (is.null(tr$edge.length)) tr <- ape::compute.brlen(tr, method = "Grafen")
    return(tr)
  }
  message("as.phylo formula failed; using hclust fallback tree")
  if (!is.null(abundance) && nrow(abundance) == length(tips)) {
    mat <- as.matrix(abundance)
    if (any(mat < 0)) mat <- abs(mat)
    mat <- log1p(mat)
    d <- stats::dist(mat, method = "euclidean")
    hc <- stats::hclust(d, method = "average")
    tr <- ape::as.phylo(hc)
    tr$tip.label <- tips
    return(tr)
  }
  hc <- stats::hclust(stats::dist(matrix(seq_along(tips), ncol = 1)), method = "average")
  tr <- ape::as.phylo(hc)
  tr$tip.label <- tips
  tr
}

fill_na_last_classified <- function(tax_df, ranks) {
  tax_df <- as.data.frame(tax_df, stringsAsFactors = FALSE)
  if (!length(ranks)) return(tax_df)
  mat <- as.matrix(tax_df[, ranks, drop = FALSE])
  storage.mode(mat) <- "character"
  mat[is.na(mat) | mat == ""] <- NA_character_
  last <- mat[, 1L]
  for (j in seq_len(ncol(mat))) {
    cur <- mat[, j]
    miss <- is.na(cur)
    if (any(miss)) cur[miss] <- last[miss]
    has <- !is.na(cur)
    if (any(has)) last[has] <- cur[has]
    mat[, j] <- cur
  }
  for (rk in ranks) tax_df[[rk]] <- mat[, rk]
  tax_df
}

RANK_COLS <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
RANK_COLS_LC <- c("kingdom", "phylum", "class", "order", "family", "genus", "species")

parse_qiime_taxon_string <- function(taxon) {
  # d__Bacteria;p__... or k__Bacteria;p__...
  parts <- strsplit(as.character(taxon), ";", fixed = TRUE)[[1]]
  out <- setNames(rep(NA_character_, length(RANK_COLS)), RANK_COLS)
  map <- c(
    d = "Kingdom", k = "Kingdom", p = "Phylum", c = "Class",
    o = "Order", f = "Family", g = "Genus", s = "Species"
  )
  for (part in parts) {
    part <- trimws(part)
    if (!nzchar(part)) next
    m <- regmatches(part, regexec("^([dkpcofgs])__(.*)$", part, perl = TRUE))[[1]]
    if (length(m) >= 3) {
      rk <- map[[m[[2]]]]
      val <- m[[3]]
      if (!is.null(rk) && nzchar(val)) out[[rk]] <- val
    }
  }
  out
}

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

discover_16s <- function(indir) {
  indir <- normalizePath(indir, mustWork = TRUE)
  files <- list.files(indir, recursive = TRUE, full.names = TRUE)
  bn <- basename(files)
  pick <- function(patterns) {
    hit <- files[Reduce(`|`, lapply(patterns, function(p) grepl(p, bn, ignore.case = TRUE)))]
    if (length(hit)) hit else character(0)
  }
  list(
    indir = indir,
    metadata = {
      m <- pick(c("^sample-metadata\\.(tsv|txt|csv)$", "^metadata.*\\.(tsv|csv)$"))
      # prefer sample-metadata.tsv
      pref <- m[grepl("sample-metadata", basename(m), ignore.case = TRUE)]
      if (length(pref)) pref[[1]] else if (length(m)) m[[1]] else NA_character_
    },
    features_qza = {
      m <- pick(c("^table\\.qza$", "^feature-table\\.qza$"))
      if (length(m)) m[[1]] else NA_character_
    },
    taxonomy_qza = {
      m <- pick(c("^taxonomy.*\\.qza$", "^taxonomy-silva\\.qza$"))
      # prefer silva if present
      silva <- m[grepl("silva", basename(m), ignore.case = TRUE)]
      if (length(silva)) silva[[1]] else if (length(m)) m[[1]] else NA_character_
    },
    tree_qza = {
      m <- pick(c("^rooted-tree\\.qza$", "^tree\\.qza$"))
      rooted <- m[grepl("rooted", basename(m), ignore.case = TRUE)]
      if (length(rooted)) rooted[[1]] else if (length(m)) m[[1]] else NA_character_
    },
    sequences_qza = {
      m <- pick(c("^rep-seqs\\.qza$", "^sequences\\.qza$", "rep-seq"))
      if (length(m)) m[[1]] else NA_character_
    },
    sequences_fasta = {
      m <- pick(c("\\.(fa|fasta|fna)(\\.gz)?$"))
      if (length(m)) m[[1]] else NA_character_
    },
    taxonomy_tsv = {
      m <- pick(c("^taxonomy\\.tsv$", "^taxonomy\\.txt$"))
      if (length(m)) m[[1]] else NA_character_
    },
    feature_table_tsv = {
      m <- pick(c("^feature-table\\.tsv$", "^otu.*\\.tsv$", "^asv.*\\.tsv$"))
      if (length(m)) m[[1]] else NA_character_
    },
    tree_nwk = {
      m <- pick(c("\\.(nwk|newick|tre|tree)$"))
      # exclude iqtree if multiple
      if (length(m)) m[[1]] else NA_character_
    }
  )
}

discover_wgs <- function(indir) {
  indir <- normalizePath(indir, mustWork = TRUE)
  files <- list.files(indir, recursive = TRUE, full.names = TRUE)
  bn <- basename(files)
  list(
    indir = indir,
    metadata = {
      m <- files[grepl("sample-metadata\\.(csv|tsv)$|^sra\\.csv$", bn, ignore.case = TRUE)]
      if (length(m)) m[[1]] else {
        m2 <- files[grepl("metadata|sra\\.csv|legends", files, ignore.case = TRUE) & grepl("\\.(csv|tsv)$", bn)]
        if (length(m2)) m2[[1]] else NA_character_
      }
    },
    bracken_genus = files[grepl("\\.nt\\.G\\.bracken$", bn, ignore.case = TRUE)],
    bracken_species_report = files[grepl("\\.nt\\.bracken\\.S\\.report$", bn, ignore.case = TRUE)],
    bracken_genus_report = files[grepl("\\.nt\\.bracken\\.G\\.report$", bn, ignore.case = TRUE)],
    sample_map = {
      m <- files[grepl("bracken_sample_map\\.csv$", bn, ignore.case = TRUE)]
      if (length(m)) m[[1]] else NA_character_
    }
  )
}

validate_16s_discovery <- function(d) {
  if (is.na(d$metadata) || !nzchar(d$metadata) || !file.exists(d$metadata)) {
    fail_missing_metadata("16S sample-metadata.tsv/csv not found")
  }
  issues <- character(0)
  has_seq <- (!is.na(d$sequences_qza) && nzchar(d$sequences_qza)) ||
    (!is.na(d$sequences_fasta) && nzchar(d$sequences_fasta))
  has_tax <- (!is.na(d$taxonomy_qza) && nzchar(d$taxonomy_qza)) ||
    (!is.na(d$taxonomy_tsv) && nzchar(d$taxonomy_tsv))
  if (!has_seq && !has_tax) {
    issues <- c(issues, "missing sequences OR taxonomy (need at least one)")
  }
  has_features <- (!is.na(d$features_qza) && nzchar(d$features_qza)) ||
    (!is.na(d$feature_table_tsv) && nzchar(d$feature_table_tsv))
  if (!has_features) {
    issues <- c(issues, "missing feature abundances (table.qza or feature-table.tsv) required for phyloseq")
  }
  if (length(issues)) fail(paste(issues, collapse = "; "))
  invisible(TRUE)
}

validate_wgs_discovery <- function(d) {
  if (is.na(d$metadata) || !nzchar(d$metadata) || !file.exists(d$metadata)) {
    fail_missing_metadata("WGS sample-metadata / sra.csv not found")
  }
  n_rep <- length(d$bracken_genus) + length(d$bracken_species_report) + length(d$bracken_genus_report)
  if (n_rep < 1) {
    fail("missing Bracken/Kraken taxonomy report (*.nt.G.bracken or *.bracken.*.report)")
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Phyloseq completeness (obligatory slots)
# ---------------------------------------------------------------------------

report_phyloseq_structure <- function(ps) {
  otu <- !is.null(phyloseq::otu_table(ps, errorIfNULL = FALSE))
  tax <- !is.null(phyloseq::tax_table(ps, errorIfNULL = FALSE))
  sam <- !is.null(phyloseq::sample_data(ps, errorIfNULL = FALSE))
  tre <- !is.null(phyloseq::phy_tree(ps, errorIfNULL = FALSE))
  structure <- list(
    tax_table = tax,
    otu_table = otu,
    sam_data = sam,
    tree_data = tre,
    n_taxa = if (otu) phyloseq::ntaxa(ps) else 0L,
    n_samples = if (sam) phyloseq::nsamples(ps) else 0L,
    n_tree_tips = if (tre) length(phyloseq::phy_tree(ps)$tip.label) else 0L,
    sample_names = if (sam) phyloseq::sample_names(ps) else character(0),
    complete = isTRUE(otu && tax && sam && tre)
  )
  message(
    "phyloseq structure: tax_table=", tax,
    " otu_table=", otu,
    " sam_data=", sam,
    " tree_data=", tre,
    " [", structure$n_taxa, " taxa × ", structure$n_samples, " samples]"
  )
  structure
}

assert_complete_phyloseq <- function(ps) {
  st <- report_phyloseq_structure(ps)
  missing <- c(
    if (!st$tax_table) "tax table",
    if (!st$otu_table) "otu table",
    if (!st$sam_data) "sam data",
    if (!st$tree_data) "tree data (novel or found)"
  )
  if (length(missing)) {
    fail(
      "Final phyloseq MUST have tax table, otu table, sam data, tree data. Missing: ",
      paste(missing, collapse = ", ")
    )
  }
  invisible(st)
}

# ---------------------------------------------------------------------------
# Tree attach / reconstruct via taxonomy-tree hook
# ---------------------------------------------------------------------------

call_taxonomy_tree_for_taxids <- function(taxids, outdir, email = Sys.getenv("NCBI_EMAIL", "import-hook@local.dev")) {
  root <- project_root()
  ensure_dir(outdir)
  taxid_path <- file.path(outdir, "import_taxids.tsv")
  utils::write.table(
    data.frame(taxon = paste0("tax_", taxids), taxid = as.integer(taxids), stringsAsFactors = FALSE),
    taxid_path, sep = "\t", row.names = FALSE, quote = FALSE
  )
  status <- system2(
    "Rscript",
    c(
      file.path(root, ".cursor/skills/taxonomy-tree/scripts/taxonomy_tree.R"),
      "--taxids", taxid_path,
      "--outdir", outdir,
      "--mode", "taxids",
      "--email", email
    ),
    stdout = TRUE,
    stderr = TRUE
  )
  nwk <- file.path(outdir, "tree_taxids.nwk")
  rds <- file.path(outdir, "tree_taxids.rds")
  if (!file.exists(nwk) && !file.exists(rds)) {
    message(paste(status, collapse = "\n"))
    fail("taxonomy-tree hook failed to produce a tree in ", outdir)
  }
  if (file.exists(rds)) {
    obj <- readRDS(rds)
    tr <- obj$tree
    # Remap scientific tip names → tax_<id> for phyloseq taxa_names
    if (!is.null(obj$lineage) && all(c("taxa_id", "tip_name") %in% names(obj$lineage))) {
      map <- setNames(as.character(obj$lineage$taxa_id), as.character(obj$lineage$tip_name))
      tr$tip.label <- unname(ifelse(tr$tip.label %in% names(map), map[tr$tip.label], tr$tip.label))
    }
    return(tr)
  }
  ape::read.tree(nwk)
}

attach_or_build_tree <- function(ps, tree_path = NA_character_, lineage_taxids = NULL, outdir,
                                 prefer_formula = TRUE, taxonomy_tree_max = 40L) {
  # 1) Existing tree file
  if (!is.na(tree_path) && nzchar(tree_path) && file.exists(tree_path) &&
      !grepl("\\.qza$", tree_path, ignore.case = TRUE)) {
    tr <- if (grepl("\\.rds$", tree_path, ignore.case = TRUE)) {
      obj <- readRDS(tree_path); if (!is.null(obj$tree)) obj$tree else obj
    } else {
      ape::read.tree(tree_path)
    }
    common <- intersect(tr$tip.label, phyloseq::taxa_names(ps))
    if (length(common) >= 2) {
      tr <- ape::keep.tip(tr, common)
      ps <- phyloseq::prune_taxa(common, ps)
      if (is.null(tr$edge.length)) tr <- ape::compute.brlen(tr, method = "Grafen")
      phyloseq::phy_tree(ps) <- tr
      return(ps)
    }
    message("Existing tree tips do not match taxa_names; rebuilding")
  }

  # 2) taxonomy-tree hook for small tip sets (NCBI lineage)
  taxids <- lineage_taxids[!is.na(lineage_taxids)]
  if (length(taxids) >= 2L && length(taxids) <= as.integer(taxonomy_tree_max)) {
    message("Tree missing — calling taxonomy-tree hook for ", length(taxids), " taxids…")
    tr <- tryCatch(
      call_taxonomy_tree_for_taxids(taxids, outdir = file.path(outdir, "taxonomy-tree")),
      error = function(e) {
        message("taxonomy-tree failed: ", conditionMessage(e))
        NULL
      }
    )
    if (!is.null(tr)) {
      # tips may be tax_* or scientific names — try both
      taxa <- phyloseq::taxa_names(ps)
      tip <- tr$tip.label
      common <- intersect(tip, taxa)
      if (length(common) < 2) {
        tip2 <- paste0("tax_", gsub("^tax_", "", tip))
        common <- intersect(tip2, taxa)
        if (length(common) >= 2) tr$tip.label <- tip2
      }
      if (length(common) >= 2) {
        tr <- ape::keep.tip(tr, common)
        ps <- phyloseq::prune_taxa(common, ps)
        if (is.null(tr$edge.length)) tr <- ape::compute.brlen(tr, method = "Grafen")
        phyloseq::phy_tree(ps) <- tr
        message("Attached taxonomy-tree (", length(tr$tip.label), " tips)")
        return(ps)
      }
      message("taxonomy-tree tips did not match taxa_names; falling back")
    }
  }

  # 3) Formula taxonomy tree / hclust fallback (offline; large tables)
  if (prefer_formula) {
    tt <- as.data.frame(phyloseq::tax_table(ps), stringsAsFactors = FALSE)
    tt$taxa_id <- phyloseq::taxa_names(ps)
    ranks <- intersect(c(RANK_COLS, RANK_COLS_LC), names(tt))
    if (length(ranks) >= 2) {
      tt2 <- fill_na_last_classified(tt, ranks)
      tt2$taxa_id <- tt$taxa_id
      otu <- as(phyloseq::otu_table(ps), "matrix")
      if (!phyloseq::taxa_are_rows(ps)) otu <- t(otu)
      tr2 <- build_formula_or_hclust_tree(tt2[, c(ranks, "taxa_id"), drop = FALSE],
                                         tip_col = "taxa_id", abundance = otu)
      if (!is.null(tr2) && length(tr2$tip.label) >= 2) {
        common <- intersect(tr2$tip.label, phyloseq::taxa_names(ps))
        if (length(common) >= 2) {
          tr2 <- ape::keep.tip(tr2, common)
          ps <- phyloseq::prune_taxa(common, ps)
          if (is.null(tr2$edge.length)) tr2 <- ape::compute.brlen(tr2, method = "Grafen")
          phyloseq::phy_tree(ps) <- tr2
          message("Attached taxonomy/hclust tree (", length(tr2$tip.label), " tips)")
          return(ps)
        }
      }
    }
  }

  message("Could not attach tree; returning phyloseq without phy_tree")
  ps
}

require_tree <- function(ps, tree_path = NA_character_, lineage_taxids = NULL, outdir) {
  ps <- attach_or_build_tree(ps, tree_path = tree_path, lineage_taxids = lineage_taxids, outdir = outdir)
  if (is.null(phyloseq::phy_tree(ps, errorIfNULL = FALSE))) {
    fail("Tree reconstruction failed — final phyloseq MUST have tree data (novel or found)")
  }
  ps
}
