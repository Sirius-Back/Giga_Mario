#!/usr/bin/env Rscript
# fix-metadata skill — find metadata + data; align sample IDs; write metadata_fixed.csv
suppressPackageStartupMessages({
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
})

root <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) {
    script <- normalizePath(sub("^--file=", "", f[[1]]), mustWork = FALSE)
    # .cursor/skills/<skill>/scripts/*.R → five dirname levels to project root
    return(dirname(dirname(dirname(dirname(dirname(script))))))
  }
  getwd()
})
source(file.path(root, ".cursor/skills/_shared/import/import_common.R"))

guess_data_sample_ids <- function(indir) {
  indir <- normalizePath(indir, mustWork = TRUE)
  files <- list.files(indir, recursive = TRUE, full.names = TRUE)
  bn <- basename(files)
  ids <- character(0)
  types <- list()

  # Bracken genus tables
  g <- files[grepl("\\.nt\\.G\\.bracken$", bn, ignore.case = TRUE)]
  if (length(g)) {
    ids_g <- sub("\\.nt\\.G\\.bracken$", "", bn[grepl("\\.nt\\.G\\.bracken$", bn, ignore.case = TRUE)], ignore.case = TRUE)
    ids <- c(ids, ids_g)
    types$bracken_genus <- list(n = length(g), sample_ids = ids_g, files = g)
  }
  # Bracken species reports
  s <- files[grepl("\\.nt\\.bracken\\.S\\.report$", bn, ignore.case = TRUE)]
  if (length(s)) {
    ids_s <- sub("\\.nt\\.bracken\\.S\\.report$", "", basename(s), ignore.case = TRUE)
    ids <- c(ids, ids_s)
    types$bracken_species_report <- list(n = length(s), sample_ids = ids_s, files = s)
  }
  # Bracken genus reports
  gr <- files[grepl("\\.nt\\.bracken\\.G\\.report$", bn, ignore.case = TRUE)]
  if (length(gr)) {
    ids_gr <- sub("\\.nt\\.bracken\\.G\\.report$", "", basename(gr), ignore.case = TRUE)
    ids <- c(ids, ids_gr)
    types$bracken_genus_report <- list(n = length(gr), sample_ids = ids_gr, files = gr)
  }
  # Feature table TSV columns
  ft <- files[grepl("^feature-table\\.tsv$", bn, ignore.case = TRUE)]
  if (length(ft)) {
    dt <- utils::read.delim(ft[[1]], check.names = FALSE, nrows = 1, stringsAsFactors = FALSE)
    ids_ft <- setdiff(names(dt), names(dt)[1])
    ids <- c(ids, ids_ft)
    types$feature_table_tsv <- list(n = 1L, sample_ids = ids_ft, files = ft[[1]])
  }
  # QIIME metadata already co-located — samples from sample-metadata if present for inventory only
  list(sample_ids = unique(as.character(ids)), types = types, files = files)
}

find_metadata_candidates <- function(indir, search_parents = TRUE) {
  indir <- normalizePath(indir, mustWork = TRUE)
  roots <- indir
  if (search_parents) {
    cur <- indir
    proj <- tryCatch(project_root(), error = function(e) indir)
    for (i in seq_len(6)) {
      parent <- dirname(cur)
      if (identical(parent, cur)) break
      roots <- c(roots, parent)
      cur <- parent
      # never ascend above project root
      if (normalizePath(cur, mustWork = FALSE) == normalizePath(proj, mustWork = FALSE)) break
    }
  }
  cands <- character(0)
  for (r in unique(roots)) {
    # non-recursive in parents; recursive only in indir itself
    recursive <- identical(normalizePath(r), normalizePath(indir))
    files <- list.files(r, recursive = recursive, full.names = TRUE)
    bn <- basename(files)
    hit <- files[grepl(
      "sample-metadata\\.(tsv|csv|txt)$|^metadata.*\\.(tsv|csv)$|^sra\\.csv$|bracken_sample_map\\.csv$",
      bn,
      ignore.case = TRUE
    )]
    hit2 <- files[grepl("legends", dirname(files), ignore.case = TRUE) & grepl("\\.(csv|tsv)$", bn)]
    cands <- c(cands, hit, hit2)
  }
  unique(cands)
}

try_align <- function(data_ids, meta_df, id_col) {
  meta_ids <- as.character(meta_df[[id_col]])
  direct <- intersect(data_ids, meta_ids)
  if (length(direct) > 0) {
    return(list(ok = TRUE, method = "direct", id_col = id_col,
                overlap = length(direct), mapped = data.frame(data_id = direct, sampleID = direct, stringsAsFactors = FALSE)))
  }
  # strip common suffixes from data ids
  stripped <- sub("\\.(fastq|fq|bam|sam).*", "", data_ids, ignore.case = TRUE)
  stripped <- sub("_S[0-9]+_L[0-9]+.*", "", stripped)
  hit <- intersect(stripped, meta_ids)
  if (length(hit) > 0) {
    map <- data.frame(data_id = data_ids[match(hit, stripped)], sampleID = hit, stringsAsFactors = FALSE)
    return(list(ok = TRUE, method = "strip_suffix", id_col = id_col, overlap = length(hit), mapped = map))
  }
  # case-insensitive
  low_d <- tolower(data_ids)
  low_m <- tolower(meta_ids)
  idx <- match(low_d, low_m)
  if (any(!is.na(idx))) {
    keep <- !is.na(idx)
    map <- data.frame(data_id = data_ids[keep], sampleID = meta_ids[idx[keep]], stringsAsFactors = FALSE)
    return(list(ok = TRUE, method = "case_insensitive", id_col = id_col, overlap = nrow(map), mapped = map))
  }
  list(ok = FALSE, method = "none", id_col = id_col, overlap = 0L, mapped = NULL)
}

run_fix_metadata <- function(indir, outdir, metadata_hint = NULL) {
  ensure_dir(outdir)
  message("fix-metadata: find metadata; find all data types; check alignment")
  data_info <- guess_data_sample_ids(indir)
  cands <- find_metadata_candidates(indir, search_parents = TRUE)
  if (!is.null(metadata_hint) && nzchar(metadata_hint) && file.exists(metadata_hint)) {
    cands <- unique(c(metadata_hint, cands))
  }

  inventory <- list(
    indir = indir,
    outdir = outdir,
    data_sample_ids = data_info$sample_ids,
    n_data_samples = length(data_info$sample_ids),
    data_types = lapply(data_info$types, function(x) list(n = x$n, n_ids = length(x$sample_ids))),
    metadata_candidates = cands
  )
  write_json(inventory, file.path(outdir, "data-inventory.json"))

  if (!length(data_info$sample_ids)) {
    fail("No data sample IDs discovered under ", indir)
  }
  if (!length(cands)) {
    fail("No metadata candidates found. Provide --metadata path.")
  }

  best <- NULL
  attempts <- list()
  for (cand in cands) {
    ext <- tolower(tools::file_ext(cand))
    df <- tryCatch(
      if (ext == "csv") utils::read.csv(cand, stringsAsFactors = FALSE, check.names = FALSE)
      else utils::read.delim(cand, stringsAsFactors = FALSE, check.names = FALSE),
      error = function(e) NULL
    )
    if (is.null(df) || !ncol(df)) next
    id_cols <- intersect(
      c("sampleID", "SampleID", "sample-id", "sample_id", "Run", "sample", "bracken_id", "fastq_id"),
      names(df)
    )
    if (!length(id_cols)) {
      # first column as fallback
      id_cols <- names(df)[1]
    }
    for (ic in id_cols) {
      al <- try_align(data_info$sample_ids, df, ic)
      attempts[[length(attempts) + 1L]] <- list(
        metadata = cand, id_col = ic, method = al$method, overlap = al$overlap
      )
      if (isTRUE(al$ok) && (is.null(best) || al$overlap > best$overlap)) {
        best <- c(al, list(metadata = cand, df = df))
      }
    }
  }

  if (is.null(best) || best$overlap < 1) {
    write_json(list(inventory = inventory, attempts = attempts, aligned = FALSE),
               file.path(outdir, "fix-metadata-report.json"))
    fail(
      "Could not align metadata with data sample IDs. ",
      "Data IDs (head): ", paste(head(data_info$sample_ids, 8), collapse = ","),
      ". Candidates tried: ", length(cands),
      ". Investigate metadata manually (subagent) and re-run."
    )
  }

  # Build metadata_fixed.csv: one row per overlapping data sample
  df <- best$df
  map <- best$mapped
  # join: keep metadata rows matching sampleID in map
  meta_ids <- as.character(df[[best$id_col]])
  rows <- lapply(seq_len(nrow(map)), function(i) {
    row <- df[match(map$sampleID[[i]], meta_ids), , drop = FALSE]
    row$sampleID <- map$data_id[[i]]
    row$sampleID_original <- map$sampleID[[i]]
    row$alignment_method <- best$method
    row
  })
  fixed <- do.call(rbind, rows)
  # prefer sampleID first
  other <- setdiff(names(fixed), c("sampleID", "sampleID_original", "alignment_method"))
  fixed <- fixed[, c("sampleID", "sampleID_original", "alignment_method", other), drop = FALSE]

  fixed_path <- file.path(outdir, "metadata_fixed.csv")
  utils::write.csv(fixed, fixed_path, row.names = FALSE)

  # If source was already aligned directly and co-located, still write fixed copy for reproducibility
  report <- list(
    aligned = TRUE,
    method = best$method,
    metadata_source = best$metadata,
    id_col = best$id_col,
    overlap = best$overlap,
    n_data_samples = length(data_info$sample_ids),
    coverage = best$overlap / max(1, length(data_info$sample_ids)),
    metadata_fixed = fixed_path,
    data_types = names(data_info$types),
    attempts = attempts,
    note = if (best$method == "direct" && best$overlap == length(data_info$sample_ids)) {
      "Metadata already aligned; metadata_fixed.csv is a normalized copy with sampleID."
    } else {
      "Metadata was not aligned with data directly — copied & edited to metadata_fixed.csv"
    }
  )
  write_json(report, file.path(outdir, "fix-metadata-report.json"))
  message(report$note)
  message("Wrote ", fixed_path, " (", nrow(fixed), " rows; method=", best$method, ")")
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  system2("python3", c(".cursor/skills/mock-data/scripts/mock_data.py", "--out", "test", "--target", "wgs", "--self-test"))
  # mock already aligned
  r1 <- run_fix_metadata("test/wgs", "test/fix-metadata/mock")
  if (!isTRUE(r1$aligned)) stop("mock should align")
  if (!file.exists(r1$metadata_fixed)) stop("missing metadata_fixed.csv")

  # misaligned: rename metadata sampleIDs
  tmp <- "test/fix-metadata/misaligned"
  ensure_dir(tmp)
  file.copy(list.files("test/wgs", full.names = TRUE), tmp, overwrite = TRUE)
  md <- utils::read.csv(file.path(tmp, "sample-metadata.csv"), stringsAsFactors = FALSE)
  md$sampleID <- paste0("BAD_", md$sampleID)
  md$Run <- md$sampleID
  utils::write.csv(md, file.path(tmp, "sample-metadata.csv"), row.names = FALSE)
  # also write a recoverable map via Run-like column with original ids in another file
  # Create secondary metadata with Run = original bracken basenames
  good <- data.frame(
    Run = c("mockwgs_S1", "mockwgs_S2", "mockwgs_S3"),
    group = c("A", "B", "A"),
    stringsAsFactors = FALSE
  )
  utils::write.csv(good, file.path(tmp, "legends_sra.csv"), row.names = FALSE)
  # put legends file where finder looks — rename to sra.csv in subdir
  ensure_dir(file.path(tmp, "legends"))
  file.copy(file.path(tmp, "legends_sra.csv"), file.path(tmp, "legends", "sra.csv"), overwrite = TRUE)

  r2 <- run_fix_metadata(tmp, file.path(tmp, "out"))
  if (!isTRUE(r2$aligned) || r2$overlap < 3) stop("misaligned recovery failed")
  fixed <- utils::read.csv(r2$metadata_fixed, stringsAsFactors = FALSE)
  if (!all(c("mockwgs_S1", "mockwgs_S2", "mockwgs_S3") %in% fixed$sampleID)) {
    stop("fixed sampleIDs incorrect")
  }
  message("misaligned → metadata_fixed.csv OK")

  # real honey: k2 + parent legends
  honey_data <- "/mnt/tank/scratch/dsmutin/bee/honey/data"
  if (dir.exists(honey_data)) {
    # limit cost: use subset indir already staged if present, else honey data root with hint
    honey_sub <- "test/metagenomic-import/honey-subset"
    if (dir.exists(honey_sub)) {
      r3 <- run_fix_metadata(honey_sub, "test/fix-metadata/honey-subset")
      if (!isTRUE(r3$aligned)) stop("honey subset align failed")
      message("honey subset fix-metadata OK: overlap=", r3$overlap)
    }
  }
  message("SELF-TEST OK")
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test(); return(invisible(0))
  }
  indir <- args$indir %||% args$input %||% "test/wgs"
  outdir <- args$outdir %||% "test/fix-metadata/run"
  run_fix_metadata(indir, outdir, metadata_hint = args$metadata)
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
