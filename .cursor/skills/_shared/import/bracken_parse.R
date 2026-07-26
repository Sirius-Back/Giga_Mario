#!/usr/bin/env Rscript
# bracken-parse hook — fast Bracken / Kraken-report parsing → wide count matrices
suppressPackageStartupMessages({
  stopifnot(requireNamespace("data.table", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(data.table)
})

root <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) {
    script <- normalizePath(sub("^--file=", "", f[[1]]), mustWork = FALSE)
    # .cursor/skills/_shared/import/*.R → five dirname levels to project root
    return(dirname(dirname(dirname(dirname(dirname(script))))))
  }
  getwd()
})
source(file.path(root, ".cursor/skills/_shared/import/import_common.R"))
root <- project_root()


parse_bracken_genus_table <- function(path) {
  dt <- data.table::fread(path, sep = "\t", header = TRUE, showProgress = FALSE)
  need <- c("taxonomy_id", "name", "new_est_reads")
  if (!all(need %in% names(dt))) {
    fail("Bracken genus table missing columns ", paste(need, collapse = ","), ": ", path)
  }
  dt <- dt[, .(taxonomy_id = as.integer(taxonomy_id), name = as.character(name),
               new_est_reads = as.numeric(new_est_reads))]
  dt <- dt[!is.na(taxonomy_id) & taxonomy_id > 0]
  sample <- sub("\\.nt\\.G\\.bracken$", "", basename(path), ignore.case = TRUE)
  dt[, sample := sample]
  dt
}

parse_bracken_kraken_report <- function(path, keep_ranks = c("S", "G")) {
  dt <- data.table::fread(
    path, sep = "\t", header = FALSE, showProgress = FALSE,
    col.names = c("pct", "reads_clade", "reads_direct", "rank", "taxonomy_id", "name"),
    fill = TRUE, quote = ""
  )
  if (ncol(dt) < 6) fail("Kraken/Bracken report needs 6 columns: ", path)
  dt[, name := trimws(name)]
  dt[, taxonomy_id := as.integer(taxonomy_id)]
  dt <- dt[rank %in% keep_ranks & !is.na(taxonomy_id) & taxonomy_id > 0]
  dt[, new_est_reads := fifelse(reads_direct > 0, as.numeric(reads_direct), as.numeric(reads_clade))]
  sample <- basename(path)
  sample <- sub("\\.nt\\.bracken\\.[SG]\\.report$", "", sample, ignore.case = TRUE)
  sample <- sub("\\.nt\\.k2\\.report$", "", sample, ignore.case = TRUE)
  sample <- sub("\\.report$", "", sample, ignore.case = TRUE)
  dt[, sample := sample]
  dt[, .(sample, taxonomy_id, name, rank, new_est_reads)]
}

detect_and_parse_one <- function(path) {
  bn <- basename(path)
  if (grepl("\\.nt\\.G\\.bracken$", bn, ignore.case = TRUE)) {
    x <- parse_bracken_genus_table(path)
    x[, rank := "G"]
    return(x[, .(sample, taxonomy_id, name, rank, new_est_reads)])
  }
  if (grepl("\\.nt\\.bracken\\.S\\.report$", bn, ignore.case = TRUE)) {
    return(parse_bracken_kraken_report(path, keep_ranks = "S"))
  }
  if (grepl("\\.nt\\.bracken\\.G\\.report$", bn, ignore.case = TRUE)) {
    return(parse_bracken_kraken_report(path, keep_ranks = "G"))
  }
  # generic report: keep S then G
  if (grepl("\\.report$", bn, ignore.case = TRUE)) {
    return(parse_bracken_kraken_report(path, keep_ranks = c("S", "G")))
  }
  fail("Unrecognized Bracken/report format: ", path)
}

build_wide_matrix <- function(long_dt) {
  meta <- unique(long_dt[, .(taxonomy_id, name, rank)])
  # Prefer species over genus when both present for same taxid
  setorder(meta, taxonomy_id, -rank)
  meta <- meta[, .SD[1], by = taxonomy_id]
  mat <- data.table::dcast(
    long_dt,
    taxonomy_id ~ sample,
    value.var = "new_est_reads",
    fun.aggregate = sum,
    fill = 0
  )
  tax_ids <- mat$taxonomy_id
  mat[, taxonomy_id := NULL]
  m <- as.matrix(mat)
  rownames(m) <- paste0("tax_", tax_ids)
  list(
    counts = m,
    taxonomy_id = tax_ids,
    name = meta$name[match(tax_ids, meta$taxonomy_id)],
    rank = meta$rank[match(tax_ids, meta$taxonomy_id)],
    samples = colnames(m)
  )
}

cleanup_host_rows <- function(parsed) {
  drop <- is_host_taxon_name(parsed$name)
  # also drop explicit host taxids
  drop <- drop | parsed$taxonomy_id %in% c(9606L, 9605L, 33208L)
  if (any(drop)) {
    message("Cleanup: dropping ", sum(drop), " host/Chordata-like taxa")
    keep <- !drop
    parsed$counts <- parsed$counts[keep, , drop = FALSE]
    parsed$taxonomy_id <- parsed$taxonomy_id[keep]
    parsed$name <- parsed$name[keep]
    parsed$rank <- parsed$rank[keep]
  }
  parsed
}

run_bracken_parse <- function(indir, outdir, pattern = NULL) {
  ensure_dir(outdir)
  d <- discover_wgs(indir)
  files <- c(d$bracken_genus, d$bracken_species_report, d$bracken_genus_report)
  if (!is.null(pattern) && nzchar(pattern)) {
    files <- files[grepl(pattern, basename(files))]
  }
  if (!length(files)) fail("No Bracken files found under ", indir)
  message("Parsing ", length(files), " Bracken/report files…")
  long_list <- lapply(files, detect_and_parse_one)
  long_dt <- data.table::rbindlist(long_list, use.names = TRUE, fill = TRUE)
  parsed <- build_wide_matrix(long_dt)
  parsed <- cleanup_host_rows(parsed)

  long_path <- file.path(outdir, "bracken_long.tsv")
  wide_path <- file.path(outdir, "bracken_counts.tsv")
  meta_path <- file.path(outdir, "bracken_taxa.tsv")
  rds_path <- file.path(outdir, "bracken_parsed.rds")

  data.table::fwrite(long_dt, long_path, sep = "\t")
  wide <- data.table::as.data.table(parsed$counts, keep.rownames = "taxa_id")
  data.table::fwrite(wide, wide_path, sep = "\t")
  data.table::fwrite(
    data.table::data.table(
      taxa_id = rownames(parsed$counts),
      taxonomy_id = parsed$taxonomy_id,
      name = parsed$name,
      rank = parsed$rank
    ),
    meta_path, sep = "\t"
  )
  saveRDS(parsed, rds_path)

  report <- list(
    n_files = length(files),
    n_taxa = nrow(parsed$counts),
    n_samples = ncol(parsed$counts),
    samples = parsed$samples,
    files = files,
    outputs = list(long = long_path, counts = wide_path, taxa = meta_path, rds = rds_path)
  )
  write_json(report, file.path(outdir, "bracken-parse-report.json"))
  message("Wrote ", rds_path, " (", report$n_taxa, " taxa × ", report$n_samples, " samples)")
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  system2("python3", c(".cursor/skills/mock-data/scripts/mock_data.py", "--out", "test", "--target", "wgs", "--self-test"))
  out <- "test/metagenomic-import/bracken-parse-mock"
  rep <- run_bracken_parse("test/wgs", out)
  if (rep$n_samples < 1 || rep$n_taxa < 1) stop("mock parse empty")
  # real honey subset
  honey_k2 <- "/mnt/tank/scratch/dsmutin/bee/honey/data/annotations/k2"
  if (dir.exists(honey_k2)) {
    # limit to 2 known files for speed
    out2 <- "test/metagenomic-import/bracken-parse-honey"
    ensure_dir(out2)
    files <- file.path(honey_k2, c("ERR2592241.nt.G.bracken", "ERR2592240.nt.G.bracken"))
    files <- files[file.exists(files)]
    if (length(files)) {
      long_list <- lapply(files, detect_and_parse_one)
      long_dt <- data.table::rbindlist(long_list, use.names = TRUE, fill = TRUE)
      parsed <- cleanup_host_rows(build_wide_matrix(long_dt))
      saveRDS(parsed, file.path(out2, "bracken_parsed.rds"))
      if (ncol(parsed$counts) < 1) stop("honey parse empty")
      message("honey subset OK: ", nrow(parsed$counts), " taxa × ", ncol(parsed$counts), " samples")
    }
  }
  message("SELF-TEST OK")
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  indir <- args$indir %||% args$input %||% "test/wgs"
  outdir <- args$outdir %||% "test/metagenomic-import/bracken-parse"
  run_bracken_parse(indir, outdir, pattern = args$pattern)
}

if (sys.nframe() == 0L) {
  tryCatch({ main(); quit(save = "no", status = 0) },
           error = function(e) { message("ERROR: ", conditionMessage(e)); quit(save = "no", status = 1) })
}
