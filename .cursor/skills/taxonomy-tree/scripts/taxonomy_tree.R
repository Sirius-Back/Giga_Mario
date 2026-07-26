#!/usr/bin/env Rscript
# taxonomy-tree hook
# Rebuild a taxonomy tree from taxon names and/or NCBI taxids (rentrez → lineage → ape),
# matching codebase sanitize rules: fill NA with last classified rank; as.character → factor
# before ape::as.phylo. Emit lineage tables, Newick, and ggtree figures for review.

suppressPackageStartupMessages({
  stopifnot(requireNamespace("rentrez", quietly = TRUE))
  stopifnot(requireNamespace("ape", quietly = TRUE))
  stopifnot(requireNamespace("jsonlite", quietly = TRUE))
  library(rentrez)
  library(ape)
})

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !all(is.na(a))) a else b

# Canonical ranks used in ape formula (codebase style)
RANK_COLS <- c("kingdom", "phylum", "class", "order", "family", "genus", "species")

# NCBI rank aliases → RANK_COLS
RANK_MAP <- c(
  "superkingdom" = "kingdom",
  "domain" = "kingdom",
  "kingdom" = "kingdom",
  "phylum" = "phylum",
  "class" = "class",
  "order" = "order",
  "family" = "family",
  "genus" = "genus",
  "species" = "species",
  "species group" = "species",
  "subspecies" = "species"
)

parse_args <- function(argv = commandArgs(trailingOnly = TRUE)) {
  out <- list(
    taxons = NULL,
    taxids = NULL,
    outdir = "test/taxonomy-tree",
    email = Sys.getenv("NCBI_EMAIL", unset = "taxonomy-tree-hook@local.dev"),
    mode = "both", # names | taxids | both
    self_test = FALSE
  )
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--self-test") {
      out$self_test <- TRUE
      i <- i + 1L
      next
    }
    if (i == length(argv)) stop("Missing value for ", key)
    val <- argv[[i + 1L]]
    if (key == "--taxons") out$taxons <- val
    else if (key == "--taxids") out$taxids <- val
    else if (key == "--outdir") out$outdir <- val
    else if (key == "--email") out$email <- val
    else if (key == "--mode") out$mode <- val
    else stop("Unknown argument: ", key)
    i <- i + 2L
  }
  out
}

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
        dir.exists(file.path(cur, ".cursor", "skills"))) {
      return(cur)
    }
    parent <- dirname(cur)
    if (identical(parent, cur)) break
    cur <- parent
  }
  getwd()
}

mock_data_script <- function() {
  file.path(project_root(), ".cursor/skills/mock-data/scripts/mock_data.py")
}


ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

# ---------------------------------------------------------------------------
# Input loaders (mock-compatible)
# ---------------------------------------------------------------------------

load_taxon_names <- function(path) {
  if (is.null(path) || !file.exists(path)) return(character(0))
  if (grepl("\\.json$", path, ignore.case = TRUE)) {
    x <- jsonlite::fromJSON(path)
    return(as.character(unlist(x, use.names = FALSE)))
  }
  # plain text / tsv first column
  lines <- readLines(path, warn = FALSE)
  lines <- trimws(lines)
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  # drop header-like
  if (length(lines) && tolower(lines[[1]]) %in% c("taxon", "taxons", "name", "names")) {
    lines <- lines[-1]
  }
  # if TSV, take first column
  vapply(strsplit(lines, "\t", fixed = TRUE), `[[`, character(1), 1L)
}

load_taxids <- function(path) {
  if (is.null(path) || !file.exists(path)) {
    return(data.frame(taxon = character(), taxid = integer(), stringsAsFactors = FALSE))
  }
  if (grepl("\\.json$", path, ignore.case = TRUE)) {
    x <- jsonlite::fromJSON(path)
    if (is.list(x) && !is.null(x$taxid)) {
      return(data.frame(
        taxon = as.character(x$taxon %||% x$name %||% NA_character_),
        taxid = as.integer(x$taxid),
        stringsAsFactors = FALSE
      ))
    }
    # bare vector of ids
    return(data.frame(
      taxon = NA_character_,
      taxid = as.integer(unlist(x)),
      stringsAsFactors = FALSE
    ))
  }
  df <- utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  # flexible columns
  taxid_col <- intersect(c("taxid", "taxonomy_id", "TaxID", "id"), names(df))
  taxon_col <- intersect(c("taxon", "name", "scientific_name", "Taxon"), names(df))
  if (!length(taxid_col)) stop("taxids file lacks taxid column: ", path)
  out <- data.frame(
    taxon = if (length(taxon_col)) as.character(df[[taxon_col[[1]]]]) else NA_character_,
    taxid = suppressWarnings(as.integer(df[[taxid_col[[1]]]])),
    stringsAsFactors = FALSE
  )
  # drop empty / unresolved
  out <- out[!is.na(out$taxid) & out$taxid > 0L, , drop = FALSE]
  out
}

# ---------------------------------------------------------------------------
# rentrez: resolve names, fetch records + parents
# ---------------------------------------------------------------------------

resolve_name_to_taxid <- function(name, pause = 0.34) {
  name <- trimws(name)
  if (!nzchar(name)) return(NA_integer_)
  # "Genus sp." / "Genus spp." → resolve genus
  genus_sp <- grepl("\\ssp+\\.?$", name, ignore.case = TRUE)
  if (genus_sp) {
    genus <- sub("\\ssp+\\.?$", "", name, ignore.case = TRUE)
    genus <- trimws(genus)
    if (nzchar(genus)) {
      tid <- resolve_name_to_taxid(genus, pause = pause)
      if (!is.na(tid)) return(tid)
    }
  }
  # Prefer scientific name query; fall back to free text
  queries <- c(
    sprintf("%s[Scientific Name]", name),
    sprintf("%s[Organism]", name),
    name
  )
  for (q in queries) {
    Sys.sleep(pause)
    res <- tryCatch(
      entrez_search(db = "taxonomy", term = q, retmax = 5),
      error = function(e) NULL
    )
    if (is.null(res) || !length(res$ids)) next
    ids <- as.integer(res$ids)
    if (length(ids) == 1L) return(ids[[1]])
    # disambiguate: fetch summaries and prefer exact scientific name match
    Sys.sleep(pause)
    sm <- tryCatch(
      entrez_summary(db = "taxonomy", id = ids),
      error = function(e) NULL
    )
    if (is.null(sm)) return(ids[[1]])
    sm_list <- if (inherits(sm, "esummary_list")) sm else list(sm)
    exact <- vapply(sm_list, function(s) {
      sn <- s$scientificname %||% s$scientific_name %||% ""
      tolower(sn) == tolower(name)
    }, logical(1))
    if (any(exact)) return(as.integer(sm_list[[which(exact)[1]]]$uid %||% ids[which(exact)[1]]))
    # prefer genus-rank hit when query looks like a genus
    ranks <- vapply(sm_list, function(s) tolower(as.character(s$rank %||% "")), character(1))
    if ("genus" %in% ranks) return(as.integer(sm_list[[which(ranks == "genus")[1]]]$uid %||% ids[which(ranks == "genus")[1]]))
    return(ids[[1]])
  }
  NA_integer_
}

# Minimal XML extractors (avoid hard xml2 dependency)
xml_first <- function(xml, tag) {
  m <- regmatches(xml, regexpr(sprintf("<%s>([^<]+)</%s>", tag, tag), xml, perl = TRUE))
  if (!length(m) || !nzchar(m)) return(NA_character_)
  sub(sprintf("<%s>([^<]+)</%s>", tag, tag), "\\1", m, perl = TRUE)
}

parse_lineage_ex <- function(xml) {
  # Extract Taxon blocks inside LineageEx
  block <- sub("(?s).*?<LineageEx>(.*?)</LineageEx>.*", "\\1", xml, perl = TRUE)
  if (identical(block, xml)) return(data.frame(taxid = integer(), name = character(), rank = character()))
  parts <- strsplit(block, "<Taxon>", fixed = TRUE)[[1]]
  parts <- parts[nzchar(trimws(parts))]
  rows <- lapply(parts, function(p) {
    data.frame(
      taxid = suppressWarnings(as.integer(xml_first(p, "TaxId"))),
      name = xml_first(p, "ScientificName"),
      rank = xml_first(p, "Rank"),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out[!is.na(out$taxid), , drop = FALSE]
}

fetch_taxon_xml <- function(taxid, pause = 0.34) {
  Sys.sleep(pause)
  tryCatch(
    entrez_fetch(db = "taxonomy", id = as.character(taxid), rettype = "xml"),
    error = function(e) NA_character_
  )
}

parse_taxon_record <- function(taxid, xml) {
  if (is.na(xml) || !nzchar(xml)) {
    return(list(
      taxid = as.integer(taxid),
      name = NA_character_,
      rank = NA_character_,
      parent = NA_integer_,
      lineage_ex = data.frame(taxid = integer(), name = character(), rank = character())
    ))
  }
  list(
    taxid = as.integer(taxid),
    name = xml_first(xml, "ScientificName"),
    rank = xml_first(xml, "Rank"),
    parent = suppressWarnings(as.integer(xml_first(xml, "ParentTaxId"))),
    lineage_ex = parse_lineage_ex(xml)
  )
}

#' Fetch tip records and walk all unique parents via ParentTaxId / LineageEx (rentrez).
#' Tip XMLs are fetched individually; remaining parent IDs are fetched in batches.
fetch_records_and_parents <- function(tip_taxids, pause = 0.34) {
  tip_taxids <- unique(as.integer(tip_taxids))
  tip_taxids <- tip_taxids[!is.na(tip_taxids) & tip_taxids > 0L]
  cache <- new.env(parent = emptyenv())

  store <- function(rec) {
    cache[[as.character(rec$taxid)]] <- rec
  }

  # 1) Fetch tips
  for (tid in tip_taxids) {
    xml <- fetch_taxon_xml(tid, pause = pause)
    rec <- parse_taxon_record(tid, xml)
    store(rec)
    # Seed parent stubs from LineageEx (names/ranks already from rentrez tip payload)
    if (nrow(rec$lineage_ex)) {
      for (i in seq_len(nrow(rec$lineage_ex))) {
        pid <- rec$lineage_ex$taxid[[i]]
        key <- as.character(pid)
        if (!exists(key, envir = cache, inherits = FALSE)) {
          store(list(
            taxid = as.integer(pid),
            name = rec$lineage_ex$name[[i]],
            rank = rec$lineage_ex$rank[[i]],
            parent = NA_integer_,
            lineage_ex = data.frame(taxid = integer(), name = character(), rank = character())
          ))
        }
      }
    }
  }

  # 2) Collect parent ids still missing full fetch (have ParentTaxId chain)
  known <- as.integer(ls(cache))
  parent_ids <- integer(0)
  for (tid in tip_taxids) {
    rec <- cache[[as.character(tid)]]
    parent_ids <- c(parent_ids, rec$parent, rec$lineage_ex$taxid)
  }
  parent_ids <- unique(as.integer(parent_ids))
  parent_ids <- parent_ids[!is.na(parent_ids) & parent_ids > 0L]
  missing <- setdiff(parent_ids, known)

  # 3) Batch-fetch any parents not already represented
  if (length(missing)) {
    chunk_size <- 20L
    for (start in seq(1L, length(missing), by = chunk_size)) {
      chunk <- missing[start:min(start + chunk_size - 1L, length(missing))]
      Sys.sleep(pause)
      xml <- tryCatch(
        entrez_fetch(db = "taxonomy", id = as.character(chunk), rettype = "xml"),
        error = function(e) NA_character_
      )
      if (is.na(xml) || !nzchar(xml)) next
      # split concatenated TaxaSet into Taxon blocks
      parts <- strsplit(xml, "<Taxon>", fixed = TRUE)[[1]]
      parts <- parts[nzchar(trimws(parts))]
      for (p in parts) {
        block <- paste0("<Taxon>", p)
        tid <- suppressWarnings(as.integer(xml_first(block, "TaxId")))
        if (is.na(tid)) next
        store(parse_taxon_record(tid, block))
      }
    }
  }

  seen <- as.integer(ls(cache))
  records <- lapply(as.character(seen), function(k) cache[[k]])
  names(records) <- as.character(seen)
  list(records = records, tip_taxids = tip_taxids, all_taxids = seen)
}

# ---------------------------------------------------------------------------
# Lineage table → fill last classified → sanitize → ape tree
# ---------------------------------------------------------------------------

map_rank <- function(rank) {
  if (is.null(rank) || length(rank) == 0 || is.na(rank) || !nzchar(rank)) return(NULL)
  if (!rank %in% names(RANK_MAP)) return(NULL)
  RANK_MAP[[rank]]
}

record_to_rank_named <- function(rec) {
  out <- setNames(rep(NA_character_, length(RANK_COLS)), RANK_COLS)
  # lineage parents
  if (nrow(rec$lineage_ex)) {
    for (i in seq_len(nrow(rec$lineage_ex))) {
      rk <- map_rank(rec$lineage_ex$rank[[i]])
      if (!is.null(rk) && is.na(out[[rk]])) out[[rk]] <- rec$lineage_ex$name[[i]]
    }
  }
  # tip itself
  rk_tip <- map_rank(rec$rank)
  if (!is.null(rk_tip)) out[[rk_tip]] <- rec$name %||% out[[rk_tip]]
  out
}

build_lineage_df <- function(tip_taxids, records) {
  rows <- lapply(tip_taxids, function(tid) {
    rec <- records[[as.character(tid)]]
    if (is.null(rec)) {
      r <- setNames(rep(NA_character_, length(RANK_COLS)), RANK_COLS)
    } else {
      r <- record_to_rank_named(rec)
      # Ensure tip name appears at finest available rank
      if (!is.na(rec$name)) {
        if (is.na(r[["species"]]) && identical(rec$rank, "species")) r[["species"]] <- rec$name
        if (is.na(r[["genus"]]) && identical(rec$rank, "genus")) r[["genus"]] <- rec$name
        if (is.na(r[["family"]]) && identical(rec$rank, "family")) r[["family"]] <- rec$name
        # family-level tip with empty species: keep family filled; tip_label carries name
      }
    }
    data.frame(
      taxa_id = paste0("tax_", tid),
      taxid = as.integer(tid),
      tip_name = if (!is.null(rec)) rec$name %||% paste0("tax_", tid) else paste0("tax_", tid),
      tip_rank = if (!is.null(rec)) rec$rank %||% NA_character_ else NA_character_,
      as.list(r),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

#' Fill NA ranks with the last classified (non-NA) higher rank — codebase rule.
fill_na_last_classified <- function(tax_df, ranks = RANK_COLS) {
  tax_df <- as.data.frame(tax_df, stringsAsFactors = FALSE)
  for (i in seq_len(nrow(tax_df))) {
    last <- NA_character_
    for (rk in ranks) {
      v <- tax_df[[rk]][i]
      if (is.null(v) || length(v) == 0 || is.na(v) || !nzchar(as.character(v))) {
        tax_df[[rk]][i] <- if (!is.na(last) && nzchar(last)) last else NA_character_
      } else {
        last <- as.character(v)
      }
    }
  }
  tax_df
}

#' Obligatory sanitize before ape::as.phylo (as.character → factor; empty→Unclassified).
sanitize_tax_df_for_ape <- function(tax_df, cols) {
  tax_df <- as.data.frame(tax_df, stringsAsFactors = FALSE)
  keep <- tax_df[, cols, drop = FALSE]
  as.data.frame(
    lapply(keep, function(x) {
      x <- as.character(x)
      x[x == "" | is.na(x)] <- "Unclassified"
      x <- gsub("/", "_", x)
      factor(x, levels = unique(x))
    }),
    stringsAsFactors = FALSE
  )
}

build_taxonomy_tree <- function(lineage_df) {
  ranks <- RANK_COLS
  df <- fill_na_last_classified(lineage_df, ranks = ranks)
  # tip column for formula
  df$taxa_id <- as.character(lineage_df$taxa_id)
  ape_cols <- c(ranks, "taxa_id")
  tax_df_phy <- sanitize_tax_df_for_ape(df, ape_cols)
  # ape formula tree (obligatory path)
  tr <- ape::as.phylo(
    data = tax_df_phy,
    ~ kingdom / phylum / class / order / family / genus / species / taxa_id
  )
  # human-readable tip labels
  tip_map <- setNames(as.character(lineage_df$tip_name), as.character(lineage_df$taxa_id))
  tr$tip.label <- ifelse(
    tr$tip.label %in% names(tip_map),
    tip_map[tr$tip.label],
    tr$tip.label
  )
  list(tree = tr, tax_df_phy = tax_df_phy, lineage = df)
}

# ---------------------------------------------------------------------------
# ggtree report — tip style by rank
# species → italic; genus → italic + " sp."; else plain
# (aligned with .cursor/skills/_shared/import/taxon_format.R)
# ---------------------------------------------------------------------------

normalize_tip_rank <- function(rank) {
  r <- tolower(trimws(as.character(rank %||% "")))
  if (!nzchar(r) || is.na(r)) return(NA_character_)
  if (r %in% c("species", "subspecies", "varietas", "forma")) return("species")
  if (r %in% c("genus")) return("genus")
  r
}

strip_sp_suffix <- function(name) {
  name <- trimws(as.character(name %||% ""))
  asv <- regmatches(name, regexpr(" ASV[0-9]+$", name))
  base <- sub(" ASV[0-9]+$", "", name)
  base <- sub("\\s+spp?\\.?$", "", base, ignore.case = TRUE)
  if (length(asv) && nzchar(asv)) paste0(base, asv) else base
}

#' Build display label + fontface for each tip from lineage tip_rank / tip_name.
format_tiplab_style <- function(tree, lineage_df) {
  tips <- as.character(tree$tip.label)
  by_name <- match(tips, as.character(lineage_df$tip_name))
  by_id <- match(tips, as.character(lineage_df$taxa_id))
  idx <- ifelse(!is.na(by_name), by_name, by_id)

  display <- tips
  fontface <- rep("plain", length(tips))
  rank_norm <- rep(NA_character_, length(tips))

  for (i in seq_along(tips)) {
    j <- idx[[i]]
    nm <- if (!is.na(j)) as.character(lineage_df$tip_name[[j]]) else tips[[i]]
    rk <- if (!is.na(j)) normalize_tip_rank(lineage_df$tip_rank[[j]]) else NA_character_
    rank_norm[[i]] <- rk
    asv <- regmatches(nm, regexpr(" ASV[0-9]+$", nm))
    asv <- if (length(asv) && nzchar(asv)) asv else ""
    core <- sub(" ASV[0-9]+$", "", nm)
    if (identical(rk, "species")) {
      display[[i]] <- paste0(core, asv)
      fontface[[i]] <- "italic"
    } else if (identical(rk, "genus")) {
      base <- sub("\\s+spp?\\.?$", "", trimws(core), ignore.case = TRUE)
      display[[i]] <- paste0(base, " sp.", asv)
      fontface[[i]] <- "italic"
    } else {
      display[[i]] <- paste0(core, asv)
      fontface[[i]] <- "plain"
    }
  }

  data.frame(
    label = tips,
    display = display,
    fontface = fontface,
    tip_rank = rank_norm,
    stringsAsFactors = FALSE
  )
}

plot_ggtree_save <- function(tree, outfile_pdf, outfile_png, title, lineage_df = NULL) {
  # Formula taxonomy trees often lack edge lengths; ggtree needs them for layout
  if (is.null(tree$edge.length) || !length(tree$edge.length) || all(is.na(tree$edge.length))) {
    tree <- ape::compute.brlen(tree, method = "Grafen")
  }

  tip_style <- if (!is.null(lineage_df)) {
    format_tiplab_style(tree, lineage_df)
  } else {
    data.frame(
      label = as.character(tree$tip.label),
      display = as.character(tree$tip.label),
      fontface = "plain",
      tip_rank = NA_character_,
      stringsAsFactors = FALSE
    )
  }
  # Keep tree tip.label as keys for ggtree join; show styled display via aesthetic
  font_int <- ifelse(tip_style$fontface == "italic", 3L, 1L)

  h <- max(4, 0.45 * length(tree$tip.label) + 2)
  w <- 9

  # Always write a reliable base-R ape PDF + PNG (env-safe visual QA)
  ape_pdf <- sub("_ggtree\\.pdf$", "_ape.pdf", outfile_pdf)
  ape_png <- sub("_ggtree\\.png$", "_ape.png", outfile_png)

  plot_ape_styled <- function() {
    op <- par(no.readonly = TRUE)
    on.exit(par(op), add = TRUE)
    par(mar = c(2, 1, 3, 12))
    plot(tree, show.tip.label = FALSE, main = title, x.lim = c(0, 1.85))
    ord <- match(tree$tip.label, tip_style$label)
    ape::tiplabels(
      tip_style$display[ord],
      adj = c(0, 0.5),
      frame = "none",
      font = font_int[ord],
      cex = 0.9
    )
  }

  grDevices::pdf(ape_pdf, width = w, height = h)
  plot_ape_styled()
  grDevices::dev.off()

  # Prefer Cairo/ragg for PNGs when available (avoids blank X11/png devices)
  open_png <- function(path, width_in, height_in, dpi = 150) {
    px_w <- as.integer(width_in * dpi)
    px_h <- as.integer(height_in * dpi)
    if (requireNamespace("ragg", quietly = TRUE)) {
      ragg::agg_png(path, width = px_w, height = px_h, res = dpi, units = "px")
      return("ragg")
    }
    if (isTRUE(capabilities("cairo"))) {
      grDevices::png(path, width = px_w, height = px_h, res = dpi, type = "cairo")
      return("cairo")
    }
    grDevices::png(path, width = px_w, height = px_h, res = dpi)
    "png"
  }

  open_png(ape_png, w, h)
  plot_ape_styled()
  grDevices::dev.off()

  if (!requireNamespace("ggtree", quietly = TRUE) || !requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggtree/ggplot2 missing; copied ape PNG to ggtree path")
    file.copy(ape_png, outfile_png, overwrite = TRUE)
    file.copy(ape_pdf, outfile_pdf, overwrite = TRUE)
    return(invisible(NULL))
  }

  suppressPackageStartupMessages({
    library(ggtree)
    library(ggplot2)
  })

  tip_df <- tip_style
  tip_df$lab <- tip_df$label
  tip_df$is_italic <- tip_df$fontface == "italic"
  tip_df$is_plain <- tip_df$fontface == "plain"

  p <- ggtree(tree) %<+% tip_df +
    ggtree::geom_tiplab(
      ggplot2::aes(subset = isTip & is_italic, label = display),
      fontface = "italic",
      size = 3.2
    ) +
    ggtree::geom_tiplab(
      ggplot2::aes(subset = isTip & is_plain, label = display),
      fontface = "plain",
      size = 3.2
    ) +
    ggtree::hexpand(0.55) +
    ggplot2::ggtitle(title) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", size = 12))

  ggplot2::ggsave(outfile_pdf, p, width = w, height = h, device = grDevices::pdf)

  open_png(outfile_png, w, h)
  print(p)
  grDevices::dev.off()

  info <- tryCatch(file.info(outfile_png)$size, error = function(e) 0)
  if (is.na(info) || info < 2000) {
    warning("ggtree PNG looks empty (", info, " bytes); replacing with ape PNG")
    file.copy(ape_png, outfile_png, overwrite = TRUE)
  }
  invisible(p)
}

write_html_report <- function(outdir, panels) {
  html <- file.path(outdir, "taxonomy-tree-report.html")
  esc <- function(x) {
    x <- as.character(x %||% "")
    x <- gsub("&", "&amp;", x, fixed = TRUE)
    x <- gsub("<", "&lt;", x, fixed = TRUE)
    x <- gsub(">", "&gt;", x, fixed = TRUE)
    x
  }
  body <- paste(vapply(panels, function(p) {
    sprintf(
      paste0(
        "<section style='margin-bottom:2rem;padding:1rem;border:1px solid #ddd;border-radius:8px;'>",
        "<h2>%s</h2>",
        "<ul>",
        "<li>Tips: <b>%s</b></li>",
        "<li>Newick: <code>%s</code></li>",
        "<li>Lineage TSV: <code>%s</code></li>",
        "<li>ggtree PNG/PDF: <code>%s</code> / <code>%s</code></li>",
        "<li>ape PNG (fallback): <code>%s</code></li>",
        "<li>Unresolved: %s</li>",
        "</ul>",
        "<img src='%s' alt='%s' style='max-width:100%%;height:auto;border:1px solid #eee;'/>",
        "<p>ape fallback:</p>",
        "<img src='%s' alt='%s ape' style='max-width:100%%;height:auto;border:1px solid #eee;'/>",
        "</section>"
      ),
      esc(p$title), esc(p$n_tips), esc(basename(p$nwk)), esc(basename(p$lineage)),
      esc(basename(p$png)), esc(basename(p$pdf)),
      esc(basename(sub("_ggtree\\.png$", "_ape.png", p$png))),
      esc(p$unresolved_msg),
      esc(basename(p$png)), esc(p$title),
      esc(basename(sub("_ggtree\\.png$", "_ape.png", p$png))), esc(p$title)
    )
  }, character(1)), collapse = "\n")

  doc <- paste0(
    "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
    "<title>taxonomy-tree report</title>",
    "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;}",
    "code{background:#f5f5f5;padding:0.1rem 0.3rem;border-radius:4px;}</style>",
    "</head><body>",
    "<h1>taxonomy-tree hook report</h1>",
    "<p>Trees rebuilt with rentrez lineages → fill_na(last classified) → ",
    "<code>as.character</code>→<code>factor</code> → <code>ape::as.phylo</code> → ggtree.</p>",
    body,
    "</body></html>"
  )
  writeLines(doc, html)
  html
}

# ---------------------------------------------------------------------------
# Pipeline for one input mode
# ---------------------------------------------------------------------------

run_mode <- function(mode, tip_taxids, unresolved, outdir, label, pause = 0.34) {
  if (!length(tip_taxids)) {
    warning("No tip taxids for mode ", mode)
    return(NULL)
  }
  message("[", label, "] fetching ", length(tip_taxids), " tip taxids + parents via rentrez…")
  fetched <- fetch_records_and_parents(tip_taxids, pause = pause)
  lineage <- build_lineage_df(fetched$tip_taxids, fetched$records)
  built <- build_taxonomy_tree(lineage)

  prefix <- file.path(outdir, paste0("tree_", mode))
  lineage_path <- paste0(prefix, "_lineage.tsv")
  phy_path <- paste0(prefix, "_tax_df_phy.tsv")
  nwk_path <- paste0(prefix, ".nwk")
  rds_path <- paste0(prefix, ".rds")
  pdf_path <- paste0(prefix, "_ggtree.pdf")
  png_path <- paste0(prefix, "_ggtree.png")
  parents_path <- paste0(prefix, "_all_taxids.tsv")

  utils::write.table(built$lineage, lineage_path, sep = "\t", row.names = FALSE, quote = FALSE)
  # write factor table as characters for inspection
  phy_chr <- as.data.frame(lapply(built$tax_df_phy, as.character), stringsAsFactors = FALSE)
  utils::write.table(phy_chr, phy_path, sep = "\t", row.names = FALSE, quote = FALSE)
  ape::write.tree(built$tree, file = nwk_path)
  saveRDS(list(tree = built$tree, lineage = built$lineage, tax_df_phy = built$tax_df_phy,
               tip_style = format_tiplab_style(built$tree, built$lineage),
               all_taxids = fetched$all_taxids, records = fetched$records),
          rds_path)

  parents_df <- data.frame(
    taxid = fetched$all_taxids,
    name = vapply(as.character(fetched$all_taxids), function(k) fetched$records[[k]]$name %||% NA_character_, character(1)),
    rank = vapply(as.character(fetched$all_taxids), function(k) fetched$records[[k]]$rank %||% NA_character_, character(1)),
    is_tip = fetched$all_taxids %in% fetched$tip_taxids,
    stringsAsFactors = FALSE
  )
  utils::write.table(parents_df, parents_path, sep = "\t", row.names = FALSE, quote = FALSE)

  if (length(unresolved)) {
    utils::write.table(
      data.frame(input = unresolved, stringsAsFactors = FALSE),
      paste0(prefix, "_unresolved.tsv"),
      sep = "\t", row.names = FALSE, quote = FALSE
    )
  }

  message("[", label, "] plotting ggtree…")
  plot_ggtree_save(
    built$tree,
    outfile_pdf = pdf_path,
    outfile_png = png_path,
    title = sprintf("taxonomy-tree (%s) - %d tips", label, length(built$tree$tip.label)),
    lineage_df = built$lineage
  )

  list(
    title = sprintf("Input: %s", label),
    n_tips = length(built$tree$tip.label),
    nwk = nwk_path,
    lineage = lineage_path,
    png = png_path,
    pdf = pdf_path,
    unresolved_msg = if (length(unresolved)) paste(unresolved, collapse = "; ") else "(none)",
    tree = built$tree
  )
}

run_pipeline <- function(args) {
  root <- project_root()
  setwd(root)
  options(rentrez.email = args$email)
  outdir <- ensure_dir(args$outdir)

  default_taxons <- file.path(root, "test/misc/taxons.json")
  default_taxids <- file.path(root, "test/misc/taxids.tsv")
  taxons_path <- args$taxons %||% default_taxons
  taxids_path <- args$taxids %||% default_taxids

  # Ensure mocks exist
  mock <- mock_data_script()
  if (!file.exists(mock)) {
    stop("mock_data.py not found at ", mock)
  }
  if ((!file.exists(taxons_path) || !file.exists(taxids_path)) && file.exists(mock)) {
    message("Generating mock misc fixtures…")
    system2("python3", c(mock, "--out", "test", "--target", "misc", "--self-test"))
  }

  panels <- list()
  summary <- list(email = args$email, outdir = outdir, modes = list())

  if (args$mode %in% c("names", "both")) {
    names_in <- load_taxon_names(taxons_path)
    message("Resolving ", length(names_in), " taxon names via rentrez…")
    resolved <- integer(length(names_in))
    unresolved <- character(0)
    for (i in seq_along(names_in)) {
      tid <- resolve_name_to_taxid(names_in[[i]])
      resolved[[i]] <- tid
      if (is.na(tid)) {
        unresolved <- c(unresolved, names_in[[i]])
        message("  unresolved: ", names_in[[i]])
      } else {
        message("  ", names_in[[i]], " → ", tid)
      }
    }
    map_df <- data.frame(taxon = names_in, taxid = resolved, stringsAsFactors = FALSE)
    utils::write.table(map_df, file.path(outdir, "names_to_taxid.tsv"),
                       sep = "\t", row.names = FALSE, quote = FALSE)
    tip_ids <- unique(resolved[!is.na(resolved)])
    panel <- run_mode("names", tip_ids, unresolved, outdir, "taxon names (rentrez)", pause = 0.34)
    if (!is.null(panel)) {
      panels <- c(panels, list(panel))
      summary$modes$names <- list(
        n_input = length(names_in),
        n_resolved = length(tip_ids),
        unresolved = unresolved,
        nwk = panel$nwk,
        png = panel$png,
        pdf = panel$pdf
      )
    }
  }

  if (args$mode %in% c("taxids", "both")) {
    id_df <- load_taxids(taxids_path)
    message("Loaded ", nrow(id_df), " taxids from ", taxids_path)
    tip_ids <- unique(id_df$taxid)
    unresolved <- character(0)
    panel <- run_mode("taxids", tip_ids, unresolved, outdir, "taxids list", pause = 0.34)
    if (!is.null(panel)) {
      panels <- c(panels, list(panel))
      summary$modes$taxids <- list(
        n_input = length(tip_ids),
        n_resolved = length(tip_ids),
        unresolved = unresolved,
        nwk = panel$nwk,
        png = panel$png,
        pdf = panel$pdf
      )
    }
  }

  if (!length(panels)) stop("No trees produced — check inputs / NCBI access")

  html <- write_html_report(outdir, panels)
  summary$report_html <- html
  summary$report_pdfs <- vapply(panels, `[[`, character(1), "pdf")
  summary$report_pngs <- vapply(panels, `[[`, character(1), "png")

  report_json <- file.path(outdir, "taxonomy-tree-report.json")
  writeLines(jsonlite::toJSON(summary, auto_unbox = TRUE, pretty = TRUE), report_json)
  message("HTML report: ", html)
  message("JSON report: ", report_json)
  invisible(summary)
}

self_test <- function() {
  root <- project_root()
  setwd(root)
  # Ensure mocks
  system2("python3", c(mock_data_script(), "--out", "test", "--target", "misc", "--self-test"))
  args <- list(
    taxons = "test/misc/taxons.json",
    taxids = "test/misc/taxids.tsv",
    outdir = "test/taxonomy-tree",
    email = Sys.getenv("NCBI_EMAIL", unset = "taxonomy-tree-hook@local.dev"),
    mode = "both",
    self_test = TRUE
  )
  summary <- run_pipeline(args)
  # Validate outputs
  need <- c(
    "test/taxonomy-tree/tree_names.nwk",
    "test/taxonomy-tree/tree_taxids.nwk",
    "test/taxonomy-tree/tree_names_ggtree.pdf",
    "test/taxonomy-tree/tree_taxids_ggtree.pdf",
    "test/taxonomy-tree/tree_names_ggtree.png",
    "test/taxonomy-tree/tree_taxids_ggtree.png",
    "test/taxonomy-tree/taxonomy-tree-report.html"
  )
  missing <- need[!file.exists(need)]
  if (length(missing)) {
    stop("SELF-TEST FAIL missing: ", paste(missing, collapse = ", "))
  }
  # Trees readable
  t1 <- ape::read.tree("test/taxonomy-tree/tree_names.nwk")
  t2 <- ape::read.tree("test/taxonomy-tree/tree_taxids.nwk")
  if (is.null(t1$tip.label) || !length(t1$tip.label)) stop("names tree empty")
  if (is.null(t2$tip.label) || !length(t2$tip.label)) stop("taxids tree empty")
  st <- readRDS("test/taxonomy-tree/tree_names.rds")$tip_style
  if (is.null(st)) stop("tip_style missing from RDS")
  gen <- st[st$tip_rank == "genus", , drop = FALSE]
  sp <- st[st$tip_rank == "species", , drop = FALSE]
  oth <- st[!st$tip_rank %in% c("genus", "species") & !is.na(st$tip_rank), , drop = FALSE]
  if (nrow(gen) && !all(grepl(" sp\\.$", gen$display))) {
    stop("genus tips must end with ' sp.': ", paste(gen$display, collapse = ", "))
  }
  if (nrow(gen) && !all(gen$fontface == "italic")) stop("genus tips must be italic")
  if (nrow(sp) && !all(sp$fontface == "italic")) stop("species tips must be italic")
  if (nrow(oth) && !all(oth$fontface == "plain")) stop("non-genus/species tips must be plain")
  message("SELF-TEST OK")
  message("  names tips: ", paste(t1$tip.label, collapse = ", "))
  message("  taxids tips: ", paste(t2$tip.label, collapse = ", "))
  message("  tip styles: ", paste(sprintf("%s[%s/%s]", st$display, st$tip_rank, st$fontface), collapse = "; "))
  message("  report: test/taxonomy-tree/taxonomy-tree-report.html")
  invisible(summary)
}

main <- function() {
  args <- parse_args()
  if (isTRUE(args$self_test)) {
    self_test()
  } else {
    run_pipeline(args)
  }
  invisible(0L)
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
