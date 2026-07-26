# Shared taxon display formatting for plots
# species → italic; genus → italic + " sp."; else plain
# Prefer ggtext markdown labels (*name*) when available.

normalize_plot_rank <- function(rank) {
  r <- tolower(trimws(as.character(rank %||% "")))
  if (!nzchar(r) || is.na(r)) return(NA_character_)
  if (r %in% c("species", "subspecies", "varietas", "forma")) return("species")
  if (r %in% c("genus")) return("genus")
  r
}

strip_sp_suffix <- function(name) {
  name <- trimws(as.character(name %||% ""))
  # keep trailing " ASVk" for disambiguation
  asv <- regmatches(name, regexpr(" ASV[0-9]+$", name))
  base <- sub(" ASV[0-9]+$", "", name)
  base <- sub("\\s+spp?\\.?$", "", base, ignore.case = TRUE)
  if (length(asv) && nzchar(asv)) paste0(base, asv) else base
}

#' One taxon → display string + fontface ("italic"|"plain") + optional markdown.
format_taxon_plot_label <- function(name, rank = NA_character_) {
  nm <- as.character(name %||% "")
  rk <- normalize_plot_rank(rank)
  # Infer genus/species from tip_rank or name shape when rank missing
  if (is.na(rk)) {
    if (grepl("\\s", sub(" ASV[0-9]+$", "", nm)) &&
        !grepl("\\ssp\\.?$", sub(" ASV[0-9]+$", "", nm), ignore.case = TRUE)) {
      rk <- "species"
    }
  }
  asv <- regmatches(nm, regexpr(" ASV[0-9]+$", nm))
  asv <- if (length(asv) && nzchar(asv)) asv else ""
  core <- sub(" ASV[0-9]+$", "", nm)

  if (identical(rk, "species")) {
    display <- paste0(core, asv)
    fontface <- "italic"
    markdown <- paste0("*", core, "*", asv)
  } else if (identical(rk, "genus")) {
    base <- strip_sp_suffix(core)
    # strip_sp_suffix may keep ASV on base — ensure ASV once
    base <- sub(" ASV[0-9]+$", "", base)
    display <- paste0(base, " sp.", asv)
    fontface <- "italic"
    markdown <- paste0("*", base, "* sp.", asv)
  } else {
    display <- paste0(core, asv)
    fontface <- "plain"
    markdown <- display
  }
  list(display = display, fontface = fontface, markdown = markdown, rank = rk)
}

#' Vectorized labels from phyloseq tax_table for OTU ids.
taxon_plot_labels_from_ps <- function(ps, otu_ids) {
  tax <- phyloseq::tax_table(ps, errorIfNULL = FALSE)
  if (is.null(tax)) {
    return(data.frame(
      otu = otu_ids, display = otu_ids, fontface = "plain",
      markdown = otu_ids, tip_rank = NA_character_, stringsAsFactors = FALSE
    ))
  }
  td <- as.data.frame(tax, stringsAsFactors = FALSE)
  ranks <- intersect(c(RANK_COLS, RANK_COLS_LC, "tip_rank"), names(td))
  deepest <- intersect(c("Species", "species", "Genus", "genus"), names(td))
  deepest <- if (length(deepest)) deepest[[1]] else names(td)[1]

  rows <- lapply(otu_ids, function(id) {
    if (!id %in% rownames(td)) {
      return(data.frame(
        otu = id, display = id, fontface = "plain", markdown = id,
        tip_rank = NA_character_, stringsAsFactors = FALSE
      ))
    }
    row <- td[id, , drop = FALSE]
    nm <- as.character(row[[deepest]][[1]])
    if (is.na(nm) || !nzchar(nm)) {
      for (col in rev(setdiff(names(row), "tip_rank"))) {
        v <- as.character(row[[col]][[1]])
        if (!is.na(v) && nzchar(v)) {
          nm <- v
          break
        }
      }
    }
    rk <- if ("tip_rank" %in% names(row)) as.character(row$tip_rank[[1]]) else NA_character_
    fmt <- format_taxon_plot_label(nm, rk)
    data.frame(
      otu = id,
      display = fmt$display,
      fontface = fmt$fontface,
      markdown = fmt$markdown,
      tip_rank = fmt$rank %||% rk,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
