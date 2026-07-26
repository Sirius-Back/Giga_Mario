#!/usr/bin/env Rscript
# network — coexistence (SparCC), NetCoMi, tidygraph/df2chord, igraph
suppressPackageStartupMessages({
  for (p in c(
    "phyloseq", "ggplot2", "dplyr", "tidyr", "jsonlite", "igraph",
    "ggraph", "viridis", "corrplot"
  )) {
    if (!requireNamespace(p, quietly = TRUE)) stop("Missing package: ", p, call. = FALSE)
  }
  library(phyloseq)
  library(ggplot2)
  library(dplyr)
  library(igraph)
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
source(file.path(root, ".cursor/skills/_shared/import/taxon_format.R"))
# Vendored aRchiteutis df2chord (verbatim)
source(file.path(root, ".cursor/skills/network/vendor/aRchiteutis/df2chord.R"))

`%||%` <- function(a, b) if (!is.null(a) && length(a) && !all(is.na(a))) a else b

load_ps <- function(path) {
  if (!file.exists(path)) fail("RDS not found: ", path)
  obj <- readRDS(path)
  meta <- list(path = path, target = NA_character_, rarefied = FALSE, rarefaction_depth = NA_real_)
  if (inherits(obj, "phyloseq")) return(list(ps = obj, meta = meta))
  if (is.list(obj) && !is.null(obj$phyloseq) && inherits(obj$phyloseq, "phyloseq")) {
    meta$target <- obj$target %||% NA_character_
    meta$rarefaction_depth <- obj$rarefaction_depth %||% NA_real_
    meta$rarefied <- !is.null(obj$rarefaction_depth) || grepl("_rare|phyloseq_rare_", basename(path))
    return(list(ps = obj$phyloseq, meta = meta))
  }
  fail("RDS must be phyloseq or list with $phyloseq: ", path)
}

resolve_input_rds <- function(rds = NULL) {
  candidates <- c(
    "test/code-review-phyloseq/grazing_phyloseq_rare.rds",
    "test/rarefaction-analysis/grazing/phyloseq_rare_1187.rds",
    "test/code-review-phyloseq/grazing_phyloseq.rds"
  )
  if (!is.null(rds) && nzchar(rds)) {
    loaded <- load_ps(rds)
    loaded$notes <- character(0)
    return(loaded)
  }
  for (p in candidates) {
    if (file.exists(p)) {
      loaded <- load_ps(p)
      loaded$notes <- paste0("auto-resolved: ", p)
      return(loaded)
    }
  }
  fail("No phyloseq RDS; pass --rds")
}

otu_matrix_taxa_rows <- function(ps) {
  m <- as(otu_table(ps), "matrix")
  if (!taxa_are_rows(ps)) m <- t(m)
  storage.mode(m) <- "double"
  m
}

#' Keep taxa with mean relative abundance > min_mean_rel (default 0.01% = 1e-4).
#' Optional --top-n caps after that filter (NULL / Inf = no cap).
filter_taxa_abundance <- function(ps, min_mean_rel = 1e-4, top_n = NULL) {
  m <- otu_matrix_taxa_rows(ps)
  rel <- sweep(m, 2, pmax(colSums(m), 1e-12), "/")
  mean_rel <- rowMeans(rel)
  keep <- names(mean_rel)[mean_rel > as.numeric(min_mean_rel)]
  if (!length(keep)) {
    fail(
      "No taxa with mean relative abundance > ", min_mean_rel,
      " (try lowering --min-mean-rel)"
    )
  }
  keep <- keep[order(mean_rel[keep], decreasing = TRUE)]
  if (!is.null(top_n) && is.finite(as.numeric(top_n)) && as.integer(top_n) > 0L) {
    keep <- keep[seq_len(min(as.integer(top_n), length(keep)))]
  }
  message(
    "Abundance filter: mean_rel > ", min_mean_rel,
    " → ", length(keep), " taxa (of ", nrow(m), ")"
  )
  prune_taxa(keep, ps)
}

#' Tip display labels from finalized taxonomy (ASV disambiguation, never make.unique dots).
tip_label_table <- function(ps) {
  ids <- taxa_names(ps)
  lab <- taxon_plot_labels_from_ps(ps, ids)
  disp <- as.character(lab$display)
  # Only disambiguate colliding displays; leave unique names without ASV postfix
  base <- sub(" ASV[0-9]+$", "", disp)
  for (b in unique(base)) {
    idx <- which(base == b)
    if (length(idx) == 1L) {
      disp[idx] <- b
    } else {
      for (k in seq_along(idx)) {
        disp[idx[[k]]] <- paste0(b, " ASV", k)
      }
    }
  }
  lab$display <- disp
  tt <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
  pcol <- names(tt)[tolower(names(tt)) == "phylum"]
  pcol <- if (length(pcol)) pcol[[1]] else NA_character_
  ph <- if (!is.na(pcol)) {
    as.character(tt[lab$otu, pcol])
  } else {
    rep(NA_character_, nrow(lab))
  }
  ph[is.na(ph) | !nzchar(ph) | grepl("^unclassified$", ph, ignore.case = TRUE)] <- "Unknown"
  lab$phylum <- ph
  lab
}

#' Resolve node metadata when adj dimnames are OTU ids OR display/Species labels.
resolve_node_meta <- function(node_ids, ps) {
  labs <- tip_label_table(ps)
  tt <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
  ids <- as.character(node_ids)

  otu_of <- rep(NA_character_, length(ids))
  # 1) exact taxa_names
  hit <- match(ids, labs$otu)
  otu_of[!is.na(hit)] <- labs$otu[hit[!is.na(hit)]]
  # 2) display / Species label
  miss <- which(is.na(otu_of))
  if (length(miss)) {
    hit2 <- match(ids[miss], labs$display)
    otu_of[miss[!is.na(hit2)]] <- labs$otu[hit2[!is.na(hit2)]]
  }
  # 3) make.names(taxa_names)
  miss <- which(is.na(otu_of))
  if (length(miss)) {
    mn <- make.names(labs$otu)
    hit3 <- match(ids[miss], mn)
    otu_of[miss[!is.na(hit3)]] <- labs$otu[hit3[!is.na(hit3)]]
  }
  # 4) deepest taxonomy column
  miss <- which(is.na(otu_of))
  if (length(miss) && nrow(tt)) {
    deep <- intersect(c("Species", "species", "Genus", "genus"), names(tt))
    if (length(deep)) {
      sp <- as.character(tt[[deep[[1]]]])
      hit4 <- match(ids[miss], sp)
      otu_of[miss[!is.na(hit4)]] <- rownames(tt)[hit4[!is.na(hit4)]]
    }
  }

  display <- labs$display[match(otu_of, labs$otu)]
  display[is.na(display) | !nzchar(display)] <- ids[is.na(display) | !nzchar(display)]
  phylum <- labs$phylum[match(otu_of, labs$otu)]
  phylum[is.na(phylum) | !nzchar(phylum)] <- "Unknown"
  fontface <- labs$fontface[match(otu_of, labs$otu)]
  fontface[is.na(fontface) | !nzchar(fontface)] <- "plain"
  data.frame(
    node = ids, otu = otu_of, display = display,
    phylum = phylum, fontface = fontface,
    stringsAsFactors = FALSE
  )
}

save_gg <- function(plot_obj, prefix, width = 8, height = 8) {
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

save_base_plot <- function(expr, prefix, width = 8, height = 8) {
  # Capture unevaluated so PDF and PNG each draw the plot (force() would cache once)
  expr <- substitute(expr)
  pdf <- paste0(prefix, ".pdf")
  png <- paste0(prefix, ".png")
  grDevices::pdf(pdf, width = width, height = height)
  eval(expr, envir = parent.frame())
  grDevices::dev.off()
  png_out <- NULL
  tryCatch({
    grDevices::png(png, width = width * 300, height = height * 300, res = 300, type = "cairo")
    eval(expr, envir = parent.frame())
    grDevices::dev.off()
    png_out <- png
  }, error = function(e) {
    if (grDevices::dev.cur() > 1) grDevices::dev.off()
    message("PNG skip: ", conditionMessage(e))
  })
  list(pdf = pdf, png = png_out)
}

write_adj <- function(mat, path) {
  df <- as.data.frame(as.matrix(mat))
  df <- cbind(taxon = rownames(df), df)
  utils::write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

require_netcomi <- function() {
  if (!requireNamespace("NetCoMi", quietly = TRUE)) {
    fail("NetCoMi required for coexistence/netcomi methods")
  }
}

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

phylum_palette <- function(phyla) {
  phyla <- unique(as.character(phyla))
  phyla <- phyla[!is.na(phyla) & nzchar(phyla)]
  n <- length(phyla)
  if (!n) return(character(0))
  cols <- if (n <= 12L) {
    RColorBrewer::brewer.pal(max(3L, n), "Set3")[seq_len(n)]
  } else {
    grDevices::colorRampPalette(RColorBrewer::brewer.pal(12, "Set3"))(n)
  }
  stats::setNames(cols, phyla)
}

#' Taxonomy display labels for NetCoMi analyze/native plots (not QIIME ASV hashes).
netcomi_display_labels <- function(props, ps) {
  ids <- NULL
  if (!is.null(props$input$adjaMat1)) ids <- colnames(props$input$adjaMat1)
  if (is.null(ids) && !is.null(props$input$assoMat1)) ids <- colnames(props$input$assoMat1)
  if (is.null(ids) || !length(ids)) return(TRUE)
  meta <- resolve_node_meta(ids, ps)
  labs <- meta$display
  labs[is.na(labs) | !nzchar(labs)] <- meta$node
  names(labs) <- meta$node
  labs
}

#' ggrepel layout positions + shadowtext draw (Metaanalyse-style halo, not label boxes).
shadowtext_repel_layers <- function(nd, size = 2.6) {
  if (!requireNamespace("ggrepel", quietly = TRUE) ||
      !requireNamespace("ggplot2", quietly = TRUE)) {
    return(NULL)
  }
  if (!requireNamespace("shadowtext", quietly = TRUE)) {
    # No shadowtext: single ggrepel layer with white halo (still not geom_label_repel)
    return(list(
      nd = nd,
      layers = list(
        ggrepel::geom_text_repel(
          data = nd,
          ggplot2::aes(
            x = .data$x, y = .data$y, label = .data$label, fontface = .data$fontface
          ),
          size = size, max.overlaps = Inf, seed = 1L,
          bg.colour = "white", bg.r = 0.12,
          box.padding = 0.3, point.padding = 0.25,
          min.segment.length = 0, segment.size = 0.2
        )
      )
    ))
  }
  tmp <- ggplot2::ggplot(
    nd, ggplot2::aes(x = .data$x, y = .data$y, label = .data$label)
  ) +
    ggrepel::geom_text_repel(
      size = size, max.overlaps = Inf, seed = 1L,
      box.padding = 0.3, point.padding = 0.25,
      min.segment.length = 0, force = 1, force_pull = 1
    )
  built <- tryCatch(ggplot2::ggplot_build(tmp), error = function(e) NULL)
  if (is.null(built) || !length(built$data)) return(NULL)
  rep <- built$data[[1]]
  if (is.null(rep$x) || is.null(rep$y) || nrow(rep) != nrow(nd)) return(NULL)
  nd$xlab <- as.numeric(rep$x)
  nd$ylab <- as.numeric(rep$y)
  list(
    nd = nd,
    layers = list(
      ggplot2::geom_segment(
        data = nd,
        ggplot2::aes(x = .data$x, y = .data$y, xend = .data$xlab, yend = .data$ylab),
        colour = "gray50", linewidth = 0.2, alpha = 0.85
      ),
      shadowtext::geom_shadowtext(
        data = nd,
        ggplot2::aes(
          x = .data$xlab, y = .data$ylab, label = .data$label,
          fontface = .data$fontface
        ),
        size = size, colour = "black", bg.colour = "white", bg.r = 0.12
      )
    )
  )
}

#' igraph / ggplot plot: larger opaque nodes; geom_shadowtext + ggrepel positions.
plot_netcomi_igraph <- function(
    adj, asso = NULL, ps, out_prefix,
    label_mode = c("shadowtext", "numbered"),
    title = "NetCoMi network"
) {
  label_mode <- match.arg(label_mode)
  adj <- as.matrix(adj)
  adj[is.na(adj)] <- 0
  diag(adj) <- 0
  if (is.null(asso)) asso <- adj
  asso <- as.matrix(asso)
  asso[is.na(asso)] <- 0

  ids <- colnames(adj)
  if (is.null(ids)) ids <- rownames(adj)
  if (is.null(ids)) ids <- as.character(seq_len(nrow(adj)))
  rownames(adj) <- colnames(adj) <- ids
  if (!is.null(dim(asso)) && all(dim(asso) == dim(adj))) {
    rownames(asso) <- colnames(asso) <- ids
  }

  meta <- resolve_node_meta(ids, ps)
  if (all(meta$phylum == "Unknown")) {
    message("WARNING: all node phyla unresolved — check tax_table Phylum vs adj dimnames")
  }

  g <- igraph::graph_from_adjacency_matrix(
    abs(adj), mode = "undirected", weighted = TRUE, diag = FALSE
  )
  igraph::V(g)$name <- ids
  igraph::V(g)$display <- meta$display
  igraph::V(g)$phylum <- meta$phylum
  igraph::V(g)$fontface <- meta$fontface

  el <- igraph::as_edgelist(g, names = TRUE)
  ew <- vapply(seq_len(nrow(el)), function(i) {
    a <- el[i, 1]; b <- el[i, 2]
    if (a %in% rownames(asso) && b %in% colnames(asso)) asso[a, b] else 0
  }, numeric(1))
  igraph::E(g)$weight_signed <- ew
  igraph::E(g)$color <- ifelse(ew >= 0, "#2ca02c", "#d62728")

  pcols <- phylum_palette(igraph::V(g)$phylum)
  igraph::V(g)$color <- ifelse(
    !is.na(igraph::V(g)$phylum) & igraph::V(g)$phylum %in% names(pcols),
    unname(pcols[igraph::V(g)$phylum]),
    "gray80"
  )
  igraph::V(g)$color[is.na(igraph::V(g)$color)] <- "gray80"

  lay <- igraph::layout_with_fr(g)
  xlim <- range(lay[, 1]); ylim <- range(lay[, 2])
  pad_x <- diff(xlim) * 0.12; pad_y <- diff(ylim) * 0.12
  if (!is.finite(pad_x) || pad_x == 0) pad_x <- 1
  if (!is.finite(pad_y) || pad_y == 0) pad_y <- 1
  figs <- list()
  node_size <- 14

  use_ggplot <- identical(label_mode, "shadowtext") &&
    requireNamespace("ggrepel", quietly = TRUE) &&
    requireNamespace("ggplot2", quietly = TRUE)

  if (use_ggplot) {
    labs <- igraph::V(g)$display
    labs[is.na(labs) | !nzchar(labs)] <- igraph::V(g)$name
    nd <- data.frame(
      x = lay[, 1], y = lay[, 2],
      label = labs,
      phylum = igraph::V(g)$phylum,
      fontface = ifelse(igraph::V(g)$fontface == "italic", "italic", "plain"),
      stringsAsFactors = FALSE
    )
    ed <- data.frame(
      x = lay[match(el[, 1], igraph::V(g)$name), 1],
      y = lay[match(el[, 1], igraph::V(g)$name), 2],
      xend = lay[match(el[, 2], igraph::V(g)$name), 1],
      yend = lay[match(el[, 2], igraph::V(g)$name), 2],
      edge_col = ifelse(ew >= 0, "#2ca02c", "#d62728"),
      edge_w = pmax(0.4, abs(ew) * 2.5),
      stringsAsFactors = FALSE
    )
    p <- ggplot2::ggplot() +
      ggplot2::geom_segment(
        data = ed,
        ggplot2::aes(
          x = .data$x, y = .data$y, xend = .data$xend, yend = .data$yend,
          linewidth = .data$edge_w
        ),
        colour = ed$edge_col, alpha = 1
      ) +
      ggplot2::geom_point(
        data = nd,
        ggplot2::aes(x = .data$x, y = .data$y, fill = .data$phylum),
        shape = 21, size = 5.5, alpha = 1, colour = "gray20", stroke = 0.35
      ) +
      ggplot2::scale_fill_manual(values = pcols, name = "Phylum", na.value = "gray80") +
      ggplot2::guides(linewidth = "none")
    st <- shadowtext_repel_layers(nd, size = 2.6)
    if (!is.null(st)) {
      for (ly in st$layers) {
        if (!is.null(ly)) p <- p + ly
      }
    } else if (requireNamespace("shadowtext", quietly = TRUE)) {
      p <- p + shadowtext::geom_shadowtext(
        data = nd,
        ggplot2::aes(
          x = .data$x, y = .data$y, label = .data$label, fontface = .data$fontface
        ),
        size = 2.6, colour = "black", bg.colour = "white", bg.r = 0.12
      )
    } else {
      p <- p + ggrepel::geom_text_repel(
        data = nd,
        ggplot2::aes(
          x = .data$x, y = .data$y, label = .data$label, fontface = .data$fontface
        ),
        size = 2.6, max.overlaps = Inf, seed = 1L,
        bg.colour = "white", bg.r = 0.12,
        box.padding = 0.3, point.padding = 0.25,
        min.segment.length = 0, segment.size = 0.2
      )
    }
    p <- p +
      ggplot2::coord_equal(
        xlim = xlim + c(-pad_x, pad_x), ylim = ylim + c(-pad_y, pad_y),
        expand = FALSE
      ) +
      ggplot2::labs(title = title) +
      ggplot2::theme_void(base_size = 11) +
      ggplot2::theme(
        legend.position = "bottom",
        legend.box = "vertical",
        plot.title = ggplot2::element_text(face = "bold", hjust = 0.5)
      )
    figs$main <- list(
      pdf = paste0(out_prefix, ".pdf"),
      png = paste0(out_prefix, ".png")
    )
    ggplot2::ggsave(figs$main$pdf, p, width = 10, height = 10)
    tryCatch(
      ggplot2::ggsave(figs$main$png, p, width = 10, height = 10, dpi = 300),
      error = function(e) message("PNG skip: ", conditionMessage(e))
    )
  } else if (identical(label_mode, "shadowtext")) {
    figs$main <- save_base_plot(
      {
        plot(
          g, layout = lay, rescale = FALSE,
          xlim = xlim + c(-pad_x, pad_x), ylim = ylim + c(-pad_y, pad_y),
          vertex.size = node_size,
          vertex.label = NA,
          vertex.color = igraph::V(g)$color,
          vertex.frame.color = "gray20",
          edge.color = igraph::E(g)$color,
          edge.width = pmax(0.5, abs(igraph::E(g)$weight_signed) * 3),
          main = title
        )
        labs <- igraph::V(g)$display
        labs[is.na(labs) | !nzchar(labs)] <- igraph::V(g)$name
        fonts <- ifelse(igraph::V(g)$fontface == "italic", 3L, 1L)
        cex <- 0.65
        ux <- graphics::strwidth("M", cex = cex) * 0.12
        uy <- graphics::strheight("M", cex = cex) * 0.12
        for (dx in c(-1, 0, 1)) {
          for (dy in c(-1, 0, 1)) {
            if (dx == 0L && dy == 0L) next
            graphics::text(
              lay[, 1] + dx * ux, lay[, 2] + dy * uy,
              labels = labs, cex = cex, col = "white", font = fonts
            )
          }
        }
        graphics::text(
          lay[, 1], lay[, 2], labels = labs, cex = cex, col = "black", font = fonts
        )
        if (length(pcols)) {
          legend(
            "bottomleft", legend = names(pcols), fill = unname(pcols),
            cex = 0.6, title = "Phylum", bty = "n"
          )
        }
      },
      out_prefix, width = 10, height = 10
    )
  } else {
    igraph::V(g)$number <- seq_len(igraph::vcount(g))
    figs$main <- save_base_plot(
      {
        op <- par(mar = c(2, 2, 3, 10))
        on.exit(par(op), add = TRUE)
        plot(
          g, layout = lay, rescale = FALSE,
          xlim = xlim + c(-pad_x, pad_x), ylim = ylim + c(-pad_y, pad_y),
          vertex.size = node_size,
          vertex.label = igraph::V(g)$number,
          vertex.label.cex = 0.6,
          vertex.color = igraph::V(g)$color,
          edge.color = igraph::E(g)$color,
          edge.width = pmax(0.5, abs(igraph::E(g)$weight_signed) * 3),
          main = title
        )
        legend_txt <- paste0(igraph::V(g)$number, ": ", igraph::V(g)$display)
        legend(
          "right", inset = c(-0.35, 0), xpd = TRUE,
          legend = legend_txt, cex = 0.45, bty = "n", title = "Nodes"
        )
        if (length(pcols)) {
          legend(
            "bottomleft", legend = names(pcols), fill = unname(pcols),
            cex = 0.55, title = "Phylum", bty = "n"
          )
        }
      },
      paste0(out_prefix, "_numbered"), width = 12, height = 10
    )
  }
  figs
}

run_coexistence <- function(ps, outdir, thresh = 0.3, seed = 123L,
                            label_mode = "shadowtext") {
  require_netcomi()
  message("coexistence: NetCoMi SparCC + igraph/shadowtext labels")
  set.seed(as.integer(seed))
  full_net <- NetCoMi::netConstruct(
    ps,
    measure = "sparcc",
    zeroMethod = "pseudo",
    sparsMethod = "threshold",
    thresh = as.numeric(thresh),
    verbose = 1,
    seed = as.integer(seed)
  )
  full_props <- NetCoMi::netAnalyze(
    full_net,
    clustMethod = "cluster_fast_greedy",
    hubPar = "eigenvector"
  )
  adj <- full_net$adjaMat1
  if (is.null(adj)) adj <- abs(full_net$assoMat1)
  asso <- full_net$assoMat1
  if (is.null(asso)) asso <- adj
  adj_path <- write_adj(asso, file.path(outdir, "adjacency_coexistence.tsv"))
  figs <- plot_netcomi_igraph(
    adj, asso, ps,
    out_prefix = file.path(outdir, "network_coexistence"),
    label_mode = label_mode,
    title = "Coexistence network (SparCC)"
  )
  # Also keep NetCoMi native plot (taxonomy display labels, not ASV hashes)
  labs_n <- netcomi_display_labels(full_props, ps)
  figs$netcomi_native <- save_base_plot(
    {
      plot(
        full_props, repulsion = 0.95, nodeSize = "fractions",
        rmSingles = TRUE, borderWidth = 0.5,
        labels = labs_n, labelScale = FALSE, cexLabels = 0.55
      )
    },
    file.path(outdir, "network_coexistence_netcomi"),
    width = 10, height = 10
  )
  saveRDS(list(net = full_net, props = full_props), file.path(outdir, "coexistence_net.rds"))
  list(
    adjacency = adj_path, figures = figs,
    net_rds = file.path(outdir, "coexistence_net.rds"),
    measure = "sparcc", thresh = thresh, label_mode = label_mode
  )
}

run_netcomi <- function(ps, outdir, measure = "sparcc", thresh = 0.3, seed = 13075L,
                       label_mode = "shadowtext") {
  require_netcomi()
  # Intended Locked default is SpiecEasi; SparCC kept for speed unless --measure spieceasi
  message("netcomi: measure=", measure, " (igraph viz with phylum/edge colors)")
  set.seed(as.integer(seed))
  args <- list(
    data = ps,
    filtTax = "numbSamp",
    filtTaxPar = list(numbSamp = 0.1),
    measure = measure,
    dissFunc = "signed",
    thresh = as.numeric(thresh),
    sparsMethod = "threshold",
    verbose = 1,
    seed = as.integer(seed)
  )
  if (identical(measure, "sparcc")) {
    args$zeroMethod <- "pseudo"
  }
  netcomi_net <- do.call(NetCoMi::netConstruct, args)
  props <- tryCatch(
    NetCoMi::netAnalyze(netcomi_net, clustMethod = "cluster_fast_greedy", hubPar = "eigenvector"),
    error = function(e) {
      message("netAnalyze failed: ", conditionMessage(e))
      NULL
    }
  )
  adj <- netcomi_net$adjaMat1
  if (is.null(adj)) adj <- abs(netcomi_net$assoMat1)
  asso <- netcomi_net$assoMat1
  if (is.null(asso)) asso <- adj
  adj_path <- write_adj(asso, file.path(outdir, paste0("adjacency_netcomi_", measure, ".tsv")))
  figs <- plot_netcomi_igraph(
    adj, asso, ps,
    out_prefix = file.path(outdir, "network_netcomi"),
    label_mode = label_mode,
    title = paste0("NetCoMi (", measure, ")")
  )
  if (!is.null(props)) {
    labs_n <- netcomi_display_labels(props, ps)
    figs$analyze <- save_base_plot(
      {
        plot(
          props, rmSingles = TRUE,
          labels = labs_n, labelScale = FALSE, cexLabels = 0.55
        )
      },
      file.path(outdir, "network_netcomi_analyze"),
      width = 10, height = 10
    )
  }
  saveRDS(list(net = netcomi_net, props = props), file.path(outdir, "netcomi_net.rds"))
  list(
    adjacency = adj_path, figures = figs,
    net_rds = file.path(outdir, "netcomi_net.rds"),
    measure = measure, thresh = thresh, label_mode = label_mode
  )
}

#' Convert NetCoMi microNetProps → tidygraph (ticks_metaanalyse.Rmd qgraph2tidy).
qgraph2tidy <- function(microNetProps) {
  if (!requireNamespace("tidygraph", quietly = TRUE)) {
    fail("tidygraph required for netcomi-ggraph method")
  }
  grDevices::pdf(NULL)
  on.exit({
    if (grDevices::dev.cur() > 1) grDevices::dev.off()
  }, add = TRUE)
  qplot <- plot(microNetProps, rmSingles = TRUE)
  if (is.null(qplot$q1)) fail("NetCoMi plot() did not return $q1 for tidygraph conversion")
  net_graph <- igraph::as.igraph(qplot$q1)
  lab <- igraph::V(net_graph)$label
  if (is.null(lab) || !length(lab)) lab <- igraph::V(net_graph)$name
  igraph::V(net_graph)$label <- lab
  adj_matrix <- microNetProps$input$adjaMat1[lab, lab, drop = FALSE]
  ord <- order(stats::prcomp(adj_matrix)$x[, 1])
  srt <- rownames(adj_matrix)[ord]
  net_graph <- igraph::permute(net_graph, match(lab, srt))
  lab2 <- igraph::V(net_graph)$label
  clust <- unlist(microNetProps$clustering$clust1)[lab2]
  igraph::V(net_graph)$clust <- clust
  tidygraph::as_tbl_graph(net_graph)
}

#' Circular ggraph from NetCoMi props (ticks_metaanalyse.Rmd tidygraphize).
tidygraphize_netcomi <- function(microNetProps, ps = NULL) {
  if (!requireNamespace("ggraph", quietly = TRUE) ||
      !requireNamespace("tidygraph", quietly = TRUE)) {
    fail("ggraph + tidygraph required for netcomi-ggraph method")
  }
  tidy_net <- qgraph2tidy(microNetProps)
  g0 <- tidygraph::as.igraph(tidy_net)
  LV <- igraph::vcount(g0)
  angle <- 90 - 360 * seq_len(LV) / LV
  hjust <- ifelse(angle < -90, 1, 0)
  angle <- ifelse(angle < -90, angle + 180, angle)

  labs_raw <- igraph::V(g0)$label
  if (is.null(labs_raw)) labs_raw <- igraph::V(g0)$name
  display <- labs_raw
  face <- rep("italic", length(display))
  if (!is.null(ps)) {
    meta <- resolve_node_meta(labs_raw, ps)
    display <- meta$display
    display[is.na(display) | !nzchar(display)] <- labs_raw
    face <- ifelse(meta$fontface == "italic", "italic", "plain")
  }
  igraph::V(g0)$display <- display
  igraph::V(g0)$fontface <- face
  tidy_net <- tidygraph::as_tbl_graph(g0)

  LVC <- length(unique(stats::na.omit(igraph::V(g0)$clust)))
  if (LVC > 12L) {
    scale_color_custom <- ggplot2::scale_color_manual(
      "cluster",
      values = grDevices::colorRampPalette(RColorBrewer::brewer.pal(12, "Set3"))(LVC),
      guide = "none"
    )
  } else {
    scale_color_custom <- ggplot2::scale_color_brewer(
      "cluster", palette = "Set3", guide = "none"
    )
  }

  ecol <- igraph::E(g0)$color
  has_ecolor <- !is.null(ecol) && length(ecol) && !all(is.na(ecol))
  if (!has_ecolor) {
    # Fall back to signed association from weight if qgraph did not carry colour
    ww <- igraph::E(g0)$weight
    if (is.null(ww)) ww <- rep(0, igraph::ecount(g0))
    ecol <- ifelse(ww >= 0, "#2ca02c", "#d62728")
    igraph::E(g0)$color <- ecol
    tidy_net <- tidygraph::as_tbl_graph(g0)
  }

  gg <- ggraph::ggraph(tidy_net, layout = "linear", circular = TRUE) +
    ggraph::geom_edge_arc0(
      ggplot2::aes(color = .data$color, alpha = abs(.data$weight)),
      alpha = 0.7
    ) +
    ggraph::geom_node_point(
      ggplot2::aes(
        color = as.character(.data$clust),
        x = .data$x * 1.1, y = .data$y * 1.1
      )
    ) +
    ggraph::geom_node_text(
      ggplot2::aes(
        label = .data$display,
        x = .data$x * 1.2, y = .data$y * 1.2,
        fontface = .data$fontface
      ),
      angle = angle, hjust = hjust, size = 2.8
    ) +
    ggraph::theme_graph(base_family = "sans") +
    ggplot2::coord_fixed() +
    scale_color_custom +
    ggraph::scale_edge_color_identity(guide = "none") +
    ggraph::scale_edge_alpha_continuous("correlation", guide = "none", range = c(0.5, 1)) +
    ggplot2::expand_limits(x = c(-3.2, 3.2), y = c(-3.2, 3.2)) +
    ggplot2::theme(
      text = ggplot2::element_text(family = "sans"),
      plot.background = ggplot2::element_rect(fill = "white", colour = NA)
    )
  gg
}

run_netcomi_ggraph <- function(ps, outdir, measure = "sparcc", thresh = 0.3,
                              seed = 13075L) {
  require_netcomi()
  message("netcomi-ggraph: NetCoMi → tidygraph → circular ggraph (Metaanalyse tidygraphize)")
  set.seed(as.integer(seed))
  args <- list(
    data = ps,
    filtTax = "numbSamp",
    filtTaxPar = list(numbSamp = 0.1),
    measure = measure,
    dissFunc = "signed",
    thresh = as.numeric(thresh),
    sparsMethod = "threshold",
    verbose = 1,
    seed = as.integer(seed)
  )
  if (identical(measure, "sparcc")) args$zeroMethod <- "pseudo"
  netcomi_net <- do.call(NetCoMi::netConstruct, args)
  props <- NetCoMi::netAnalyze(
    netcomi_net, clustMethod = "cluster_fast_greedy", hubPar = "eigenvector"
  )
  asso <- netcomi_net$assoMat1
  if (is.null(asso)) asso <- netcomi_net$adjaMat1
  adj_path <- write_adj(asso, file.path(outdir, paste0("adjacency_netcomi_ggraph_", measure, ".tsv")))
  gg <- tidygraphize_netcomi(props, ps = ps)
  figs <- save_gg(gg, file.path(outdir, "network_netcomi_ggraph"), width = 10, height = 10)
  saveRDS(list(net = netcomi_net, props = props), file.path(outdir, "netcomi_ggraph_net.rds"))
  list(
    adjacency = adj_path, figures = figs,
    net_rds = file.path(outdir, "netcomi_ggraph_net.rds"),
    measure = measure, thresh = thresh, method = "netcomi-ggraph"
  )
}

run_tidygraph_chord <- function(ps, outdir, k_means = 5L, coenf_level = 0.7) {
  message("tidygraph: aRchiteutis df2chord with tip_rank-aware fonts")
  suppressPackageStartupMessages({
    library(corrplot)
    library(tidyr)
    library(ggraph)
    library(viridis)
    library(igraph)
  })
  m <- otu_matrix_taxa_rows(ps)
  labs <- tip_label_table(ps)
  # Unique rownames for cor(); keep otu→display map
  rn <- labs$display[match(rownames(m), labs$otu)]
  rn[is.na(rn)] <- rownames(m)[is.na(rn)]
  # collision-safe without dots
  if (anyDuplicated(rn)) {
    for (d in unique(rn[duplicated(rn) | duplicated(rn, fromLast = TRUE)])) {
      idx <- which(rn == d)
      rn[idx] <- paste0(rn[idx], " ASV", seq_along(idx))
    }
  }
  rownames(m) <- rn
  face <- labs$fontface[match(sub(" ASV[0-9]+$", "", rn), sub(" ASV[0-9]+$", "", labs$display))]
  # better match by otu order
  face <- labs$fontface[match(taxa_names(ps), labs$otu)]
  names(face) <- rn

  df <- as.data.frame(m)
  # Build chord like df2chord but fontface not all italic
  corm <- stats::cor(t(as.matrix(df)), use = "pairwise.complete.obs")
  df_order <- corrplot::corrMatOrder(corm, order = "hclust")
  corhcl <- stats::hclust(stats::dist(corm))
  groups <- stats::cutree(corhcl, k = as.integer(k_means))
  corm <- corm[df_order, df_order]
  groups <- groups[df_order]
  rdf <- colnames(corm)
  ldfr <- length(rdf)
  h_remove <- c()
  for (i in seq_len(ldfr)) {
    h_remove <- c(h_remove, (ldfr * (i - 1) + 1):(ldfr * (i - 1) + (i - 1)))
  }
  h_remove <- h_remove[-(1:2)]
  vertices <- tidyr::pivot_longer(as.data.frame(corm), cols = dplyr::all_of(rdf))
  angle <- 90 - 360 * 0:(ldfr - 1) / ldfr
  hjust <- ifelse(angle < -90, 1, 0)
  angle <- ifelse(angle < -90, angle + 180, angle)
  vals <- vertices[-h_remove, 2] %>% unlist()
  if (!isFALSE(coenf_level) && !identical(coenf_level, FALSE)) {
    vals[abs(vals) < as.numeric(coenf_level)] <- NA
  }
  gg_net <- igraph::graph_from_adjacency_matrix(corm, mode = "undirected", weighted = TRUE)
  face_ord <- unname(face[rdf])
  face_ord[is.na(face_ord)] <- "plain"

  gg <- ggraph::ggraph(gg_net, layout = "linear", circular = TRUE) +
    ggraph::geom_edge_arc(aes(alpha = (vals)^2, color = vals), show.legend = FALSE) +
    ggraph::geom_node_point(
      aes(x = x * 1.05, y = y * 1.05, color = as.character(groups)),
      show.legend = FALSE
    ) +
    ggraph::geom_node_text(
      aes(
        x = x * 1.1, y = y * 1.1, label = name,
        angle = angle, hjust = hjust,
        fontface = face_ord
      )
    ) +
    ggplot2::scale_color_manual(values = viridis::viridis(as.integer(k_means))) +
    ggraph::scale_edge_color_gradient2(
      low = "red", mid = "white", high = "blue", na.value = "transparent"
    ) +
    ggraph::scale_edge_alpha_continuous(range = c(0, 0.5)) +
    ggplot2::coord_fixed() +
    ggplot2::theme_void() +
    ggplot2::expand_limits(x = c(-3, 3), y = c(-3, 3))

  adj_path <- write_adj(corm, file.path(outdir, "adjacency_chord.tsv"))
  figs <- save_gg(gg, file.path(outdir, "network_chord"), width = 10, height = 10)
  list(adjacency = adj_path, figures = figs, k_means = k_means, coenf_level = coenf_level)
}

get_centr <- function(graph_obj) {
  data.frame(
    Degree = igraph::degree(graph_obj),
    Betweenness = igraph::betweenness(graph_obj),
    Closeness = igraph::closeness(graph_obj, normalized = TRUE),
    Eigenvector = igraph::eigen_centrality(graph_obj)$vector,
    row.names = igraph::V(graph_obj)$name
  )
}

run_igraph <- function(ps, outdir, thresh = 0.3, seed = 123L, label_mode = "shadowtext") {
  # Alias: builds SparCC netcomi-style graph (combined with NetCoMi viz)
  message("igraph: delegated to NetCoMi SparCC + phylum/edge coloring")
  run_netcomi(
    ps, outdir, measure = "sparcc", thresh = thresh, seed = seed,
    label_mode = label_mode
  )
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

run_network <- function(
    rds = NULL,
    outdir = "test/network/run",
    method = "netcomi",
    top_n = NULL,
    min_mean_rel = 1e-4,
    thresh = NULL,
    measure = "sparcc",
    k_means = 5L,
    coenf_level = 0.7,
    seed = 123L,
    label_mode = "shadowtext"
) {
  setwd(project_root())
  ensure_dir(outdir)
  loaded <- resolve_input_rds(rds)
  ps0 <- ensure_finalized_taxonomy(loaded$ps)
  ps <- filter_taxa_abundance(
    ps0,
    min_mean_rel = as.numeric(min_mean_rel),
    top_n = top_n
  )
  message("Using ", ntaxa(ps), " taxa / ", nsamples(ps), " samples")

  methods <- if (identical(tolower(method), "all")) {
    c("coexistence", "netcomi", "netcomi-ggraph", "tidygraph")
  } else {
    trimws(strsplit(method, ",", fixed = TRUE)[[1]])
  }

  results <- list()
  for (m in methods) {
    m <- tolower(m)
    if (m == "coexistence") {
      th <- thresh %||% 0.3
      results[[m]] <- run_coexistence(
        ps, outdir, thresh = th, seed = seed, label_mode = label_mode
      )
    } else if (m == "netcomi") {
      th <- thresh %||% 0.3
      results[[m]] <- run_netcomi(
        ps, outdir, measure = measure, thresh = th, seed = seed,
        label_mode = label_mode
      )
    } else if (m %in% c("netcomi-ggraph", "ggraph", "tidygraphize")) {
      th <- thresh %||% 0.3
      results[["netcomi-ggraph"]] <- run_netcomi_ggraph(
        ps, outdir, measure = measure, thresh = th, seed = seed
      )
    } else if (m %in% c("tidygraph", "chord", "df2chord")) {
      results[["tidygraph"]] <- run_tidygraph_chord(
        ps, outdir, k_means = k_means, coenf_level = coenf_level
      )
    } else if (m == "igraph") {
      th <- thresh %||% 0.3
      results[[m]] <- run_igraph(
        ps, outdir, thresh = th, seed = seed, label_mode = label_mode
      )
    } else {
      fail("Unknown method: ", m,
           " (coexistence|netcomi|netcomi-ggraph|tidygraph|igraph|all)")
    }
  }

  report <- list(
    input_rds = loaded$meta$path,
    notes = loaded$notes,
    n_taxa_input = ntaxa(ps0),
    n_taxa_used = ntaxa(ps),
    n_samples = nsamples(ps),
    min_mean_rel = as.numeric(min_mean_rel),
    top_n = top_n,
    methods = methods,
    measure = measure,
    label_mode = label_mode,
    seed = as.integer(seed),
    results = results
  )
  write_json(report, file.path(outdir, "network-report.json"))
  message("Network OK: methods=", paste(methods, collapse = ","), " outdir=", outdir)
  invisible(report)
}

self_test <- function() {
  setwd(project_root())
  rds <- "test/code-review-phyloseq/grazing_phyloseq_rare.rds"
  out1 <- "test/network/grazing-self-test-chord"
  r1 <- run_network(
    rds = rds, outdir = out1, method = "tidygraph",
    min_mean_rel = 1e-4, top_n = 30L, coenf_level = 0.5
  )
  if (!file.exists(r1$results$tidygraph$figures$pdf)) stop("missing chord PDF")
  out2 <- "test/network/grazing-self-test-netcomi"
  r2 <- run_network(
    rds = rds, outdir = out2, method = "netcomi", measure = "sparcc",
    min_mean_rel = 1e-4, top_n = 25L, thresh = 0.3
  )
  if (is.null(r2$results$netcomi$figures$main)) stop("missing netcomi igraph fig")
  out_g <- "test/network/grazing-self-test-netcomi-ggraph"
  rg <- run_network(
    rds = rds, outdir = out_g, method = "netcomi-ggraph", measure = "sparcc",
    min_mean_rel = 1e-4, top_n = 20L, thresh = 0.3
  )
  if (!file.exists(rg$results[["netcomi-ggraph"]]$figures$pdf)) {
    stop("missing netcomi-ggraph PDF")
  }
  out3 <- "test/network/grazing-self-test-coexistence"
  r3 <- run_network(
    rds = rds, outdir = out3, method = "coexistence",
    min_mean_rel = 1e-4, top_n = 20L, thresh = 0.3
  )
  if (!file.exists(r3$results$coexistence$adjacency)) stop("missing coexistence adj")
  message("SELF-TEST OK")
  invisible(list(chord = r1, netcomi = r2, ggraph = rg, coexistence = r3))
}

main <- function() {
  args <- parse_kv_args()
  if (isTRUE(args$self_test)) {
    self_test()
    return(invisible(0))
  }
  top_n <- if (!is.null(args$top_n) && nzchar(args$top_n)) as.integer(args$top_n) else NULL
  run_network(
    rds = args$rds,
    outdir = args$outdir %||% "test/network/run",
    method = args$method %||% "netcomi",
    top_n = top_n,
    min_mean_rel = as.numeric(args$min_mean_rel %||% 1e-4),
    thresh = if (!is.null(args$thresh)) as.numeric(args$thresh) else NULL,
    measure = args$measure %||% "sparcc",
    k_means = as.integer(args$k_means %||% 5L),
    coenf_level = as.numeric(args$coenf_level %||% 0.7),
    seed = as.integer(args$seed %||% 123L),
    label_mode = args$label_mode %||% "shadowtext"
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
