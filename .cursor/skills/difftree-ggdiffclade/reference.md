# difftree-ggdiffclade — reference

## Exact codebase locations

### Primary (PacBio)

`/mnt/tank/scratch/dsmutin/ixodes/PacBio/Article.Rmd`

| Block | Approx. lines | Content |
|-------|---------------|---------|
| `colors_2` | 185 | `c("lightgreen","pink")` |
| Genus DE | 1876–1970 | `diff_analysis` → `ggdiffclade` → `ggdiffbox` → `ggarrange` |
| Sex DE | 2099–2160+ | Same pattern, `classgroup = "sex.stage"` |

Also mirrored in `/mnt/tank/scratch/dsmutin/ixodes/PacBio/rec.Rmd` (~1601+).

### Copy (Kristina)

`/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/codebase.Rmd`

| Block | Approx. lines |
|-------|---------------|
| Genus DE | 1709–1803 |
| Sex DE | 1932–2026 |

## PacBio core snippet (genus)

```r
deres <- diff_analysis(
  obj = psfr,
  classgroup = "genus",
  mlfun = "lda",
  filtermod = "pvalue",
  firstcomfun = "kruskal_test",
  firstalpha = 0.05,
  strictmod = TRUE,
  secondcomfun = "wilcox_test",
  subclmin = 3,
  subclwilc = TRUE,
  secondalpha = 0.01,
  lda = 3
)

diffclade_genus <- ggdiffclade(
  obj = deres,
  alpha = 0.3, linewd = 0.15, skpointsize = 0.6,
  layout = "radial", taxlevel = 3,
  removeUnknown = TRUE, reduce = TRUE
) +
  scale_fill_diff_cladogram(values = colors_2) +
  theme(...) +
  scale_size_binned("p-value, -lg", range = c(1, 3)) +
  guides(color = guide_none(), fill = guide_legend(position = "bottom"),
         size = guide_legend(position = "bottom"))

# shadowtext_data from layers[[3]]; ggdiffbox; join LDAmean; markdown y labels

gg_DE_genus <- ggarrange(
  diffclade_genus, diffbox,
  align = "hv", widths = c(.7, 1)
)
```

## Skill deviation (legend placement)

PacBio leaves fill/size guides on the **left** cladogram. This skill moves **all** definitions & quantities to the **right** boxplot stack (boxplot group legend + attached cladogram fill/size legend), left panel `legend.position = "none"`.

## Dependencies

- MicrobiotaProcess ≥ 1.18
- coin (exports `kruskal_test` / `wilcox_test` used by `diff_analysis`)
- ggpubr, ggtext, dplyr, stringr, forcats, ggplot2, phyloseq

## Preprocess notes

- Drop `tip_rank` and other non-`Kingdom`…`Species` columns (MP concatenates them into lineages).
- Strip `^[a-z]__` SILVA prefixes.
- Rebuild phyloseq **without** `phy_tree` (avoids ape/tidytree `reorderRcpp` crashes with dual `phylo` class).
- Coerce target to `factor` (integer-coded grazing breaks `rlang::sym` inside MP).
