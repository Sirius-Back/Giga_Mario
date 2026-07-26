# UpSet — reference

## Default: target only

User requirement: **combine samples only by target variable**.

```r
# Equivalent to MicrobiotaProcess::get_upset(ps, factorNames = target)
# Taxon × target-level presence (1 if any sample in that level has count > 0)
```

Do not build PacBio-style `full = paste(genus, sex.stage, "(", region, ")")` unless `--factors` is set.

## Codebase sources

### PacBio / Kristina (`Article.Rmd`)

```r
upsetda <- get_upset(obj = ps, factorNames = "full")  # multi-factor in those papers
ComplexUpset::upset(
  upsetda,
  intersect = colnames(upsetda)[1:n],
  stripes = "white",
  min_size = 3,
  base_annotations = list(... intersection_size ...),
  set_sizes = upset_set_size() + ...,
  queries = uq_list,
  sort_intersections_by = c("degree", "cardinality"),
  sort_intersections = "ascending"
)
```

### ticks_metaanalyse.Rmd

Same `ComplexUpset::upset` API on a pre-built cohort 0/1 table; `uq <- function(N) upset_query(set = ..., fill = ...)`.

## Escape hatch

`--factors col1,col2` → temporary sample_data column of pasted levels → `get_upset` on that column (PacBio multi-group mode).
