#!/usr/bin/env python3
"""setup hook: create or audit Article.Rmd setup (themes, palettes, libs, TARGET/BATCH).

First run (no setup chunk / missing Article.Rmd):
  Create Article.Rmd with a single canonical setup chunk.

Later runs (setup present):
  Ensure plots use theme_main() and setup palettes only; rewrite deviations.

All theme modifications live in setup as theme_main(); plots use `+ theme_main()`.
Palettes default to RColorBrewer families by visualization type (overridable).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BREWER = {
    "discrete": "Set2",  # groups, bars, general categorical
    "composition": "Paired",  # stacked composition
    "sequential": "YlGnBu",  # heatmaps, abundance gradients
    "diverging": "RdYlBu",  # z-scores / signed effects
    "qualitative_large": "Set3",  # many categories (>8)
    "binary": "Dark2",  # two-class contrasts
}

CRAN_LIBS = [
    "tidyverse",
    "ggplot2",
    "ggpubr",
    "RColorBrewer",
    "scales",
    "vegan",
    "ape",
]

BIOC_LIBS = [
    "phyloseq",
]

# Theme constructors that must not appear outside setup (except theme_main / theme())
FORBIDDEN_THEME_CALLS = [
    "theme_bw",
    "theme_minimal",
    "theme_classic",
    "theme_gray",
    "theme_grey",
    "theme_light",
    "theme_dark",
    "theme_void",
    "theme_linedraw",
    "theme_pubr",
    "theme_superlight",
    "theme_honey",
    "theme_classic2",
    "theme_transparent",
]

# Ad-hoc scale helpers → full call rewrite to setup scales (drop foreign args)
ADHOC_SCALE_CALLS = [
    (re.compile(r"scale_fill_brewer\s*\([^)]*\)"), "scale_fill_discrete_main()"),
    (re.compile(r"scale_colour_brewer\s*\([^)]*\)"), "scale_color_discrete_main()"),
    (re.compile(r"scale_color_brewer\s*\([^)]*\)"), "scale_color_discrete_main()"),
    (re.compile(r"scale_fill_viridis_c\s*\([^)]*\)"), "scale_fill_sequential_main()"),
    (re.compile(r"scale_color_viridis_c\s*\([^)]*\)"), "scale_color_sequential_main()"),
    (re.compile(r"scale_colour_viridis_c\s*\([^)]*\)"), "scale_color_sequential_main()"),
    (re.compile(r"scale_fill_viridis_d\s*\([^)]*\)"), "scale_fill_discrete_main()"),
    (re.compile(r"scale_color_viridis_d\s*\([^)]*\)"), "scale_color_discrete_main()"),
    (re.compile(r"scale_colour_viridis_d\s*\([^)]*\)"), "scale_color_discrete_main()"),
    (re.compile(r"scale_fill_distiller\s*\([^)]*\)"), "scale_fill_sequential_main()"),
    (re.compile(r"scale_color_distiller\s*\([^)]*\)"), "scale_color_sequential_main()"),
    (re.compile(r"scale_colour_distiller\s*\([^)]*\)"), "scale_color_sequential_main()"),
]


@dataclass
class SetupReport:
    action: str
    article: str
    created: bool = False
    modified: bool = False
    target: str = "group"
    batch: str = "batch"
    palettes: dict[str, str] = field(default_factory=dict)
    issues_found: list[str] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def project_root() -> Path:
    here = Path(__file__).resolve()
    for cur in [here.parent, *here.parents]:
        if (cur / ".cursor").is_dir() and (
            (cur / "artifact-registry.md").is_file() or (cur / ".cursor" / "skills").is_dir()
        ):
            return cur
    return Path.cwd()


# ---------------------------------------------------------------------------
# Article.Rmd template
# ---------------------------------------------------------------------------

def render_setup_chunk(
    *,
    target: str,
    batch: str,
    palettes: dict[str, str],
    seed: int = 7,
    base_size: int = 11,
) -> str:
    pal = {**DEFAULT_BREWER, **palettes}
    cran = ",\n  ".join(f'"{x}"' for x in CRAN_LIBS)
    bioc = ",\n  ".join(f'"{x}"' for x in BIOC_LIBS)
    return f'''```{{r setup, include=FALSE}}
# =============================================================================
# CANONICAL SETUP — edit themes/palettes/TARGET/BATCH ONLY here
# Plots must use: + theme_main() and setup scale_*_main() / pal_*() helpers
# =============================================================================
set.seed({seed})

knitr::opts_chunk$set(
  echo = FALSE,
  warning = FALSE,
  message = FALSE,
  fig.align = "center",
  dpi = 300
)

# --- Libraries ---------------------------------------------------------------
cran_libs <- c(
  {cran}
)
for (lib in cran_libs) {{
  if (!requireNamespace(lib, quietly = TRUE)) {{
    install.packages(lib)
  }}
  library(lib, character.only = TRUE)
}}

bioc_libs <- c(
  {bioc}
)
if (!requireNamespace("BiocManager", quietly = TRUE)) {{
  install.packages("BiocManager")
}}
for (lib in bioc_libs) {{
  if (!requireNamespace(lib, quietly = TRUE)) {{
    BiocManager::install(lib, ask = FALSE, update = FALSE)
  }}
  library(lib, character.only = TRUE)
}}
rm(lib, cran_libs, bioc_libs)

# --- Analysis design variables ----------------------------------------------
# TARGET: primary grouping / contrast column in sample metadata
# BATCH: secondary blocking / study batch column (e.g. BioProject, plate)
TARGET <- "{target}"
BATCH <- "{batch}"

# Paths (project-relative; override per dataset)
path.data <- "data"
path.results <- "results"
path.figures <- "figures"
path.tables <- "tables"
dir.create(path.results, showWarnings = FALSE, recursive = TRUE)
dir.create(path.figures, showWarnings = FALSE, recursive = TRUE)
dir.create(path.tables, showWarnings = FALSE, recursive = TRUE)

# --- Palettes (RColorBrewer; one family per visualization class) ------------
PAL_DISCRETE <- "{pal['discrete']}"
PAL_COMPOSITION <- "{pal['composition']}"
PAL_SEQUENTIAL <- "{pal['sequential']}"
PAL_DIVERGING <- "{pal['diverging']}"
PAL_QUAL_LARGE <- "{pal['qualitative_large']}"
PAL_BINARY <- "{pal['binary']}"

.brewer_n <- function(n, palette) {{
  n <- as.integer(n)
  if (is.na(n) || n < 1L) n <- 1L
  max_n <- RColorBrewer::brewer.pal.info[palette, "maxcolors"]
  base <- RColorBrewer::brewer.pal(max(3L, min(max_n, max(n, 3L))), palette)
  if (n <= length(base)) {{
    return(base[seq_len(n)])
  }}
  grDevices::colorRampPalette(base)(n)
}}

pal_discrete <- function(n) .brewer_n(n, PAL_DISCRETE)
pal_composition <- function(n) .brewer_n(n, PAL_COMPOSITION)
pal_sequential <- function(n) .brewer_n(n, PAL_SEQUENTIAL)
pal_diverging <- function(n) .brewer_n(n, PAL_DIVERGING)
pal_qual_large <- function(n) .brewer_n(n, PAL_QUAL_LARGE)
pal_binary <- function(n = 2L) .brewer_n(n, PAL_BINARY)

scale_fill_discrete_main <- function(...) {{
  ggplot2::discrete_scale(aesthetics = "fill", palette = pal_discrete, ...)
}}
scale_color_discrete_main <- function(...) {{
  ggplot2::discrete_scale(aesthetics = "colour", palette = pal_discrete, ...)
}}
scale_fill_composition_main <- function(...) {{
  ggplot2::discrete_scale(aesthetics = "fill", palette = pal_composition, ...)
}}
scale_color_composition_main <- function(...) {{
  ggplot2::discrete_scale(aesthetics = "colour", palette = pal_composition, ...)
}}
scale_fill_sequential_main <- function(...) {{
  ggplot2::scale_fill_gradientn(colours = pal_sequential(9L), ...)
}}
scale_color_sequential_main <- function(...) {{
  ggplot2::scale_color_gradientn(colours = pal_sequential(9L), ...)
}}
scale_fill_diverging_main <- function(...) {{
  ggplot2::scale_fill_gradientn(colours = pal_diverging(11L), ...)
}}
scale_color_diverging_main <- function(...) {{
  ggplot2::scale_color_gradientn(colours = pal_diverging(11L), ...)
}}

# --- Theme (ALL modifications here; plots use + theme_main()) ---------------
theme_main <- function(base_size = {base_size}, ...) {{
  ggpubr::theme_pubr(base_size = base_size) +
    ggplot2::theme(
      legend.position = "bottom",
      legend.title = ggplot2::element_text(face = "bold"),
      strip.background = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      ...
    )
}}
ggplot2::theme_set(theme_main())
```'''


def render_article_rmd(
    *,
    target: str,
    batch: str,
    palettes: dict[str, str],
    title: str = "Article",
) -> str:
    setup = render_setup_chunk(target=target, batch=batch, palettes=palettes)
    return f'''---
title: "{title}"
output:
  html_document:
    toc: true
    toc_depth: 3
    fig_caption: true
  pdf_document:
    toc: true
    fig_caption: true
editor_options:
  chunk_output_type: console
---

{setup}

# Overview

This notebook was initialized by the **setup** hook.

- Primary grouping (`TARGET`): see setup chunk
- Batch / study block (`BATCH`): see setup chunk
- Plots must use `+ theme_main()` and setup `scale_*_main()` / `pal_*()` helpers only

```{{r smoke-plot, fig.cap="Setup smoke plot (discrete + theme_main)"}}
# Reproducible smoke check — uses ONLY setup theme/palettes
set.seed(7)
df <- data.frame(
  x = rep(letters[1:4], each = 10),
  y = rnorm(40),
  g = rep(c("A", "B"), 20)
)
ggplot(df, aes(x, y, fill = g)) +
  geom_boxplot(outlier.size = 0.6) +
  scale_fill_discrete_main() +
  labs(x = NULL, y = "value", fill = TARGET) +
  theme_main()
```
'''


# ---------------------------------------------------------------------------
# Parse / audit Article.Rmd
# ---------------------------------------------------------------------------

CHUNK_RE = re.compile(
    r"(?P<header>^```\{r\s+(?P<name>[^,}\s]+)(?P<opts>[^}]*)\}\s*\n)"
    r"(?P<body>.*?)(?P<footer>^```\s*$)",
    re.MULTILINE | re.DOTALL,
)


def find_setup_chunk(text: str) -> Optional[re.Match[str]]:
    for m in CHUNK_RE.finditer(text):
        if m.group("name") == "setup":
            return m
    return None


def extract_assignment(setup_body: str, name: str) -> Optional[str]:
    m = re.search(
        rf'^\s*{re.escape(name)}\s*<-\s*["\']([^"\']+)["\']\s*$',
        setup_body,
        re.MULTILINE,
    )
    return m.group(1) if m else None


def has_theme_main(setup_body: str) -> bool:
    return bool(re.search(r"theme_main\s*<-\s*function", setup_body))


def has_required_palette_helpers(setup_body: str) -> list[str]:
    required = [
        "pal_discrete",
        "pal_composition",
        "pal_sequential",
        "pal_diverging",
        "scale_fill_discrete_main",
        "scale_color_discrete_main",
        "scale_fill_sequential_main",
        "scale_fill_diverging_main",
        "theme_main",
        "TARGET",
        "BATCH",
    ]
    missing = []
    for name in required:
        if name in ("TARGET", "BATCH"):
            if not re.search(rf"^\s*{name}\s*<-", setup_body, re.MULTILINE):
                missing.append(name)
        elif not re.search(rf"{re.escape(name)}\s*<-", setup_body):
            missing.append(name)
    return missing


def split_setup_and_rest(text: str) -> tuple[Optional[str], str, str, Optional[re.Match[str]]]:
    """Return (setup_body, before, after, match) for the setup chunk."""
    m = find_setup_chunk(text)
    if not m:
        return None, text, "", None
    setup_body = m.group("body")
    before = text[: m.start()]
    after = text[m.end() :]
    return setup_body, before, after, m


def audit_and_fix_body(body: str, report: SetupReport) -> str:
    """Rewrite non-setup R code for theme/palette consistency."""
    out = body
    # Replace forbidden full themes with theme_main()
    for name in FORBIDDEN_THEME_CALLS:
        # theme_bw() or theme_bw(...)
        pat = re.compile(rf"\b{name}\s*\([^)]*\)")
        if pat.search(out):
            out2, n = pat.subn("theme_main()", out)
            if n:
                report.issues_found.append(f"found {n}× {name}()")
                report.fixes_applied.append(f"replaced {name}() → theme_main() ({n})")
                out = out2

    # theme_set(something other than theme_main())
    for m in re.finditer(r"theme_set\s*\(([^)]*)\)", out):
        arg = m.group(1).strip()
        if "theme_main" not in arg:
            report.issues_found.append(f"theme_set({arg})")
            out = out.replace(m.group(0), "theme_set(theme_main())")
            report.fixes_applied.append("theme_set(...) → theme_set(theme_main())")

    # Ad-hoc scale_* helpers
    for pat, repl in ADHOC_SCALE_CALLS:
        if pat.search(out):
            out2, n = pat.subn(repl, out)
            if n:
                report.issues_found.append(f"ad-hoc scale matched {pat.pattern} ×{n}")
                report.fixes_applied.append(f"{pat.pattern} → {repl} ({n})")
                out = out2

    # ggplot chains missing theme_main when they already have a theme_* replacement
    # Soft check: ggplot(...) ... without theme_main in same expression block
    ggplot_blocks = re.findall(
        r"ggplot\s*\(.*?$(?:\n(?:.*?)*)?(?=\n\n|\n```|\n# |\Z)",
        out,
        re.MULTILINE,
    )
    # Simpler line-based: if a chunk has ggplot( and geom_ but no theme_main, warn
    if re.search(r"\bgplot\s*\(", out) and "theme_main" not in out:
        # Only auto-append if there is a clear trailing plot assignment ending
        # Prefer warning — auto-append is fragile for multi-plot chunks
        report.warnings.append(
            "ggplot present without theme_main() in this region; prefer `+ theme_main()`"
        )

    # Hard-coded hex palettes in scale_*_manual(values = c("#...
    hex_manual = re.compile(
        r"scale_(fill|color|colour)_manual\s*\(\s*values\s*=\s*c\([^)]*#[0-9A-Fa-f]{3,8}"
    )
    if hex_manual.search(out):
        report.warnings.append(
            "ad-hoc hex colours in scale_*_manual; prefer scale_*_discrete_main() "
            "or pal_*() from setup"
        )

    return out


def ensure_setup_chunk(
    text: str,
    *,
    target: str,
    batch: str,
    palettes: dict[str, str],
    report: SetupReport,
) -> str:
    setup_body, before, after, match = split_setup_and_rest(text)
    new_setup = render_setup_chunk(target=target, batch=batch, palettes=palettes)

    if setup_body is None:
        # Insert after YAML if present
        yaml_end = re.search(r"^---\s*$", text, re.MULTILINE)
        if yaml_end and text.startswith("---"):
            # find closing ---
            m2 = re.search(r"^---\s*\n.*?^---\s*\n", text, re.MULTILINE | re.DOTALL)
            if m2:
                insert_at = m2.end()
                report.fixes_applied.append("inserted missing setup chunk after YAML")
                return text[:insert_at] + "\n" + new_setup + "\n" + text[insert_at:]
        report.fixes_applied.append("prepended setup chunk")
        return new_setup + "\n\n" + text

    missing = has_required_palette_helpers(setup_body)
    needs_replace = False
    if missing:
        report.issues_found.append(f"setup missing: {', '.join(missing)}")
        needs_replace = True
    if not has_theme_main(setup_body):
        report.issues_found.append("setup lacks theme_main()")
        needs_replace = True

    # Preserve user TARGET/BATCH if present and CLI did not force? We use CLI/defaults.
    # If setup incomplete, replace entire setup chunk with canonical one but keep
    # existing TARGET/BATCH string values when found.
    existing_target = extract_assignment(setup_body, "TARGET") or target
    existing_batch = extract_assignment(setup_body, "BATCH") or batch
    report.target = existing_target
    report.batch = existing_batch

    if needs_replace:
        new_setup = render_setup_chunk(
            target=existing_target, batch=existing_batch, palettes=palettes
        )
        assert match is not None
        report.fixes_applied.append("replaced incomplete setup chunk with canonical setup")
        text = before + new_setup + after
        setup_body, before, after, match = split_setup_and_rest(text)

    # Audit non-setup regions chunk by chunk
    pieces: list[str] = []
    last = 0
    assert match is not None
    # Re-find after possible replacement
    setup_match = find_setup_chunk(text)
    assert setup_match is not None
    for m in CHUNK_RE.finditer(text):
        pieces.append(text[last : m.start()])
        name = m.group("name")
        header = m.group("header")
        body = m.group("body")
        footer = m.group("footer")
        if name == "setup":
            pieces.append(header + body + footer)
        else:
            fixed = audit_and_fix_body(body, report)
            pieces.append(header + fixed + footer)
        last = m.end()
    pieces.append(text[last:])
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_setup(
    article: Path,
    *,
    target: str = "group",
    batch: str = "batch",
    palettes: Optional[dict[str, str]] = None,
    title: str = "Article",
    dry_run: bool = False,
) -> SetupReport:
    palettes = {**DEFAULT_BREWER, **(palettes or {})}
    report = SetupReport(
        action="create",
        article=str(article),
        target=target,
        batch=batch,
        palettes=palettes,
    )

    if not article.exists():
        content = render_article_rmd(
            target=target, batch=batch, palettes=palettes, title=title
        )
        report.action = "create"
        report.created = True
        report.fixes_applied.append(f"created {article}")
        if not dry_run:
            article.parent.mkdir(parents=True, exist_ok=True)
            article.write_text(content, encoding="utf-8")
            report.modified = True
        return report

    text = article.read_text(encoding="utf-8")
    setup = find_setup_chunk(text)
    if setup is None:
        report.action = "insert-setup"
        report.issues_found.append("Article.Rmd exists but setup chunk missing")
        new_text = ensure_setup_chunk(
            text, target=target, batch=batch, palettes=palettes, report=report
        )
    else:
        report.action = "audit"
        new_text = ensure_setup_chunk(
            text, target=target, batch=batch, palettes=palettes, report=report
        )

    if new_text != text:
        report.modified = True
        if not dry_run:
            article.write_text(new_text, encoding="utf-8")
            report.fixes_applied.append(f"wrote fixes to {article}")
    else:
        report.fixes_applied.append("no content changes required")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--article",
        type=Path,
        default=None,
        help="Path to Article.Rmd (default: ./Article.Rmd)",
    )
    p.add_argument("--target", default="group", help="TARGET column name")
    p.add_argument("--batch", default="batch", help="BATCH column name")
    p.add_argument(
        "--palette",
        action="append",
        default=[],
        metavar="KEY=BrewerName",
        help="Override palette, e.g. discrete=Set1 (repeatable)",
    )
    p.add_argument("--title", default="Article", help="YAML title when creating")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON report (default: ./test/setup/setup-report.json)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run create+audit fixture test under ./test/setup/",
    )
    return p


def parse_palette_args(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    valid = set(DEFAULT_BREWER)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --palette {item!r}; expected KEY=BrewerName")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in valid:
            raise ValueError(f"Unknown palette key {k!r}; choose from {sorted(valid)}")
        out[k] = v
    return out


def self_test(root: Path) -> int:
    """Create + audit fixtures; return 0 on success."""
    test_dir = root / "test" / "setup"
    test_dir.mkdir(parents=True, exist_ok=True)
    article_new = test_dir / "Article_create.Rmd"
    article_audit = test_dir / "Article_audit.Rmd"
    if article_new.exists():
        article_new.unlink()

    r1 = run_setup(
        article_new,
        target="Condition",
        batch="BioProject",
        palettes={"discrete": "Set1"},
    )
    if not article_new.is_file():
        print("SELF-TEST FAIL: Article not created", file=sys.stderr)
        return 1
    text = article_new.read_text(encoding="utf-8")
    for needle in (
        "theme_main",
        "TARGET <- \"Condition\"",
        "BATCH <- \"BioProject\"",
        "PAL_DISCRETE <- \"Set1\"",
        "scale_fill_discrete_main",
        "theme_pubr",
        "+ theme_main()",
    ):
        if needle not in text:
            print(f"SELF-TEST FAIL: missing {needle!r}", file=sys.stderr)
            return 1

    # Broken notebook for audit path
    broken = '''---
title: "Broken"
---

```{r setup, include=FALSE}
library(ggplot2)
TARGET <- "group"
BATCH <- "batch"
```

```{r bad-plot}
ggplot(mtcars, aes(wt, mpg, colour = factor(cyl))) +
  geom_point() +
  scale_color_brewer(palette = "Dark2") +
  theme_bw()
```
'''
    article_audit.write_text(broken, encoding="utf-8")
    r2 = run_setup(article_audit, target="group", batch="batch")
    fixed = article_audit.read_text(encoding="utf-8")
    if "theme_bw()" in fixed:
        print("SELF-TEST FAIL: theme_bw still present", file=sys.stderr)
        return 1
    if "theme_main()" not in fixed:
        print("SELF-TEST FAIL: theme_main not injected", file=sys.stderr)
        return 1
    if "scale_color_discrete_main(" not in fixed:
        print("SELF-TEST FAIL: scale_color_brewer not rewritten", file=sys.stderr)
        return 1
    if "theme_main <- function" not in fixed:
        print("SELF-TEST FAIL: canonical setup not installed", file=sys.stderr)
        return 1

    # Second audit should be idempotent (no theme_bw left to fix)
    r3 = run_setup(article_audit)
    if any("theme_bw" in x for x in r3.issues_found):
        print("SELF-TEST FAIL: idempotent audit still finds theme_bw", file=sys.stderr)
        return 1

    report_path = test_dir / "setup-report.json"
    report_path.write_text(
        json.dumps(
            {"create": asdict(r1), "audit": asdict(r2), "reaudit": asdict(r3)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("SELF-TEST OK")
    print(f"  created: {article_new}")
    print(f"  audited: {article_audit}")
    print(f"  report:  {report_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()

    if args.self_test:
        return self_test(root)

    try:
        palettes = parse_palette_args(args.palette)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    article = (args.article or (root / "Article.Rmd")).resolve()
    report = run_setup(
        article,
        target=args.target,
        batch=args.batch,
        palettes=palettes,
        title=args.title,
        dry_run=args.dry_run,
    )

    report_path = args.report or (root / "test" / "setup" / "setup-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print(json.dumps(asdict(report), indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
