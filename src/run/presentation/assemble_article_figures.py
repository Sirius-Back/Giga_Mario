#!/usr/bin/env python3
"""Assemble curated Nature-style article figure set under figures/article/.

1. Regenerates clean Fig. 1 / 1b / 5 / 6 via ``plot_article_clean``
   (readable Spearman + LegNet pairwise embed heatmaps).
2. Copies presentation Fig. 2–4 and Extended Data sources.
3. Writes MANIFEST.md.

Usage::

  python -m src.run.presentation.assemble_article_figures
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Copied (not regenerated) main-text figures
COPY_MAIN: tuple[tuple[str, str, str, str], ...] = (
    (
        "Fig02_split_stratification",
        "Fig. 2",
        "Fold OG/PG stratification (sd_random, L_hom) vs embedding D_hom_emb",
        "figures/presentation/split_stratification_en",
    ),
    (
        "Fig03_paralog_ortholog_dissim",
        "Fig. 3",
        "LegNet embedding D_hom_emb ranking (paralog vs ortholog geometry)",
        "figures/presentation/paralog_ortholog_dissim_en",
    ),
    (
        "Fig04_leak_breaks_early_stopping",
        "Fig. 4",
        "Leakage / early-stopping diagnostic for homolog-aware splits",
        "figures/presentation/leak_breaks_early_stopping_en",
    ),
)

# Produced by plot_article_clean into outdir (stems already final names)
CLEAN_MAIN: tuple[tuple[str, str, str], ...] = (
    (
        "Fig01_best_checkpoint_spearman",
        "Fig. 1",
        "Clean horizontal best-checkpoint Spearman (test vs ZSV) by model family",
    ),
    (
        "Fig01b_test_vs_zsv_scatter",
        "Fig. 1b",
        "Test vs ZSV Spearman scatter (direct runs)",
    ),
    (
        "Fig05_embed_pairwise_rsa_pooled",
        "Fig. 5",
        "LegNet pairwise RSA heatmap (pooled, centered cosine)",
    ),
    (
        "Fig06_embed_pairwise_cka_pooled",
        "Fig. 6",
        "LegNet pairwise linear CKA heatmap (pooled)",
    ),
)

ED_FIGURES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "ED_Fig01_best_checkpoint_multimetric",
        "ED Fig. 1",
        "Legacy dense multi-metric best-checkpoint panels (crowded; secondary)",
        ("figures/best_models_compare/Figure_01_best_models_train_val_test_zsv",),
    ),
    (
        "ED_Fig02_zsv_by_model",
        "ED Fig. 2",
        "Zero-shot vertebrate (ZSV) performance by model (train-viz overview)",
        (
            "figures/train-viz/all_completed_direct/Figure_40_zsv_by_model",
            "figures/train-viz/all_completed_direct/Figure_31_zsv_by_model",
            "figures/train-viz/all_completed_adversarial/Figure_31_zsv_by_model",
        ),
    ),
    (
        "ED_Fig03_legacy_spearman_grouped_bars",
        "ED Fig. 3",
        "Legacy grouped-bar Spearman (superseded by Fig. 1 for readability)",
        ("figures/best_models_compare/Figure_02_best_models_spearman",),
    ),
)


def _resolve_stem(root: Path, candidates: tuple[str, ...]) -> Path:
    for rel in candidates:
        stem = root / rel
        if stem.with_suffix(".pdf").is_file() or stem.with_suffix(".svg").is_file():
            return stem
    raise FileNotFoundError(
        "Missing required figure source (need .pdf or .svg): "
        + ", ".join(candidates)
    )


def _copy_stem(src_stem: Path, dst_stem: Path) -> list[Path]:
    written: list[Path] = []
    for ext in (".pdf", ".svg", ".png"):
        src = src_stem.with_suffix(ext)
        if not src.is_file():
            continue
        dst = dst_stem.with_suffix(ext)
        shutil.copy2(src, dst)
        written.append(dst)
    if not written:
        raise FileNotFoundError(f"No .pdf/.svg/.png found for {src_stem}")
    return written


def _formats(outdir: Path, stem: str) -> str:
    exts = []
    for ext in (".pdf", ".svg", ".png"):
        if (outdir / f"{stem}{ext}").is_file():
            exts.append(ext)
    return ", ".join(exts) if exts else "(missing)"


def assemble(root: Path, outdir: Path, *, regenerate_clean: bool = True) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if regenerate_clean:
        from src.run.presentation.plot_article_clean import main as clean_main

        rc = clean_main(["--root", str(root), "-o", str(outdir)])
        if rc != 0:
            raise RuntimeError(f"plot_article_clean failed with code {rc}")

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Article figure manifest",
        "",
        f"Generation date: {gen}",
        "",
        "Curated English Nature-style figure set. Fig. 1 / 1b / 5 / 6 are regenerated",
        "clean plots (`plot_article_clean`); Fig. 2–4 are copied presentation assets.",
        "",
        "| Label | Article file stem | Source | Role | Formats |",
        "|-------|-------------------|--------|------|---------|",
    ]

    for stem, label, role in CLEAN_MAIN:
        if not (outdir / f"{stem}.pdf").is_file() and not (outdir / f"{stem}.svg").is_file():
            raise FileNotFoundError(f"Clean figure missing after regenerate: {stem}")
        lines.append(
            f"| {label} | `{stem}` | `src.run.presentation.plot_article_clean` | {role} | {_formats(outdir, stem)} |"
        )

    for stem, label, role, src_rel in COPY_MAIN:
        src = _resolve_stem(root, (src_rel,))
        _copy_stem(src, outdir / stem)
        lines.append(
            f"| {label} | `{stem}` | `{src.relative_to(root)}` | {role} | {_formats(outdir, stem)} |"
        )

    for stem, label, role, candidates in ED_FIGURES:
        src = _resolve_stem(root, candidates)
        _copy_stem(src, outdir / stem)
        lines.append(
            f"| {label} | `{stem}` | `{src.relative_to(root)}` | {role} | {_formats(outdir, stem)} |"
        )

    lines += [
        "",
        "## Main-text order",
        "",
        "1. Fig. 1 — clean test vs ZSV Spearman (horizontal, by family)",
        "2. Fig. 1b — test vs ZSV scatter",
        "3. Fig. 2 — fold stratification vs embedding scores",
        "4. Fig. 3 — D_hom_emb ranking",
        "5. Fig. 4 — leakage / early stopping",
        "6. Fig. 5 — pairwise embed RSA (pooled)",
        "7. Fig. 6 — pairwise embed CKA (pooled)",
        "",
        "## Extended Data",
        "",
        "1. ED Fig. 1 — legacy dense multi-metric panels",
        "2. ED Fig. 2 — ZSV by model (train-viz)",
        "3. ED Fig. 3 — legacy grouped-bar Spearman (superseded by Fig. 1)",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python -m src.run.presentation.assemble_article_figures",
        "# or clean plots only:",
        "python -m src.run.presentation.plot_article_clean",
        "```",
        "",
    ]
    manifest = outdir / "MANIFEST.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("-o", "--outdir", type=Path, default=None)
    ap.add_argument(
        "--no-regenerate-clean",
        action="store_true",
        help="Only copy presentation/ED assets; keep existing clean Fig.1/5/6",
    )
    args = ap.parse_args(argv)
    root = args.root.resolve()
    outdir = (args.outdir or (root / "figures" / "article")).resolve()
    manifest = assemble(
        root, outdir, regenerate_clean=not args.no_regenerate_clean
    )
    print(f"wrote {manifest}", flush=True)
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            print(f"  {p.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
