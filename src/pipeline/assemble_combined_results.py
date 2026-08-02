#!/usr/bin/env python3
"""Assemble a self-contained master results markdown.

Sources (verified paths only):
  - ready_legnet / ready_caduceus ``parse.md`` + ``parse_data_stats.json``
  - docs/run_results_{direct,adversarial}.md (from collect_run_results)
  - docs/best_models_compare_report.md (from compare_best_models)
  - results/embed_legnet homology ranking + pairwise / validation sidecars
  - figure paths under figures/best_models_compare and figures/presentation

Writes ``docs/combined_results.md`` by default. Observations only; gaps go to
Missing Information.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

SKIP_NAME_TOKENS = (
    "ARCHIVED",
    "FAILED",
    "BAD_",
    "adversarial_FAILED",
    "_hung_",
    "probe",
    "legacy_legnet",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _strip_md_title(body: str) -> str:
    """Drop a leading H1 so sections nest under our H2."""
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines).rstrip() + "\n"


def _panel_section(root: Path, name: str) -> tuple[str, list[str]]:
    """Return markdown block + missing notes for one ready panel."""
    panel = root / name
    missing: list[str] = []
    lines = [f"### `{name}`", ""]
    if not panel.is_dir():
        missing.append(f"Panel directory missing: `{name}`")
        lines.append(f"_Panel directory not found: `{panel}`._")
        lines.append("")
        return "\n".join(lines), missing

    parse_md = panel / "parse.md"
    stats_path = panel / "parse_data_stats.json"
    parse_body = _read_text(parse_md)
    if parse_body is None:
        missing.append(f"`{name}/parse.md` missing or unreadable")
        lines.append("_`parse.md` missing._")
        lines.append("")
    else:
        lines.append(f"Source: `{parse_md.relative_to(root)}`")
        lines.append("")
        lines.append(_strip_md_title(parse_body).rstrip())
        lines.append("")

    stats = _load_json(stats_path)
    if stats is None:
        missing.append(f"`{name}/parse_data_stats.json` missing or unreadable")
        lines.append("_`parse_data_stats.json` missing._")
        lines.append("")
    else:
        lines.append(f"Source: `{stats_path.relative_to(root)}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(stats, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines), missing


def _discover_metric_gaps(root: Path) -> list[str]:
    """best_meta without best_split; Caduceus missing metrics_summary.md; empty ZSV."""
    notes: list[str] = []
    unif = root / "runs_unif"
    if not unif.is_dir():
        notes.append("`runs_unif/` missing — cannot audit train-dir metric coverage.")
        return notes

    no_bs: list[str] = []
    no_md_cad: list[str] = []
    no_zsv: list[str] = []

    for fam in ("legnet", "caduceus"):
        fam_dir = unif / fam
        if not fam_dir.is_dir():
            continue
        for run in sorted(fam_dir.iterdir()):
            if not run.is_dir() or run.name.startswith("."):
                continue
            if any(tok in run.name for tok in SKIP_NAME_TOKENS):
                continue
            for stage, tdir in (
                ("direct", run / "direct"),
                ("adv", run / "adversarial" / "train"),
            ):
                if not (tdir / "best_model" / "best_meta.json").is_file():
                    continue
                label = f"`runs_unif/{fam}/{run.name}:{stage}`"
                if not (tdir / "best_split_metrics.json").is_file():
                    no_bs.append(label)
                if fam == "caduceus" and not (tdir / "metrics_summary.md").is_file():
                    no_md_cad.append(label)
                zsv_paths = (
                    tdir / "zero_shot_metrics.json",
                    tdir / "zsv" / "zero_shot_metrics.json",
                )
                bs = _load_json(tdir / "best_split_metrics.json") or {}
                has_zsv_file = any(p.is_file() for p in zsv_paths)
                zsv_in_bs = False
                if isinstance(bs, dict):
                    sp = bs.get("spearman")
                    if isinstance(sp, dict) and sp.get("zsv") is not None:
                        zsv_in_bs = True
                    for key in ("metrics_by_split", "metrics", "splits"):
                        block = bs.get(key)
                        if isinstance(block, dict) and isinstance(block.get("zsv"), dict):
                            zsv_in_bs = True
                            break
                if not has_zsv_file and not zsv_in_bs:
                    no_zsv.append(label)

    if no_bs:
        notes.append(
            f"**`best_model` present but `best_split_metrics.json` missing ({len(no_bs)}):** "
            + ", ".join(no_bs)
        )
    if no_md_cad:
        notes.append(
            f"**Caduceus train dirs without `metrics_summary.md` ({len(no_md_cad)}):** "
            + ", ".join(no_md_cad)
            + ". Caduceus metrics are taken from `best_split_metrics.json` / best-epoch jsonl when present."
        )
    if no_zsv:
        notes.append(
            f"**No ZSV metrics file and no `zsv` key in `best_split_metrics.json` ({len(no_zsv)}):** "
            + ", ".join(no_zsv)
        )
    return notes


def _embed_section(root: Path) -> tuple[str, list[str]]:
    missing: list[str] = []
    emb = root / "results" / "embed_legnet"
    lines = ["## Embed analyses (LegNet)", ""]
    if not emb.is_dir():
        missing.append("`results/embed_legnet/` missing")
        lines.append("_Embed results directory not found._")
        lines.append("")
        return "\n".join(lines), missing

    complete = _load_json(emb / "run_complete.json")
    if complete:
        lines.append(f"Source: `results/embed_legnet/run_complete.json`")
        lines.append("")
        lines.append(
            f"- finished_at: `{complete.get('finished_at', '—')}`"
        )
        stages = complete.get("stages")
        if isinstance(stages, list):
            lines.append(f"- stages: {', '.join(str(s) for s in stages)}")
        if "n_runs_discovered" in complete:
            lines.append(f"- n_runs_discovered: {complete['n_runs_discovered']}")
        lines.append("")
    else:
        missing.append("`results/embed_legnet/run_complete.json` missing or unreadable")

    validation = _load_json(emb / "validation_report.json")
    if validation:
        lines.append("Source: `results/embed_legnet/validation_report.json`")
        lines.append("")
        lines.append(
            f"- n_total={validation.get('n_total')} n_ready={validation.get('n_ready')} "
            f"n_skipped={validation.get('n_skipped')} n_failed={validation.get('n_failed')}"
        )
        runs = validation.get("runs")
        if isinstance(runs, list):
            for r in runs:
                if not isinstance(r, dict):
                    continue
                lines.append(
                    f"- run `{r.get('key')}`: status={r.get('status')} "
                    f"n_train={r.get('n_train')} n_test={r.get('n_test')} n_val={r.get('n_val')}"
                )
        lines.append("")
    else:
        missing.append("`results/embed_legnet/validation_report.json` missing or unreadable")

    ranking = emb / "homology_dissim" / "ranking.tsv"
    lines.append("### Homology embedding dissimilarity ranking (`D_hom_emb`, pooled)")
    lines.append("")
    lines.append(f"Source: `{ranking.relative_to(root)}`")
    lines.append("")
    # Compact primary columns for readability
    md = None
    if ranking.is_file():
        try:
            with ranking.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                fieldnames = reader.fieldnames or []
                keep = [
                    c
                    for c in (
                        "rank",
                        "run",
                        "split_method",
                        "mean_d_ortho",
                        "mean_d_para",
                        "D_hom_emb",
                        "coverage",
                        "n_og",
                        "n_pg",
                    )
                    if c in fieldnames
                ]
                rows = list(reader)
        except OSError:
            rows = []
            keep = []
        if keep and rows:
            lines_tbl = [
                "| " + " | ".join(keep) + " |",
                "| " + " | ".join("---" for _ in keep) + " |",
            ]
            for row in rows:
                cells = []
                for c in keep:
                    raw = row.get(c, "")
                    try:
                        v = float(raw)
                        if c in {"rank", "n_og", "n_pg"}:
                            cells.append(str(int(v)))
                        else:
                            cells.append(f"{v:.4f}")
                    except (TypeError, ValueError):
                        cells.append(str(raw).replace("|", "\\|"))
                lines_tbl.append("| " + " | ".join(cells) + " |")
            md = "\n".join(lines_tbl) + "\n"
    if md is None:
        missing.append("`results/embed_legnet/homology_dissim/ranking.tsv` missing or empty")
        lines.append("_Homology ranking TSV not available._")
        lines.append("")
    else:
        lines.append(md)

    pairwise = emb / "pairwise" / "pairwise_compare.tsv"
    lines.append("### Pairwise embed compare")
    lines.append("")
    if pairwise.is_file():
        # count rows only; full matrix is large
        try:
            with pairwise.open(encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1
        except OSError:
            n = -1
        lines.append(f"Source: `{pairwise.relative_to(root)}` ({max(n, 0)} pair rows).")
        lines.append("")
        lines.append(
            "Full pairwise RSA/CKA/Procrustes table retained as TSV; not inlined here."
        )
        lines.append("")
    else:
        missing.append("`results/embed_legnet/pairwise/pairwise_compare.tsv` missing")
        lines.append("_Pairwise compare TSV not found._")
        lines.append("")

    return "\n".join(lines), missing


def _figure_section(root: Path) -> str:
    lines = ["## Figures", ""]
    globs = [
        root / "figures" / "best_models_compare",
        root / "figures" / "presentation",
    ]
    found = False
    for d in globs:
        if not d.is_dir():
            continue
        files = sorted(
            p
            for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".svg", ".png"}
        )
        if not files:
            continue
        found = True
        lines.append(f"### `{d.relative_to(root)}`")
        lines.append("")
        for p in files:
            lines.append(f"- `{p.relative_to(root)}`")
        lines.append("")
    if not found:
        lines.append("_No PDF/SVG/PNG figures found under `figures/best_models_compare` or `figures/presentation`._")
        lines.append("")
    return "\n".join(lines)


def assemble(root: Path) -> str:
    missing: list[str] = []
    gen = _utc_now()
    parts: list[str] = [
        "# Combined results (ready_legnet / ready_caduceus)",
        "",
        f"Generation date: {gen}",
        "",
        "Self-contained master pack of panel provenance, training result tables, "
        "best-checkpoint Spearman comparison, and LegNet embed analyses. "
        "Values are copied from the listed source artifacts; no numbers are invented.",
        "",
        "## Source inventory",
        "",
        "| Artifact | Role |",
        "|----------|------|",
        "| `ready_legnet/parse.md`, `parse_data_stats.json` | LegNet panel preprocess |",
        "| `ready_caduceus/parse.md`, `parse_data_stats.json` | Caduceus panel preprocess |",
        "| `docs/run_results_direct.md` | Direct train metrics table |",
        "| `docs/run_results_adversarial.md` | Adversarial train metrics table |",
        "| `docs/best_models_compare_report.md` | Best-checkpoint Spearman |",
        "| `results/embed_legnet/homology_dissim/ranking.tsv` | `D_hom_emb` ranking |",
        "| `results/embed_legnet/pairwise/pairwise_compare.tsv` | Pairwise embed compare |",
        "| `results/embed_legnet/run_complete.json`, `validation_report.json` | Embed pipeline status |",
        "| `figures/best_models_compare/`, `figures/presentation/` | Figure paths |",
        "",
        "## Panel preparation",
        "",
    ]

    for name in ("ready_legnet", "ready_caduceus"):
        block, miss = _panel_section(root, name)
        parts.append(block)
        missing.extend(miss)

    # Direct / adversarial
    for title, rel in (
        ("Direct training results", "docs/run_results_direct.md"),
        ("Adversarial training results", "docs/run_results_adversarial.md"),
    ):
        path = root / rel
        body = _read_text(path)
        parts.append(f"## {title}")
        parts.append("")
        if body is None:
            missing.append(f"`{rel}` missing — run `python -m src.pipeline.collect_run_results`")
            parts.append(f"_Source `{rel}` not found._")
            parts.append("")
        else:
            parts.append(f"Source: `{rel}`")
            parts.append("")
            parts.append(_strip_md_title(body))

    # Best-checkpoint
    best_rel = "docs/best_models_compare_report.md"
    best_body = _read_text(root / best_rel)
    parts.append("## Best-checkpoint Spearman comparison")
    parts.append("")
    if best_body is None:
        missing.append(
            f"`{best_rel}` missing — run `python -m src.pipeline.compare_best_models`"
        )
        parts.append(f"_Source `{best_rel}` not found._")
        parts.append("")
    else:
        parts.append(f"Source: `{best_rel}`")
        parts.append("")
        parts.append(_strip_md_title(best_body))

    emb_block, emb_miss = _embed_section(root)
    parts.append(emb_block)
    missing.extend(emb_miss)

    parts.append(_figure_section(root))

    # Coverage audit
    missing.extend(_discover_metric_gaps(root))

    parts.append("## Missing Information Report")
    parts.append("")
    if not missing:
        parts.append("No gaps detected by the assembler audit.")
        parts.append("")
    else:
        parts.append(
            "The following items were missing, unreadable, or incomplete at generation time. "
            "No substitute values were inserted."
        )
        parts.append("")
        for i, note in enumerate(missing, 1):
            parts.append(f"{i}. {note}")
        parts.append("")
        parts.append(
            "GPU `best_split_metrics` repredict was **not** run in this assembly pass "
            "(per combined-results plan)."
        )
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Default: <root>/docs/combined_results.md",
    )
    args = ap.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or (root / "docs" / "combined_results.md")).resolve()
    text = assemble(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
