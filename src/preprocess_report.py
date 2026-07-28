"""Non-agentic preprocess report writer.

Plain Python only: validate preprocess stage artifacts and write ``parse.md``
under an outdir. No LLM calls.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.common import ID_CSV_COLUMNS, read_csv


REQUIRED_ID_COLS = list(ID_CSV_COLUMNS)
PREDICT_ID_COL = "id"
PREDICT_VALUE_COL = "predict_var1"


def _count_ext(directory: Path, pattern: str = "*.ext") -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(pattern) if _.is_file())


def _count_fa(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    n = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".fa", ".fasta"}:
            n += 1
    return n


def _pipe_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        row = next(csv.reader(fh, delimiter="|"), None)
    return list(row or [])


def check_id_csv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "ok": False, "n_rows": 0, "errors": []}
    if not path.is_file():
        out["errors"].append("missing ID.csv")
        return out
    try:
        header = _pipe_header(path)
        rows = read_csv(path)
    except Exception as exc:  # noqa: BLE001 — report, do not invent
        out["errors"].append(f"unreadable: {exc}")
        return out
    missing = [c for c in REQUIRED_ID_COLS if c not in header]
    if missing:
        out["errors"].append(f"missing columns {missing}")
    ids = [r.get("ID", "").strip() for r in rows]
    if any(not i for i in ids):
        out["errors"].append("blank ID values")
    if len(ids) != len(set(ids)):
        out["errors"].append("duplicate ID values")
    out["n_rows"] = len(rows)
    out["ok"] = not out["errors"]
    return out


def check_marked(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "ok": False, "n_files": 0, "errors": []}
    root = path / "MARKED" if (path / "MARKED").is_dir() else path
    if not root.is_dir():
        out["errors"].append("MARKED directory missing")
        return out
    n = _count_fa(root)
    out["n_files"] = n
    if n < 1:
        out["errors"].append("no MARKED FASTA files")
    out["ok"] = not out["errors"]
    return out


def check_parsed(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "ok": False, "n_files": 0, "errors": []}
    root = path / "PARSED" if (path / "PARSED").is_dir() else path
    if not root.is_dir():
        out["errors"].append("PARSED directory missing")
        return out
    n = _count_ext(root)
    out["n_files"] = n
    if n < 1:
        out["errors"].append("no PARSED/*.ext files")
    out["ok"] = not out["errors"]
    return out


def check_predict(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "n_rows": 0,
        "mapped": False,
        "errors": [],
    }
    root = path / "PREDICT" if (path / "PREDICT").is_dir() else path
    csv_path = root / "predict.csv"
    if not csv_path.is_file():
        out["errors"].append("predict.csv missing")
        return out
    try:
        rows = read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"unreadable predict.csv: {exc}")
        return out
    if not rows:
        out["errors"].append("empty predict.csv")
        return out
    if PREDICT_ID_COL not in rows[0]:
        out["errors"].append("predict.csv missing id")
    if PREDICT_VALUE_COL not in rows[0]:
        out["errors"].append("predict.csv missing predict_var1")
    out["mapped"] = "sample_id" in rows[0]
    ids = [r.get("id", "").strip() for r in rows]
    if any(not i for i in ids):
        out["errors"].append("blank predict.csv id")
    # Merged: id unique; mapped source may repeat region id across samples
    if not out["mapped"] and len(ids) != len(set(ids)):
        out["errors"].append("duplicate predict.csv id (merged layout)")
    out["n_rows"] = len(rows)
    out["n_ext"] = _count_ext(root) + sum(
        _count_ext(p) for p in root.iterdir() if p.is_dir()
    )
    out["ok"] = not out["errors"]
    return out


def check_fold_csv(path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path) if path else "", "ok": True, "skipped": path is None, "errors": []}
    if path is None:
        return out
    if not path.is_file():
        out["ok"] = False
        out["errors"].append("fold.csv missing")
        return out
    rows = read_csv(path)
    if not rows or "ID" not in rows[0] or "fold" not in rows[0]:
        out["ok"] = False
        out["errors"].append("fold.csv needs ID and fold columns")
        return out
    out["n_rows"] = len(rows)
    out["ok"] = True
    return out


def collect_preprocess_checks(
    *,
    outdir: Path,
    id_csv: Path | None = None,
    require_fold: bool = False,
) -> dict[str, Any]:
    """Inspect a preprocess outdir (or stage subdirs) and return structured checks."""
    outdir = Path(outdir)
    id_path = Path(id_csv) if id_csv else outdir / "ID.csv"
    if not id_path.is_file():
        # common layout: outdir/id_gen/ID.csv
        alt = outdir / "id_gen" / "ID.csv"
        if alt.is_file():
            id_path = alt
    fold_path = outdir / "fold.csv"
    if not fold_path.is_file():
        alt_fold = outdir / "generate_fold" / "fold.csv"
        fold_path = alt_fold if alt_fold.is_file() else fold_path

    checks = {
        "outdir": str(outdir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "id_csv": check_id_csv(id_path),
        "marked": check_marked(outdir / "adapt" if (outdir / "adapt" / "MARKED").is_dir() else outdir),
        "parsed": check_parsed(
            outdir / "parse_data" if (outdir / "parse_data" / "PARSED").is_dir() else outdir
        ),
        "predict": check_predict(
            next(
                (
                    p
                    for p in (
                        outdir / "parse_target",
                        outdir / "parse_target_merged",
                        outdir / "parse_target_mapped",
                        outdir,
                    )
                    if (p / "PREDICT").is_dir() or (p / "predict.csv").is_file()
                ),
                outdir,
            )
        ),
        "fold": check_fold_csv(fold_path if fold_path.is_file() or require_fold else None),
    }
    checks["ok"] = all(
        checks[k].get("ok", False) or checks[k].get("skipped")
        for k in ("id_csv", "marked", "parsed", "predict", "fold")
    )
    return checks


def render_parse_md(checks: dict[str, Any]) -> str:
    """Render plain markdown for ``parse.md`` (no agentic content)."""
    lines = [
        "# Preprocess report (`parse.md`)",
        "",
        f"- Generated (UTC): {checks.get('generated_at', '')}",
        f"- Outdir: `{checks.get('outdir', '')}`",
        f"- Overall OK: {checks.get('ok', False)}",
        "",
        "## Checks",
        "",
    ]
    for key in ("id_csv", "marked", "parsed", "predict", "fold"):
        block = checks.get(key) or {}
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(f"- path: `{block.get('path', '')}`")
        lines.append(f"- ok: {block.get('ok', False)}")
        if block.get("skipped"):
            lines.append("- skipped: true")
        for field in ("n_rows", "n_files", "n_ext", "mapped"):
            if field in block:
                lines.append(f"- {field}: {block[field]}")
        errs = block.get("errors") or []
        if errs:
            lines.append("- errors:")
            for err in errs:
                lines.append(f"  - {err}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This file is produced by `src.preprocess_report` (plain code). "
        "It does not invent biological conclusions."
    )
    lines.append("")
    return "\n".join(lines)


def write_parse_md(
    outdir: Path,
    *,
    id_csv: Path | None = None,
    require_fold: bool = False,
    filename: str = "parse.md",
) -> Path:
    """Validate preprocess artifacts and write ``{outdir}/parse.md``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checks = collect_preprocess_checks(
        outdir=outdir, id_csv=id_csv, require_fold=require_fold
    )
    path = outdir / filename
    path.write_text(render_parse_md(checks), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write non-agentic preprocess parse.md")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--id-csv", type=Path, default=None)
    p.add_argument("--require-fold", action="store_true")
    args = p.parse_args(argv)
    path = write_parse_md(
        args.outdir, id_csv=args.id_csv, require_fold=args.require_fold
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
