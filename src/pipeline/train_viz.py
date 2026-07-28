"""Visualize pipeline training logs through the canonical ``src.train_viz``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ensure_dir


def run_train_viz(logs_dir: Path, *, outdir: Path) -> Path:
    logs_dir = Path(logs_dir)
    candidates = [
        logs_dir / "train_metrics.jsonl",
        logs_dir / "logs" / "train_metrics.jsonl",
    ]
    metrics = next((p for p in candidates if p.is_file()), None)
    if metrics is None:
        raise FileNotFoundError(f"No train_metrics.jsonl under {logs_dir}")

    rows = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty metrics log")

    outdir = ensure_dir(Path(outdir))
    try:
        from src.train_viz.viz import main as canonical_main

        status = canonical_main(["--logs", str(metrics), "--outdir", str(outdir)])
        if status not in (None, 0):
            raise RuntimeError(f"src.train_viz.viz.main returned {status}")
        return outdir
    except Exception as exc:
        # A smoke log intentionally has no invented loss/quality metrics, so the
        # publication plotter may have nothing numeric to plot. Preserve its
        # diagnostic and emit structural—not biological—fallback artifacts.
        (outdir / "train_viz_fallback_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        summary = {
            "n_records": len(rows),
            "source": str(metrics),
            "last_epoch": rows[-1].get("epoch"),
            "smoke": bool(rows[-1].get("smoke", False)),
            "status": "structural fallback; no model-quality plot was generated",
        }
        (outdir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='480' height='100'>"
            "<text x='10' y='50'>Structural smoke log: no model-quality metrics</text>"
            "</svg>\n"
        )
        (outdir / "loss_curve.svg").write_text(svg, encoding="utf-8")
        (outdir / "train_metrics.csv").write_text("epoch,smoke\n" + "\n".join(
            f"{row.get('epoch', '')},{str(bool(row.get('smoke', False))).lower()}"
            for row in rows
        ) + "\n", encoding="utf-8")
    return outdir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="train-viz from logs")
    p.add_argument("--logs", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    args = p.parse_args(argv)
    print(run_train_viz(args.logs, outdir=args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
