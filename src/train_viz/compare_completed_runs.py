#!/usr/bin/env python3
"""Build unified train-viz figures across completed ``runs/*/`` train stages.

Uses epoch metrics already exported for TensorBoard (``train_metrics_epochs.jsonl``
preferentially, else ``train_metrics.jsonl``). Does not invent metrics.

Completion rule (strict): jsonl has ≥1 integer epoch with train/validation blocks
**and** a ``epoch == "final"`` record (or ``pipeline_done.json`` at the pipeline root
with status COMPLETED / completed for that stage). In-progress jobs with early
checkpoints are excluded.

Example::

  conda run -n caduceus_env python -m src.train_viz.compare_completed_runs \\
    --runs-root runs -o figures/train-viz
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.train_viz.viz import main as train_viz_main

# Probe / legacy trees are never treated as manuscript runs.
_SKIP_NAME_SUBSTR = ("probe", "legacy")


def _full_metrics_jsonl(train_dir: Path) -> Path | None:
    full = train_dir / "logs" / "train_metrics.jsonl"
    if full.is_file() and full.stat().st_size > 0:
        return full
    return None


def _pick_plot_log(train_dir: Path) -> Path | None:
    """Prefer epoch-only jsonl for curves; fall back to full metrics."""
    epochs = train_dir / "logs" / "train_metrics_epochs.jsonl"
    if epochs.is_file() and epochs.stat().st_size > 0:
        return epochs
    return _full_metrics_jsonl(train_dir)


def _jsonl_complete(jsonl: Path) -> bool:
    """Completion uses the full metrics file (includes ``epoch: final`` / ZSV)."""
    has_epoch = False
    has_final = False
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        ep = rec.get("epoch")
        if ep == "final":
            has_final = True
            continue
        if isinstance(ep, int) and not rec.get("smoke"):
            if "train" in rec or "validation" in rec:
                has_epoch = True
    return has_epoch and has_final


def _has_epoch_blocks(jsonl: Path) -> bool:
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(rec, dict)
            and isinstance(rec.get("epoch"), int)
            and not rec.get("smoke")
            and ("train" in rec or "validation" in rec)
        ):
            return True
    return False


def _pipeline_marked_complete(pipeline_root: Path) -> bool:
    done = pipeline_root / "pipeline_done.json"
    if not done.is_file():
        return False
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    status = str(payload.get("status", "")).upper()
    return status in {"COMPLETED", "COMPLETE", "OK", "DONE"}


def discover_completed_stages(
    runs_root: Path,
) -> dict[str, list[tuple[str, Path, Path]]]:
    """Return ``{"direct": [(run_id, train_dir, plot_log), ...], "adversarial": [...]}``."""
    out: dict[str, list[tuple[str, Path, Path]]] = {"direct": [], "adversarial": []}
    if not runs_root.is_dir():
        return out
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if any(s in name for s in _SKIP_NAME_SUBSTR):
            continue
        for stage, rel in (
            ("direct", Path("direct")),
            ("adversarial", Path("adversarial") / "train"),
        ):
            train_dir = run_dir / rel
            full = _full_metrics_jsonl(train_dir)
            plot_log = _pick_plot_log(train_dir)
            if full is None or plot_log is None:
                continue
            # Prefer strict final-record completion; allow pipeline_done only for
            # classification adv where ZSV/final may be skipped by design.
            complete = _jsonl_complete(full)
            if not complete and stage == "adversarial" and _pipeline_marked_complete(run_dir):
                complete = _has_epoch_blocks(full)
            if not complete:
                continue
            out[stage].append((name, train_dir, plot_log))
    return out


def _run_unified(
    *,
    stage: str,
    entries: list[tuple[str, Path, Path]],
    outdir: Path,
    title: str,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    if not entries:
        manifest = {
            "stage": stage,
            "outdir": str(outdir),
            "status": "no_completed_runs",
            "n_runs": 0,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        (outdir / "compare_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    logs: list[str] = []
    models: list[str] = []
    argv: list[str] = ["--logs"]
    for run_id, _train_dir, log in entries:
        mid = f"{run_id}_{stage}"
        logs.append(str(log))
        models.append(mid)
    argv.extend(logs)
    for mid in models:
        argv.extend(["--model", mid, "--label", mid])
    argv.extend(
        [
            "-o",
            str(outdir),
            "--title",
            title,
            "--ribbon",
            "none",
            "--max-epoch",
            "23",
        ]
    )
    rc = train_viz_main(argv)
    status = "ok" if rc in (0, None) else f"train_viz_failed:{rc}"
    pdfs = sorted(str(p) for p in outdir.rglob("*.pdf"))
    pngs = sorted(str(p) for p in outdir.rglob("*.png"))
    manifest: dict[str, Any] = {
        "stage": stage,
        "outdir": str(outdir),
        "status": status,
        "n_runs": len(entries),
        "runs": [
            {"run_id": rid, "train_dir": str(td), "log": str(log)}
            for rid, td, log in entries
        ],
        "n_pdf": len(pdfs),
        "n_png": len(pngs),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "compare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_missing_monitors(runs_root: Path) -> list[dict[str, Any]]:
    """Render per-run train_monitor for completed stages missing PNG figures."""
    from src.train_viz.train_monitor import refresh_train_monitor

    discovered = discover_completed_stages(runs_root)
    results: list[dict[str, Any]] = []
    for stage, entries in discovered.items():
        for run_id, train_dir, _log in entries:
            mon = train_dir / "figures" / "train_monitor"
            n_png = len(list(mon.rglob("*.png"))) if mon.is_dir() else 0
            if n_png > 0:
                continue
            man = refresh_train_monitor(
                train_dir,
                model=f"{run_id}_{stage}",
                title=f"{run_id} {stage} — train monitor",
                include_split_compare=True,
            )
            results.append(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "train_dir": str(train_dir),
                    "monitor": man,
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("figures/train-viz"),
        help="Parent outdir; writes all_completed_direct/ and all_completed_adversarial/",
    )
    p.add_argument(
        "--skip-refresh-missing",
        action="store_true",
        help="Do not fill missing per-run train_monitor figures first",
    )
    p.add_argument(
        "--skip-tb-compare",
        action="store_true",
        help="Skip TensorBoard compare tree + HTML dashboard",
    )
    p.add_argument(
        "--start-tb-server",
        action="store_true",
        help="After TB compare export, start TensorBoard (see tb_compare)",
    )
    p.add_argument("--tb-port", type=int, default=6006)
    args = p.parse_args(argv)

    refreshed: list[dict[str, Any]] = []
    if not args.skip_refresh_missing:
        refreshed = refresh_missing_monitors(args.runs_root)

    discovered = discover_completed_stages(args.runs_root)
    direct = _run_unified(
        stage="direct",
        entries=discovered["direct"],
        outdir=args.outdir / "all_completed_direct",
        title="Completed runs — direct (from TB/jsonl metrics)",
    )
    adv = _run_unified(
        stage="adversarial",
        entries=discovered["adversarial"],
        outdir=args.outdir / "all_completed_adversarial",
        title="Completed runs — adversarial (from TB/jsonl metrics)",
    )
    tb_bundle: dict[str, Any] | None = None
    if not args.skip_tb_compare:
        from src.train_viz.tb_compare import main as tb_compare_main

        tb_argv = [
            "--runs-root",
            str(args.runs_root),
            "-o",
            str(args.outdir),
            "--port",
            str(args.tb_port),
            "--stop-existing",
        ]
        if args.start_tb_server:
            tb_argv.append("--start-server")
        tb_rc = tb_compare_main(tb_argv)
        bundle_path = args.outdir / "tb_compare_bundle.json"
        if bundle_path.is_file():
            tb_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        elif tb_rc not in (0, None):
            tb_bundle = {"status": f"tb_compare_failed:{tb_rc}"}
    inventory = {
        "runs_root": str(args.runs_root),
        "outdir": str(args.outdir),
        "refreshed_missing_monitors": refreshed,
        "direct": direct,
        "adversarial": adv,
        "tb_compare": tb_bundle,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    inv_path = args.outdir / "all_completed_inventory.json"
    args.outdir.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Unified train-viz inventory",
        "",
        f"- Generated: `{inventory['written_at']}`",
        f"- Runs root: `{args.runs_root}`",
        f"- Refreshed missing per-run monitors: **{len(refreshed)}**",
        "",
        "## Direct",
        f"- Status: `{direct.get('status')}` · n={direct.get('n_runs')} · "
        f"pdf={direct.get('n_pdf')} png={direct.get('n_png')}",
        f"- Outdir: `{direct.get('outdir')}`",
        "",
        "## Adversarial",
        f"- Status: `{adv.get('status')}` · n={adv.get('n_runs')} · "
        f"pdf={adv.get('n_pdf')} png={adv.get('n_png')}",
        f"- Outdir: `{adv.get('outdir')}`",
        "",
        "## Included runs",
        "",
        "### Direct",
    ]
    for rid, td, log in discovered["direct"]:
        lines.append(f"- `{rid}` ← `{log}`")
    lines.extend(["", "### Adversarial"])
    for rid, td, log in discovered["adversarial"]:
        lines.append(f"- `{rid}` ← `{log}`")
    if refreshed:
        lines.extend(["", "## Per-run monitors refreshed"])
        for item in refreshed:
            st = (item.get("monitor") or {}).get("status")
            lines.append(f"- `{item['run_id']}` / `{item['stage']}` → `{st}`")
    if tb_bundle:
        lines.extend(
            [
                "",
                "## TensorBoard compare",
                f"- Dashboard: `{args.outdir / 'compare_index.html'}`",
                f"- Event trees: `{args.outdir / 'tb_compare'}`",
            ]
        )
        tb_meta = tb_bundle.get("tensorboard") if isinstance(tb_bundle, dict) else None
        if isinstance(tb_meta, dict) and tb_meta.get("url"):
            lines.append(f"- Live URL: {tb_meta['url']}")
    (args.outdir / "all_completed_inventory.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, indent=2))
    ok = direct.get("status") in {"ok", "no_completed_runs"} and adv.get("status") in {
        "ok",
        "no_completed_runs",
    }
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
