#!/usr/bin/env python3
"""Unified TensorBoard + HTML dashboard for comparing completed runs.

Writes under ``figures/train-viz/tb_compare/``:

- ``direct/<run_id>/`` and ``adversarial/<run_id>/`` — SummaryWriter event
  trees with aligned tags (``train/loss``, ``validation/pearson``, …) so
  TensorBoard can overlay runs on one chart
- ``index.html`` — static dashboard (multimodel Altair panels + TB launch link)
- ``launch_tensorboard.sh`` — one-command TB server for both stages

Completion discovery reuses ``compare_completed_runs.discover_completed_stages``.

Example::

  conda run -n caduceus_env python -m src.train_viz.tb_compare \\
    --runs-root runs -o figures/train-viz --start-server --port 6006
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from src.train_viz.compare_completed_runs import discover_completed_stages
from src.train_viz.tensorboard_metrics import write_tensorboard_from_jsonl

SPLIT_ALIASES = (
    ("train", "train"),
    ("validation", "validation"),
    ("val", "validation"),
    ("test", "test"),
    ("zero-shot-validation", "zero-shot-validation"),
    ("zero_shot", "zero-shot-validation"),
)


def _host_ip() -> str:
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True).strip()
        if out:
            return out.split()[0]
    except (OSError, subprocess.CalledProcessError):
        pass
    return "127.0.0.1"


def _port_free(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _find_port(preferred: int = 6006, span: int = 20) -> int:
    for p in range(preferred, preferred + span):
        if _port_free(p):
            return p
    raise RuntimeError(f"No free TCP port in {preferred}..{preferred + span - 1}")


def export_run_compare_events(
    train_dir: Path,
    dest_run_dir: Path,
    *,
    run_label: str,
) -> dict[str, Any]:
    """Write a clean SummaryWriter tree for one run into ``dest_run_dir``.

    Tags are ``{split}/{metric}`` with step = epoch (or a large step for final/ZSV),
    matching ``write_tensorboard_from_jsonl`` so overlays align across models.
    """
    from torch.utils.tensorboard import SummaryWriter

    train_dir = Path(train_dir)
    dest_run_dir = Path(dest_run_dir)
    if dest_run_dir.exists():
        shutil.rmtree(dest_run_dir)
    dest_run_dir.mkdir(parents=True, exist_ok=True)

    jsonl = train_dir / "logs" / "train_metrics.jsonl"
    manifest: dict[str, Any] = {
        "run_label": run_label,
        "train_dir": str(train_dir),
        "dest": str(dest_run_dir),
        "status": "ok",
        "n_scalars": 0,
    }
    if not jsonl.is_file() or jsonl.stat().st_size == 0:
        manifest["status"] = "no_metrics"
        return manifest

    # Keep per-run TB under the train dir fresh as well (source of truth).
    try:
        write_tensorboard_from_jsonl(train_dir, purge=True)
    except Exception as exc:  # noqa: BLE001
        manifest["source_tb_error"] = f"{type(exc).__name__}: {exc}"

    writer = SummaryWriter(log_dir=str(dest_run_dir))
    n = 0
    try:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("smoke"):
                continue
            ep = rec.get("epoch")
            if isinstance(ep, int):
                step = int(ep)
            elif ep == "final":
                step = int(rec.get("global_step", 10_000_000))
            else:
                continue
            for split_key, tag in SPLIT_ALIASES:
                block = rec.get(split_key)
                if not isinstance(block, dict):
                    continue
                for k, v in block.items():
                    if k == "n":
                        continue
                    try:
                        writer.add_scalar(f"{tag}/{k}", float(v), step)
                        n += 1
                    except (TypeError, ValueError):
                        continue
        zsv_path = train_dir / "logs" / "zero_shot_metrics.json"
        if zsv_path.is_file():
            try:
                zsv = json.loads(zsv_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                zsv = None
            metrics = zsv.get("metrics") if isinstance(zsv, dict) else None
            if isinstance(metrics, dict) and not zsv.get("skipped"):
                for k, v in metrics.items():
                    if k == "n":
                        continue
                    try:
                        writer.add_scalar(f"zero-shot-validation/{k}", float(v), 10_000_000)
                        n += 1
                    except (TypeError, ValueError):
                        continue
        writer.flush()
    finally:
        writer.close()

    manifest["n_scalars"] = n
    if n == 0:
        manifest["status"] = "empty"
    (dest_run_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_tb_compare_trees(
    discovered: dict[str, list[tuple[str, Path, Path]]],
    outdir: Path,
) -> dict[str, Any]:
    """Populate ``outdir/tb_compare/{direct,adversarial}/<label>/`` event dirs."""
    root = Path(outdir) / "tb_compare"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "root": str(root),
        "stages": {},
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    for stage, entries in discovered.items():
        stage_dir = root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_runs: list[dict[str, Any]] = []
        for run_id, train_dir, _log in entries:
            label = f"{run_id}_{stage}"
            dest = stage_dir / label
            man = export_run_compare_events(train_dir, dest, run_label=label)
            stage_runs.append(man)
        result["stages"][stage] = {
            "dir": str(stage_dir),
            "n_runs": len(stage_runs),
            "runs": stage_runs,
        }
    (root / "tb_compare_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def write_launch_script(root: Path, port: int) -> Path:
    """Write a shell helper that starts TensorBoard over the compare tree."""
    script = root / "launch_tensorboard.sh"
    body = f"""#!/usr/bin/env bash
# Launch TensorBoard overlay for completed-run comparison.
# Each subdirectory under direct/ and adversarial/ is one model run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${{1:-{port}}}"
exec tensorboard \\
  --host 0.0.0.0 \\
  --port "$PORT" \\
  --reload_interval 30 \\
  --logdir "$ROOT"
"""
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def _collect_multimodel_html(figures_stage_dir: Path) -> list[Path]:
    if not figures_stage_dir.is_dir():
        return []
    return sorted(figures_stage_dir.glob("Figure_*_multimodel_*_altair.html"))


def write_index_html(
    outdir: Path,
    *,
    tb_manifest: dict[str, Any],
    tb_url: str | None,
    port: int,
    host: str,
    static_dashboard_url: str | None = None,
) -> Path:
    """Static dashboard linking TensorBoard + multimodel Altair panels."""
    outdir = Path(outdir)
    tb_root = outdir / "tb_compare"
    index = outdir / "compare_index.html"
    dash_url = static_dashboard_url or f"http://{host}:6007/compare_index.html"

    sections: list[str] = []
    for stage in ("direct", "adversarial"):
        fig_dir = outdir / f"all_completed_{stage}"
        htmls = _collect_multimodel_html(fig_dir)
        runs = (tb_manifest.get("stages") or {}).get(stage, {}).get("runs") or []
        run_labels = [r.get("run_label", "?") for r in runs if r.get("status") == "ok"]
        cards = []
        for h in htmls:
            rel = h.relative_to(outdir).as_posix()
            title = h.name.replace("_altair.html", "").replace("Figure_", "").replace("_", " ")
            cards.append(
                f'<li><a href="{escape(rel)}" target="_blank" rel="noopener">'
                f"{escape(title)}</a></li>"
            )
        iframe_bits = []
        for prefer in (
            "multimodel_pearson_validation",
            "multimodel_loss_validation",
            "multimodel_spearman_validation",
        ):
            hits = [h for h in htmls if prefer in h.name]
            if not hits:
                continue
            rel = hits[0].relative_to(outdir).as_posix()
            iframe_bits.append(
                f'<figure><figcaption>{escape(prefer)}</figcaption>'
                f'<iframe src="{escape(rel)}" loading="lazy" '
                f'title="{escape(prefer)}"></iframe></figure>'
            )
        sections.append(
            f"""
<section>
  <h2>{escape(stage.title())} — model overlays</h2>
  <p>Runs ({len(run_labels)}): {escape(", ".join(run_labels)) or "none"}</p>
  <p>TensorBoard logdir: <code>{escape(str((tb_root / stage).resolve()))}</code></p>
  <h3>Interactive multimodel panels (Altair)</h3>
  <ul>{"".join(cards) if cards else "<li><em>No multimodel HTML yet — run compare_completed_runs first.</em></li>"}</ul>
  <div class="iframes">{"".join(iframe_bits)}</div>
  <p>Manuscript PDFs: <a href="all_completed_{stage}/manuscript/">{escape(f"all_completed_{stage}/manuscript/")}</a></p>
</section>
"""
        )

    links = [
        f'<p class="hero-link"><a href="{escape(dash_url)}" target="_blank" rel="noopener">'
        f"Open static comparison dashboard → {escape(dash_url)}</a></p>"
    ]
    if tb_url:
        links.append(
            f'<p class="hero-link"><a href="{escape(tb_url)}" target="_blank" rel="noopener">'
            f"Open TensorBoard comparison → {escape(tb_url)}</a></p>"
        )
        links.append(
            "<p>In TensorBoard: select several runs in the left sidebar, open a scalar "
            "(e.g. <code>validation/pearson</code>), enable overlay of multiple runs.</p>"
        )
    else:
        links.append(
            f"""<p>Start TensorBoard locally:</p>
  <pre><code>bash {escape(str((tb_root / "launch_tensorboard.sh").resolve()))} {port}
# then open http://{escape(host)}:{port}/
</code></pre>"""
        )
    tb_block = "\n".join(links)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Completed runs — model comparison</title>
  <style>
    :root {{
      --bg: #f7f5f0;
      --ink: #1c1a16;
      --accent: #0b6e4f;
      --card: #fff;
      --muted: #5c574e;
    }}
    body {{
      margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(160deg, #f7f5f0 0%, #e8efe9 50%, #f0ebe3 100%);
      color: var(--ink); line-height: 1.45;
    }}
    header {{
      padding: 2.2rem 1.5rem 1.2rem; max-width: 1100px; margin: 0 auto;
    }}
    header h1 {{ font-size: 1.85rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; }}
    header p {{ color: var(--muted); margin: 0.25rem 0; }}
    .hero-link a {{
      display: inline-block; margin-top: 0.8rem; padding: 0.65rem 1rem;
      background: var(--accent); color: #fff; text-decoration: none;
      font-weight: 600; border-radius: 4px;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 0 1.5rem 3rem; }}
    section {{
      background: var(--card); border: 1px solid #ddd6c8; border-radius: 6px;
      padding: 1.1rem 1.25rem; margin: 1rem 0;
    }}
    code, pre {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.86rem; }}
    pre {{ background: #f0ece4; padding: 0.75rem; overflow-x: auto; }}
    iframe {{
      width: 100%; height: 420px; border: 1px solid #ddd6c8; border-radius: 4px;
      background: #fff;
    }}
    figure {{ margin: 1rem 0; }}
    figcaption {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 0.35rem; }}
    ul {{ columns: 2; gap: 1.5rem; }}
    @media (max-width: 800px) {{ ul {{ columns: 1; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Completed runs — cross-model comparison</h1>
    <p>Metrics from TensorBoard-synced jsonl (aligned tags). Generated
      {escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}.</p>
    {tb_block}
  </header>
  <main>
    {"".join(sections)}
    <section>
      <h2>Reproduce</h2>
      <pre><code>conda run -n caduceus_env python -m src.train_viz.compare_completed_runs \\
  --runs-root runs -o figures/train-viz --skip-refresh-missing
conda run -n caduceus_env python -m src.train_viz.tb_compare \\
  --runs-root runs -o figures/train-viz --start-server --port {port}
</code></pre>
      <p>Inventory: <a href="all_completed_inventory.md">all_completed_inventory.md</a></p>
    </section>
  </main>
</body>
</html>
"""
    index.write_text(html, encoding="utf-8")
    return index


def start_tensorboard(
    tb_root: Path,
    *,
    port: int,
    log_path: Path,
) -> dict[str, Any]:
    """Start TensorBoard in the background; return URL + PID metadata."""
    tb_root = Path(tb_root)
    if not tb_root.is_dir():
        return {"status": "no_logdirs", "port": port}

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tensorboard",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--reload_interval",
        "30",
        "--logdir",
        str(tb_root.resolve()),
    ]
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    url = None
    host = _host_ip()
    for _ in range(40):
        time.sleep(0.25)
        if proc.poll() is not None:
            return {
                "status": "failed",
                "pid": proc.pid,
                "port": port,
                "log": str(log_path),
                "returncode": proc.returncode,
            }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                url = f"http://{host}:{port}/"
                break
            except OSError:
                continue
    meta = {
        "status": "running" if url else "starting",
        "pid": proc.pid,
        "port": port,
        "url": url or f"http://{host}:{port}/",
        "log": str(log_path),
        "logdir": str(tb_root.resolve()),
        "cmd": cmd,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (tb_root / "tensorboard.pid").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def stop_existing_tb(tb_root: Path) -> None:
    pid_file = Path(tb_root) / "tensorboard.pid"
    if not pid_file.is_file():
        return
    try:
        meta = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = int(meta.get("pid", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return
    if pid <= 0:
        return
    try:
        os.kill(pid, 0)
    except OSError:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("-o", "--outdir", type=Path, default=Path("figures/train-viz"))
    p.add_argument("--port", type=int, default=6006)
    p.add_argument(
        "--start-server",
        action="store_true",
        help="Start TensorBoard in background and write URL into index.html",
    )
    p.add_argument(
        "--stop-existing",
        action="store_true",
        help="Stop previous tb_compare TensorBoard PID if still running",
    )
    args = p.parse_args(argv)

    discovered = discover_completed_stages(args.runs_root)
    tb_manifest = build_tb_compare_trees(discovered, args.outdir)
    port = args.port
    if args.start_server:
        if not _port_free(port):
            port = _find_port(args.port)
    write_launch_script(Path(tb_manifest["root"]), port)

    server_meta: dict[str, Any] | None = None
    tb_url = None
    host = _host_ip()
    if args.stop_existing:
        stop_existing_tb(Path(tb_manifest["root"]))
    if args.start_server:
        server_meta = start_tensorboard(
            Path(tb_manifest["root"]),
            port=port,
            log_path=Path(tb_manifest["root"]) / "tensorboard_server.log",
        )
        tb_url = server_meta.get("url")

    index = write_index_html(
        args.outdir,
        tb_manifest=tb_manifest,
        tb_url=tb_url,
        port=port,
        host=host,
    )
    # Also drop a short pointer next to TB root
    (Path(tb_manifest["root"]) / "README.md").write_text(
        "\n".join(
            [
                "# TensorBoard compare root",
                "",
                f"- Dashboard: `{index}`",
                f"- Launch: `bash {Path(tb_manifest['root']) / 'launch_tensorboard.sh'}`",
                f"- URL: {tb_url or f'http://{host}:{port}/ (after launch)'}",
                "",
                "Each subdirectory under `direct/` / `adversarial/` is one model run.",
                "Open a scalar tag (e.g. `validation/pearson`) and enable multiple runs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = {
        "tb_compare": tb_manifest,
        "index_html": str(index),
        "tensorboard": server_meta,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "tb_compare_bundle.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    if server_meta and server_meta.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
