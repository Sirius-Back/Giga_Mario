#!/usr/bin/env python3
"""Collect direct/adversarial run results from ``runs/`` and ``runs_unif/``.

Writes wide CSVs + markdown tables with columns:
  run, run_id, model, train, test, val, split type, split params,
  best epoch, final epoch, zsv/val/test/train pearson,
  done (run|run_unif|submit|unif), is legacy, additional
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
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

COLUMNS = [
    "run",
    "run_id",
    "model",
    "train",
    "test",
    "val",
    "split type",
    "split params",
    "best epoch",
    "final epoch",
    "zsv pearson",
    "val pearson",
    "test pearson",
    "train pearson",
    "done",
    "is legacy",
    "additional",
]


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or v == "":
        return ""
    fv = _finite(v)
    if fv is None:
        return str(v)
    return f"{fv:.{nd}f}"


def _parse_run_number(run_id: str) -> str:
    m = re.match(r"run(\d+)", run_id)
    return m.group(1) if m else ""


def _infer_model(run_id: str, train_dir: Path | None, run_root: Path) -> str:
    low = run_id.lower()
    if "caduceus" in low:
        return "caduceus"
    if "legnet" in low:
        return "legnet"
    if train_dir is not None:
        if (train_dir / "caduceus_input").is_dir():
            return "caduceus"
        if any((train_dir / "best_model").glob("pearson-*.ckpt")):
            return "legnet"
        rc = _load_json(train_dir / "logs" / "run_config.json") or {}
        skill = str(rc.get("skill") or rc.get("model") or "").lower()
        if "caduceus" in skill:
            return "caduceus"
        if "legnet" in skill:
            return "legnet"
    # path under runs_unif/<model>/
    parts = run_root.parts
    if "runs_unif" in parts:
        i = parts.index("runs_unif")
        if len(parts) > i + 1 and parts[i + 1] in {"caduceus", "legnet"}:
            return parts[i + 1]
    return "unknown"


def _infer_split_type(run_id: str, run_root: Path) -> str:
    low = run_id.lower()
    for key in (
        "hashfrag",
        "pangenome",
        "mmseqs",
        "blastp",
        "paralog",
        "random",
        "kmer",
        "gc",
    ):
        if key in low:
            if key == "paralog":
                return "paralogs"
            return key
    for meta_name in (
        "hashfrag_split_meta.json",
        "kmer_split_meta.json",
        "pangenome_split_meta.json",
        "gc_split_meta.json",
        "mmseqs_split_meta.json",
        "blastp_split_meta.json",
        "random_split_meta.json",
        "paralogs_split_meta.json",
    ):
        meta = _load_json(run_root / meta_name)
        if meta and meta.get("split_id"):
            return str(meta["split_id"])
    hydra = run_root / "hydra_resolved_config.yaml"
    if hydra.is_file():
        for line in hydra.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("split:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def _compact_params(d: dict[str, Any], keys: list[str]) -> str:
    parts: list[str] = []
    for k in keys:
        if k not in d or d[k] is None:
            continue
        v = d[k]
        if isinstance(v, (dict, list)) and k not in {"k", "window"}:
            continue
        if k == "k" and isinstance(v, list):
            v = ",".join(str(x) for x in v)
        if k == "window" and isinstance(v, dict):
            v = f"[{v.get('pos1')},{v.get('pos2')}]"
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def _split_params(run_id: str, run_root: Path, split_type: str) -> str:
    # Prefer dedicated meta
    candidates = [
        run_root / f"{split_type}_split_meta.json",
        run_root / "hashfrag_split_meta.json",
        run_root / "kmer_split_meta.json",
        run_root / "pangenome_split_meta.json",
        run_root / "gc_split_meta.json",
        run_root / "mmseqs_split_meta.json",
        run_root / "split_cpu_meta.json",
    ]
    meta: dict[str, Any] | None = None
    for c in candidates:
        meta = _load_json(c)
        if meta:
            break

    if meta:
        # Flatten nested window from marked_source
        flat = dict(meta)
        ms = meta.get("marked_source")
        if isinstance(ms, dict) and isinstance(ms.get("window"), dict):
            flat["window"] = ms["window"]
        keys = [
            "k",
            "seed",
            "threshold",
            "p_train",
            "p_test",
            "min_shared",
            "n_clusters",
            "n_edges",
            "window",
            "identity",
            "coverage",
            "method",
        ]
        # method from assign_meta
        am = meta.get("assign_meta")
        if isinstance(am, dict):
            if am.get("method_used"):
                flat["method"] = am["method_used"]
            cluster = am.get("cluster")
            if isinstance(cluster, dict) and cluster.get("method_requested"):
                flat.setdefault("method", cluster["method_requested"])
        out = _compact_params(flat, keys)
        if out:
            return out

    # From run_id tokens
    bits: list[str] = []
    m = re.search(r"_k(\d+)", run_id)
    if m:
        bits.append(f"k={m.group(1)}")
    m = re.search(r"_w(-?\d+)_(-?\d+)", run_id)
    if m:
        bits.append(f"window=[{m.group(1)},{m.group(2)}]")
    m = re.search(r"_wm(-?\d+)_(-?\d+)", run_id)
    if m:
        bits.append(f"window=[{m.group(1)},{m.group(2)}]")
    m = re.search(r"id0?(\d+)", run_id)
    if m:
        bits.append(f"identity=0.{m.group(1)}" if len(m.group(1)) <= 2 else f"identity={m.group(1)}")
    # hydra kmer_size
    hydra = run_root / "hydra_resolved_config.yaml"
    if hydra.is_file() and not any(b.startswith("k=") for b in bits):
        for line in hydra.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("kmer_size:"):
                bits.append(f"k={line.split(':', 1)[1].strip()}")
            if line.strip().startswith("seed:"):
                bits.append(f"seed={line.split(':', 1)[1].strip()}")
    return "; ".join(bits)


def _split_counts(run_root: Path) -> dict[str, int | None]:
    out: dict[str, int | None] = {"train": None, "val": None, "test": None}
    for name in (
        "hashfrag_split_meta.json",
        "kmer_split_meta.json",
        "pangenome_split_meta.json",
        "gc_split_meta.json",
        "mmseqs_split_meta.json",
        "blastp_split_meta.json",
        "random_split_meta.json",
        "paralogs_split_meta.json",
    ):
        meta = _load_json(run_root / name)
        if not meta:
            continue
        counts = meta.get("counts")
        if isinstance(counts, dict):
            for k in ("train", "val", "test"):
                fv = _finite(counts.get(k))
                if fv is not None:
                    out[k] = int(fv)
            if any(out[k] is not None for k in out):
                return out

    split_csv = run_root / "split.csv"
    if split_csv.is_file():
        try:
            # ID|train_test|fold  OR  ID,train_test,fold
            first = split_csv.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
            if not first:
                return out
            sep = "|" if "|" in first[0] else ","
            hdr = first[0].split(sep)
            # handle sticky header ID|train_test|fold as one cell
            if len(hdr) == 1 and "|" in hdr[0]:
                hdr = hdr[0].split("|")
                sep = "|"
            try:
                idx = hdr.index("train_test")
            except ValueError:
                idx = 1
            c: Counter[str] = Counter()
            with split_csv.open(encoding="utf-8", errors="ignore") as fh:
                next(fh, None)
                for line in fh:
                    parts = line.rstrip("\n").split(sep)
                    if len(parts) <= idx:
                        continue
                    c[parts[idx]] += 1
            for k in ("train", "val", "test"):
                if k in c:
                    out[k] = c[k]
        except OSError:
            pass
    return out


def _jsonl_epochs(train_dir: Path) -> tuple[dict[int, dict[str, Any]], int | None]:
    """Return epoch→record map and max numeric epoch."""
    by_ep: dict[int, dict[str, Any]] = {}
    max_ep: int | None = None
    for name in ("train_metrics.jsonl", "train_metrics_epochs.jsonl"):
        path = train_dir / "logs" / name
        if not path.is_file():
            path = train_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = rec.get("epoch")
            if isinstance(ep, int):
                by_ep[ep] = rec
                max_ep = ep if max_ep is None else max(max_ep, ep)
    return by_ep, max_ep


def _pearson_from_block(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    for key in ("pearson", "Pearson", "val_pearson"):
        if key in block:
            return _finite(block[key])
    return None


def _accuracy_from_block(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    return _finite(block.get("accuracy"))


def _extract_pearsons(train_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "train": None,
        "val": None,
        "test": None,
        "zsv": None,
    }
    best = _load_json(train_dir / "best_split_metrics.json") or {}
    by_split = best.get("metrics_by_split")
    if isinstance(by_split, dict):
        for split in ("train", "val", "test", "zsv"):
            out[split] = _pearson_from_block(by_split.get(split))

    meta = _load_json(train_dir / "best_model" / "best_meta.json") or {}
    best_ep = meta.get("epoch")
    if out["val"] is None and meta.get("metric") == "val_pearson":
        out["val"] = _finite(meta.get("val_pearson") or meta.get("value"))

    by_ep, _ = _jsonl_epochs(train_dir)
    if best_ep is not None and int(best_ep) in by_ep:
        rec = by_ep[int(best_ep)]
        if out["train"] is None:
            out["train"] = _pearson_from_block(rec.get("train"))
        if out["val"] is None:
            out["val"] = _pearson_from_block(rec.get("validation") or rec.get("val"))
        if out["test"] is None:
            out["test"] = _pearson_from_block(rec.get("test"))
    elif best_ep is None and by_ep:
        # No best_meta: pick epoch with max validation pearson
        best_v = None
        best_rec = None
        for rec in by_ep.values():
            vp = _pearson_from_block(rec.get("validation") or rec.get("val"))
            if vp is None:
                continue
            if best_v is None or vp > best_v:
                best_v = vp
                best_rec = rec
        if best_rec is not None:
            out["val"] = out["val"] if out["val"] is not None else best_v
            if out["train"] is None:
                out["train"] = _pearson_from_block(best_rec.get("train"))
            if out["test"] is None:
                out["test"] = _pearson_from_block(best_rec.get("test"))

    # ZSV file
    if out["zsv"] is None:
        zsv = _load_json(train_dir / "logs" / "zero_shot_metrics.json") or _load_json(
            train_dir / "zero_shot_metrics.json"
        )
        if isinstance(zsv, dict) and not zsv.get("skipped"):
            block = zsv.get("metrics") if isinstance(zsv.get("metrics"), dict) else zsv
            out["zsv"] = _pearson_from_block(block)

    # metrics_summary fallbacks
    summary = _load_json(train_dir / "metrics_summary.json") or {}
    if out["test"] is None:
        out["test"] = _pearson_from_block(summary.get("test"))
    if out["val"] is None:
        light = summary.get("lightning") if isinstance(summary.get("lightning"), dict) else {}
        out["val"] = _finite(light.get("best_val_pearson"))

    return out


def _classification_acc_note(train_dir: Path) -> str:
    """If task is classification, return accuracy summary for additional."""
    rc = _load_json(train_dir / "logs" / "run_config.json") or {}
    task = str(rc.get("task") or "").lower()
    if task != "classification":
        return ""
    meta = _load_json(train_dir / "best_model" / "best_meta.json") or {}
    best_ep = meta.get("epoch")
    by_ep, _ = _jsonl_epochs(train_dir)
    rec = by_ep.get(int(best_ep)) if best_ep is not None else None
    if not rec:
        return "task=classification (no pearson)"
    parts = ["task=classification"]
    for split, key in (("train", "train"), ("val", "validation"), ("test", "test")):
        block = rec.get(key) if key != "validation" else (rec.get("validation") or rec.get("val"))
        acc = _accuracy_from_block(block)
        if acc is not None:
            parts.append(f"acc_{split}={acc:.4f}")
    return "; ".join(parts)


def _best_final_epochs(train_dir: Path) -> tuple[int | None, int | None]:
    meta = _load_json(train_dir / "best_model" / "best_meta.json") or {}
    best_ep = meta.get("epoch")
    best_ep_i = int(best_ep) if best_ep is not None else None
    by_ep, max_ep = _jsonl_epochs(train_dir)
    if best_ep_i is None and by_ep:
        best_v = None
        for ep, rec in by_ep.items():
            vp = _pearson_from_block(rec.get("validation") or rec.get("val"))
            if vp is None:
                continue
            if best_v is None or vp > best_v:
                best_v = vp
                best_ep_i = ep
    summary = _load_json(train_dir / "metrics_summary.json") or {}
    light = summary.get("lightning") if isinstance(summary.get("lightning"), dict) else {}
    last = light.get("last_logged") if isinstance(light.get("last_logged"), dict) else {}
    last_ep = last.get("epoch")
    if isinstance(last_ep, int):
        max_ep = last_ep if max_ep is None else max(max_ep, last_ep)
    # checkpoint dirs model_<ep>_epoch
    for p in train_dir.glob("model_*_epoch"):
        m = re.search(r"model_(\d+)_epoch", p.name)
        if m:
            ep = int(m.group(1))
            max_ep = ep if max_ep is None else max(max_ep, ep)
    # epoch dirs under logs
    logs = train_dir / "logs"
    if logs.is_dir():
        for p in logs.iterdir():
            if p.name.startswith("epoch") and p.name[5:].isdigit():
                ep = int(p.name[5:])
                max_ep = ep if max_ep is None else max(max_ep, ep)
    return best_ep_i, max_ep


def _done_label(
    *,
    is_legacy: bool,
    train_dir: Path | None,
    run_root: Path,
    stage: str,
    queue_text: str,
) -> str:
    """Map to run | run_unif | submit | unif."""
    has_best = bool(
        train_dir
        and (train_dir / "best_model" / "best_meta.json").is_file()
    )
    has_metrics = bool(
        train_dir
        and (
            (train_dir / "logs" / "train_metrics.jsonl").is_file()
            or (train_dir / "metrics_summary.json").is_file()
            or (train_dir / "best_split_metrics.json").is_file()
        )
    )
    pipe = _load_json(run_root / "pipeline_done.json")
    pipe_ok = isinstance(pipe, dict) and str(pipe.get("status", "")).upper() in {
        "COMPLETED",
        "OK",
        "DONE",
    }

    run_id = run_root.name
    in_queue_running = bool(
        re.search(
            rf"###\s+\S*{re.escape(run_id)}\S*[^\n]*—\s*RUNNING",
            queue_text,
            flags=re.I,
        )
    )

    if is_legacy:
        if has_best or pipe_ok or has_metrics:
            return "run"
        if in_queue_running:
            return "submit"
        return "run" if (run_root / "SPLIT").is_dir() else "submit"

    # unified tree
    if (has_best or has_metrics) and pipe_ok:
        return "unif"
    if has_best or has_metrics:
        return "run_unif"
    if in_queue_running or (run_root / "split_done.json").is_file():
        return "submit"
    return "submit"


def _additional(
    *,
    train_dir: Path | None,
    run_root: Path,
    stage: str,
    pearsons: dict[str, float | None],
) -> str:
    notes: list[str] = []
    if train_dir is None:
        notes.append("no train dir")
        return "; ".join(notes)

    if not (train_dir / "best_model" / "best_meta.json").is_file():
        notes.append("no best_model")

    clf = _classification_acc_note(train_dir)
    if clf:
        notes.append(clf)

    best = _load_json(train_dir / "best_split_metrics.json") or {}
    if best.get("note"):
        notes.append(str(best["note"]))
    src = best.get("source")
    if src:
        notes.append(f"src={src}")

    meta = _load_json(train_dir / "best_model" / "best_meta.json") or {}
    if meta.get("selection"):
        notes.append(str(meta["selection"]))
    if meta.get("metric") and meta["metric"] != "val_pearson":
        notes.append(f"select={meta['metric']}")

    # Missing pearsons (skip listing when classification — expected)
    if not clf:
        missing = [k for k, v in pearsons.items() if v is None]
        if missing and (train_dir / "best_model" / "best_meta.json").is_file():
            notes.append("missing:" + ",".join(missing))

    # Failed adversarial sibling
    if stage == "adversarial":
        fails = list(run_root.glob("adversarial_FAILED_*"))
        if fails:
            notes.append(f"prior_fail={fails[-1].name}")

    pipe = _load_json(run_root / "pipeline_done.json")
    if pipe and pipe.get("status"):
        notes.append(f"pipeline={pipe['status']}")

    return "; ".join(notes)


def _skip_run_name(name: str) -> bool:
    if name.startswith("."):
        return True
    return any(tok in name for tok in SKIP_NAME_TOKENS)


def discover_runs(root: Path) -> list[tuple[Path, bool]]:
    """Return list of (run_root, is_legacy)."""
    out: list[tuple[Path, bool]] = []
    runs = root / "runs"
    if runs.is_dir():
        for p in sorted(runs.iterdir()):
            if p.is_dir() and not _skip_run_name(p.name):
                out.append((p, True))
    unif = root / "runs_unif"
    if unif.is_dir():
        for model_dir in sorted(unif.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            if model_dir.name not in {"caduceus", "legnet"}:
                continue
            for p in sorted(model_dir.iterdir()):
                if p.is_dir() and not _skip_run_name(p.name):
                    out.append((p, False))
    return out


def collect_row(
    run_root: Path,
    *,
    is_legacy: bool,
    stage: str,
    queue_text: str,
) -> dict[str, Any] | None:
    run_id = run_root.name
    if stage == "direct":
        train_dir = run_root / "direct"
    else:
        train_dir = run_root / "adversarial" / "train"

    # Include row if stage exists or split exists (partial)
    stage_exists = train_dir.is_dir() or (
        stage == "adversarial" and (run_root / "adversarial").is_dir()
    )
    if not stage_exists and not (run_root / "split.csv").is_file():
        return None
    # For adversarial: skip entirely if no adversarial dir at all
    if stage == "adversarial" and not (run_root / "adversarial").is_dir():
        return None

    tdir: Path | None = train_dir if train_dir.is_dir() else None
    if tdir and not (tdir / "best_model" / "best_meta.json").is_file():
        # still include if any logs / metrics
        has_any = (tdir / "logs").is_dir() or (tdir / "metrics_summary.json").is_file()
        if not has_any and stage == "adversarial":
            # empty adversarial stub
            if not list((run_root / "adversarial").glob("**/best_meta.json")):
                # keep row as incomplete if adversarial dir has content beyond empty
                pass

    model = _infer_model(run_id, tdir, run_root)
    split_type = _infer_split_type(run_id, run_root)
    params = _split_params(run_id, run_root, split_type)
    counts = _split_counts(run_root)
    pearsons = (
        _extract_pearsons(tdir)
        if tdir
        else {"train": None, "val": None, "test": None, "zsv": None}
    )
    best_ep, final_ep = _best_final_epochs(tdir) if tdir else (None, None)
    done = _done_label(
        is_legacy=is_legacy,
        train_dir=tdir,
        run_root=run_root,
        stage=stage,
        queue_text=queue_text,
    )
    additional = _additional(
        train_dir=tdir, run_root=run_root, stage=stage, pearsons=pearsons
    )

    return {
        "run": _parse_run_number(run_id),
        "run_id": run_id,
        "model": model,
        "train": counts["train"] if counts["train"] is not None else "",
        "test": counts["test"] if counts["test"] is not None else "",
        "val": counts["val"] if counts["val"] is not None else "",
        "split type": split_type,
        "split params": params,
        "best epoch": best_ep if best_ep is not None else "",
        "final epoch": final_ep if final_ep is not None else "",
        "zsv pearson": _fmt(pearsons["zsv"]),
        "val pearson": _fmt(pearsons["val"]),
        "test pearson": _fmt(pearsons["test"]),
        "train pearson": _fmt(pearsons["train"]),
        "done": done,
        "is legacy": "yes" if is_legacy else "no",
        "additional": additional,
    }


def rows_to_markdown(rows: list[dict[str, Any]], title: str) -> str:
    lines = [f"# {title}", "", f"Generation date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", ""]
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("|" + "|".join(["---"] * len(COLUMNS)) + "|")
    for r in rows:
        cells = []
        for c in COLUMNS:
            v = r.get(c, "")
            s = "" if v is None else str(v)
            s = s.replace("|", "\\|")
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def collect_all(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queue_path = project_root / "queue.md"
    queue_text = queue_path.read_text(encoding="utf-8", errors="ignore") if queue_path.is_file() else ""
    direct: list[dict[str, Any]] = []
    adv: list[dict[str, Any]] = []
    for run_root, is_legacy in discover_runs(project_root):
        drow = collect_row(
            run_root, is_legacy=is_legacy, stage="direct", queue_text=queue_text
        )
        if drow:
            # Skip pure probes already filtered; skip if absolutely empty
            direct.append(drow)
        arow = collect_row(
            run_root, is_legacy=is_legacy, stage="adversarial", queue_text=queue_text
        )
        if arow:
            adv.append(arow)

    def sort_key(r: dict[str, Any]) -> tuple:
        try:
            n = int(r["run"]) if r["run"] != "" else 10**9
        except ValueError:
            n = 10**9
        return (0 if r["is legacy"] == "no" else 1, n, r["run_id"], r["model"])

    direct.sort(key=sort_key)
    adv.sort(key=sort_key)
    return direct, adv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <root>/docs",
    )
    args = ap.parse_args(argv)
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "docs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    direct, adv = collect_all(root)

    write_csv(out_dir / "run_results_direct.csv", direct)
    write_csv(out_dir / "run_results_adversarial.csv", adv)
    (out_dir / "run_results_direct.md").write_text(
        rows_to_markdown(direct, "Run results — direct"), encoding="utf-8"
    )
    (out_dir / "run_results_adversarial.md").write_text(
        rows_to_markdown(adv, "Run results — adversarial"), encoding="utf-8"
    )

    # Combined JSON for canvas / downstream
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_direct": len(direct),
        "n_adversarial": len(adv),
        "direct": direct,
        "adversarial": adv,
    }
    (out_dir / "run_results_tables.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print(f"direct={len(direct)} adversarial={len(adv)} → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
