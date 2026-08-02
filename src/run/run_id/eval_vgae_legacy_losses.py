"""Re-score existing VGAE split.csv under all L_hom aggregations + SD balance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.splits.vgae.homology_loss import (
    evaluate_split_all_aggs,
    load_homology_groups,
    sd_group_balance_report,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "VGAE" / "legacy_eval_existing_models.json"


def _load_split_labels(split_csv: Path, ids: list[str]) -> list[str]:
    id_to_label: dict[str, str] = {}
    with split_csv.open("r", encoding="utf-8", newline="") as fh:
        # Support both CSV and pipe-delimited SBS-style tables
        sample = fh.read(4096)
        fh.seek(0)
        delim = "|" if sample.count("|") >= sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delim)
        for row in reader:
            rid = (
                row.get("id")
                or row.get("ID")
                or row.get("region_id")
                or ""
            ).strip()
            raw = (
                row.get("split")
                or row.get("label")
                or row.get("role")
                or row.get("train_test")
                or row.get("fold")
                or ""
            ).strip().lower()
            lab = raw
            if "train" in raw:
                lab = "train"
            elif "test" in raw:
                lab = "test"
            elif "val" in raw:
                lab = "val"
            if rid and lab in {"train", "test", "val"}:
                id_to_label[rid] = lab
    missing = [i for i in ids if i not in id_to_label]
    if missing:
        raise KeyError(
            f"{split_csv}: {len(missing)} ids missing labels; example={missing[0]!r}"
        )
    return [id_to_label[i] for i in ids]


def eval_run(run_dir: Path) -> dict:
    split = run_dir / "split.csv"
    if not split.is_file():
        return {"run": run_dir.name, "status": "SKIPPED", "reason": "missing split"}
    # Region-level split.csv is the homology panel (Stage2 hash pack ids are hash_*)
    ids: list[str] = []
    labels: list[str] = []
    with split.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = "|" if sample.count("|") >= sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delim)
        for row in reader:
            rid = (
                row.get("id") or row.get("ID") or row.get("region_id") or ""
            ).strip()
            raw = (
                row.get("split")
                or row.get("label")
                or row.get("role")
                or row.get("train_test")
                or row.get("fold")
                or ""
            ).strip().lower()
            if "train" in raw:
                lab = "train"
            elif "test" in raw:
                lab = "test"
            elif "val" in raw:
                lab = "val"
            else:
                continue
            if rid:
                ids.append(rid)
                labels.append(lab)
    if not ids:
        return {"run": run_dir.name, "status": "SKIPPED", "reason": "empty split"}
    groups = load_homology_groups(ids)
    all_aggs = evaluate_split_all_aggs(labels, groups)
    balance = sd_group_balance_report(labels, groups, max_groups=None, seed=42)
    meta_path = run_dir / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "run": run_dir.name,
        "status": "COMPLETED",
        "loss_mode": meta.get("loss_mode"),
        "reported_best_l_hom": meta.get("best_l_hom"),
        "reported_best_epoch": meta.get("best_epoch"),
        "n_regions": len(ids),
        "all_aggs": all_aggs,
        "sd_balance": balance,
        "counts": {
            "train": labels.count("train"),
            "test": labels.count("test"),
            "val": labels.count("val"),
        },
    }


def main() -> int:
    runs = sorted(
        p
        for p in (ROOT / "VGAE").iterdir()
        if p.is_dir() and p.name.startswith("stage") and (p / "split.csv").is_file()
    )
    rows = [eval_run(p) for p in runs]
    OUT.write_text(json.dumps({"models": rows}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for r in rows:
        if r["status"] != "COMPLETED":
            print(f"  {r['run']}: {r['status']}")
            continue
        mean = r["all_aggs"]["mean"]["l_hom"]
        w = r["all_aggs"]["weighted"]["l_hom"]
        rob = r["all_aggs"]["robust"]["l_hom"]
        logb = r["all_aggs"]["log_balance"]["l_hom"]
        o = r["sd_balance"]["ortho"]
        print(
            f"  {r['run']}: legacy={mean:.4f} weighted={w:.4f} "
            f"robust={rob:.4f} logbal={logb:.4f} | "
            f"ortho p90/med={o['p90_over_median']:.2f} "
            f"top5%mass={o['top5pct_mass_frac']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
