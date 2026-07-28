"""Build an adversarial panel with the same structural contracts.

After a new random split, call ``apply_fold_class_targets`` so PREDICT becomes
M2-style fold-class encodings (train/val/test → 0/1/2) before materialize/train.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.splits.random import M1_FOLD_TO_CLASS

from .common import ensure_dir, read_csv, sanitize_filename, write_csv
from .generate_fold import is_zsv_fold


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _replace_linked_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link_or_copy(path, target)


def _resolve_stage_dir(path: Path, name: str) -> Path:
    resolved = path / name if (path / name).is_dir() else path
    if not resolved.is_dir():
        raise FileNotFoundError(f"Expected {name}/ at {path / name} or {path}")
    return resolved


def _validate_panel(
    split_csv: Path, parsed_target: Path, parsed_data: Path, *, intersect_allow: bool
) -> None:
    from .common import checkout_ids_before_split

    split_rows = read_csv(split_csv)
    if not split_rows:
        raise ValueError(f"split.csv is empty: {split_csv}")
    required_split = {"ID", "train_test", "fold"}
    if required_split - set(split_rows[0]):
        raise ValueError(f"split.csv missing {sorted(required_split - set(split_rows[0]))}")

    split_ids = [row["ID"].strip() for row in split_rows]
    # Pre-split style checkout: allows mapped source; enforces region presence.
    checkout_ids_before_split(
        predict_root=parsed_target,
        parsed_root=parsed_data,
        split_ids=split_ids,
        intersect_allow=intersect_allow,
    )


def _break_write(path: Path, text: str) -> None:
    """Write ``text`` without mutating hardlinked source inodes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_text(text, encoding="utf-8")


def apply_fold_class_targets(
    *,
    predict_root: Path,
    split_csv: Path,
    class_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Rewrite non-ZSV PREDICT targets to fold-class encodings.

    Uses the Locked M1/M2 map ``train→0, val→1, test→2`` (``M1_FOLD_TO_CLASS``).
    ZSV rows keep their existing continuous ``predict_var1``. Destination files
    are unlinked before write so hardlinked source panels stay intact.
    """
    predict_root = Path(predict_root)
    if (predict_root / "PREDICT").is_dir():
        predict_root = predict_root / "PREDICT"
    if not predict_root.is_dir():
        raise FileNotFoundError(f"PREDICT root missing: {predict_root}")

    mapping = dict(class_map or M1_FOLD_TO_CLASS)
    split_rows = read_csv(Path(split_csv))
    if not split_rows:
        raise ValueError(f"split.csv is empty: {split_csv}")

    id_to_class: dict[str, int] = {}
    zsv_ids: set[str] = set()
    for row in split_rows:
        rid = row["ID"].strip()
        tt = row["train_test"].strip().lower()
        if tt in {"validation", "val"}:
            tt = "val"
        if is_zsv_fold(tt) or is_zsv_fold(row.get("fold", "")):
            zsv_ids.add(rid)
            continue
        if tt not in mapping:
            raise ValueError(
                f"Cannot map train_test={tt!r} for ID={rid!r}; "
                f"expected one of {sorted(mapping)}"
            )
        id_to_class[rid] = int(mapping[tt])

    predict_csv = predict_root / "predict.csv"
    if not predict_csv.is_file():
        raise FileNotFoundError(f"Missing {predict_csv}")
    rows = read_csv(predict_csv)
    if not rows or "id" not in rows[0] or "predict_var1" not in rows[0]:
        raise ValueError(f"{predict_csv} must have id|predict_var1")

    mapped = 0
    kept_zsv = 0
    missing: list[str] = []
    for row in rows:
        rid = row["id"].strip()
        if rid in zsv_ids:
            kept_zsv += 1
            continue
        if rid not in id_to_class:
            missing.append(rid)
            continue
        new_val = str(id_to_class[rid])
        row["predict_var1"] = new_val
        mapped += 1
        # Prefer flat PREDICT/{id}.ext; fall back to mapped sample subdirs.
        ext = predict_root / f"{sanitize_filename(rid)}.ext"
        if not ext.is_file() and "sample_id" in row and row["sample_id"].strip():
            ext = (
                predict_root
                / sanitize_filename(row["sample_id"].strip())
                / f"{sanitize_filename(rid)}.ext"
            )
        if not ext.is_file():
            # Composite mapped ids already sanitized in stem form.
            candidates = list(predict_root.rglob(f"{sanitize_filename(rid)}.ext"))
            if len(candidates) == 1:
                ext = candidates[0]
            else:
                raise FileNotFoundError(f"PREDICT .ext missing for {rid}")
        _break_write(ext, new_val + "\n")

    if missing:
        raise ValueError(
            "split.csv IDs missing from class assignment (non-ZSV): "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
        )

    fields = list(rows[0].keys())
    if predict_csv.exists():
        predict_csv.unlink()
    write_csv(predict_csv, rows, fields)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fold_class",
        "class_map": mapping,
        "n_mapped": mapped,
        "n_zsv_kept_continuous": kept_zsv,
        "predict_root": str(predict_root),
        "split_csv": str(split_csv),
    }
    meta_path = predict_root / "predict_target.json"
    if meta_path.exists() or meta_path.is_symlink():
        meta_path.unlink()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def run_adversarial(
    *,
    outdir_new: Path,
    outdir: Path | None = None,
    split_csv: Path | None = None,
    parsed_target: Path | None = None,
    parsed_data: Path | None = None,
    intersect_allow: bool = False,
) -> Path:
    """
    Copy panel structure into `outdir_new` for adversarial re-split + class targets.

    Accepts either a prior `outdir` containing PREDICT/PARSED/split.csv,
    or explicit split_csv + parsed_target + parsed_data.

    Does **not** rewrite PREDICT by itself — call ``apply_fold_class_targets``
    after the adversarial ``split_predict`` produces the new ``split.csv``.
    """
    if outdir is not None:
        src = Path(outdir)
        split_csv = src / "split.csv"
        parsed_target = src / "PREDICT"
        parsed_data = src / "PARSED"
    else:
        if not (split_csv and parsed_target and parsed_data):
            raise ValueError("Provide outdir OR (split_csv, parsed_target, parsed_data)")
        split_csv = Path(split_csv)
        parsed_target = _resolve_stage_dir(Path(parsed_target), "PREDICT")
        parsed_data = _resolve_stage_dir(Path(parsed_data), "PARSED")

    assert split_csv is not None and parsed_target is not None and parsed_data is not None
    split_csv = Path(split_csv)
    parsed_target = _resolve_stage_dir(Path(parsed_target), "PREDICT")
    parsed_data = _resolve_stage_dir(Path(parsed_data), "PARSED")
    _validate_panel(
        split_csv, parsed_target, parsed_data, intersect_allow=intersect_allow
    )

    outdir_new = Path(outdir_new)
    if outdir_new.resolve() in {
        split_csv.parent.resolve(), parsed_target.parent.resolve(), parsed_data.parent.resolve()
    }:
        raise ValueError("outdir_new must differ from the source panel")
    ensure_dir(outdir_new)
    _link_or_copy(split_csv, outdir_new / "split.csv")
    _replace_linked_tree(parsed_target, outdir_new / "PREDICT")
    _replace_linked_tree(parsed_data, outdir_new / "PARSED")

    # Carry only real optional upstream context, preserving parse-target inputs.
    source_root = Path(outdir) if outdir is not None else split_csv.parent
    for name in ("MARKED", "ID.csv", "intersect.csv"):
        source = source_root / name
        if source.is_dir():
            _replace_linked_tree(source, outdir_new / name)
        elif source.is_file():
            _link_or_copy(source, outdir_new / name)

    return outdir_new


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="adversarial panel copy")
    p.add_argument("--outdir-new", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--parsed-target", type=Path, default=None)
    p.add_argument("--parsed-data", type=Path, default=None)
    p.add_argument("--intersect-allow", action="store_true")
    args = p.parse_args(argv)
    print(
        run_adversarial(
            outdir_new=args.outdir_new,
            outdir=args.outdir,
            split_csv=args.split_csv,
            parsed_target=args.parsed_target,
            parsed_data=args.parsed_data,
            intersect_allow=args.intersect_allow,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
