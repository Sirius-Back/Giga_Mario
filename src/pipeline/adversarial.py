"""Build an adversarial panel with the same structural contracts.

After a new random split for *training folds*, call ``apply_fold_class_targets``
with the **previous (direct / M1) ``split.csv``** so PREDICT becomes M2-style
fold-class encodings (previous train/val/test → 0/1/2) before materialize/train.

Using the new adversarial split for labels would make each train/val/test bucket
constant (all-0 / all-1 / all-2) and break the fold-membership task.
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


def write_id_csv_from_split(split_csv: Path, out_csv: Path) -> Path:
    """Write ``ID.csv`` listing exactly the IDs in ``split.csv``.

    Use for adversarial random re-split so assignment stays ⊆ the direct/M1
    label set (e.g. pangenome MARKED∩PARSED subsets that omit some panel IDs).
    """
    split_csv = Path(split_csv)
    out_csv = Path(out_csv)
    rows = read_csv(split_csv)
    if not rows:
        raise ValueError(f"split.csv is empty: {split_csv}")
    if "ID" not in rows[0]:
        raise ValueError(f"split.csv missing ID column: {split_csv}")
    out_rows = [{"ID": r["ID"].strip()} for r in rows if r.get("ID", "").strip()]
    if not out_rows:
        raise ValueError(f"no IDs in split.csv: {split_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, out_rows, ["ID"])
    return out_csv


def write_fold_csv_from_split(split_csv: Path, out_csv: Path) -> Path:
    """Write ``fold.csv`` (ID|fold) from ``split.csv`` for the same ID set.

    Preserves ZSV fold labels when adversarial random assignment must not
    reassign zero-shot IDs, while staying ⊆ the direct split ID set.
    """
    split_csv = Path(split_csv)
    out_csv = Path(out_csv)
    rows = read_csv(split_csv)
    if not rows:
        raise ValueError(f"split.csv is empty: {split_csv}")
    required = {"ID", "fold"}
    if required - set(rows[0]):
        raise ValueError(
            f"split.csv missing {sorted(required - set(rows[0]))}: {split_csv}"
        )
    out_rows = [
        {"ID": r["ID"].strip(), "fold": r["fold"].strip()}
        for r in rows
        if r.get("ID", "").strip()
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, out_rows, ["ID", "fold"])
    return out_csv


def apply_fold_class_targets(
    *,
    predict_root: Path,
    split_csv: Path | None = None,
    label_split_csv: Path | None = None,
    class_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Rewrite non-ZSV PREDICT targets to **previous** fold-class encodings.

    ``label_split_csv`` (preferred) or ``split_csv`` must be the **previous**
    direct/M1 ``split.csv`` (train/val/test membership to encode). Do **not**
    pass the new adversarial training split here.

    Uses the Locked M1/M2 map ``train→0, val→1, test→2`` (``M1_FOLD_TO_CLASS``).
    ZSV rows keep their existing continuous ``predict_var1``. Destination files
    are unlinked before write so hardlinked source panels stay intact.

    Panel IDs absent from the previous split (e.g. outside a pangenome
    MARKED∩PARSED subset) are left continuous and counted as
    ``n_skipped_unlabeled`` — callers should restrict the adversarial random
    ``id_csv`` to ``write_id_csv_from_split(label_split_csv, …)`` so those IDs
    are not assigned train/test/val.
    """
    predict_root = Path(predict_root)
    if (predict_root / "PREDICT").is_dir():
        predict_root = predict_root / "PREDICT"
    if not predict_root.is_dir():
        raise FileNotFoundError(f"PREDICT root missing: {predict_root}")

    label_csv = Path(label_split_csv or split_csv or "")
    if not label_csv.is_file():
        raise FileNotFoundError(
            "label_split_csv/split_csv missing — pass the previous (direct/M1) "
            f"split.csv for fold-class labels (got {label_split_csv!r} / {split_csv!r})"
        )

    mapping = dict(class_map or M1_FOLD_TO_CLASS)
    split_rows = read_csv(label_csv)
    if not split_rows:
        raise ValueError(f"split.csv is empty: {label_csv}")

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
    skipped_unlabeled = 0
    for row in rows:
        rid = row["id"].strip()
        if rid in zsv_ids:
            kept_zsv += 1
            continue
        if rid not in id_to_class:
            skipped_unlabeled += 1
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

    if mapped == 0:
        raise ValueError(
            "no PREDICT IDs mapped to fold-class labels from "
            f"{label_csv} (n_skipped_unlabeled={skipped_unlabeled})"
        )

    fields = list(rows[0].keys())
    if predict_csv.exists():
        predict_csv.unlink()
    write_csv(predict_csv, rows, fields)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fold_class",
        "label_source": "previous_split_m1",
        "class_map": mapping,
        "n_mapped": mapped,
        "n_zsv_kept_continuous": kept_zsv,
        "n_skipped_unlabeled": skipped_unlabeled,
        "predict_root": str(predict_root),
        "split_csv": str(label_csv),
        "label_split_csv": str(label_csv),
    }
    meta_path = predict_root / "predict_target.json"
    if meta_path.exists() or meta_path.is_symlink():
        meta_path.unlink()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def setup_adversarial_random_fold_class(
    *,
    adv_root: Path,
    label_split_csv: Path,
    parsed_target: Path,
    parsed_data: Path,
    fold_csv: Path | None = None,
    seed: int,
    ratios: tuple[float, float, float],
    intersect_allow: bool = True,
    build_legnet_input: bool = True,
) -> tuple[Path, Path]:
    """Copy panel → random re-split on label IDs → fold-class PREDICT → materialize.

    Returns ``(adv_split_csv, adv_legnet_tsv_or_split)``.

    ``fold_csv`` is ignored for assignment scope: fold labels are taken from
    ``label_split_csv`` so ZSV stays marked and IDs stay ⊆ the direct split
    (avoids full-panel fold.csv vs subset id_csv mismatch).

    ``build_legnet_input``: when True (LegNet runs), write ``legnet_input/all.tsv``
    (requires 230 bp sequences). Set **False for Caduceus** — Caduceus trains from
    ``SPLIT/`` via ``adapt_split_for_caduceus`` and must not run the 230 bp TSV
    builder on long/short Caduceus panels.
    """
    _ = fold_csv
    from .split import run_split
    from .split_predict import run_split_predict

    adv_root = Path(adv_root)
    label_split_csv = Path(label_split_csv)
    run_adversarial(
        outdir_new=adv_root,
        split_csv=label_split_csv,
        parsed_target=parsed_target,
        parsed_data=parsed_data,
        intersect_allow=intersect_allow,
    )
    id_csv = write_id_csv_from_split(label_split_csv, adv_root / "ID_from_direct_split.csv")
    fold_from_split = write_fold_csv_from_split(
        label_split_csv, adv_root / "fold_from_direct_split.csv"
    )
    adv_split = run_split_predict(
        outdir=adv_root,
        type="random",
        seed=seed,
        id_csv=id_csv,
        fold_csv=fold_from_split,
        ratios=ratios,
    )
    apply_fold_class_targets(
        predict_root=adv_root / "PREDICT",
        label_split_csv=label_split_csv,
    )
    run_split(
        adv_split,
        parsed_target=adv_root / "PREDICT",
        parsed_data=adv_root / "PARSED",
        outdir=adv_root,
        strategy="traintestval",
        intersect_allow=intersect_allow,
        id_csv=id_csv,
    )
    if not build_legnet_input:
        # Caduceus (and other non-LegNet) train on SPLIT trees, not LegNet TSV.
        return adv_split, adv_root / "SPLIT"
    from .legnet_input import build_legnet_tsv

    adv_tsv = build_legnet_tsv(
        split_root=adv_root / "SPLIT",
        out_tsv=adv_root / "legnet_input" / "all.tsv",
    )
    return adv_split, Path(adv_tsv)


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
    # Always byte-copy split.csv (never hardlink): adversarial re-split must not
    # mutate the previous/direct split used for fold-class labels.
    dest_split = outdir_new / "split.csv"
    if dest_split.exists() or dest_split.is_symlink():
        dest_split.unlink()
    shutil.copy2(split_csv, dest_split)
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
