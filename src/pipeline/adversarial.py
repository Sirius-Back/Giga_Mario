"""Build an adversarial panel with the same structural contracts."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .common import ensure_dir, read_csv


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    Copy panel structure into `outdir_new` so parse_target can be re-run.

    Accepts either a prior `outdir` containing PREDICT/PARSED/split.csv,
    or explicit split_csv + parsed_target + parsed_data.
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
