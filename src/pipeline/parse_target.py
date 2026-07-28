"""Map raw TARGET tables onto PREDICT artifacts (Caduceus / LegNet-ready labels)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.get_mpra import read_wide_row
from src.preprocessing import genome_prefix

from .common import ensure_dir, read_csv, sanitize_filename, write_csv
from .id_rule import run_id_rule


def _load_wide_target(path: Path) -> dict[str, float]:
    """Read a validated wide target CSV using the shared MPRA reader."""
    header, values = read_wide_row(path)
    out: dict[str, float] = {}
    for name, value in zip(header, values):
        key = name.strip()
        if key and key not in out:
            out[key] = float(value)
    return out


def _discover_genome_csvs(folder: Path) -> dict[str, Path]:
    """Map genome accession / stem to a preferred wide target CSV."""
    found: dict[str, Path] = {}
    for path in sorted(folder.glob("*.csv")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        stem = path.stem
        # Preserve exact names first, then make GCF accession matching available
        # for assembly-stem files such as GCF_000005845.2_ASM584v2.csv.
        found.setdefault(stem, path)
        accession = genome_prefix(stem)
        if accession.startswith("GCF_"):
            found.setdefault(accession, path)
    return found


def _read_mappings(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Mappings CSV missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"id", "tpm", "genome"}
        fields = set(reader.fieldnames or [])
        if missing := required - fields:
            raise ValueError(f"Mappings CSV missing columns {sorted(missing)}: {path}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Mappings CSV has no rows: {path}")
    if missing_ids := [row["id"] for row in rows if not row["id"]]:
        raise ValueError(f"Mappings CSV has empty sample ids: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Mappings CSV has duplicate sample ids: {path}")
    return rows


def _resolve_mapping_target(mapping: dict[str, str], target_path: Path) -> Path:
    """Resolve a mapping target, preferring the MPRA-panel basename."""
    source = Path(mapping["tpm"])
    preferred = target_path / source.name
    candidates = [preferred]
    if source.is_absolute():
        candidates.append(source)
    else:
        candidates.extend([Path.cwd() / source, source])

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"Mapping sample {mapping['id']!r} has no readable wide CSV; "
        f"tried MPRA basename {preferred} and mapping path {source}"
    )


def _ids_for_genome(id_rows: list[dict[str, str]], genome: str) -> list[dict[str, str]]:
    """Match mapping genomes to ID.csv accessions without losing version suffixes."""
    wanted = genome_prefix(genome)
    return [row for row in id_rows if genome_prefix(row["genome"]) == wanted]


def _values_by_panel_id(target: dict[str, float], id_csv: Path) -> dict[str, float]:
    """Resolve TARGET header keys → panel IDs via ``id_rule``; do not invent IDs.

    For each distinct target key, map with
    ``run_id_rule([key], id_csv, id_col_1="raw_target_ID", id_col_2="ID")``;
    if empty, retry with ``id_col_1="gene_nameORnon_coding_ID"``. Multi-hit
    expansion assigns the same scalar to every matched ID. First assignment
    wins so raw_target_ID remaps take priority over gene-name remaps. Keys that
    match no ID.csv row are ignored (callers use absent→0).
    """
    by_id: dict[str, float] = {}
    pending: list[str] = []
    for key, value in target.items():
        ids = run_id_rule([key], id_csv, id_col_1="raw_target_ID", id_col_2="ID")
        if ids:
            scalar = float(value)
            for rid in ids:
                by_id.setdefault(str(rid), scalar)
        else:
            pending.append(key)
    for key in pending:
        ids = run_id_rule(
            [key], id_csv, id_col_1="gene_nameORnon_coding_ID", id_col_2="ID"
        )
        scalar = float(target[key])
        for rid in ids:
            by_id.setdefault(str(rid), scalar)
    return by_id


def run_parse_target(
    target_path: Path,
    *,
    outdir: Path,
    id_csv: Path,
    input_type: str = "folder",
    to_type: str = "caduceus",
    mappings: Path | None = None,
) -> dict[str, Path]:
    """
    Write `{outdir}/PREDICT/ID.ext` and `{outdir}/PREDICT/predict.csv`.

    `to_type`:
      - caduceus: predict_var1 == scalar target; per-ID `.ext` is one float
      - legnet: predict_var1 == mean_value scalar target; `.ext` is one float

    With mappings, each mapping row is an independent sample.  Its artifacts
    live at `PREDICT/{sample_id}/{ID}.ext`, while `predict.csv` records
    `id|sample_id|predict_var1`.
    """
    if to_type not in {"caduceus", "legnet"}:
        raise ValueError(f"to_type must be caduceus|legnet, got {to_type!r}")
    if input_type != "folder":
        raise ValueError(f"input_type={input_type!r} not implemented (use folder)")

    target_path = Path(target_path)
    if not target_path.is_dir():
        raise FileNotFoundError(f"TARGET folder missing: {target_path}")

    id_csv = Path(id_csv)
    id_rows = read_csv(id_csv)
    if not id_rows:
        raise ValueError(f"Empty ID.csv: {id_csv}")

    predict_dir = ensure_dir(Path(outdir) / "PREDICT")
    predict_rows: list[dict[str, Any]] = []

    if mappings is not None:
        for mapping in _read_mappings(Path(mappings)):
            target = _load_wide_target(_resolve_mapping_target(mapping, target_path))
            by_id = _values_by_panel_id(target, id_csv)
            sample_id = mapping["id"]
            sample_dir = ensure_dir(predict_dir / sanitize_filename(sample_id))
            for row in _ids_for_genome(id_rows, mapping["genome"]):
                rid = row["ID"]
                value = float(by_id.get(str(rid), 0.0))
                (sample_dir / f"{sanitize_filename(rid)}.ext").write_text(
                    f"{value}\n", encoding="utf-8"
                )
                predict_rows.append(
                    {"id": rid, "sample_id": sample_id, "predict_var1": value}
                )
        fields = ["id", "sample_id", "predict_var1"]
    else:
        csv_index = _discover_genome_csvs(target_path)
        if not csv_index:
            raise FileNotFoundError(f"No non-empty target CSV files under {target_path}")
        targets_by_genome = {
            key: _load_wide_target(path) for key, path in csv_index.items()
        }
        by_id_by_genome = {
            key: _values_by_panel_id(target, id_csv)
            for key, target in targets_by_genome.items()
        }
        for row in id_rows:
            genome = row["genome"]
            by_id = (
                by_id_by_genome.get(genome)
                or by_id_by_genome.get(genome_prefix(genome))
            )
            rid = row["ID"]
            value = float(by_id.get(str(rid), 0.0)) if by_id is not None else 0.0
            (predict_dir / f"{sanitize_filename(rid)}.ext").write_text(
                f"{value}\n", encoding="utf-8"
            )
            predict_rows.append({"id": rid, "predict_var1": value})
        fields = ["id", "predict_var1"]

    predict_csv = predict_dir / "predict.csv"
    write_csv(predict_csv, predict_rows, fields)
    return {"predict_dir": predict_dir, "predict_csv": predict_csv}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="parse_target → PREDICT/")
    p.add_argument("--target", required=True, type=Path, help="Folder of wide target CSVs")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--id-csv", required=True, type=Path)
    p.add_argument("--input-type", default="folder")
    p.add_argument("--to-type", default="caduceus", choices=["caduceus", "legnet"])
    p.add_argument(
        "--mappings",
        type=Path,
        default=None,
        help="Optional comma-delimited sample mapping CSV (id,tpm,genome)",
    )
    args = p.parse_args(argv)
    out = run_parse_target(
        args.target,
        outdir=args.outdir,
        id_csv=args.id_csv,
        input_type=args.input_type,
        to_type=args.to_type,
        mappings=args.mappings,
    )
    print(out["predict_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
