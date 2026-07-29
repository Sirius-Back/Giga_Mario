"""Shared CSV / ID helpers for the universal pipeline."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

ID_CSV_COLUMNS = [
    "genome",
    "chr",
    "pos1",
    "pos2",
    "gene_nameORnon_coding_ID",
    "raw_target_ID",
    "ID",
]

FOLD_CSV_MIN = ["ID", "fold"]
STRAT_CSV_MIN = ["ID", "strat1"]
SPLIT_CSV_COLUMNS = ["ID", "train_test", "fold"]
PREDICT_CSV_ID_COL = "id"
INTERSECT_COLUMNS = ["ID1", "ID2", "intersection_size"]

MARKED_HEADER_FIELDS = [
    "genome",
    "chr",
    "pos1",
    "pos2",
    "gene_nameORnon_coding_ID",
    "raw_target_ID",
    "ID",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str], *, delimiter: str = "|") -> None:
    ensure_dir(path.parent)
    path = Path(path)
    # Break hardlinks so rewriting adversarial/split.csv cannot mutate the source panel.
    if path.exists() or path.is_symlink():
        path.unlink()
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: Path, *, delimiter: str = "|") -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def require_columns(rows: list[dict[str, str]], required: Iterable[str], *, label: str) -> None:
    if not rows:
        # header-only files still need validation of fieldnames via separate check
        return
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{label} missing columns {missing}; have {list(rows[0])}")


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "sample"


def make_mapped_unique_id(sample_id: str, region_id: str) -> str:
    """Composite training ID for a mapped ``(sample_id, region_id)`` pair.

    Used as the unique ``predict.csv`` ``id`` and as the shared filename stem for
    SPLIT/PREDICT and SPLIT/FASTA artifacts.
    """
    return f"{sanitize_filename(sample_id)}__{sanitize_filename(str(region_id))}"


def load_unique_id_csv_ids(id_csv: Path) -> list[str]:
    """Load ID.csv and return ordered unique ``ID`` values (blank/dupes raise)."""
    rows = read_csv(Path(id_csv))
    if not rows:
        raise ValueError(f"ID.csv is empty: {id_csv}")
    if "ID" not in rows[0]:
        raise ValueError(f"ID.csv missing column 'ID'; have {list(rows[0])}")
    ids: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        rid = str(row["ID"]).strip()
        if not rid:
            raise ValueError(f"ID.csv has blank ID at row {row_number}")
        if rid in seen:
            raise ValueError(f"ID.csv has duplicate ID {rid!r} at row {row_number}")
        seen.add(rid)
        ids.append(rid)
    return ids


def index_unique_predict_rows(
    rows: list[dict[str, str]], *, label: str = "predict.csv"
) -> dict[str, dict[str, str]]:
    """Index predict rows by ``id``; blank or duplicate IDs raise."""
    by_id: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        rid = str(row.get("id", "")).strip()
        if not rid:
            raise ValueError(f"{label} has blank id at row {row_number}")
        if rid in by_id:
            raise ValueError(f"{label} has duplicate id {rid!r} at row {row_number}")
        by_id[rid] = row
    if not by_id:
        raise ValueError(f"{label} contains no usable prediction IDs")
    return by_id


def assert_matching_artifact_ids(
    *,
    fasta_dir: Path,
    predict_dir: Path,
    predict_by_id: dict[str, dict[str, str]],
    bucket: str,
) -> None:
    """Require identical unique ID sets for FASTA, PREDICT files, and predict.csv."""
    fasta_ids = {p.stem for p in fasta_dir.glob("*.ext") if p.is_file()}
    pred_file_ids = {p.stem for p in predict_dir.glob("*.ext") if p.is_file()}
    csv_ids = set(predict_by_id)
    if fasta_ids != pred_file_ids or fasta_ids != csv_ids:
        only_fasta = sorted(fasta_ids - csv_ids)[:5]
        only_pred = sorted(pred_file_ids - fasta_ids)[:5]
        only_csv = sorted(csv_ids - fasta_ids)[:5]
        raise ValueError(
            f"{bucket}: IDs must be unique and identical across "
            f"FASTA/*.ext, PREDICT/*.ext, and predict.csv; "
            f"n_fasta={len(fasta_ids)} n_predict_files={len(pred_file_ids)} "
            f"n_predict_csv={len(csv_ids)}; "
            f"examples only_fasta={only_fasta} only_predict_files={only_pred} "
            f"only_predict_csv={only_csv}"
        )


def checkout_ids_before_split(
    *,
    predict_root: Path,
    parsed_root: Path,
    split_ids: list[str],
    sample_id: str | None = None,
    id_csv: Path | None = None,
    intersect_allow: bool = True,
) -> tuple[bool, list[dict[str, str]]]:
    """
    Pre-split checkout of region IDs and prediction rows.

    - ``ID.csv`` (optional) must have unique ``ID`` values; split IDs ⊆ ID.csv.
    - Merged PREDICT: ``predict.csv`` ``id`` must already be unique.
    - Mapped PREDICT: source may repeat region ``id`` across samples; materialize
      will emit composite unique IDs. Optional ``sample_id`` filters samples.
    """
    if id_csv is not None:
        id_csv_set = set(load_unique_id_csv_ids(id_csv))
        unknown = [i for i in split_ids if i not in id_csv_set]
        if unknown:
            raise ValueError(
                f"split.csv contains ID absent from ID.csv (first: {unknown[0]!r})"
            )

    predict_csv = Path(predict_root) / "predict.csv"
    rows = read_csv(predict_csv)
    if not rows:
        raise ValueError(f"predict.csv is empty: {predict_csv}")
    if "id" not in rows[0]:
        raise ValueError(f"predict.csv missing column 'id'; have {list(rows[0])}")

    mapped = "sample_id" in rows[0]
    if mapped and sample_id is not None:
        rows = [r for r in rows if str(r.get("sample_id", "")).strip() == sample_id]
        if not rows:
            raise ValueError(
                f"No predict.csv rows for sample_id={sample_id!r} under {predict_csv}"
            )

    if mapped:
        seen_keys: set[tuple[str, str]] = set()
        cleaned: list[dict[str, str]] = []
        for row_number, row in enumerate(rows, start=2):
            rid = str(row.get("id", "")).strip()
            sid = str(row.get("sample_id", "")).strip()
            if not rid:
                raise ValueError(f"{predict_csv} has blank id at row {row_number}")
            if not sid:
                raise ValueError(f"{predict_csv} has blank sample_id at row {row_number}")
            key = (rid, sid)
            if key in seen_keys:
                raise ValueError(
                    f"{predict_csv} has duplicate (id, sample_id)=({rid!r}, {sid!r})"
                )
            seen_keys.add(key)
            row = dict(row)
            row["id"] = rid
            row["sample_id"] = sid
            cleaned.append(row)
        rows = cleaned
    else:
        # Ensures source id column is already unique for merged panels.
        index_unique_predict_rows(rows, label=str(predict_csv))

    split_set = set(split_ids)
    present_regions = {str(r["id"]).strip() for r in rows}
    missing_regions = sorted(split_set - present_regions)
    if missing_regions and not intersect_allow:
        raise FileNotFoundError(
            f"split IDs missing from predict.csv (first: {missing_regions[0]!r})"
        )

    # Index mapped rows by region id once — avoid O(|split| × |predict|) scans.
    rows_by_region: dict[str, list[dict[str, str]]] = {}
    if mapped:
        for row in rows:
            rows_by_region.setdefault(row["id"], []).append(row)

    # Strict mode only: verify PARSED/PREDICT files exist (intersect_allow skips).
    if not intersect_allow:
        for rid in split_ids:
            if rid not in present_regions:
                continue
            safe = sanitize_filename(rid)
            parsed_path = Path(parsed_root) / f"{safe}.ext"
            if not parsed_path.is_file():
                raise FileNotFoundError(f"PARSED/{safe}.ext missing for ID {rid!r}")
            if mapped:
                for prow in rows_by_region.get(rid, []):
                    pred_path = (
                        Path(predict_root)
                        / sanitize_filename(prow["sample_id"])
                        / f"{safe}.ext"
                    )
                    if not pred_path.is_file():
                        raise FileNotFoundError(
                            f"PREDICT/{sanitize_filename(prow['sample_id'])}/{safe}.ext "
                            f"missing for ID {rid!r}"
                        )
            else:
                pred_path = Path(predict_root) / f"{safe}.ext"
                if not pred_path.is_file():
                    raise FileNotFoundError(f"PREDICT/{safe}.ext missing for ID {rid!r}")

    return mapped, rows


def parse_gtf_attrs(attr_field: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\s+"([^"]*)"', attr_field):
        out[m.group(1)] = m.group(2)
    return out


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks).upper()
    return seqs


def write_fasta_record(path: Path, header_fields: Sequence[str], sequence: str) -> None:
    ensure_dir(path.parent)
    header = "|" + "|".join(str(x) for x in header_fields)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            fh.write(sequence[i : i + 80] + "\n")


def parse_marked_header(header: str) -> dict[str, str]:
    h = header[1:] if header.startswith(">") else header
    if h.startswith("|"):
        h = h[1:]
    parts = h.split("|")
    if len(parts) < 7:
        raise ValueError(f"Marked FASTA header needs 7 fields, got {parts!r}")
    return {
        "genome": parts[0],
        "chr": parts[1],
        "pos1": parts[2],
        "pos2": parts[3],
        "gene_nameORnon_coding_ID": parts[4],
        "raw_target_ID": parts[5],
        "ID": parts[6],
    }
