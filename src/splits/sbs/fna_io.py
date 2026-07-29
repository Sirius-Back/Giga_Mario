"""FNA loaders for SBS distance backends.

Supported modes:
  - ``directory`` (default): one FASTA file per region under a folder
  - ``file``: single multi-FASTA (hook; implemented for completeness)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

FastaMode = Literal["directory", "file", "auto"]

# Include Caduceus/LegNet PARSED ``.ext`` (raw sequence, one file per region).
_FASTA_SUFFIXES = {".fa", ".fna", ".fasta", ".fas", ".ext"}


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    """Return list of (header_token, sequence) from one FASTA path.

    Also accepts raw single-sequence files (no ``>`` header), common for
    ``PARSED/*.ext`` panels — header falls back to the filename stem.
    """
    if not path.is_file():
        raise FileNotFoundError(f"FASTA missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"FASTA is empty: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if ">" not in text:
        seq = "".join(text.split()).upper()
        if not seq:
            raise ValueError(f"empty sequence in {path}")
        return [(path.stem, seq)]
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    return records


def _region_id_from_header(header: str, *, fallback: str) -> str:
    """Prefer trailing pipe field (MARKED / ID.csv layout); else first token."""
    if "|" in header:
        parts = [p.strip() for p in header.split("|") if p.strip()]
        if parts:
            return parts[-1]
    token = header.split()[0] if header.split() else fallback
    return token or fallback


def _region_id_from_filename(path: Path) -> str:
    return path.stem


def iter_fasta_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"FNA directory missing: {directory}")
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _FASTA_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no FASTA files under {directory}")
    return paths


def load_fna_directory(
    directory: Path,
    *,
    ids: Iterable[str] | None = None,
    max_ids: int | None = None,
) -> dict[str, str]:
    """Load ``{region_id: sequence}`` from one-file-per-region directory.

    Region id = filename stem (MARKED uses ``{ID}.fa``). When ``ids`` is set,
    only those stems are loaded (missing files raise).
    """
    directory = Path(directory)
    allow = set(ids) if ids is not None else None
    out: dict[str, str] = {}
    for path in iter_fasta_paths(directory):
        rid = _region_id_from_filename(path)
        if allow is not None and rid not in allow:
            continue
        records = _read_fasta_records(path)
        # One record expected; if multi-record, keep first matching stem/header.
        seq = records[0][1]
        if not seq:
            raise ValueError(f"empty sequence in {path}")
        out[rid] = seq
        if max_ids is not None and len(out) >= max_ids:
            break
    if allow is not None:
        missing = sorted(allow - set(out))
        if missing:
            raise FileNotFoundError(
                f"FNA directory missing {len(missing)} id(s); "
                f"example={missing[0]!r} under {directory}"
            )
    if not out:
        raise ValueError(f"no sequences loaded from {directory}")
    return out


def load_fna_file(
    path: Path,
    *,
    ids: Iterable[str] | None = None,
    max_ids: int | None = None,
) -> dict[str, str]:
    """Load ``{region_id: sequence}`` from a multi-FASTA file."""
    path = Path(path)
    allow = set(ids) if ids is not None else None
    out: dict[str, str] = {}
    for idx, (header, seq) in enumerate(_read_fasta_records(path)):
        rid = _region_id_from_header(header, fallback=f"rec{idx}")
        if allow is not None and rid not in allow:
            continue
        if not seq:
            raise ValueError(f"empty sequence for {rid!r} in {path}")
        if rid in out:
            raise ValueError(f"duplicate FASTA id {rid!r} in {path}")
        out[rid] = seq
        if max_ids is not None and len(out) >= max_ids:
            break
    if allow is not None:
        missing = sorted(allow - set(out))
        if missing:
            raise ValueError(
                f"multi-FASTA missing {len(missing)} id(s); example={missing[0]!r}"
            )
    if not out:
        raise ValueError(f"no sequences loaded from {path}")
    return out


def resolve_fna_mode(fna: Path, mode: FastaMode = "auto") -> FastaMode:
    fna = Path(fna)
    if mode != "auto":
        return mode
    if fna.is_dir():
        return "directory"
    if fna.is_file():
        return "file"
    raise FileNotFoundError(f"FNA path not found: {fna}")


def load_fna_sequences(
    fna: Path,
    *,
    mode: FastaMode = "auto",
    ids: Iterable[str] | None = None,
    max_ids: int | None = None,
) -> dict[str, str]:
    """Unified loader: directory (default MARKED layout) or single multi-FASTA."""
    resolved = resolve_fna_mode(Path(fna), mode)
    if resolved == "directory":
        return load_fna_directory(Path(fna), ids=ids, max_ids=max_ids)
    return load_fna_file(Path(fna), ids=ids, max_ids=max_ids)
