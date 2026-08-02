"""Completeness and ID-consistency checks for a LegNet run unit."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.embed import SEQ_LEN
from src.embed.discover import LegNetRun
from src.pipeline.common import SPLIT_CSV_COLUMNS, read_csv, require_columns

ROLE_ALIASES = {
    "train": "train",
    "test": "test",
    "val": "val",
    "validation": "val",
    "valid": "val",
}

# human_legnet CV fold in legnet_input/all.tsv (see src/pipeline/legnet_input.py)
FOLD_TO_ROLE = {"3": "train", "1": "test", "2": "val"}


@dataclass
class ValidationResult:
    key: str
    status: str  # READY | SKIPPED | FAILED
    reasons: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    n_val: int = 0
    n_tsv: int = 0
    in_ch: int | None = None
    ef_block_sizes: list[int] | None = None
    ckpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_role(raw: str) -> str | None:
    return ROLE_ALIASES.get(str(raw).strip().lower())


def panel_id_from_seq_id(seq_id: str) -> str:
    """Map LegNet ``seq_id`` (e.g. ``ENCSR…__123``) → panel ``split.csv`` ID."""
    if "__" not in seq_id:
        raise ValueError(f"seq_id lacks __panel_id suffix: {seq_id!r}")
    return seq_id.rsplit("__", 1)[-1]


def load_split_roles(split_csv: Path) -> dict[str, set[str]]:
    """Return ``{role: set(panel_ids)}`` for train/test/val."""
    rows = read_csv(split_csv, delimiter="|")
    if not rows:
        with split_csv.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("|")
        missing = [c for c in SPLIT_CSV_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{split_csv} missing columns {missing}")
        return {"train": set(), "test": set(), "val": set()}
    require_columns(rows, SPLIT_CSV_COLUMNS, label=str(split_csv))
    out: dict[str, set[str]] = {"train": set(), "test": set(), "val": set()}
    for row in rows:
        role = _normalize_role(row["train_test"])
        if role is None or role not in out:
            continue
        out[role].add(str(row["ID"]).strip())
    return out


@dataclass
class TsvPanel:
    """rev==0 rows with sequences and roles from CV fold."""

    seq_by_id: dict[str, str]  # seq_id → seq
    role_by_id: dict[str, str]  # seq_id → train|test|val
    panel_id_by_seq: dict[str, str]  # seq_id → panel ID


def load_tsv_panel(
    tsv_path: Path, *, seq_len: int = SEQ_LEN
) -> tuple[TsvPanel, list[str]]:
    """Load rev==0 rows; roles from fold∈{1,2,3}."""
    issues: list[str] = []
    seq_by_id: dict[str, str] = {}
    role_by_id: dict[str, str] = {}
    panel_id_by_seq: dict[str, str] = {}
    with tsv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in {tsv_path}")
        required = {"seq_id", "seq", "mean_value", "fold", "rev"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{tsv_path} missing columns {missing}")
        for i, row in enumerate(reader, start=2):
            try:
                rev = int(float(row["rev"]))
            except (TypeError, ValueError):
                issues.append(f"line {i}: bad rev={row.get('rev')!r}")
                continue
            if rev != 0:
                continue
            sid = str(row["seq_id"]).strip()
            seq = row["seq"]
            if len(seq) != seq_len:
                issues.append(
                    f"line {i}: seq_id={sid} len={len(seq)} != {seq_len}"
                )
                continue
            fold = str(row["fold"]).strip()
            role = FOLD_TO_ROLE.get(fold)
            if role is None:
                issues.append(f"line {i}: unexpected fold={fold!r}")
                continue
            try:
                pid = panel_id_from_seq_id(sid)
            except ValueError as exc:
                issues.append(f"line {i}: {exc}")
                continue
            if sid in seq_by_id:
                issues.append(f"duplicate seq_id={sid}")
                continue
            seq_by_id[sid] = seq
            role_by_id[sid] = role
            panel_id_by_seq[sid] = pid
    return (
        TsvPanel(
            seq_by_id=seq_by_id,
            role_by_id=role_by_id,
            panel_id_by_seq=panel_id_by_seq,
        ),
        issues,
    )


# Back-compat alias used by older tests / callers
def load_tsv_index(
    tsv_path: Path, *, seq_len: int = SEQ_LEN
) -> tuple[dict[str, str], list[str]]:
    panel, issues = load_tsv_panel(tsv_path, seq_len=seq_len)
    return panel.seq_by_id, issues


def validate_run(run: LegNetRun, *, load_ckpt: bool = False) -> ValidationResult:
    """Validate one discover unit. Does not require GPU unless load_ckpt."""
    res = ValidationResult(key=run.key, status="READY", ckpt=str(run.ckpt_path))
    reasons: list[str] = []

    if not run.split_csv.is_file():
        reasons.append(f"missing split.csv: {run.split_csv}")
    if not run.legnet_tsv.is_file():
        reasons.append(f"missing legnet TSV: {run.legnet_tsv}")
    if not run.config_json.is_file():
        reasons.append(f"missing config.json: {run.config_json}")
    if not run.ckpt_path.is_file():
        reasons.append(f"missing checkpoint: {run.ckpt_path}")

    if reasons:
        res.status = "SKIPPED"
        res.reasons = reasons
        return res

    try:
        cfg = json.loads(run.config_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        res.status = "FAILED"
        res.reasons = [f"config.json invalid: {exc}"]
        return res

    in_ch = 4 + int(bool(cfg.get("use_reverse_channel", False)))
    res.in_ch = in_ch
    ef = cfg.get("ef_block_sizes", [80, 96, 112, 128])
    res.ef_block_sizes = list(ef) if isinstance(ef, list) else None
    if in_ch != 4:
        reasons.append(f"unexpected in_ch={in_ch} (expected 4)")
    if list(ef) != [80, 96, 112, 128]:
        reasons.append(f"non-default ef_block_sizes={ef}")

    try:
        split_roles = load_split_roles(run.split_csv)
    except (OSError, ValueError) as exc:
        res.status = "FAILED"
        res.reasons = [f"split.csv: {exc}"]
        return res

    try:
        panel, tsv_issues = load_tsv_panel(run.legnet_tsv)
    except (OSError, ValueError) as exc:
        res.status = "FAILED"
        res.reasons = [f"legnet TSV: {exc}"]
        return res

    reasons.extend(tsv_issues[:20])
    if len(tsv_issues) > 20:
        reasons.append(f"… +{len(tsv_issues) - 20} more TSV issues")

    res.n_tsv = len(panel.seq_by_id)
    res.n_train = sum(1 for r in panel.role_by_id.values() if r == "train")
    res.n_test = sum(1 for r in panel.role_by_id.values() if r == "test")
    res.n_val = sum(1 for r in panel.role_by_id.values() if r == "val")
    if res.n_train == 0:
        reasons.append("train role empty in TSV fold=3")
    if res.n_test == 0:
        reasons.append("test role empty in TSV fold=1")

    # Cross-check TSV roles vs split.csv via panel ID suffix
    mismatches = 0
    missing_in_split = 0
    for sid, role in panel.role_by_id.items():
        pid = panel.panel_id_by_seq[sid]
        # find which split role owns pid
        split_role = None
        for rn, ids in split_roles.items():
            if pid in ids:
                split_role = rn
                break
        if split_role is None:
            missing_in_split += 1
            continue
        if split_role != role:
            mismatches += 1
    if missing_in_split:
        reasons.append(
            f"{missing_in_split} TSV panel IDs absent from split.csv train/test/val"
        )
    if mismatches:
        reasons.append(f"{mismatches} TSV fold vs split.csv role mismatches")

    # split train/test should be covered by TSV (ZSV and other roles may exist
    # only in split.csv — ignore those).
    tsv_panel_ids = set(panel.panel_id_by_seq.values())
    for role_name in ("train", "test", "val"):
        missing = split_roles[role_name] - tsv_panel_ids
        # Allow incomplete val; require train/test coverage ≥ 99%
        if role_name in ("train", "test") and missing:
            frac_missing = len(missing) / max(len(split_roles[role_name]), 1)
            if frac_missing > 0.01:
                reasons.append(
                    f"split {role_name}: {len(missing)}/{len(split_roles[role_name])} "
                    "IDs missing from TSV (>1% threshold)"
                )
            elif missing:
                reasons.append(
                    f"split {role_name}: {len(missing)} IDs missing from TSV "
                    f"(≤1%, soft)"
                )

    if load_ckpt:
        try:
            from src.embed.legnet_extract import load_lit_model

            lit = load_lit_model(run, map_location="cpu")
            _ = lit.model
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"checkpoint load failed: {type(exc).__name__}: {exc}")

    soft_prefixes = (
        "non-default ef_block_sizes",
        "split train: ",  # only soft when ≤1% — handled above with soft wording
        "split test: ",
        "split val: ",
    )
    # Soft if message contains "(≤1%, soft)"
    hard = [
        r
        for r in reasons
        if not r.startswith("non-default ef_block_sizes")
        and "(≤1%, soft)" not in r
    ]
    res.reasons = reasons
    if not hard:
        res.status = "READY"
        return res

    skip_markers = (
        "missing split.csv",
        "missing legnet TSV",
        "missing config",
        "missing checkpoint",
    )
    if any(any(m in r for m in skip_markers) for r in hard):
        res.status = "SKIPPED"
    else:
        res.status = "FAILED"
    return res


def validate_all(
    runs: list[LegNetRun], *, load_ckpt: bool = False
) -> list[ValidationResult]:
    return [validate_run(r, load_ckpt=load_ckpt) for r in runs]


def write_validation_report(
    results: list[ValidationResult], path: Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_total": len(results),
        "n_ready": sum(1 for r in results if r.status == "READY"),
        "n_skipped": sum(1 for r in results if r.status == "SKIPPED"),
        "n_failed": sum(1 for r in results if r.status == "FAILED"),
        "runs": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
