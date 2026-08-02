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


def load_split_roles(split_csv: Path) -> dict[str, dict[str, set[str]]]:
    """Return ``{role: set(ids)}`` for train/test/val (string IDs)."""
    rows = read_csv(split_csv, delimiter="|")
    if not rows:
        # header-only: still need columns from file
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


def load_tsv_index(
    tsv_path: Path, *, seq_len: int = SEQ_LEN
) -> tuple[dict[str, str], list[str]]:
    """Map seq_id → seq for rev==0 rows; return (index, issues)."""
    issues: list[str] = []
    index: dict[str, str] = {}
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
            if sid in index:
                issues.append(f"duplicate seq_id={sid}")
                continue
            index[sid] = seq
    return index, issues


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
        roles = load_split_roles(run.split_csv)
    except (OSError, ValueError) as exc:
        res.status = "FAILED"
        res.reasons = [f"split.csv: {exc}"]
        return res

    res.n_train = len(roles["train"])
    res.n_test = len(roles["test"])
    res.n_val = len(roles["val"])
    if res.n_train == 0:
        reasons.append("train role empty")
    if res.n_test == 0:
        reasons.append("test role empty")
    inter_tt = roles["train"] & roles["test"]
    if inter_tt:
        reasons.append(f"train∩test non-empty: n={len(inter_tt)}")
    inter_tv = roles["train"] & roles["val"]
    if inter_tv:
        reasons.append(f"train∩val non-empty: n={len(inter_tv)}")
    inter_ev = roles["test"] & roles["val"]
    if inter_ev:
        reasons.append(f"test∩val non-empty: n={len(inter_ev)}")

    try:
        tsv_index, tsv_issues = load_tsv_index(run.legnet_tsv)
    except (OSError, ValueError) as exc:
        res.status = "FAILED"
        res.reasons = [f"legnet TSV: {exc}"]
        return res
    res.n_tsv = len(tsv_index)
    # Cap issue spam
    reasons.extend(tsv_issues[:20])
    if len(tsv_issues) > 20:
        reasons.append(f"… +{len(tsv_issues) - 20} more TSV issues")

    needed = roles["train"] | roles["test"] | roles["val"]
    missing = sorted(needed - set(tsv_index), key=str)
    if missing:
        reasons.append(
            f"{len(missing)} split IDs missing from TSV (rev==0); "
            f"e.g. {missing[:5]}"
        )

    if load_ckpt:
        try:
            from src.embed.legnet_extract import load_lit_model

            lit = load_lit_model(run, map_location="cpu")
            _ = lit.model
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"checkpoint load failed: {type(exc).__name__}: {exc}")

    soft_prefixes = ("non-default ef_block_sizes",)
    hard = [r for r in reasons if not any(r.startswith(p) for p in soft_prefixes)]
    res.reasons = reasons
    if not hard:
        res.status = "READY"
        return res

    # Missing required files already returned SKIPPED above; remaining hard issues
    # are data/consistency failures.
    skip_markers = ("missing split.csv", "missing legnet TSV", "missing config", "missing checkpoint")
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
