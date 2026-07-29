"""hashFrag homology-aware split strategy.

Caption: ``splits/hashfrag.md``. Wired into ``split-predict`` as ``type=hashfrag``.

Flow:
  MARKED/ (+ optional fold.csv ZSV) → multi-FASTA → ``hashFrag create_orthogonal_splits``
  → parse train/test TSV → carve val from train pool → ``split.csv``.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.common import write_csv
from src.pipeline.generate_fold import is_zsv_fold, normalize_fold_label
from src.splits.common import TEST_FRACTION, VAL_FRACTION_OF_TRAINPOOL
from src.splits.sbs.fna_io import load_fna_directory

__all__ = (
    "SPLIT_ID",
    "HF_FASTA_PREFIX",
    "run_hashfrag_split_assign",
    "marked_to_multifasta",
    "parse_hashfrag_split_tsv",
    "to_fasta_token",
    "from_fasta_token",
)

SPLIT_ID = "hashfrag"
HF_FASTA_PREFIX = "hf_"
_REVERSED_SUFFIX = "_Reversed"


def to_fasta_token(region_id: str) -> str:
    """Stable non-numeric FASTA / BLAST id for a region (avoids hashFrag int bug)."""
    return f"{HF_FASTA_PREFIX}{region_id}"


def from_fasta_token(token: str) -> str | None:
    """Map hashFrag/BLAST id back to region id; drop RC mates."""
    tok = token.strip()
    if tok.endswith(_REVERSED_SUFFIX):
        return None
    if not tok.startswith(HF_FASTA_PREFIX):
        return None
    rid = tok[len(HF_FASTA_PREFIX) :]
    return rid or None


def marked_to_multifasta(
    marked_dir: Path,
    out_fa: Path,
    *,
    ids: list[str],
) -> Path:
    """Write multi-FASTA ``>{hf_ID}`` for the given region ids from MARKED/."""
    marked_dir = Path(marked_dir)
    if not marked_dir.is_dir():
        raise FileNotFoundError(f"MARKED directory missing: {marked_dir}")
    if not ids:
        raise ValueError("no IDs to write into multi-FASTA")
    seqs = load_fna_directory(marked_dir, ids=ids)
    out_fa = Path(out_fa)
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    with out_fa.open("w", encoding="utf-8") as fh:
        for rid in ids:
            seq = seqs[rid]
            if not seq:
                raise ValueError(f"empty sequence for ID {rid!r} under {marked_dir}")
            fh.write(f">{to_fasta_token(rid)}\n")
            for chunk in textwrap.wrap(seq, 80) or [""]:
                fh.write(chunk + "\n")
    return out_fa


def parse_hashfrag_split_tsv(path: Path) -> dict[str, str]:
    """Parse ``hashFrag.*.split_*.tsv`` → ``{region_id: train|test}`` (no RC)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"hashFrag split TSV missing: {path}")
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(
                f"hashFrag split TSV must have columns id,split; got {reader.fieldnames}"
            )
        for row in reader:
            rid = from_fasta_token(str(row["id"]))
            if rid is None:
                continue
            label = str(row["split"]).strip().lower()
            if label not in ("train", "test"):
                raise ValueError(f"unexpected hashFrag split label {label!r} for {rid}")
            if rid in out and out[rid] != label:
                raise ValueError(f"conflicting hashFrag labels for {rid}: {out[rid]} vs {label}")
            out[rid] = label
    if not out:
        raise ValueError(f"no forward (non-RC) IDs parsed from {path}")
    return out


def parse_homologous_groups_tsv(path: Path | None) -> dict[str, str]:
    """Optional ``hashFrag.homologous_groups.tsv`` → ``{region_id: group_id}``."""
    if path is None or not Path(path).is_file():
        return {}
    out: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        # No header in smoke output: id\tgroup
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if parts[0].lower() == "id":
                continue
            rid = from_fasta_token(parts[0])
            if rid is None:
                continue
            out[rid] = parts[1].strip()
    return out


def _find_split_tsv(work_dir: Path) -> Path:
    matches = sorted(Path(work_dir).glob("hashFrag.*.split_*.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"no hashFrag.*.split_*.tsv under {work_dir}; hashFrag may have failed"
        )
    return matches[0]


def _require_tools() -> tuple[str, str, str]:
    hashfrag = shutil.which("hashFrag")
    blastn = shutil.which("blastn")
    makeblastdb = shutil.which("makeblastdb")
    missing = [
        name
        for name, path in (
            ("hashFrag", hashfrag),
            ("blastn", blastn),
            ("makeblastdb", makeblastdb),
        )
        if path is None
    ]
    if missing:
        raise EnvironmentError(
            "hashFrag strategy requires tools on PATH: "
            + ", ".join(missing)
            + ". Install hashFrag (pip) and NCBI BLAST+."
        )
    assert hashfrag and blastn and makeblastdb
    return hashfrag, blastn, makeblastdb


def _tool_version(exe: str) -> str:
    try:
        proc = subprocess.run(
            [exe, "-version"] if "blast" in Path(exe).name else [exe, "-h"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        text = (proc.stdout or proc.stderr or "").strip().splitlines()
        return text[0][:200] if text else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _carve_val_from_train(
    train_ids: list[str],
    *,
    seed: int,
    val_fraction_of_trainpool: float = VAL_FRACTION_OF_TRAINPOOL,
) -> dict[str, str]:
    """Split hashFrag train pool into train + val (Caduceus-aligned default)."""
    if not train_ids:
        raise ValueError("hashFrag returned empty train pool; cannot carve val")
    rng = random.Random(seed)
    order = list(train_ids)
    rng.shuffle(order)
    if len(order) < 2:
        # Keep single-id pool as train; val empty is invalid for project — promote none
        return {order[0]: "train"}
    n_val = max(1, int(round(len(order) * val_fraction_of_trainpool)))
    n_val = min(n_val, len(order) - 1)
    labels = {rid: "val" for rid in order[:n_val]}
    labels.update({rid: "train" for rid in order[n_val:]})
    return labels


def _load_zsv_ids(fold_csv: Path | None, id_set: set[str]) -> set[str]:
    if fold_csv is None:
        return set()
    from src.pipeline.common import read_csv

    rows = read_csv(Path(fold_csv))
    zsv: set[str] = set()
    for row in rows:
        rid = row["ID"].strip()
        if rid not in id_set:
            continue
        fold = normalize_fold_label(row.get("fold", "0"))
        if is_zsv_fold(fold):
            zsv.add(rid)
    return zsv


def run_hashfrag_split_assign(
    *,
    outdir: Path,
    marked: Path,
    threshold: float,
    id_csv: Path | None = None,
    fold_csv: Path | None = None,
    seed: int = 42,
    max_ids: int | None = None,
    ids: list[str] | None = None,
    p_train: float | None = None,
    p_test: float | None = None,
    threads: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """MARKED → hashFrag orthogonal splits → ``outdir/split.csv``.

    Parameters
    ----------
    threshold:
        Obligatory hashFrag alignment-score threshold (``-t``).
    p_train / p_test:
        Optional hashFrag proportions (must sum to 1). Default: Caduceus-like
        test fraction as ``p_test``, remainder as hashFrag ``p_train`` (train+val pool).
    """
    hashfrag_bin, blastn_bin, makeblastdb_bin = _require_tools()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "hashfrag_work"
    work.mkdir(parents=True, exist_ok=True)
    fasta_dir = outdir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    if threshold is None:
        raise ValueError("hashFrag strategy requires an explicit homology threshold")
    # hashFrag CLI parses -t as int
    threshold_cli = int(round(float(threshold)))
    if threshold_cli < 1:
        raise ValueError(f"hashFrag threshold must be >= 1; got {threshold}")

    if ids is None:
        if id_csv is None:
            raise ValueError("Provide ids= or id_csv= for hashfrag split")
        from src.pipeline.common import read_csv

        id_rows = read_csv(Path(id_csv))
        if not id_rows or "ID" not in id_rows[0]:
            raise ValueError(f"id_csv missing ID column: {id_csv}")
        ids = [r["ID"].strip() for r in id_rows if r["ID"].strip()]
    if not ids:
        raise ValueError("empty ID list for hashfrag split")
    if max_ids is not None:
        ids = ids[: int(max_ids)]

    id_set = set(ids)
    zsv_ids = sorted(_load_zsv_ids(fold_csv, id_set))
    zsv_set = set(zsv_ids)
    assignable = [i for i in ids if i not in zsv_set]
    if len(assignable) < 3:
        raise ValueError(
            f"need >=3 non-ZSV IDs for hashFrag orthogonal splits; got {len(assignable)}"
        )

    if p_test is None and p_train is None:
        p_test = float(TEST_FRACTION)
        p_train = 1.0 - p_test
    elif p_train is None or p_test is None:
        raise ValueError("provide both p_train and p_test, or neither")
    if abs((p_train + p_test) - 1.0) > 1e-6:
        raise ValueError(f"p_train + p_test must equal 1; got {p_train}+{p_test}")

    all_fa = marked_to_multifasta(marked, fasta_dir / "all.fa", ids=assignable)

    cmd = [
        hashfrag_bin,
        "create_orthogonal_splits",
        "-f",
        str(all_fa),
        "-t",
        str(threshold_cli),
        "--p-train",
        str(p_train),
        "--p-test",
        str(p_test),
        "-n",
        "1",
        "-s",
        str(seed),
        "-T",
        str(max(1, int(threads))),
        "-o",
        str(work),
    ]
    if force:
        cmd.append("--force")

    log_path = outdir / "hashfrag_cli.log"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log_path.write_text(
        "CMD: " + " ".join(cmd) + "\n\nSTDOUT:\n" + (proc.stdout or "")
        + "\n\nSTDERR:\n" + (proc.stderr or ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hashFrag create_orthogonal_splits failed (rc={proc.returncode}); "
            f"see {log_path}"
        )

    split_tsv = _find_split_tsv(work)
    hf_labels = parse_hashfrag_split_tsv(split_tsv)
    missing = [i for i in assignable if i not in hf_labels]
    if missing:
        raise RuntimeError(
            f"hashFrag split missing {len(missing)} assignable ID(s); "
            f"example={missing[0]!r}"
        )
    extra = sorted(set(hf_labels) - set(assignable))
    if extra:
        raise RuntimeError(
            f"hashFrag split has {len(extra)} unexpected ID(s); example={extra[0]!r}"
        )

    train_pool = sorted(i for i, lab in hf_labels.items() if lab == "train")
    test_ids = sorted(i for i, lab in hf_labels.items() if lab == "test")
    carved = _carve_val_from_train(train_pool, seed=seed)

    groups = parse_homologous_groups_tsv(work / "hashFrag.homologous_groups.tsv")

    rows: list[dict[str, str]] = []
    for rid in ids:
        if rid in zsv_set:
            rows.append({"ID": rid, "train_test": "zsv", "fold": "zsv"})
            continue
        if rid in test_ids:
            tt = "test"
        else:
            tt = carved[rid]
        fold = groups.get(rid, "0")
        rows.append({"ID": rid, "train_test": tt, "fold": fold})

    split_csv = outdir / "split.csv"
    write_csv(split_csv, rows, ["ID", "train_test", "fold"])

    counts = {
        "train": sum(1 for r in rows if r["train_test"] == "train"),
        "val": sum(1 for r in rows if r["train_test"] == "val"),
        "test": sum(1 for r in rows if r["train_test"] == "test"),
        "zsv": sum(1 for r in rows if r["train_test"] == "zsv"),
    }
    summary: dict[str, Any] = {
        "split_id": SPLIT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "threshold": threshold_cli,
        "p_train": p_train,
        "p_test": p_test,
        "threads": threads,
        "marked": str(marked),
        "all_fa": str(all_fa),
        "hashfrag_work": str(work),
        "split_tsv": str(split_tsv),
        "split_csv": str(split_csv),
        "n_ids": len(ids),
        "n_assignable": len(assignable),
        "counts": counts,
        "tools": {
            "hashFrag": _tool_version(hashfrag_bin),
            "blastn": _tool_version(blastn_bin),
            "makeblastdb": _tool_version(makeblastdb_bin),
        },
        "cli_log": str(log_path),
    }
    (outdir / "hashfrag_split_meta.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
