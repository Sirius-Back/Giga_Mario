"""hashFrag homology-aware split strategy.

Caption: ``splits/hashfrag.md``. Wired into ``split-predict`` as ``type=hashfrag``.

Flow:
  MARKED/ (+ optional fold.csv ZSV) → multi-FASTA → hashFrag BLAST modules
  → homologous groups → **project orthogonal assigner** (fixes upstream
  ``round(N*p_train)+round(N*p_test)`` off-by-one) → carve val → ``split.csv``.

When ``force=false`` and ``hashFrag.homologous_groups.tsv`` already exists,
BLAST/process/identify are skipped (resume path).
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import textwrap
from collections import defaultdict
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
    "assign_orthogonal_from_groups",
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


def assign_orthogonal_from_groups(
    homology_path: Path,
    *,
    p_train: float,
    p_test: float,
    seed: int,
) -> dict[str, str]:
    """Assign train/test from hashFrag homologous groups (forward IDs only).

    Mirrors ``create_orthogonal_splits_module`` (whole group → one split) but
    fixes the upstream off-by-one::

        n_train = round(N * p_train)
        n_test = N - n_train   # not round(N * p_test)

    ``N`` counts all tokens in the groups file (including ``_Reversed``), matching
    hashFrag's size targets; returned labels map **region ids** (RC dropped).
    """
    if abs((p_train + p_test) - 1.0) > 1e-6:
        raise ValueError(f"p_train + p_test must equal 1; got {p_train}+{p_test}")
    homology_path = Path(homology_path)
    if not homology_path.is_file():
        raise FileNotFoundError(f"homologous groups missing: {homology_path}")

    groups_map: dict[str, set[str]] = defaultdict(set)
    n_tokens = 0
    with homology_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2 or parts[0].lower() == "id":
                continue
            sample_id, group_id = parts[0].strip(), parts[1].strip()
            groups_map[group_id].add(sample_id)
            n_tokens += 1
    if n_tokens < 3 or len(groups_map) < 1:
        raise ValueError(f"insufficient homologous groups in {homology_path}")

    # Largest-remainder fix (upstream rounds both halves independently).
    n_train = int(round(n_tokens * p_train))
    n_train = min(max(n_train, 1), n_tokens - 1)
    n_test = n_tokens - n_train

    rng = random.Random(seed)
    groups: list[set[str]] = list(groups_map.values())
    test_tokens: set[str] = set()
    # Same loop as hashFrag: grow test by whole groups until target size.
    while len(test_tokens) < n_test:
        if not groups:
            break
        idx = rng.randint(0, len(groups) - 1)
        group = groups.pop(idx)
        test_tokens.update(group)
    train_tokens: set[str] = set()
    for group in groups:
        train_tokens.update(group)

    if not train_tokens:
        raise RuntimeError(
            "orthogonal assign left empty train pool; try another seed or threshold"
        )
    if not test_tokens:
        raise RuntimeError("orthogonal assign left empty test pool")

    labels: dict[str, str] = {}
    for tok in train_tokens:
        rid = from_fasta_token(tok)
        if rid is not None:
            labels[rid] = "train"
    for tok in test_tokens:
        rid = from_fasta_token(tok)
        if rid is not None:
            labels[rid] = "test"  # group-grain: test wins if both (should not)
    if not labels:
        raise RuntimeError(f"no forward IDs labeled from {homology_path}")
    return labels


def write_orthogonal_split_tsv(
    labels: dict[str, str],
    work_dir: Path,
) -> Path:
    """Write hashFrag-compatible ``hashFrag.train_X.test_Y.split_001.tsv``."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    n_train = sum(1 for v in labels.values() if v == "train")
    n_test = sum(1 for v in labels.values() if v == "test")
    out_path = work_dir / f"hashFrag.train_{n_train}.test_{n_test}.split_001.tsv"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("id\tsplit\n")
        for rid, lab in sorted(labels.items(), key=lambda kv: (kv[1], kv[0])):
            fh.write(f"{to_fasta_token(rid)}\t{lab}\n")
    return out_path


def _find_split_tsv(work_dir: Path) -> Path:
    matches = sorted(Path(work_dir).glob("hashFrag.*.split_*.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"no hashFrag.*.split_*.tsv under {work_dir}; hashFrag may have failed"
        )
    return matches[0]


def _run_hashfrag_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    append: bool = False,
) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\nSTDOUT:\n" + (proc.stdout or ""))
        fh.write("\n\nSTDERR:\n" + (proc.stderr or "") + "\n---\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"hashFrag command failed (rc={proc.returncode}): {' '.join(cmd)}; "
            f"see {log_path}"
        )


def ensure_homologous_groups(
    *,
    hashfrag_bin: str,
    work: Path,
    all_fa: Path,
    threshold_cli: int,
    threads: int,
    force: bool,
    log_path: Path,
) -> Path:
    """Ensure ``hashFrag.homologous_groups.tsv`` exists; reuse BLAST when possible."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    groups_path = work / "hashFrag.homologous_groups.tsv"
    blast_out = work / "hashFrag.blastn.out"
    processed = work / "hashFrag.blastn.processed.tsv"

    if groups_path.is_file() and groups_path.stat().st_size > 0 and not force:
        log_path.write_text(
            f"RESUME: reusing existing homologous groups: {groups_path}\n",
            encoding="utf-8",
        )
        return groups_path

    if force or not blast_out.is_file() or blast_out.stat().st_size == 0:
        cmd = [
            hashfrag_bin,
            "blastn_module",
            "-f",
            str(all_fa),
            "-T",
            str(max(1, int(threads))),
            "-o",
            str(work),
            "--blastdb-label",
            "hashFrag",
        ]
        if force:
            cmd.append("--force")
        _run_hashfrag_cmd(cmd, log_path, append=False)
    else:
        log_path.write_text(
            f"RESUME: reusing existing BLAST results: {blast_out}\n",
            encoding="utf-8",
        )

    if force or not processed.is_file() or processed.stat().st_size == 0:
        _run_hashfrag_cmd(
            [
                hashfrag_bin,
                "process_blast_results_module",
                "--blastn-path",
                str(blast_out),
                "--processed-blastn-path",
                str(processed),
            ],
            log_path,
            append=True,
        )

    _run_hashfrag_cmd(
        [
            hashfrag_bin,
            "identify_homologous_groups_module",
            "-i",
            str(processed),
            "-t",
            str(threshold_cli),
            "-o",
            str(groups_path),
        ],
        log_path,
        append=True,
    )
    if not groups_path.is_file() or groups_path.stat().st_size == 0:
        raise FileNotFoundError(f"homologous groups not written: {groups_path}")
    return groups_path


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

    all_fa = fasta_dir / "all.fa"
    if force or not all_fa.is_file() or all_fa.stat().st_size == 0:
        all_fa = marked_to_multifasta(marked, all_fa, ids=assignable)

    log_path = outdir / "hashfrag_cli.log"
    groups_path = ensure_homologous_groups(
        hashfrag_bin=hashfrag_bin,
        work=work,
        all_fa=all_fa,
        threshold_cli=threshold_cli,
        threads=threads,
        force=force,
        log_path=log_path,
    )

    # Project assigner — bypasses upstream create_orthogonal_splits_module off-by-one.
    hf_labels = assign_orthogonal_from_groups(
        groups_path, p_train=p_train, p_test=p_test, seed=seed
    )
    split_tsv = write_orthogonal_split_tsv(hf_labels, work)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\nPROJECT_ASSIGN: wrote {split_tsv} "
            f"(train={sum(1 for v in hf_labels.values() if v=='train')}, "
            f"test={sum(1 for v in hf_labels.values() if v=='test')})\n"
        )

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

    groups = parse_homologous_groups_tsv(groups_path)

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
        "homologous_groups": str(groups_path),
        "split_tsv": str(split_tsv),
        "split_csv": str(split_csv),
        "n_ids": len(ids),
        "n_assignable": len(assignable),
        "counts": counts,
        "assigner": "src.splits.hashfrag.assign_orthogonal_from_groups",
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
