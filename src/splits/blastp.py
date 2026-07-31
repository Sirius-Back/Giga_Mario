"""BLASTP homology split (SBS family).

Caption: ``splits/blastp.md``. Wired into ``split-predict`` as ``type=blastp``.

Flow:
  raw (fna+gtf) → adapt MARKED_blastp → filter ∩ PARSED → CDS protein FASTA
  → DIAMOND blastp --sensitive (sparse hits) → connected components
  → fold-grain train/val/test (+ ZSV) → ``split.csv``.

Proteins come from **CDS exons** (genetic code, default universal), not raw DNA
window translation. Materialize for LegNet still uses panel PARSED/PREDICT.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.common import ID_CSV_COLUMNS, ensure_dir, read_csv, require_columns, write_csv
from src.pipeline.generate_fold import is_zsv_fold, normalize_fold_label
from src.preprocessing import Fasta, extract_forward, genome_prefix, open_text, parse_attrs
from src.splits.pangenome import (
    filter_ids_to_parsed,
    intersect_pangenome,
    materialize_marked_subset,
)
from src.splits.sbs.assign import (
    aggregate_stratification_per_fold,
    assignment_rows_to_split_csv,
    write_assignment_table,
)

__all__ = (
    "SPLIT_ID",
    "DEFAULT_GENETIC_CODE",
    "resolve_genetic_code",
    "intersect_blastp",
    "translate_cds_proteins",
    "connected_components_from_edges",
    "run_sparse_blastp",
    "run_blastp_split_assign",
    "ensure_marked_blastp",
)

SPLIT_ID = "blastp"
DEFAULT_GENETIC_CODE = "universal"
DEFAULT_EVALUE = 1e-5
DEFAULT_MAX_TARGET_SEQS = 50
DEFAULT_MIN_BITSCORE = 50.0
DEFAULT_QUERY_CHUNK = 250
DEFAULT_WINDOW = {"pos1": 0, "pos2": 0}
DEFAULT_ENVIRONMENT = "gene"

A2A_ADAPT_HINT = (
    "Invoke @preprocess / src.pipeline.adapt with blastp environment/window "
    "to produce MARKED_blastp, then filter to PARSED. "
    "Only pass reuse_panel_marked=True when the panel MARKED window matches."
)

# NCBI table id or Biopython name aliases → table id for Seq.translate
_GENETIC_CODE_ALIASES: dict[str, int] = {
    "universal": 1,
    "standard": 1,
    "standard_code": 1,
    "ncbi1": 1,
    "1": 1,
}


class BlastpAdaptRequiredError(FileNotFoundError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or A2A_ADAPT_HINT)


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def resolve_genetic_code(name: str | int | None) -> int:
    """Map caption genetic-code token → NCBI translation table id."""
    if name is None:
        return _GENETIC_CODE_ALIASES[DEFAULT_GENETIC_CODE]
    if isinstance(name, int):
        if name <= 0:
            raise ValueError(f"genetic_code table id must be positive; got {name}")
        return name
    key = str(name).strip().lower()
    if key in _GENETIC_CODE_ALIASES:
        return _GENETIC_CODE_ALIASES[key]
    if key.isdigit():
        return resolve_genetic_code(int(key))
    raise ValueError(
        f"unknown genetic_code={name!r}; use 'universal' (NCBI 1) or a table id"
    )


def intersect_blastp(
    marked_dir: Path,
    parsed_dir: Path,
    ids: Sequence[str] | None = None,
) -> list[str]:
    """Keep IDs present in both MARKED_blastp and PARSED (caption step 2)."""
    return intersect_pangenome(marked_dir, parsed_dir, ids=ids)


def _index_folder(folder: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    out: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.endswith(s) for s in suffixes):
            continue
        g = genome_prefix(path.name)
        if g in out:
            raise ValueError(f"Multiple inputs resolve to genome {g}: {out[g]}, {path}")
        out[g] = path
    return out


def _parse_cds_exons(
    gtf_path: Path,
) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    """Map (chrom, gene_id) → list of (start, end, strand) CDS exons."""
    exons: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GTF coordinates in {gtf_path}: {line.rstrip()}"
                ) from exc
            attrs = parse_attrs(parts[8])
            raw = attrs.get("gene_id") or attrs.get("gene") or attrs.get("transcript_id")
            if not raw:
                continue
            chrom = parts[0]
            strand = parts[6] if parts[6] in "+-" else "+"
            exons[(chrom, raw)].append((start, end, strand))
    return exons


def _cds_dna(
    fasta: Any,
    chrom: str,
    exon_list: list[tuple[int, int, str]],
) -> str | None:
    if not exon_list or chrom not in fasta:
        return None
    strand = exon_list[0][2]
    ordered = sorted(exon_list, key=lambda x: x[0], reverse=(strand == "-"))
    chunks: list[str] = []
    for start, end, _ in ordered:
        # GTF is 1-based inclusive; extract_forward uses 1-based inclusive.
        piece = extract_forward(fasta, chrom, start, end)
        if not piece:
            return None
        chunks.append(piece)
    dna = "".join(chunks)
    if strand == "-":
        from Bio.Seq import Seq

        dna = str(Seq(dna).reverse_complement())
    return dna


def _translate_dna(dna: str, *, table: int) -> str:
    from Bio.Seq import Seq

    seq = Seq(dna.upper().replace("U", "T"))
    # Truncate to codon multiple; CDS should already be in-frame.
    usable = len(seq) - (len(seq) % 3)
    if usable < 3:
        return ""
    prot = str(seq[:usable].translate(table=table, to_stop=False))
    prot = prot.rstrip("*")
    return prot.replace("*", "X")


def translate_cds_proteins(
    *,
    id_csv: Path,
    gtf_dir: Path,
    fna_dir: Path,
    ids: Sequence[str],
    out_faa: Path,
    genetic_code: str | int = DEFAULT_GENETIC_CODE,
    mapping_path: Path | None = None,
) -> dict[str, Any]:
    """Write protein FASTA keyed by protein_id; map region IDs → protein_id.

    Regions without CDS become singletons (no protein row); mapping records
    ``protein_id=None`` for those.
    """
    table = resolve_genetic_code(genetic_code)
    id_rows = read_csv(Path(id_csv))
    require_columns(id_rows, ID_CSV_COLUMNS, label="ID.csv")
    by_id = {r["ID"].strip(): r for r in id_rows}
    want = [str(i) for i in ids]
    missing = [i for i in want if i not in by_id]
    if missing:
        raise ValueError(f"ID.csv missing ID {missing[0]!r}")

    gtf_index = _index_folder(Path(gtf_dir), (".gtf",))
    fna_index = _index_folder(
        Path(fna_dir),
        (".fna", ".fa", ".fasta", ".fna.gz", ".fa.gz", ".fasta.gz"),
    )

    # protein_id → aa sequence; region → protein_id | None
    proteins: dict[str, str] = {}
    region_to_prot: dict[str, str | None] = {}
    gene_to_prot: dict[tuple[str, str, str], str] = {}

    by_genome: dict[str, list[str]] = defaultdict(list)
    for rid in want:
        by_genome[by_id[rid]["genome"].strip()].append(rid)

    n_cds = 0
    n_nocds = 0
    for genome, rids in by_genome.items():
        if genome not in gtf_index or genome not in fna_index:
            for rid in rids:
                region_to_prot[rid] = None
                n_nocds += 1
            continue
        exons = _parse_cds_exons(gtf_index[genome])
        # Also index by gene display name when unique on chrom.
        name_index: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for (chrom, gid), _ex in exons.items():
            # recover display name from first matching gene line is expensive;
            # ID.csv raw_target_ID / gene_name are the join keys.
            name_index[(chrom, gid)].append((chrom, gid))

        fasta = Fasta(str(fna_index[genome]), as_raw=True, sequence_always_upper=True)
        try:
            for rid in rids:
                meta = by_id[rid]
                chrom = meta["chr"].strip()
                raw = meta["raw_target_ID"].strip()
                gene_name = meta["gene_nameORnon_coding_ID"].strip()
                key = (genome, chrom, raw)
                if key in gene_to_prot:
                    region_to_prot[rid] = gene_to_prot[key]
                    continue
                exon_list = exons.get((chrom, raw))
                if exon_list is None:
                    # fallback: unique gene_name match among CDS keys is not
                    # available without gene features; try raw==gene_name key.
                    exon_list = exons.get((chrom, gene_name))
                if not exon_list:
                    region_to_prot[rid] = None
                    n_nocds += 1
                    continue
                dna = _cds_dna(fasta, chrom, exon_list)
                if not dna:
                    region_to_prot[rid] = None
                    n_nocds += 1
                    continue
                aa = _translate_dna(dna, table=table)
                if len(aa) < 5:
                    region_to_prot[rid] = None
                    n_nocds += 1
                    continue
                # BLAST -parse_seqids caps local ids at 50 chars; keep ids short.
                digest = hashlib.md5(
                    f"{genome}|{chrom}|{raw}".encode("utf-8")
                ).hexdigest()[:16]
                pid = f"p{digest}"
                # Collapse identical gene keys to one protein record.
                if pid not in proteins:
                    proteins[pid] = aa
                gene_to_prot[key] = pid
                region_to_prot[rid] = pid
                n_cds += 1
        finally:
            fasta.close()

    out_faa = Path(out_faa)
    out_faa.parent.mkdir(parents=True, exist_ok=True)
    with out_faa.open("w", encoding="utf-8") as fh:
        for pid, aa in proteins.items():
            fh.write(f">{pid}\n")
            fh.write("\n".join(textwrap.wrap(aa, 60)) + "\n")

    mapping = {
        "genetic_code": genetic_code,
        "table_id": table,
        "n_regions": len(want),
        "n_proteins": len(proteins),
        "n_with_cds": n_cds,
        "n_without_cds": n_nocds,
        "region_to_protein": region_to_prot,
    }
    dest = Path(mapping_path) if mapping_path else out_faa.with_suffix(".mapping.json")
    dest.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return mapping


def connected_components_from_edges(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> dict[str, int]:
    """Return node → component id (dense 0..C-1). Isolates get their own ids."""
    uf = _UnionFind(list(nodes))
    node_set = set(nodes)
    for a, b in edges:
        if a in node_set and b in node_set and a != b:
            uf.union(a, b)
    roots: dict[str, int] = {}
    out: dict[str, int] = {}
    next_id = 0
    for n in nodes:
        r = uf.find(n)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        out[n] = roots[r]
    return out


def _tool_version(bin_name: str) -> str:
    path = shutil.which(bin_name)
    if not path:
        return "missing"
    try:
        out = subprocess.check_output([path, "-version"], text=True, stderr=subprocess.STDOUT)
        return out.strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f"error:{exc}"


def run_sparse_blastp(
    faa: Path,
    *,
    work: Path,
    threads: int = 8,
    evalue: float = DEFAULT_EVALUE,
    max_target_seqs: int = DEFAULT_MAX_TARGET_SEQS,
    min_bitscore: float = DEFAULT_MIN_BITSCORE,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    force: bool = False,
) -> tuple[Path, list[tuple[str, str]]]:
    """DIAMOND blastp → undirected edges (production default; not NCBI BLASTP).

    ``query_chunk`` is ignored (kept for API compatibility). Single-pass DIAMOND
    replaces the old chunked NCBI blastp path.
    """
    _ = query_chunk
    diamond = shutil.which("diamond")
    if not diamond:
        raise FileNotFoundError(
            "diamond must be on PATH (conda install -c bioconda diamond)"
        )

    work = ensure_dir(Path(work))
    faa = Path(faa)
    if not faa.is_file() or faa.stat().st_size == 0:
        raise FileNotFoundError(f"protein FASTA missing/empty: {faa}")

    hits_path = work / "diamond_hits.tsv"
    edges_path = work / "diamond_edges.tsv"
    # Legacy NCBI resume files are intentionally ignored (discarded for DIAMOND).
    if hits_path.is_file() and edges_path.is_file() and not force:
        edges: list[tuple[str, str]] = []
        with edges_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("qseqid"):
                    continue
                a, b = line.split("\t")[:2]
                edges.append((a, b))
        print(f"RESUME: reusing DIAMOND hits {hits_path}", flush=True)
        return hits_path, edges

    db_prefix = work / "prot_diamond"
    db_file = Path(str(db_prefix) + ".dmnd")
    if force or not db_file.is_file():
        print(f"DIAMOND makedb ← {faa}", flush=True)
        subprocess.run(
            [
                diamond,
                "makedb",
                "--in",
                str(faa),
                "-d",
                str(db_prefix),
                "--threads",
                str(max(1, int(threads))),
            ],
            check=True,
            cwd=str(work),
        )

    print(
        f"DIAMOND blastp --sensitive evalue={evalue} -k={max_target_seqs} "
        f"threads={threads} → {hits_path}",
        flush=True,
    )
    subprocess.run(
        [
            diamond,
            "blastp",
            "-q",
            str(faa),
            "-d",
            str(db_prefix),
            "-o",
            str(hits_path),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "evalue",
            "bitscore",
            "--evalue",
            str(evalue),
            "-k",
            str(int(max_target_seqs)),
            "--threads",
            str(max(1, int(threads))),
            "--sensitive",
        ],
        check=True,
        cwd=str(work),
    )

    edge_set: set[tuple[str, str]] = set()
    with hits_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            q, s = parts[0], parts[1]
            try:
                bits = float(parts[5])
            except ValueError:
                continue
            if q == s or bits < float(min_bitscore):
                continue
            a, b = (q, s) if q < s else (s, q)
            edge_set.add((a, b))

    edges = sorted(edge_set)
    with edges_path.open("w", encoding="utf-8") as fh:
        fh.write("qseqid\tsseqid\n")
        for a, b in edges:
            fh.write(f"{a}\t{b}\n")
    print(f"DIAMOND done: n_hits_file={hits_path} n_edges={len(edges)}", flush=True)
    return hits_path, edges


def _load_fold_map(fold_csv: Path | None) -> dict[str, str]:
    if fold_csv is None:
        return {}
    rows = read_csv(Path(fold_csv))
    return {row["ID"].strip(): normalize_fold_label(row["fold"]) for row in rows}


def _load_strat_map(strat_csv: Path | None) -> dict[str, dict[str, str]]:
    if strat_csv is None:
        return {}
    rows = read_csv(Path(strat_csv))
    return {row["ID"].strip(): row for row in rows}


def _assign_folds_to_train_test(
    fold_ids: list[str],
    *,
    seed: int,
    fold_strata: dict[str, str] | None,
    ratios: tuple[float, float, float] | None,
) -> dict[str, str]:
    from src.splits.random import assign_folds_random, assign_folds_stratified

    if not fold_ids:
        return {}
    if len(fold_ids) < 3:
        labels = ["train", "val", "test"]
        return {fid: labels[i % 3] for i, fid in enumerate(sorted(fold_ids))}
    rng = random.Random(seed)
    if fold_strata:
        strata = [fold_strata[f] for f in fold_ids]
        labels = assign_folds_stratified(fold_ids, strata, rng, ratios=ratios)
        return dict(zip(fold_ids, labels))
    order = list(fold_ids)
    rng.shuffle(order)
    labels = assign_folds_random(len(order), ratios=ratios)
    return {fid: lab for fid, lab in zip(order, labels)}


def assign_from_blastp_components(
    ids: Sequence[str],
    cluster_ids: Sequence[int],
    *,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Map BLASTP connected components → assignment rows (ZSV held out)."""
    if len(ids) != len(cluster_ids):
        raise ValueError("ids and cluster_ids length mismatch")
    fold_map = _load_fold_map(fold_csv)
    strat_map = _load_strat_map(stratification_csv)

    zsv_ids: list[str] = []
    assignable: list[str] = []
    cluster_by_id = {str(rid): int(cid) for rid, cid in zip(ids, cluster_ids)}
    for rid in ids:
        rid = str(rid)
        raw = fold_map.get(rid, "0")
        if is_zsv_fold(raw):
            zsv_ids.append(rid)
        else:
            assignable.append(rid)

    rows: list[dict[str, str]] = []
    meta: dict[str, Any] = {
        "n_total": len(ids),
        "n_zsv": len(zsv_ids),
        "n_assignable": len(assignable),
        "seed": seed,
        "method_used": "blastp_cc",
    }
    for rid in zsv_ids:
        rows.append(
            {
                "region": rid,
                "cluster": "zsv",
                "train_test": "zsv",
                "fold": "zsv",
                "additional": json.dumps({"method": "blastp_cc"}),
            }
        )

    if not assignable:
        return rows, meta

    fold_members: dict[str, list[str]] = defaultdict(list)
    region_fold: dict[str, str] = {}
    for rid in assignable:
        fold_label = str(cluster_by_id[rid])
        region_fold[rid] = fold_label
        fold_members[fold_label].append(rid)

    fold_strata = None
    if strat_map:
        missing = [rid for rid in assignable if rid not in strat_map]
        if missing:
            raise ValueError(
                f"stratification.csv missing ID {missing[0]!r} "
                "(required when stratification is set)"
            )
        fold_strata = aggregate_stratification_per_fold(dict(fold_members), strat_map)

    fold_to_tt = _assign_folds_to_train_test(
        sorted(fold_members),
        seed=seed,
        fold_strata=fold_strata,
        ratios=ratios,
    )
    meta["train_test_by_fold"] = fold_to_tt
    meta["n_clusters"] = len(fold_members)

    by_region: dict[str, dict[str, str]] = {r["region"]: r for r in rows}
    for rid in assignable:
        fold_label = region_fold[rid]
        by_region[rid] = {
            "region": rid,
            "cluster": fold_label,
            "train_test": fold_to_tt[fold_label],
            "fold": fold_label,
            "additional": json.dumps(
                {"method": "blastp_cc", "cluster": int(fold_label)},
                sort_keys=True,
            ),
        }
    ordered = [by_region[str(rid)] for rid in ids if str(rid) in by_region]
    return ordered, meta


def adapt_blastp_from_raw(
    *,
    outdir: Path,
    gtf_dir: Path,
    fna_dir: Path,
    id_csv: Path,
    environment: str,
    window: dict[str, int],
    genomes: Sequence[str] | None = None,
    max_window: int | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """A2A adapt: raw → ``outdir/MARKED_blastp`` (+ intersect)."""
    from src.pipeline.adapt import run_adapt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = outdir / "_adapt_blastp_stage"
    if stage.exists():
        shutil.rmtree(stage)
    result = run_adapt(
        Path(gtf_dir),
        Path(fna_dir),
        outdir=stage,
        id_csv=Path(id_csv),
        environment=environment,
        window=window,
        max_window=max_window,
        genomes=list(genomes) if genomes else None,
        seed=seed,
    )
    marked_src = Path(result["marked_dir"])
    marked_bp = outdir / "MARKED_blastp"
    if marked_bp.exists() or marked_bp.is_symlink():
        if marked_bp.is_dir() and not marked_bp.is_symlink():
            shutil.rmtree(marked_bp)
        else:
            marked_bp.unlink()
    marked_src.rename(marked_bp)
    intersect_src = Path(result["intersect_csv"])
    intersect_dst = outdir / "intersect_blastp.csv"
    if intersect_src.is_file():
        intersect_src.replace(intersect_dst)
    if stage.is_dir():
        shutil.rmtree(stage, ignore_errors=True)
    return {"marked_blastp": marked_bp, "intersect_csv": intersect_dst}


def ensure_marked_blastp(
    *,
    outdir: Path,
    marked_blastp: Path | None = None,
    panel_marked: Path | None = None,
    reuse_panel_marked: bool = False,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    id_csv: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    genomes: Sequence[str] | None = None,
    max_window: int | None = None,
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Resolve ``MARKED_blastp`` (adapt from raw, reuse, or fail with A2A hint)."""
    import warnings

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"source": None}

    if marked_blastp is not None:
        mp = Path(marked_blastp)
        if not mp.is_dir():
            raise FileNotFoundError(f"marked_blastp missing: {mp}")
        if not any(mp.glob("*.fa")):
            raise FileNotFoundError(f"marked_blastp has no *.fa: {mp}")
        meta["source"] = "marked_blastp"
        return mp, meta

    default_mp = outdir / "MARKED_blastp"
    if default_mp.is_dir() and any(default_mp.glob("*.fa")):
        meta["source"] = "outdir/MARKED_blastp"
        return default_mp, meta

    # Explicit opt-in reuse wins over A2A adapt (panel MARKED as ID keep-set).
    if reuse_panel_marked:
        if panel_marked is None:
            raise ValueError(
                "reuse_panel_marked=True requires panel_marked=… "
                "(existing panel MARKED whose window matches blastp)"
            )
        pm = Path(panel_marked)
        if not pm.is_dir():
            raise FileNotFoundError(f"panel_marked missing: {pm}")
        warnings.warn(
            "reuse_panel_marked=True: using panel MARKED as MARKED_blastp. "
            "Only valid when the blastp window equals the panel adapt window "
            "or MARKED is used solely as an ID keep-set for CDS BLASTP.",
            UserWarning,
            stacklevel=2,
        )
        meta["source"] = "reuse_panel_marked"
        meta["panel_marked"] = str(pm)
        return pm, meta

    can_adapt = (
        gtf_dir is not None
        and fna_dir is not None
        and id_csv is not None
        and environment is not None
        and window is not None
    )
    if can_adapt:
        paths = adapt_blastp_from_raw(
            outdir=outdir,
            gtf_dir=Path(gtf_dir),
            fna_dir=Path(fna_dir),
            id_csv=Path(id_csv),
            environment=str(environment),
            window=window,
            genomes=genomes,
            max_window=max_window,
            seed=seed,
        )
        meta["source"] = "adapt_from_raw"
        meta["window"] = dict(window)
        meta["environment"] = environment
        meta["intersect_csv"] = str(paths["intersect_csv"])
        return paths["marked_blastp"], meta

    raise BlastpAdaptRequiredError(
        "MARKED_blastp not found and adapt inputs incomplete "
        "(need gtf_dir, fna_dir, id_csv, environment, window). "
        + A2A_ADAPT_HINT
    )


def run_blastp_split_assign(
    *,
    outdir: Path,
    parsed: Path | None = None,
    id_csv: Path | None = None,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    seed: int = 42,
    max_ids: int | None = None,
    ids: list[str] | None = None,
    genomes: Sequence[str] | None = None,
    ratios: tuple[float, float, float] | None = None,
    genetic_code: str | int = DEFAULT_GENETIC_CODE,
    threads: int = 8,
    evalue: float = DEFAULT_EVALUE,
    max_target_seqs: int = DEFAULT_MAX_TARGET_SEQS,
    min_bitscore: float = DEFAULT_MIN_BITSCORE,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    force: bool = False,
    # MARKED_blastp resolution
    marked_blastp: Path | None = None,
    panel_marked: Path | None = None,
    marked: Path | None = None,
    reuse_panel_marked: bool = False,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    max_window: int | None = None,
) -> dict[str, Any]:
    """Adapt/resolve MARKED_blastp → filter → CDS proteins → BLASTP CC → split.csv."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panel = panel_marked or marked
    env = environment or DEFAULT_ENVIRONMENT
    win = window if window is not None else dict(DEFAULT_WINDOW)

    marked_bp, source_meta = ensure_marked_blastp(
        outdir=outdir,
        marked_blastp=marked_blastp,
        panel_marked=panel,
        reuse_panel_marked=reuse_panel_marked,
        gtf_dir=gtf_dir,
        fna_dir=fna_dir,
        id_csv=id_csv,
        environment=env if gtf_dir and fna_dir and id_csv else environment,
        window=win if gtf_dir and fna_dir and id_csv else window,
        genomes=genomes,
        max_window=max_window,
        seed=seed,
    )

    if parsed is not None:
        parsed_dir = Path(parsed)
    elif panel is not None and (Path(panel).parent / "PARSED").is_dir():
        parsed_dir = Path(panel).parent / "PARSED"
    else:
        parsed_dir = marked_bp.parent / "PARSED"

    if id_csv is None:
        raise ValueError("id_csv is required for type=blastp (CDS gene join)")
    if gtf_dir is None or fna_dir is None:
        # Proteins need raw GTF/FNA even when MARKED_blastp already exists.
        raise ValueError(
            "gtf_dir and fna_dir are required for type=blastp CDS translation"
        )

    if ids is None:
        kept = filter_ids_to_parsed(
            marked_dir=marked_bp,
            parsed_dir=parsed_dir,
            id_csv=id_csv,
            genomes=genomes,
            max_ids=max_ids,
            seed=seed,
        )
    else:
        kept = intersect_blastp(marked_bp, parsed_dir, ids=ids)
        if max_ids is not None and len(kept) > int(max_ids):
            rng = random.Random(int(seed))
            kept = list(kept)
            rng.shuffle(kept)
            kept = sorted(kept[: int(max_ids)], key=lambda x: (len(x), x))

    marked_parsed = materialize_marked_subset(
        marked_bp, outdir / "MARKED_parsed", kept, mode="symlink"
    )

    faa = outdir / "proteins.faa"
    mapping_path = outdir / "protein_mapping.json"
    if (
        faa.is_file()
        and faa.stat().st_size > 0
        and mapping_path.is_file()
        and not force
    ):
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print(f"RESUME: reusing proteins {faa} + {mapping_path}", flush=True)
    else:
        mapping = translate_cds_proteins(
            id_csv=Path(id_csv),
            gtf_dir=Path(gtf_dir),
            fna_dir=Path(fna_dir),
            ids=kept,
            out_faa=faa,
            genetic_code=genetic_code,
            mapping_path=mapping_path,
        )
    region_to_prot: dict[str, str | None] = mapping["region_to_protein"]

    blast_work = outdir / "blastp_work"
    protein_nodes = sorted({p for p in region_to_prot.values() if p})
    edges: list[tuple[str, str]] = []
    hits_path: Path | None = None
    if protein_nodes and faa.is_file() and faa.stat().st_size > 0:
        hits_path, edges = run_sparse_blastp(
            faa,
            work=blast_work,
            threads=threads,
            evalue=evalue,
            max_target_seqs=max_target_seqs,
            min_bitscore=min_bitscore,
            query_chunk=query_chunk,
            force=force,
        )
        prot_clusters = connected_components_from_edges(protein_nodes, edges)
    else:
        prot_clusters = {}

    # Expand protein components → region cluster ids; no-CDS → unique singleton.
    next_singleton = (max(prot_clusters.values()) + 1) if prot_clusters else 0
    region_clusters: list[int] = []
    singleton_map: dict[str, int] = {}
    for rid in kept:
        pid = region_to_prot.get(rid)
        if pid and pid in prot_clusters:
            region_clusters.append(int(prot_clusters[pid]))
        else:
            if rid not in singleton_map:
                singleton_map[rid] = next_singleton
                next_singleton += 1
            region_clusters.append(singleton_map[rid])

    rows, assign_meta = assign_from_blastp_components(
        kept,
        region_clusters,
        fold_csv=fold_csv,
        stratification_csv=stratification_csv,
        seed=seed,
        ratios=ratios,
    )
    assign_path = write_assignment_table(rows, outdir / "blastp_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, outdir)

    summary = {
        "split_id": SPLIT_ID,
        "seed": seed,
        "marked_blastp": str(marked_bp),
        "marked_parsed": str(marked_parsed),
        "marked_source": source_meta,
        "parsed": str(parsed_dir),
        "panel_marked": str(panel) if panel else None,
        "n_ids": len(kept),
        "n_proteins": len(protein_nodes),
        "n_blast_edges": len(edges),
        "n_clusters": int(assign_meta.get("n_clusters", 0)),
        "genetic_code": genetic_code,
        "table_id": resolve_genetic_code(genetic_code),
        "evalue": evalue,
        "max_target_seqs": max_target_seqs,
        "min_bitscore": min_bitscore,
        "threads": threads,
        "force": force,
        "split_csv": str(split_csv),
        "assignment_csv": str(assign_path),
        "proteins_faa": str(faa),
        "blastp_hits": str(hits_path) if hits_path else None,
        "assign_meta": assign_meta,
        "tools": {
            "diamond": _tool_version("diamond"),
            "blastp": _tool_version("blastp"),
            "makeblastdb": _tool_version("makeblastdb"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "genomes": list(genomes) if genomes else None,
    }
    (outdir / "blastp_split_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
