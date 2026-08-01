"""Build sorted ortholog/paralog hash lookup table for all MARKED genes.

Schema (pipe-separated), sorted by ``id_MARKED_hash`` for binary search:

  id_MARKED|id_MARKED_hash|id|genome|orthogroup|orthogroup_hash|paragroup|paragroup_hash

- ``id_MARKED`` — panel region ID (``ready_*/MARKED/{ID}.fa``, ``ID.csv``)
- ``id`` — Ensembl gene stable ID, or empty (NULL) if no homology mapping
- ``genome`` — GCF accession from ``ID.csv``
- ``orthogroup`` — connected-component id on ortholog edges only (empty if unknown)
- ``paragroup`` — connected-component id on paralog edges only (empty if unknown)

Hashes are deterministic FNV-1a 32-bit (same algorithm as ``src.preprocessing.stable_hash``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def stable_hash(s: str) -> int:
    """Deterministic 32-bit FNV-1a (matches src.preprocessing.stable_hash)."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        self.add(x)
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def compact_labels(self, nodes: list[int]) -> dict[int, int]:
        """Map each node → dense component id in [0, n_comp)."""
        roots = sorted({self.find(n) for n in nodes})
        remap = {r: i for i, r in enumerate(roots)}
        return {n: remap[self.find(n)] for n in nodes}


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def load_id_csv(path: Path) -> dict[str, str]:
    """Return id_MARKED → genome (GCF)."""
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        required = {"ID", "genome"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"ID.csv missing columns {required}; have {reader.fieldnames}")
        for row in reader:
            mid = row["ID"].strip()
            if not mid:
                continue
            if mid in out:
                raise ValueError(f"Duplicate ID in ID.csv: {mid}")
            out[mid] = row["genome"].strip()
    if not out:
        raise ValueError(f"ID.csv empty: {path}")
    return out


def load_marked_ensembl_map(nodes_path: Path) -> dict[str, list[tuple[str, str]]]:
    """marked_id → [(ensembl_species, ensembl_gene), ...] (stable order)."""
    by_marked: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    with _open_text(nodes_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"ensembl_species", "ensembl_gene", "marked_id"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"nodes TSV missing {required}; have {reader.fieldnames}")
        for row in reader:
            mid = (row.get("marked_id") or "").strip()
            if not mid:
                continue
            sp = row["ensembl_species"].strip()
            ens = row["ensembl_gene"].strip().split(".")[0]
            key = (mid, sp, ens)
            if key in seen:
                continue
            seen.add(key)
            by_marked[mid].append((sp, ens))
    for mid in by_marked:
        by_marked[mid].sort()
    return dict(by_marked)


def load_edges_assign_groups(edges_path: Path) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int], dict]:
    """Return (orthogroup_of, paragroup_of, stats) keyed by (species, ensembl_gene)."""
    uf_ortho = UnionFind()
    uf_para = UnionFind()
    key_to_idx: dict[tuple[str, str], int] = {}
    idx_to_key: list[tuple[str, str]] = []

    def node_id(species: str, gene: str) -> int:
        key = (species, gene)
        i = key_to_idx.get(key)
        if i is not None:
            return i
        i = len(idx_to_key)
        key_to_idx[key] = i
        idx_to_key.append(key)
        uf_ortho.add(i)
        uf_para.add(i)
        return i

    n_edges = 0
    n_ortho = 0
    n_para = 0
    with _open_text(edges_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i1, s1 = header.index("gene1"), header.index("genome1")
            i2, s2 = header.index("gene2"), header.index("genome2")
            irel = header.index("relation")
        except ValueError as e:
            raise ValueError(f"edges header missing columns: {header}") from e
        for line in fh:
            if not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            g1 = p[i1].split(".")[0]
            g2 = p[i2].split(".")[0]
            sp1, sp2 = p[s1], p[s2]
            rel = p[irel]
            u, v = node_id(sp1, g1), node_id(sp2, g2)
            n_edges += 1
            if rel == "ortholog":
                uf_ortho.union(u, v)
                n_ortho += 1
            elif rel == "paralog":
                uf_para.union(u, v)
                n_para += 1
            else:
                raise ValueError(f"Unknown relation {rel!r}")

    nodes = list(range(len(idx_to_key)))
    ortho_lab = uf_ortho.compact_labels(nodes)
    para_lab = uf_para.compact_labels(nodes)
    orthogroup_of = {idx_to_key[i]: ortho_lab[i] for i in nodes}
    paragroup_of = {idx_to_key[i]: para_lab[i] for i in nodes}
    stats = {
        "n_nodes": len(nodes),
        "n_edges": n_edges,
        "n_ortholog_edges": n_ortho,
        "n_paralog_edges": n_para,
        "n_orthogroups": len(set(ortho_lab.values())),
        "n_paragroups": len(set(para_lab.values())),
    }
    return orthogroup_of, paragroup_of, stats


def build_rows(
    marked_genome: dict[str, str],
    marked_ensembl: dict[str, list[tuple[str, str]]],
    orthogroup_of: dict[tuple[str, str], int],
    paragroup_of: dict[tuple[str, str], int],
) -> tuple[list[dict[str, str]], dict]:
    rows: list[dict[str, str]] = []
    n_null = 0
    n_mapped = 0
    n_multi = 0
    for mid in marked_genome:
        genome = marked_genome[mid]
        mid_hash = str(stable_hash(mid))
        hits = marked_ensembl.get(mid)
        if not hits:
            n_null += 1
            rows.append(
                {
                    "id_MARKED": mid,
                    "id_MARKED_hash": mid_hash,
                    "id": "",
                    "genome": genome,
                    "orthogroup": "",
                    "orthogroup_hash": "",
                    "paragroup": "",
                    "paragroup_hash": "",
                }
            )
            continue
        if len(hits) > 1:
            n_multi += 1
        n_mapped += 1
        for sp, ens in hits:
            key = (sp, ens)
            og = orthogroup_of.get(key)
            pg = paragroup_of.get(key)
            # Gene in nodes map but absent from edges → still emit id, empty groups.
            og_s = "" if og is None else str(og)
            pg_s = "" if pg is None else str(pg)
            rows.append(
                {
                    "id_MARKED": mid,
                    "id_MARKED_hash": mid_hash,
                    "id": ens,
                    "genome": genome,
                    "orthogroup": og_s,
                    "orthogroup_hash": "" if og is None else str(stable_hash(og_s)),
                    "paragroup": pg_s,
                    "paragroup_hash": "" if pg is None else str(stable_hash(pg_s)),
                }
            )
    # Sort for binary search on id_MARKED_hash, then id_MARKED, then id.
    rows.sort(key=lambda r: (int(r["id_MARKED_hash"]), r["id_MARKED"], r["id"]))
    meta = {
        "n_marked": len(marked_genome),
        "n_rows": len(rows),
        "n_marked_null_id": n_null,
        "n_marked_with_ensembl": n_mapped,
        "n_marked_multi_ensembl": n_multi,
    }
    return rows, meta


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id_MARKED",
        "id_MARKED_hash",
        "id",
        "genome",
        "orthogroup",
        "orthogroup_hash",
        "paragroup",
        "paragroup_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("|".join(cols) + "\n")
        for r in rows:
            fh.write("|".join(r[c] for c in cols) + "\n")


def assert_sorted_by_hash(rows: list[dict[str, str]]) -> None:
    prev = -1
    for r in rows:
        h = int(r["id_MARKED_hash"])
        if h < prev:
            raise AssertionError("table not sorted by id_MARKED_hash")
        prev = h



def load_hash_table(path: Path) -> list[dict[str, str]]:
    """Load the pipe-separated hash table into memory."""
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="|"))


def lookup_by_marked_id(rows: list[dict[str, str]], id_marked: str) -> list[dict[str, str]]:
    """Binary-search rows sorted by ``id_MARKED_hash``; return all rows for ``id_marked``."""
    import bisect

    target_h = stable_hash(str(id_marked))
    hashes = [int(r["id_MARKED_hash"]) for r in rows]
    i = bisect.bisect_left(hashes, target_h)
    out: list[dict[str, str]] = []
    while i < len(rows) and int(rows[i]["id_MARKED_hash"]) == target_h:
        if rows[i]["id_MARKED"] == str(id_marked):
            out.append(rows[i])
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--id-csv",
        type=Path,
        default=Path("ready_legnet/ID.csv"),
        help="Panel ID.csv (genome|...|ID)",
    )
    p.add_argument(
        "--nodes",
        type=Path,
        default=Path("mag/homology_graph/maps/nodes_extract.tsv"),
        help="Ensembl↔marked map (nodes_extract or nodes_enriched)",
    )
    p.add_argument(
        "--edges",
        type=Path,
        default=Path("mag/homology_graph/edges.tsv.gz"),
        help="Full ortholog/paralog edge table",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("mag/homology_graph/maps/gene_ortho_para_hash.tsv"),
        help="Output pipe-separated table sorted by id_MARKED_hash",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: <out>.summary.json)",
    )
    args = p.parse_args(argv)

    for path, label in (
        (args.id_csv, "ID.csv"),
        (args.nodes, "nodes"),
        (args.edges, "edges"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")

    print(f"[hash] load ID.csv {args.id_csv}", flush=True)
    marked_genome = load_id_csv(args.id_csv)
    print(f"[hash] MARKED genes: {len(marked_genome)}", flush=True)

    print(f"[hash] load nodes {args.nodes}", flush=True)
    marked_ensembl = load_marked_ensembl_map(args.nodes)
    print(f"[hash] marked with ensembl: {len(marked_ensembl)}", flush=True)

    print(f"[hash] load edges + UF {args.edges}", flush=True)
    orthogroup_of, paragroup_of, graph_stats = load_edges_assign_groups(args.edges)
    print(f"[hash] graph: {json.dumps(graph_stats)}", flush=True)

    rows, row_meta = build_rows(marked_genome, marked_ensembl, orthogroup_of, paragroup_of)
    assert_sorted_by_hash(rows)
    write_table(args.out, rows)
    summary = {
        **graph_stats,
        **row_meta,
        "id_csv": str(args.id_csv),
        "nodes": str(args.nodes),
        "edges": str(args.edges),
        "out": str(args.out),
        "hash": "fnv1a_32_stable_hash",
        "sort": "id_MARKED_hash,id_MARKED,id",
        "null_id_policy": "empty string when MARKED gene has no Ensembl homology map",
    }
    summary_path = args.summary or Path(str(args.out) + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[hash] wrote {args.out} rows={len(rows)}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
