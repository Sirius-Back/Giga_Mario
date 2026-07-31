"""Build an undirected ortholog/paralog edge table from Ensembl Compara TSVs.

Reads ``Compara.<release>.protein_default.homologies.tsv.gz`` per species,
keeps edges whose both genomes are in the configured panel, collapses
``homology_type`` to ``ortholog`` / ``paralog``, and deduplicates undirected
edges (gene order canonicalized).
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping

# Scientific panel from raw/ ∩ mag/intersection.md (Ensembl production names).
PANEL_SPECIES: tuple[str, ...] = (
    "homo_sapiens",
    "mus_musculus",
    "rattus_norvegicus",
    "sus_scrofa",
    "bos_taurus",
    "equus_caballus",
    "canis_lupus_familiaris",
    "ovis_aries",
    "macaca_mulatta",
    "oryctolagus_cuniculus",
    "capra_hircus",
)

EDGE_COLUMNS = ("gene1", "genome1", "gene2", "genome2", "relation")


def classify_relation(homology_type: str) -> str | None:
    """Map Ensembl ``homology_type`` to ``ortholog`` / ``paralog`` (or None)."""
    ht = homology_type.strip().lower()
    if "ortholog" in ht:
        return "ortholog"
    if "paralog" in ht:
        return "paralog"
    return None


def _canonical_edge(
    gene_a: str,
    genome_a: str,
    gene_b: str,
    genome_b: str,
    relation: str,
) -> tuple[str, str, str, str, str]:
    """Order endpoints so undirected (a,b) == (b,a)."""
    if (gene_a, genome_a) <= (gene_b, genome_b):
        return gene_a, genome_a, gene_b, genome_b, relation
    return gene_b, genome_b, gene_a, genome_a, relation


def compara_path(ensembl_data: Path, species: str, release: int) -> Path:
    """Expected local Compara protein_default path for one species."""
    return (
        Path(ensembl_data)
        / species
        / "compara_homology"
        / f"Compara.{release}.protein_default.homologies.tsv.gz"
    )


def iter_compara_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield Compara TSV rows as dicts; fail early on missing/empty file."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty Compara file: {path}")
    with gzip.open(path, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Compara TSV has no header: {path}")
        required = {
            "gene_stable_id",
            "species",
            "homology_type",
            "homology_gene_stable_id",
            "homology_species",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Compara TSV {path} missing columns: {sorted(missing)}")
        for row in reader:
            yield row


@dataclass
class HomologyGraphResult:
    """Edge set plus compact summary statistics."""

    edges: set[tuple[str, str, str, str, str]] = field(default_factory=set)
    n_rows_read: int = 0
    n_rows_kept: int = 0
    n_self_loops_skipped: int = 0
    relation_counts: Counter[str] = field(default_factory=Counter)
    per_species_files: dict[str, str] = field(default_factory=dict)

    def sorted_edges(self) -> list[tuple[str, str, str, str, str]]:
        return sorted(self.edges)


def build_homology_graph(
    ensembl_data: Path | str,
    *,
    species: Iterable[str] = PANEL_SPECIES,
    release: int = 116,
    high_confidence_only: bool = False,
) -> HomologyGraphResult:
    """Collect unique within-panel ortholog/paralog edges.

    Args:
        ensembl_data: ``mag/ensembl/data`` root.
        species: Ensembl production names to include.
        release: Compara release number embedded in filenames.
        high_confidence_only: if True, keep only ``is_high_confidence == 1``.

    Returns:
        Deduplicated undirected edge set and counters.
    """
    ensembl_data = Path(ensembl_data)
    panel = tuple(species)
    panel_set = set(panel)
    if not panel_set:
        raise ValueError("species panel is empty")

    result = HomologyGraphResult()
    for sp in panel:
        path = compara_path(ensembl_data, sp, release)
        result.per_species_files[sp] = str(path)
        for row in iter_compara_rows(path):
            result.n_rows_read += 1
            g1 = row["gene_stable_id"].strip()
            g2 = row["homology_gene_stable_id"].strip()
            s1 = row["species"].strip()
            s2 = row["homology_species"].strip()
            if not g1 or not g2 or s1 not in panel_set or s2 not in panel_set:
                continue
            if high_confidence_only:
                conf = (row.get("is_high_confidence") or "").strip()
                if conf not in {"1", "true", "True"}:
                    continue
            relation = classify_relation(row["homology_type"])
            if relation is None:
                continue
            if g1 == g2 and s1 == s2:
                result.n_self_loops_skipped += 1
                continue
            # Ortholog edges must be cross-genome; paralog edges same-genome.
            if relation == "ortholog" and s1 == s2:
                continue
            if relation == "paralog" and s1 != s2:
                # other_paralog can be cross-species in Compara; treat as ortholog-like
                # only when genomes differ? Keep as paralog label only for same genome;
                # cross-genome other_paralog → skip to avoid polluting ortholog graph.
                continue
            edge = _canonical_edge(g1, s1, g2, s2, relation)
            if edge not in result.edges:
                result.relation_counts[relation] += 1
            result.edges.add(edge)
            result.n_rows_kept += 1
    return result


def write_edge_table(
    edges: Iterable[tuple[str, str, str, str, str]],
    path: Path | str,
    *,
    columns: tuple[str, ...] = EDGE_COLUMNS,
) -> Path:
    """Write edge table as gzipped TSV (gene1 genome1 gene2 genome2 relation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for edge in edges:
            if len(edge) != len(columns):
                raise ValueError(f"edge arity {len(edge)} != columns {len(columns)}")
            writer.writerow(edge)
    return path


def load_edge_table(path: Path | str) -> list[tuple[str, str, str, str, str]]:
    """Load edge TSV.gz written by :func:`write_edge_table`."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Edge table missing or empty: {path}")
    edges: list[tuple[str, str, str, str, str]] = []
    with gzip.open(path, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = set(EDGE_COLUMNS)
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError(
                f"Edge table {path} missing columns; need {EDGE_COLUMNS}, got {reader.fieldnames}"
            )
        for row in reader:
            edges.append(
                (
                    row["gene1"],
                    row["genome1"],
                    row["gene2"],
                    row["genome2"],
                    row["relation"],
                )
            )
    return edges


def summarize_graph(
    edges: Iterable[tuple[str, str, str, str, str]],
) -> dict[str, object]:
    """Compute node/edge/component summary for manifests."""
    edge_list = list(edges)
    nodes: set[tuple[str, str]] = set()
    by_rel: Counter[str] = Counter()
    genomes: Counter[str] = Counter()
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g1, s1, g2, s2, rel in edge_list:
        a, b = (g1, s1), (g2, s2)
        nodes.add(a)
        nodes.add(b)
        genomes[s1] += 1
        genomes[s2] += 1
        by_rel[rel] += 1
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        union(a, b)

    comps: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in nodes:
        comps[find(node)].append(node)
    sizes = sorted((len(v) for v in comps.values()), reverse=True)

    return {
        "n_edges": len(edge_list),
        "n_nodes": len(nodes),
        "n_genomes": len({s for _, s in nodes}),
        "n_components": len(comps),
        "relation_counts": dict(by_rel),
        "component_size_max": sizes[0] if sizes else 0,
        "component_size_median": sizes[len(sizes) // 2] if sizes else 0,
        "nodes_per_genome": dict(
            Counter(s for _, s in nodes)
        ),
        "endpoint_mentions_per_genome": dict(genomes),
    }


def write_summary(summary: Mapping[str, object], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
