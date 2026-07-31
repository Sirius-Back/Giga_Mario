"""Build Ensembl gene → MARKED ID map from local ID.csv + raw/gtf + gene2ensembl.

Outputs under ``mag/homology_graph/maps/``:
  - nodes_enriched.tsv
  - map_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

GCF_TO_ENSEMBL = {
    "GCF_000001405.40": "homo_sapiens",
    "GCF_000001635.27": "mus_musculus",
    "GCF_036323735.1": "rattus_norvegicus",
    "GCF_000003025.6": "sus_scrofa",
    "GCF_002263795.3": "bos_taurus",
    "GCF_002863925.1": "equus_caballus",
    "GCF_011100685.1": "canis_lupus_familiaris",
    "GCF_016772045.2": "ovis_aries",
    "GCF_049350105.2": "macaca_mulatta",
    "GCF_964237555.1": "oryctolagus_cuniculus",
    "GCF_001704415.2": "capra_hircus",
}
ENSEMBL_TO_TAXID = {
    "homo_sapiens": 9606,
    "mus_musculus": 10090,
    "rattus_norvegicus": 10116,
    "sus_scrofa": 9823,
    "bos_taurus": 9913,
    "equus_caballus": 9796,
    "canis_lupus_familiaris": 9615,
    "ovis_aries": 9940,
    "macaca_mulatta": 9544,
    "oryctolagus_cuniculus": 9986,
    "capra_hircus": 9925,
}
GENEID_RE = re.compile(r'db_xref "GeneID:(\d+)"')
GENE_ATTR_RE = re.compile(r'\bgene "([^"]+)"')


def load_id_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    required = {"genome", "gene_nameORnon_coding_ID", "ID"}
    if missing := required - set(df.columns):
        raise ValueError(f"ID.csv missing columns {missing}")
    df["genome"] = df["genome"].str.strip()
    df["gene_nameORnon_coding_ID"] = df["gene_nameORnon_coding_ID"].str.strip()
    df["ID"] = df["ID"].str.strip()
    df["ensembl_species"] = df["genome"].map(GCF_TO_ENSEMBL)
    if df["ensembl_species"].isna().any():
        bad = sorted(df.loc[df["ensembl_species"].isna(), "genome"].unique())
        raise ValueError(f"Unmapped GCF in ID.csv: {bad}")
    return df


def parse_gtf_geneid_symbol(gtf_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with gtf_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            m_id, m_gene = GENEID_RE.search(attrs), GENE_ATTR_RE.search(attrs)
            if m_id and m_gene:
                out.setdefault(m_id.group(1), m_gene.group(1))
    return out


def load_gene2ensembl(path: Path, taxids: set[int]) -> dict[tuple[int, str], str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out: dict[tuple[int, str], str] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            try:
                tax = int(cols[0])
            except ValueError:
                continue
            if tax not in taxids:
                continue
            gene_id, ens = cols[1], cols[2]
            if ens and ens != "-":
                out.setdefault((tax, gene_id), ens.split(".")[0])
    return out


def load_homology_genes(edges_path: Path) -> set[tuple[str, str]]:
    genes: set[tuple[str, str]] = set()
    opener = gzip.open if str(edges_path).endswith(".gz") else open
    with opener(edges_path, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i1, s1 = header.index("gene1"), header.index("genome1")
        i2, s2 = header.index("gene2"), header.index("genome2")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            genes.add((p[s1], p[i1].split(".")[0]))
            genes.add((p[s2], p[i2].split(".")[0]))
    return genes


def build_maps(*, id_csv: Path, gtf_dir: Path, gene2ensembl: Path, edges: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    id_df = load_id_csv(id_csv)
    symbol_to_marked: dict[tuple[str, str], str] = {}
    for row in id_df.itertuples(index=False):
        symbol_to_marked.setdefault((row.ensembl_species, row.gene_nameORnon_coding_ID), str(row.ID))

    gcf_geneid_symbol: dict[str, dict[str, str]] = {}
    for gcf, sp in GCF_TO_ENSEMBL.items():
        matches = sorted(gtf_dir.glob(f"{gcf}_*_genomic.gtf"))
        if not matches:
            raise FileNotFoundError(f"No GTF for {gcf} in {gtf_dir}")
        gcf_geneid_symbol[gcf] = parse_gtf_geneid_symbol(matches[0])
        print(f"[map] GTF {gcf}: {len(gcf_geneid_symbol[gcf])} GeneIDs", flush=True)

    if not gene2ensembl.is_file() or gene2ensembl.stat().st_size < 50_000_000:
        raise FileNotFoundError(
            f"gene2ensembl incomplete ({gene2ensembl}, "
            f"{gene2ensembl.stat().st_size if gene2ensembl.is_file() else 0} bytes); wait for download"
        )
    g2e = load_gene2ensembl(gene2ensembl, set(ENSEMBL_TO_TAXID.values()))
    print(f"[map] gene2ensembl panel rows: {len(g2e)}", flush=True)

    ens_to_symbol: dict[tuple[str, str], str] = {}
    for gcf, sp in GCF_TO_ENSEMBL.items():
        tax = ENSEMBL_TO_TAXID[sp]
        for gene_id, symbol in gcf_geneid_symbol[gcf].items():
            ens = g2e.get((tax, gene_id))
            if ens:
                ens_to_symbol.setdefault((sp, ens), symbol)

    homology_genes = load_homology_genes(edges)
    print(f"[map] homology genes: {len(homology_genes)}", flush=True)

    gcf_of = {s: g for g, s in GCF_TO_ENSEMBL.items()}
    rows = []
    matched = 0
    for sp, ens in sorted(homology_genes):
        ens0 = ens.split(".")[0]
        symbol = ens_to_symbol.get((sp, ens0), "")
        marked = symbol_to_marked.get((sp, symbol), "") if symbol else ""
        if marked:
            matched += 1
        rows.append(
            {
                "ensembl_species": sp,
                "ensembl_gene": ens0,
                "gene_symbol": symbol,
                "marked_id": marked,
                "gcf": gcf_of[sp],
            }
        )
    out_nodes = outdir / "nodes_enriched.tsv"
    pd.DataFrame(rows).to_csv(out_nodes, sep="\t", index=False)

    # Edges where both endpoints have MARKED ids (plain TSV for C++).
    marked_keys = {(r["ensembl_species"], r["ensembl_gene"]) for r in rows if r["marked_id"]}
    opener = gzip.open if str(edges).endswith(".gz") else open
    out_edges = outdir / "edges_extract.tsv"
    n_keep = 0
    with opener(edges, "rt", encoding="utf-8") as fh, out_edges.open("w", encoding="utf-8") as out:
        header = fh.readline().rstrip("\n")
        out.write(header + "\n")
        cols = header.split("\t")
        i1, s1 = cols.index("gene1"), cols.index("genome1")
        i2, s2 = cols.index("gene2"), cols.index("genome2")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            g1, g2 = p[i1].split(".")[0], p[i2].split(".")[0]
            if (p[s1], g1) in marked_keys and (p[s2], g2) in marked_keys:
                out.write(line if line.endswith("\n") else line + "\n")
                n_keep += 1
    print(f"[map] edges_extract: {n_keep}", flush=True)

    summary = {
        "n_homology_genes": len(homology_genes),
        "n_with_symbol": sum(1 for r in rows if r["gene_symbol"]),
        "n_with_marked_id": matched,
        "fraction_marked": matched / max(len(homology_genes), 1),
        "nodes_enriched": str(out_nodes),
        "edges_extract": str(out_edges),
        "n_edges_extract": n_keep,
    }
    (outdir / "map_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-csv", type=Path, default=Path("ready_legnet/ID.csv"))
    p.add_argument("--gtf-dir", type=Path, default=Path("raw/gtf"))
    p.add_argument("--gene2ensembl", type=Path, default=Path("mag/homology_graph/maps/gene2ensembl.gz"))
    p.add_argument("--edges", type=Path, default=Path("mag/homology_graph/edges.tsv.gz"))
    p.add_argument("--outdir", type=Path, default=Path("mag/homology_graph/maps"))
    args = p.parse_args(argv)
    build_maps(
        id_csv=args.id_csv,
        gtf_dir=args.gtf_dir,
        gene2ensembl=args.gene2ensembl,
        edges=args.edges,
        outdir=args.outdir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
