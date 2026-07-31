"""Visualize homology graphs: network PNG + Altair / cnsplots diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .graph import EDGE_COLUMNS, PANEL_SPECIES

# Okabe–Ito (colorblind-safe); cycle if panel grows.
_OKABE_ITO = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
    "#999999",
    "#882255",
    "#44AA99",
)


def _genome_palette(genomes: Sequence[str]) -> dict[str, str]:
    return {g: _OKABE_ITO[i % len(_OKABE_ITO)] for i, g in enumerate(sorted(genomes))}


def edges_to_frame(edges: Iterable[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(list(edges), columns=list(EDGE_COLUMNS))
    if df.empty:
        return df
    for col in EDGE_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def drop_isolate_nodes(
    edges: Iterable[tuple[str, str, str, str, str]],
) -> list[tuple[str, str, str, str, str]]:
    """Keep only edges (genes appearing in ≥1 edge). Isolates never appear."""
    return list(edges)


def connected_components(
    edges: Iterable[tuple[str, str, str, str, str]],
    *,
    relation: str | None = None,
) -> list[list[tuple[str, str]]]:
    """Union-find components; optional filter to one relation type."""
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

    for g1, s1, g2, s2, rel in edges:
        if relation is not None and rel != relation:
            continue
        a, b = (g1, s1), (g2, s2)
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        union(a, b)

    buckets: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in parent:
        buckets[find(node)].append(node)
    return sorted(buckets.values(), key=len, reverse=True)


def compute_graph_stats(edges: Iterable[tuple[str, str, str, str, str]]) -> dict[str, pd.DataFrame]:
    """Tables for characteristic plots (cluster sizes, paralogs, linked families)."""
    edge_list = list(edges)
    df = edges_to_frame(edge_list)
    if df.empty:
        empty = pd.DataFrame()
        return {
            "component_sizes": empty,
            "paralog_degree": empty,
            "ortholog_group_sizes": empty,
            "linked_paralog_clusters": empty,
            "relation_mix": empty,
        }

    # Mixed-graph components (ortholog + paralog).
    comps = connected_components(edge_list)
    comp_rows = []
    for i, nodes in enumerate(comps):
        genomes = sorted({s for _, s in nodes})
        comp_rows.append(
            {
                "component_id": i,
                "n_genes": len(nodes),
                "n_genomes": len(genomes),
                "genomes": ",".join(genomes),
            }
        )
    component_sizes = pd.DataFrame(comp_rows)

    # Per-gene paralog degree (within-species).
    para_deg: Counter[tuple[str, str]] = Counter()
    for g1, s1, g2, s2, rel in edge_list:
        if rel != "paralog":
            continue
        para_deg[(g1, s1)] += 1
        para_deg[(g2, s2)] += 1
    paralog_degree = pd.DataFrame(
        [{"gene": g, "genome": s, "n_paralogs": n} for (g, s), n in para_deg.items()]
    )

    # Ortholog-only components ≈ orthogroups.
    ortho_comps = connected_components(edge_list, relation="ortholog")
    og_rows = []
    for i, nodes in enumerate(ortho_comps):
        genomes = sorted({s for _, s in nodes})
        og_rows.append(
            {
                "orthogroup_id": i,
                "n_genes": len(nodes),
                "n_genomes": len(genomes),
            }
        )
    ortholog_group_sizes = pd.DataFrame(og_rows)

    # Paralog families (same-species components on paralog edges) linked by orthology.
    para_comps = connected_components(edge_list, relation="paralog")
    node_to_fam: dict[tuple[str, str], int] = {}
    fam_size: dict[int, int] = {}
    fam_genome: dict[int, str] = {}
    for fid, nodes in enumerate(para_comps):
        fam_size[fid] = len(nodes)
        fam_genome[fid] = nodes[0][1]
        for node in nodes:
            node_to_fam[node] = fid

    # Singleton families for genes that only have ortholog edges.
    next_fid = len(para_comps)
    for g1, s1, g2, s2, rel in edge_list:
        for node in ((g1, s1), (g2, s2)):
            if node not in node_to_fam:
                node_to_fam[node] = next_fid
                fam_size[next_fid] = 1
                fam_genome[next_fid] = node[1]
                next_fid += 1

    linked: set[tuple[int, int]] = set()
    for g1, s1, g2, s2, rel in edge_list:
        if rel != "ortholog":
            continue
        fa, fb = node_to_fam[(g1, s1)], node_to_fam[(g2, s2)]
        if fa == fb:
            continue
        a, b = sorted((fa, fb))
        linked.add((a, b))
    linked_paralog_clusters = pd.DataFrame(
        [
            {
                "family_a": a,
                "family_b": b,
                "genome_a": fam_genome[a],
                "genome_b": fam_genome[b],
                "size_a": fam_size[a],
                "size_b": fam_size[b],
                "size_ratio": fam_size[a] / fam_size[b] if fam_size[b] else np.nan,
                "size_min": min(fam_size[a], fam_size[b]),
                "size_max": max(fam_size[a], fam_size[b]),
            }
            for a, b in sorted(linked)
        ]
    )

    relation_mix = (
        df.groupby(["genome1", "relation"], as_index=False)
        .size()
        .rename(columns={"size": "n_edges", "genome1": "genome"})
    )

    return {
        "component_sizes": component_sizes,
        "paralog_degree": paralog_degree,
        "ortholog_group_sizes": ortholog_group_sizes,
        "linked_paralog_clusters": linked_paralog_clusters,
        "relation_mix": relation_mix,
    }


def _select_drawable_components(
    components: list[list[tuple[str, str]]],
    *,
    n_panels: int,
    min_nodes: int,
    max_nodes: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    """Sample components across size strata so panels are not all max-sized."""
    eligible = [c for c in components if min_nodes <= len(c) <= max_nodes]
    if not eligible:
        eligible = [c for c in components if len(c) >= 2]
    if not eligible:
        return []

    # Size bins: small / medium / large within the drawable range.
    sizes = np.array([len(c) for c in eligible], dtype=int)
    q33, q66 = np.quantile(sizes, [0.33, 0.66])
    bins: list[list[list[tuple[str, str]]]] = [[], [], []]
    for c in eligible:
        n = len(c)
        if n <= q33:
            bins[0].append(c)
        elif n <= q66:
            bins[1].append(c)
        else:
            bins[2].append(c)

    rng = np.random.default_rng(seed)
    chosen: list[list[tuple[str, str]]] = []
    # Round-robin across non-empty bins for diversity.
    pointers = [0, 0, 0]
    for b in bins:
        rng.shuffle(b)
    while len(chosen) < n_panels and any(pointers[i] < len(bins[i]) for i in range(3)):
        for i in range(3):
            if len(chosen) >= n_panels:
                break
            if pointers[i] < len(bins[i]):
                chosen.append(bins[i][pointers[i]])
                pointers[i] += 1
    return chosen


def plot_network_png(
    edges: Iterable[tuple[str, str, str, str, str]],
    out_path: Path | str,
    *,
    max_components: int = 24,
    max_nodes_per_component: int = 35,
    min_nodes: int = 3,
    edge_alpha: float = 0.18,
    dpi: int = 300,
    seed: int = 42,
) -> Path:
    """Draw a grid of mid-sized connected components (isolates already absent).

    Huge components are skipped for layout tractability; the caption in the
    companion stats figures covers full-graph size distributions.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_list = drop_isolate_nodes(edges)
    comps = _select_drawable_components(
        connected_components(edge_list),
        n_panels=max_components,
        min_nodes=min_nodes,
        max_nodes=max_nodes_per_component,
        seed=seed,
    )
    if not comps:
        raise ValueError("No drawable components after dropping isolates")

    node_set = {n for c in comps for n in c}
    # adjacency restricted to selected components
    G = nx.Graph()
    for gene, genome in node_set:
        G.add_node((gene, genome), genome=genome, gene=gene)
    for g1, s1, g2, s2, rel in edge_list:
        a, b = (g1, s1), (g2, s2)
        if a in node_set and b in node_set:
            G.add_edge(a, b, relation=rel)

    genomes = sorted({d["genome"] for _, d in G.nodes(data=True)})
    palette = _genome_palette(genomes)

    n = len(comps)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.4 * nrows),
        dpi=dpi,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(seed)

    for ax in axes_arr:
        ax.set_axis_off()

    for i, nodes in enumerate(comps):
        ax = axes_arr[i]
        sub = G.subgraph(nodes).copy()
        pos = nx.spring_layout(
            sub,
            seed=int(rng.integers(0, 10_000)),
            k=1.2 / np.sqrt(max(len(sub), 1)),
        )
        ortho = [(u, v) for u, v, d in sub.edges(data=True) if d.get("relation") == "ortholog"]
        para = [(u, v) for u, v, d in sub.edges(data=True) if d.get("relation") == "paralog"]
        nx.draw_networkx_edges(
            sub, pos, edgelist=ortho, ax=ax, edge_color="#0072B2", width=0.6, alpha=edge_alpha
        )
        nx.draw_networkx_edges(
            sub, pos, edgelist=para, ax=ax, edge_color="#D55E00", width=0.6, alpha=edge_alpha
        )
        colors = [palette[sub.nodes[n]["genome"]] for n in sub.nodes]
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=28, linewidths=0)
        ax.set_axis_off()

    legend_genomes = [
        Patch(facecolor=palette[g], edgecolor="none", label=g.replace("_", " ")) for g in genomes
    ]
    legend_edges = [
        Line2D([0], [0], color="#0072B2", lw=1.5, label="ortholog"),
        Line2D([0], [0], color="#D55E00", lw=1.5, label="paralog"),
    ]
    fig.legend(
        handles=legend_genomes + legend_edges,
        loc="lower center",
        ncol=min(4, len(legend_genomes) + 2),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Homology graph components (genes without orthologs/paralogs excluded)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _layout_component(
    graph: Any,
    *,
    seed: int,
) -> dict[Any, np.ndarray]:
    """Force-directed layout; algorithm scales with component size."""
    import networkx as nx

    n = graph.number_of_nodes()
    if n <= 1:
        return {node: np.array([0.0, 0.0], dtype=float) for node in graph.nodes}
    if n <= 80:
        pos = nx.spring_layout(graph, seed=seed, k=1.1 / np.sqrt(n), iterations=60)
    elif n <= 300:
        pos = nx.forceatlas2_layout(
            graph,
            seed=seed,
            max_iter=70,
            jitter_tolerance=1.0,
            scaling_ratio=2.0,
            gravity=1.0,
        )
    else:
        # Very large families: spectral init + short spring polish (tractable).
        try:
            init = nx.spectral_layout(graph, seed=seed)
        except Exception:
            init = None
        pos = nx.spring_layout(
            graph,
            pos=init,
            seed=seed,
            k=1.4 / np.sqrt(n),
            iterations=40,
        )
    return {k: np.asarray(v, dtype=float) for k, v in pos.items()}


def _normalize_pos(pos: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
    """Map positions into the unit square [0, 1]^2."""
    if not pos:
        return pos
    xy = np.vstack(list(pos.values()))
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return {k: (v - lo) / span for k, v in pos.items()}


def _pack_component_boxes(
    sizes: list[int],
    *,
    gap: float = 0.35,
    canvas_aspect: float = 1.35,
) -> list[tuple[float, float, float]]:
    """Shelf-pack square footprints; returns (x0, y0, side) per component.

    Footprint side ∝ sqrt(n) so large orthology/paralogy families get more space.
    """
    sides = [max(0.35, float(np.sqrt(max(n, 1)))) for n in sizes]
    total_area = sum(s * s for s in sides) * 1.25
    shelf_width = max(np.sqrt(total_area * canvas_aspect), max(sides) * 1.05)

    placements: list[tuple[float, float, float]] = []
    x = 0.0
    y = 0.0
    shelf_h = 0.0
    for side in sides:
        if x > 0 and x + side > shelf_width:
            x = 0.0
            y += shelf_h + gap
            shelf_h = 0.0
        placements.append((x, y, side))
        x += side + gap
        shelf_h = max(shelf_h, side)
    return placements


def plot_network_full_png(
    edges: Iterable[tuple[str, str, str, str, str]],
    out_path: Path | str,
    *,
    min_nodes: int = 6,
    edge_alpha: float = 0.06,
    node_size: float = 1.8,
    dpi: int = 200,
    seed: int = 42,
    max_edges_draw_per_component: int | None = None,
) -> Path:
    """Render the full homology graph for all components with ``n > 5``.

    Connected components are laid out independently (spring / ForceAtlas2), then
    packed onto one canvas. This is the complete within-panel graph excluding
    tiny components (n≤5) and isolates.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_list = list(edges)
    comps = [c for c in connected_components(edge_list) if len(c) >= min_nodes]
    if not comps:
        raise ValueError(f"No components with n>={min_nodes}")
    # Largest first: more visual weight / earlier packing slots.
    comps = sorted(comps, key=len, reverse=True)

    node_to_comp = {node: i for i, nodes in enumerate(comps) for node in nodes}
    keep = set(node_to_comp)

    # Build one NX graph for the filtered full edge set.
    G = nx.Graph()
    for gene, genome in keep:
        G.add_node((gene, genome), genome=genome)
    ortho_edges: list[tuple[Any, Any]] = []
    para_edges: list[tuple[Any, Any]] = []
    for g1, s1, g2, s2, rel in edge_list:
        a, b = (g1, s1), (g2, s2)
        if a not in keep or b not in keep:
            continue
        if node_to_comp[a] != node_to_comp[b]:
            continue
        G.add_edge(a, b, relation=rel)
        if rel == "ortholog":
            ortho_edges.append((a, b))
        elif rel == "paralog":
            para_edges.append((a, b))

    genomes = sorted({d["genome"] for _, d in G.nodes(data=True)})
    palette = _genome_palette(genomes)
    rng = np.random.default_rng(seed)

    print(
        f"[homology_full] components={len(comps)} nodes={G.number_of_nodes()} "
        f"edges={G.number_of_edges()}",
        flush=True,
    )

    global_pos: dict[Any, np.ndarray] = {}
    placements = _pack_component_boxes([len(c) for c in comps])
    for i, (nodes, (x0, y0, side)) in enumerate(zip(comps, placements)):
        sub = G.subgraph(nodes)
        local = _normalize_pos(_layout_component(sub, seed=int(rng.integers(0, 1_000_000))))
        for node, xy in local.items():
            global_pos[node] = np.array([x0 + xy[0] * side, y0 + xy[1] * side], dtype=float)
        if (i + 1) % 200 == 0 or i == 0 or i + 1 == len(comps):
            print(f"[homology_full] laid out {i + 1}/{len(comps)} (n={len(nodes)})", flush=True)

    def _segments(
        pairs: list[tuple[Any, Any]],
        *,
        limit: int | None,
    ) -> np.ndarray:
        if limit is not None and len(pairs) > limit:
            idx = rng.choice(len(pairs), size=limit, replace=False)
            pairs = [pairs[j] for j in idx]
        segs = np.empty((len(pairs), 2, 2), dtype=float)
        for k, (u, v) in enumerate(pairs):
            segs[k, 0] = global_pos[u]
            segs[k, 1] = global_pos[v]
        return segs

    # Optional global cap only if explicitly requested; default draw all edges.
    ortho_segs = _segments(ortho_edges, limit=max_edges_draw_per_component)
    para_segs = _segments(para_edges, limit=max_edges_draw_per_component)

    xs = [p[0] for p in global_pos.values()]
    ys = [p[1] for p in global_pos.values()]
    span_x = max(xs) - min(xs) + 1e-6
    span_y = max(ys) - min(ys) + 1e-6
    fig_w = 22.0
    fig_h = max(14.0, fig_w * (span_y / span_x))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if len(para_segs):
        ax.add_collection(
            LineCollection(
                para_segs,
                colors="#D55E00",
                linewidths=0.25,
                alpha=edge_alpha,
                rasterized=True,
            )
        )
    if len(ortho_segs):
        ax.add_collection(
            LineCollection(
                ortho_segs,
                colors="#0072B2",
                linewidths=0.25,
                alpha=edge_alpha,
                rasterized=True,
            )
        )

    # Nodes on top, colored by genome.
    by_genome: dict[str, list[Any]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        by_genome[data["genome"]].append(node)
    for genome, nodes in by_genome.items():
        coords = np.array([global_pos[n] for n in nodes], dtype=float)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=node_size,
            c=palette[genome],
            linewidths=0,
            alpha=0.85,
            rasterized=True,
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    ax.set_title(
        f"Full homology graph (components with n≥{min_nodes}; "
        f"{len(comps)} components, {G.number_of_nodes()} genes, {G.number_of_edges()} edges)",
        fontsize=11,
    )
    legend_genomes = [
        Patch(facecolor=palette[g], edgecolor="none", label=g.replace("_", " ")) for g in genomes
    ]
    legend_edges = [
        Line2D([0], [0], color="#0072B2", lw=1.5, label="ortholog"),
        Line2D([0], [0], color="#D55E00", lw=1.5, label="paralog"),
    ]
    ax.legend(
        handles=legend_genomes + legend_edges,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    # PNG + PDF (rasterized artists keep PDF size manageable).
    stem = out_path.with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[homology_full] wrote {png_path} and {pdf_path}", flush=True)
    return png_path


def _apply_cns_style(dpi: int = 300) -> None:
    import matplotlib.pyplot as plt
    import cnsplots as cns

    cns.settings.font_family = "sans-serif"
    cns.settings.font_sans_serif = ("DejaVu Sans", "Arial", "Helvetica")
    cns.settings.axes_spines_top = False
    cns.settings.axes_spines_right = False
    cns.settings.legend_frameon = False
    cns.settings.title_fontsize = 10
    cns.settings.legend_fontsize = 8
    cns.settings.figure_dpi = int(dpi)
    cns.setup_matplotlib(
        color_cycle="OkabeIto",
        title_fontsize=10,
        title_fontweight="regular",
        legend_fontsize=8,
        axes_linewidth=0.8,
    )
    plt.rcParams.update(
        {
            "savefig.dpi": int(dpi),
            "figure.dpi": min(int(dpi), 150),
            "axes.grid": True,
            "grid.color": "#B0B0B0",
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
        }
    )


def plot_characteristics_cnsplots(
    stats: dict[str, pd.DataFrame],
    outdir: Path | str,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Publication static figures via cnsplots (PDF/SVG/PNG)."""
    import matplotlib.pyplot as plt
    import cnsplots as cns

    from src.train_viz.plotting import save_cns_figure

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _apply_cns_style(dpi)
    written: list[Path] = []

    # 1. Component size distribution
    cs = stats["component_sizes"]
    if not cs.empty:
        cns.figure(width=360, height=260)
        plot_df = cs.copy()
        plot_df["log10_n_genes"] = np.log10(plot_df["n_genes"].clip(lower=1))
        ax = cns.histplot(data=plot_df, x="log10_n_genes", bins=40)
        ax.set_xlabel("log10(component size, genes)")
        ax.set_ylabel("Number of components")
        ax.set_title("Connected-component size distribution")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / "Figure_01_component_size_hist", dpi))

    # 2. Paralog degree
    pdg = stats["paralog_degree"]
    if not pdg.empty:
        cns.figure(width=360, height=260)
        plot_df = pdg.copy()
        plot_df["log10_n_paralogs"] = np.log10(plot_df["n_paralogs"].clip(lower=1))
        ax = cns.histplot(data=plot_df, x="log10_n_paralogs", bins=40)
        ax.set_xlabel("log10(paralog neighbors per gene)")
        ax.set_ylabel("Number of genes")
        ax.set_title("Paralog-degree distribution")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / "Figure_02_paralog_degree_hist", dpi))

    # 3. Orthogroup sizes
    og = stats["ortholog_group_sizes"]
    if not og.empty:
        cns.figure(width=360, height=260)
        plot_df = og.copy()
        plot_df["log10_n_genes"] = np.log10(plot_df["n_genes"].clip(lower=1))
        ax = cns.histplot(data=plot_df, x="log10_n_genes", bins=40)
        ax.set_xlabel("log10(orthogroup size, genes)")
        ax.set_ylabel("Number of orthogroups")
        ax.set_title("Ortholog-only component sizes")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / "Figure_03_orthogroup_size_hist", dpi))

    # 4. Linked paralog-cluster size scatter
    link = stats["linked_paralog_clusters"]
    if not link.empty:
        cns.figure(width=360, height=300)
        plot_df = link.copy()
        # subsample for overplotting if huge
        if len(plot_df) > 50_000:
            plot_df = plot_df.sample(50_000, random_state=42)
        ax = cns.scatterplot(
            data=plot_df,
            x="size_a",
            y="size_b",
            alpha=0.25,
            s=8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Paralog-cluster size A (genes)")
        ax.set_ylabel("Paralog-cluster size B (genes)")
        ax.set_title("Sizes of paralog clusters linked by orthology")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / "Figure_04_linked_cluster_sizes", dpi))

    # 5. Size ratio of linked clusters
    if not link.empty:
        cns.figure(width=360, height=260)
        plot_df = link.copy()
        plot_df["log2_size_ratio"] = np.log2(
            (plot_df["size_max"] / plot_df["size_min"]).clip(lower=1e-12)
        )
        ax = cns.histplot(data=plot_df, x="log2_size_ratio", bins=40)
        ax.set_xlabel("log2(size_max / size_min) for orthology-linked clusters")
        ax.set_ylabel("Number of linked pairs")
        ax.set_title("Size imbalance between linked paralog clusters")
        cns.setup_ax(ax)
        written.extend(save_cns_figure(outdir / "Figure_05_linked_cluster_size_ratio", dpi))

    plt.close("all")
    return written


def plot_characteristics_altair(
    stats: dict[str, pd.DataFrame],
    outdir: Path | str,
) -> list[Path]:
    """Interactive Altair charts (HTML + VL JSON + PNG when vl-convert works)."""
    import altair as alt

    from src.train_viz.plotting import save_altair_chart

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alt.data_transformers.disable_max_rows()

    cs = stats["component_sizes"]
    if not cs.empty:
        plot_df = cs.copy()
        plot_df["log10_n_genes"] = np.log10(plot_df["n_genes"].clip(lower=1))
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("log10_n_genes:Q", bin=alt.Bin(maxbins=40), title="log10(component size)"),
                y=alt.Y("count()", title="Number of components"),
                tooltip=["n_genes", "n_genomes"],
            )
            .properties(title="Connected-component size distribution", width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / "Figure_01_component_size_hist_altair"))

    pdg = stats["paralog_degree"]
    if not pdg.empty:
        plot_df = pdg.copy()
        plot_df["log10_n_paralogs"] = np.log10(plot_df["n_paralogs"].clip(lower=1))
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "log10_n_paralogs:Q",
                    bin=alt.Bin(maxbins=40),
                    title="log10(paralog neighbors)",
                ),
                y=alt.Y("count()", title="Number of genes"),
                color=alt.Color("genome:N", legend=alt.Legend(title="Genome")),
            )
            .properties(title="Paralog-degree distribution by genome", width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / "Figure_02_paralog_degree_hist_altair"))

    og = stats["ortholog_group_sizes"]
    if not og.empty:
        plot_df = og.copy()
        chart = (
            alt.Chart(plot_df)
            .mark_circle(opacity=0.35)
            .encode(
                x=alt.X("n_genomes:Q", title="Genomes in orthogroup"),
                y=alt.Y("n_genes:Q", scale=alt.Scale(type="log"), title="Genes in orthogroup"),
                tooltip=["orthogroup_id", "n_genes", "n_genomes"],
            )
            .properties(title="Orthogroup size vs genome span", width=420, height=280)
        )
        written.extend(save_altair_chart(chart, outdir / "Figure_03_orthogroup_span_altair"))

    link = stats["linked_paralog_clusters"]
    if not link.empty:
        plot_df = link if len(link) <= 50_000 else link.sample(50_000, random_state=42)
        chart = (
            alt.Chart(plot_df)
            .mark_circle(opacity=0.25, size=20)
            .encode(
                x=alt.X("size_a:Q", scale=alt.Scale(type="log"), title="Cluster size A"),
                y=alt.Y("size_b:Q", scale=alt.Scale(type="log"), title="Cluster size B"),
                tooltip=["genome_a", "genome_b", "size_a", "size_b", "size_ratio"],
            )
            .properties(
                title="Paralog-cluster sizes linked by orthology",
                width=420,
                height=320,
            )
        )
        written.extend(save_altair_chart(chart, outdir / "Figure_04_linked_cluster_sizes_altair"))

        chart2 = (
            alt.Chart(plot_df.assign(log2_ratio=np.log2((plot_df["size_max"] / plot_df["size_min"]))))
            .mark_bar()
            .encode(
                x=alt.X(
                    "log2_ratio:Q",
                    bin=alt.Bin(maxbins=40),
                    title="log2(size_max/size_min)",
                ),
                y=alt.Y("count()", title="Linked pairs"),
            )
            .properties(title="Size imbalance of orthology-linked clusters", width=420, height=280)
        )
        written.extend(
            save_altair_chart(chart2, outdir / "Figure_05_linked_cluster_size_ratio_altair")
        )

    return written


def write_stats_tables(stats: dict[str, pd.DataFrame], outdir: Path | str) -> list[Path]:
    """Persist stats tables as TSV for reuse."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in stats.items():
        path = outdir / f"{name}.tsv.gz"
        df.to_csv(path, sep="\t", index=False, compression="gzip")
        written.append(path)
    return written
