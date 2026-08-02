"""Stage-2: export hash-node pangenome graph + compositional wrap for VGAE."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.splits.pangenome_native import ensure_built, get_native_graph, reset_native_graph
from src.splits.sbs.backends.gc import gc_percent
from src.splits.sbs.fna_io import load_fna_directory
from src.splits.vgae.graph_data import (
    PackedGraph,
    assert_no_homology_features,
    build_compositional_features,
    dense_kmer_relative,
)


def _decode_hash(h: int, k: int) -> str:
    alphabet = "ACGT"
    chars = []
    val = int(h)
    for _ in range(k):
        chars.append(alphabet[val & 3])
        val >>= 2
    return "".join(reversed(chars))


def export_hash_graph_native(
    sequences: list[str],
    *,
    k: int = 5,
    min_df: int = 2,
    min_cooccur: int = 2,
    max_edges: int = 500_000,
) -> dict[str, Any]:
    """Call C++ ``pangenome_export_hash_graph`` (rebuilds .so if sources newer)."""
    import ctypes

    ensure_built(force=False)
    reset_native_graph()
    native = get_native_graph()
    lib = native._lib  # noqa: SLF001 — intentional ctypes access
    if not hasattr(lib, "pangenome_export_hash_graph"):
        ensure_built(force=True)
        reset_native_graph()
        native = get_native_graph()
        lib = native._lib

    fn = lib.pangenome_export_hash_graph
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    fn.restype = ctypes.c_int

    blob, offsets, n = native._prepare_blob(sequences)  # noqa: SLF001
    n_hashes = ctypes.c_int32(0)
    n_edges = ctypes.c_int32(0)
    n_inc = ctypes.c_int32(0)

    # Size query
    rc = fn(
        blob,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int32(n),
        ctypes.c_int(int(k)),
        ctypes.c_int32(int(min_df)),
        ctypes.c_int32(int(min_cooccur)),
        None,
        ctypes.c_int32(0),
        ctypes.byref(n_hashes),
        None,
        None,
        None,
        ctypes.c_int32(0),
        ctypes.byref(n_edges),
        None,
        None,
        ctypes.c_int32(0),
        ctypes.byref(n_inc),
    )
    if rc != 0:
        raise RuntimeError(f"pangenome_export_hash_graph size-query failed rc={rc}")

    nh = int(n_hashes.value)
    ne_all = int(n_edges.value)
    ni = int(n_inc.value)
    ne = min(ne_all, int(max_edges))

    hash_values = np.zeros(nh, dtype=np.uint64)
    edge_u = np.zeros(ne, dtype=np.int32)
    edge_v = np.zeros(ne, dtype=np.int32)
    edge_w = np.zeros(ne, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indices = np.zeros(ni, dtype=np.int32)

    rc = fn(
        blob,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int32(n),
        ctypes.c_int(int(k)),
        ctypes.c_int32(int(min_df)),
        ctypes.c_int32(int(min_cooccur)),
        hash_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_int32(nh),
        ctypes.byref(n_hashes),
        edge_u.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        edge_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        edge_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(ne),
        ctypes.byref(n_edges),
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(ni),
        ctypes.byref(n_inc),
    )
    if rc != 0:
        raise RuntimeError(f"pangenome_export_hash_graph fill failed rc={rc}")

    ne_w = int(n_edges.value)
    return {
        "hash_values": hash_values[: int(n_hashes.value)].copy(),
        "edge_u": edge_u[:ne_w].copy(),
        "edge_v": edge_v[:ne_w].copy(),
        "edge_w": edge_w[:ne_w].copy(),
        "inc_indptr": indptr.copy(),
        "inc_indices": indices[: int(n_inc.value)].copy(),
        "n_regions": n,
        "k": int(k),
        "min_df": int(min_df),
        "min_cooccur": int(min_cooccur),
        "n_edges_uncapped": ne_all,
    }


def pack_hash_graph(
    marked_dir: Path,
    region_ids: Sequence[str],
    pack_dir: Path,
    *,
    k: int = 5,
    min_df: int = 2,
    min_cooccur: int = 2,
    max_edges: int = 500_000,
    max_ids: int | None = None,
) -> tuple[PackedGraph, dict[str, Any]]:
    """Build hash-node PackedGraph + incidence for pooling back to regions."""
    marked_dir = Path(marked_dir)
    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    ids = [str(x) for x in region_ids]
    if max_ids is not None:
        ids = ids[: int(max_ids)]
    sequences_map = load_fna_directory(marked_dir, ids=ids)
    sequences = [sequences_map[i] for i in ids]

    exported = export_hash_graph_native(
        sequences,
        k=k,
        min_df=min_df,
        min_cooccur=min_cooccur,
        max_edges=max_edges,
    )
    hash_values = exported["hash_values"]
    n_h = int(len(hash_values))
    if n_h < 3:
        raise RuntimeError(f"hash graph too small: n_hashes={n_h}")

    # Region compositional features for incidence pooling
    region_x, feat_names = build_compositional_features(ids, sequences_map, k=k)
    assert_no_homology_features(feat_names)

    # Hash node features: GC of decoded k-mer + incidence-pooled region X mean
    x = np.zeros((n_h, region_x.shape[1]), dtype=np.float32)
    counts = np.zeros(n_h, dtype=np.float64)
    indptr = exported["inc_indptr"]
    indices = exported["inc_indices"]
    for r in range(len(ids)):
        a = int(indptr[r])
        b = int(indptr[r + 1])
        for hid in indices[a:b].tolist():
            x[hid] += region_x[r]
            counts[hid] += 1.0
    for h in range(n_h):
        if counts[h] > 0:
            x[h] /= counts[h]
        else:
            # Fallback: features of the k-mer string itself
            kmer = _decode_hash(int(hash_values[h]), k)
            x[h, 0] = float(gc_percent(kmer))
            x[h, 1:] = dense_kmer_relative(kmer, k)

    # Override GC column with decoded-hash GC (compositional signal of the node)
    for h in range(n_h):
        kmer = _decode_hash(int(hash_values[h]), k)
        x[h, 0] = float(gc_percent(kmer))

    edge_u = np.asarray(exported["edge_u"], dtype=np.int64)
    edge_v = np.asarray(exported["edge_v"], dtype=np.int64)
    edge_w_raw = np.asarray(exported["edge_w"], dtype=np.int32)
    w = np.log1p(edge_w_raw.astype(np.float64))
    if w.size and float(w.max()) > 0:
        w = w / float(w.max())
    edge_w = w.astype(np.float32)

    hash_ids = [f"hash_{i}" for i in range(n_h)]
    np.savez_compressed(
        pack_dir / "node_features.npz",
        x=x,
        feature_names=np.asarray(feat_names, dtype=object),
    )
    np.savez_compressed(
        pack_dir / "edges_weighted.npz",
        edge_u=edge_u.astype(np.int32),
        edge_v=edge_v.astype(np.int32),
        edge_w=edge_w,
        edge_w_raw=edge_w_raw,
    )
    np.savez_compressed(
        pack_dir / "incidence.npz",
        indptr=indptr.astype(np.int32),
        indices=indices.astype(np.int32),
        region_ids=np.asarray(ids, dtype=object),
        hash_values=hash_values,
    )
    (pack_dir / "ids.txt").write_text(
        "\n".join(hash_ids) + "\n", encoding="utf-8"
    )
    meta = {
        "format": "gigamario_vgae_pack_v1",
        "grain": "hash",
        "k": int(k),
        "n_nodes": n_h,
        "n_edges": int(len(edge_u)),
        "n_features": int(x.shape[1]),
        "feature_names": list(feat_names),
        "homology_in_encoder": False,
        "marked_dir": str(marked_dir.resolve()),
        "n_regions": len(ids),
        "min_df": int(min_df),
        "min_cooccur": int(min_cooccur),
        "n_edges_uncapped": int(exported["n_edges_uncapped"]),
        "max_ids": max_ids,
    }
    (pack_dir / "feature_meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )

    pack = PackedGraph(
        ids=tuple(hash_ids),
        x=x,
        feature_names=feat_names,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        edge_w_raw=edge_w_raw,
        k=int(k),
        meta=meta,
        pack_dir=pack_dir,
    )
    incidence = {
        "indptr": indptr,
        "indices": indices,
        "region_ids": ids,
        "hash_values": hash_values,
    }
    return pack, incidence


def pool_hash_scores_to_regions(
    hash_scores: np.ndarray,
    incidence: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    """Mean-pool hash role scores → per-region scores."""
    indptr = np.asarray(incidence["indptr"], dtype=np.int32)
    indices = np.asarray(incidence["indices"], dtype=np.int32)
    region_ids = [str(x) for x in incidence["region_ids"]]
    n_r = len(region_ids)
    kdim = hash_scores.shape[1]
    out = np.zeros((n_r, kdim), dtype=np.float64)
    for r in range(n_r):
        a = int(indptr[r])
        b = int(indptr[r + 1])
        if b <= a:
            out[r] = 1.0 / kdim
            continue
        out[r] = hash_scores[indices[a:b]].mean(axis=0)
    return region_ids, out
