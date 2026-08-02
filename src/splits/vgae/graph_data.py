"""Pack pangenome contingency graphs with GC / k-mer node features.

Homology (ortholog/paralog) columns are forbidden in the encoder feature matrix.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.splits.pangenome import load_contingency_graph
from src.splits.sbs.backends.gc import gc_percent
from src.splits.sbs.fna_io import load_fna_directory

FORBIDDEN_FEATURE_PATTERN = re.compile(
    r"(ortho|para|homology|ortholog|paralog|orthogroup|paragroup)",
    re.IGNORECASE,
)

_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass(frozen=True)
class PackedGraph:
    """Encoder-safe graph pack (topology + compositional features only)."""

    ids: tuple[str, ...]
    x: np.ndarray  # float32 (n, f)
    feature_names: tuple[str, ...]
    edge_u: np.ndarray  # int64
    edge_v: np.ndarray  # int64
    edge_w: np.ndarray  # float32 (log1p-normalized)
    edge_w_raw: np.ndarray  # int32 original weights
    k: int
    meta: dict[str, Any]
    pack_dir: Path

    @property
    def n_nodes(self) -> int:
        return len(self.ids)

    @property
    def n_edges(self) -> int:
        return int(len(self.edge_u))


def assert_no_homology_features(feature_names: Sequence[str]) -> None:
    """Raise if any feature name looks like ortholog/paralog leakage."""
    bad = [n for n in feature_names if FORBIDDEN_FEATURE_PATTERN.search(str(n))]
    if bad:
        raise ValueError(
            "homology features forbidden in VGAE encoder inputs; "
            f"offending columns: {bad[:20]}"
        )


def _kmer_index(kmer: str) -> int | None:
    code = 0
    for ch in kmer:
        b = _BASE_INDEX.get(ch)
        if b is None:
            return None
        code = (code << 2) | b
    return code


def dense_kmer_relative(sequence: str, k: int) -> np.ndarray:
    """Relative ACGT k-mer frequencies as a dense ``4**k`` vector."""
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    dim = 4**k
    counts = np.zeros(dim, dtype=np.float64)
    # Prefer native / shared counter (observed dict) then scatter into dense.
    try:
        from src.splits.sbs.backends.kmer import count_kmers

        obs = count_kmers(sequence, k, engine="auto")
        for mer, c in obs.items():
            idx = _kmer_index(mer)
            if idx is not None:
                counts[idx] += float(c)
    except Exception:
        seq = "".join(ch for ch in sequence.upper() if not ch.isspace())
        if len(seq) < k:
            return counts.astype(np.float32)
        for i in range(len(seq) - k + 1):
            idx = _kmer_index(seq[i : i + k])
            if idx is not None:
                counts[idx] += 1.0
    total = float(counts.sum())
    if total > 0.0:
        counts /= total
    return counts.astype(np.float32)


def _feature_names_for_k(k: int) -> tuple[str, ...]:
    names = ["GC_pct"]
    # Lexicographic ACGT order matching base-4 index.
    alphabet = "ACGT"

    def _rec(prefix: str, depth: int) -> None:
        if depth == 0:
            names.append(f"kmer_{prefix}")
            return
        for b in alphabet:
            _rec(prefix + b, depth - 1)

    _rec("", k)
    return tuple(names)


def _features_for_one(seq: str, k: int) -> np.ndarray:
    row = np.empty(1 + 4**int(k), dtype=np.float32)
    row[0] = float(gc_percent(seq))
    row[1:] = dense_kmer_relative(seq, k)
    return row


def build_compositional_features(
    ids: Sequence[str],
    sequences: dict[str, str],
    *,
    k: int,
    n_workers: int = 4,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build ``X = [GC_pct, k-mer freqs…]`` aligned to ``ids`` order."""
    feature_names = _feature_names_for_k(k)
    assert_no_homology_features(feature_names)
    n = len(ids)
    fdim = len(feature_names)
    x = np.zeros((n, fdim), dtype=np.float32)
    missing: list[str] = []
    seq_list: list[str] = []
    for rid in ids:
        seq = sequences.get(rid)
        if seq is None:
            missing.append(str(rid))
            seq_list.append("")
        else:
            seq_list.append(seq)
    if missing:
        raise FileNotFoundError(
            f"missing MARKED sequences for {len(missing)} graph ID(s); "
            f"example={missing[0]!r}"
        )

    workers = max(1, min(int(n_workers), 8))
    if n < 2000 or workers == 1:
        for i, seq in enumerate(seq_list):
            x[i] = _features_for_one(seq, k)
        return x, feature_names

    from concurrent.futures import ProcessPoolExecutor

    chunk = max(1, n // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Map in chunks to reduce IPC overhead
        futures = []
        ranges: list[tuple[int, int]] = []
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            ranges.append((start, end))
            futures.append(
                pool.submit(
                    _features_chunk,
                    seq_list[start:end],
                    int(k),
                )
            )
        for (start, end), fut in zip(ranges, futures):
            x[start:end] = fut.result()
    return x, feature_names


def build_compositional_features_projected(
    ids: Sequence[str],
    sequences: dict[str, str],
    *,
    k: int,
    project_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Build GC + projected k-mer features **without** a full ``(n, 4**k)`` matrix.

    Uses sparse observed k-mers (``count_kmers``) and accumulates
    ``R[idx]`` rows — O(#observed kmers · project_dim) per sequence, not
    O(4**k · project_dim).
    """
    d = int(project_dim)
    if d < 2:
        raise ValueError(f"project_dim must be >= 2; got {d}")
    k = int(k)
    f_full = 1 + 4**k
    if f_full <= d:
        x, names = build_compositional_features(ids, sequences, k=k)
        return x, names, {"applied": False, "reason": "already_small"}

    rng = np.random.default_rng(int(seed))
    out_rest = d - 1
    raw = rng.standard_normal((4**k, out_rest), dtype=np.float64)
    q, _ = np.linalg.qr(raw, mode="reduced")
    rmat = q[:, :out_rest].astype(np.float32)  # (4**k, out_rest)

    # Prefer shared counter (dict of observed kmers only)
    try:
        from src.splits.sbs.backends.kmer import count_kmers

        def _obs(seq: str) -> dict[str, float]:
            return count_kmers(seq, k, engine="auto")

    except Exception:

        def _obs(seq: str) -> dict[str, float]:
            counts: dict[str, float] = {}
            s = "".join(ch for ch in seq.upper() if not ch.isspace())
            if len(s) < k:
                return counts
            for i in range(len(s) - k + 1):
                mer = s[i : i + k]
                if all(c in _BASE_INDEX for c in mer):
                    counts[mer] = counts.get(mer, 0.0) + 1.0
            return counts

    n = len(ids)
    x = np.zeros((n, d), dtype=np.float32)
    missing: list[str] = []
    report_every = max(1, n // 20)
    for i, rid in enumerate(ids):
        seq = sequences.get(str(rid))
        if seq is None:
            missing.append(str(rid))
            continue
        x[i, 0] = float(gc_percent(seq))
        obs = _obs(seq)
        total = float(sum(obs.values()))
        if total > 0.0:
            acc = np.zeros(out_rest, dtype=np.float64)
            for mer, c in obs.items():
                idx = _kmer_index(mer)
                if idx is None:
                    continue
                acc += (float(c) / total) * rmat[idx]
            x[i, 1:] = acc.astype(np.float32)
        if (i + 1) % report_every == 0 or (i + 1) == n:
            print(
                f"[vgae-pack] projected features {i + 1}/{n} "
                f"({100.0 * (i + 1) / n:.0f}%)",
                flush=True,
            )
    if missing:
        raise FileNotFoundError(
            f"missing MARKED sequences for {len(missing)} graph ID(s); "
            f"example={missing[0]!r}"
        )
    names = ("GC_pct",) + tuple(f"kmer_proj_{j}" for j in range(out_rest))
    meta = {
        "applied": True,
        "from_dim": f_full,
        "to_dim": d,
        "seed": int(seed),
        "kept_gc": True,
        "streamed": True,
        "sparse_project": True,
    }
    return x, names, meta


def project_features_fixed(
    x: np.ndarray,
    feature_names: Sequence[str],
    *,
    project_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Seeded Gaussian projection ``X @ R`` (additive; leaves dense pack optional).

    Keeps column 0 (``GC_pct``) unprojected and projects the remaining k-mer
    block into ``project_dim - 1`` dims when ``project_dim >= 2``.
    """
    d = int(project_dim)
    if d < 2:
        raise ValueError(f"project_dim must be >= 2; got {d}")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D; got {x.shape}")
    if x.shape[1] <= d:
        return x, tuple(feature_names), {"applied": False, "reason": "already_small"}
    rng = np.random.default_rng(int(seed))
    # Preserve GC; project k-mer block
    gc = x[:, :1]
    rest = x[:, 1:]
    out_rest_dim = d - 1
    # Orthogonal-ish via QR on tall random matrix
    raw = rng.standard_normal((rest.shape[1], out_rest_dim), dtype=np.float64)
    q, _ = np.linalg.qr(raw, mode="reduced")
    proj = (rest.astype(np.float64) @ q[:, :out_rest_dim]).astype(np.float32)
    x2 = np.concatenate([gc, proj], axis=1)
    names = ("GC_pct",) + tuple(f"kmer_proj_{i}" for i in range(out_rest_dim))
    meta = {
        "applied": True,
        "from_dim": int(x.shape[1]),
        "to_dim": int(d),
        "seed": int(seed),
        "kept_gc": True,
    }
    return x2, names, meta


def build_multik_projected_features(
    ids: Sequence[str],
    sequences: dict[str, str],
    *,
    ks: Sequence[int] = (4, 5, 7),
    per_k_project_dim: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """GC once + projected k-mer spectra for each ``k`` in ``ks`` (concat).

    No homology columns. Each spectrum is projected to ``per_k_project_dim``
    (including a temporary GC that is dropped after the first block).
    """
    ks_t = tuple(int(k) for k in ks)
    if not ks_t or any(k < 1 for k in ks_t):
        raise ValueError(f"ks must be non-empty positive ints; got {ks!r}")
    d = int(per_k_project_dim)
    if d < 2:
        raise ValueError(f"per_k_project_dim must be >= 2; got {d}")

    blocks: list[np.ndarray] = []
    names: list[str] = ["GC_pct"]
    per_k_meta: list[dict[str, Any]] = []
    gc_col: np.ndarray | None = None
    for k in ks_t:
        xk, _nk, meta_k = build_compositional_features_projected(
            ids,
            sequences,
            k=k,
            project_dim=d,
            seed=int(seed) + int(k),
        )
        if gc_col is None:
            gc_col = xk[:, :1].astype(np.float32, copy=True)
        blocks.append(xk[:, 1:].astype(np.float32, copy=False))
        names.extend(f"k{k}_proj_{j}" for j in range(xk.shape[1] - 1))
        per_k_meta.append({"k": int(k), **meta_k})
    assert gc_col is not None
    x = np.concatenate([gc_col, *blocks], axis=1)
    feature_names = tuple(names)
    assert_no_homology_features(feature_names)
    meta = {
        "applied": True,
        "kind": "multik_projected",
        "ks": list(ks_t),
        "per_k_project_dim": d,
        "seed": int(seed),
        "n_features": int(x.shape[1]),
        "per_k": per_k_meta,
    }
    return x, feature_names, meta


def _union_find_components(
    n: int, edge_u: np.ndarray, edge_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(root[i], cc_size[root])`` via union–find (path compression)."""
    parent = np.arange(int(n), dtype=np.int64)
    rank = np.zeros(int(n), dtype=np.int8)
    size = np.ones(int(n), dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for u, v in zip(edge_u.tolist(), edge_v.tolist()):
        u_i, v_i = int(u), int(v)
        if u_i < 0 or v_i < 0 or u_i >= n or v_i >= n or u_i == v_i:
            continue
        ru, rv = find(u_i), find(v_i)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        if rank[ru] == rank[rv]:
            rank[ru] += 1
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = find(i)
    return roots, size


def build_structural_features(
    n_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    *,
    n_cc_hash: int = 8,
    seed: int = 42,
    max_clust_degree: int = 512,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Topology-only node features: degree, CC hash, local clustering.

    No sequence or homology labels — safe under the encoder firewall.
    Nodes with degree ``> max_clust_degree`` get clustering=0 (cost guard).
    """
    n = int(n_nodes)
    if n < 1:
        raise ValueError(f"n_nodes must be >= 1; got {n}")
    u = np.asarray(edge_u, dtype=np.int64)
    v = np.asarray(edge_v, dtype=np.int64)
    deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64)
    deg_max = float(deg.max()) if n else 0.0
    deg_norm = (deg / deg_max).astype(np.float32) if deg_max > 0 else deg.astype(np.float32)
    log_deg = np.log1p(deg).astype(np.float32)

    roots, cc_size_by_root = _union_find_components(n, u, v)
    cc_size = cc_size_by_root[roots].astype(np.float64)
    log_cc = np.log1p(cc_size).astype(np.float32)

    # Deterministic fractional hash of component root → sin/cos positional codes
    rng = np.random.default_rng(int(seed))
    # Map root id → U[0,1) via mixed multiplicative hash (seeded salt)
    salt = int(rng.integers(1, 2**31 - 1))
    frac = ((roots.astype(np.uint64) * np.uint64(salt)) % np.uint64(1_000_003)).astype(
        np.float64
    ) / 1_000_003.0
    n_hash = max(2, int(n_cc_hash))
    if n_hash % 2 == 1:
        n_hash += 1
    hash_feats = np.empty((n, n_hash), dtype=np.float32)
    for j in range(n_hash // 2):
        ang = 2.0 * np.pi * (j + 1) * frac
        hash_feats[:, 2 * j] = np.sin(ang).astype(np.float32)
        hash_feats[:, 2 * j + 1] = np.cos(ang).astype(np.float32)

    # Adjacency sets for clustering (undirected)
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        ai, bi = int(a), int(b)
        if ai == bi or ai < 0 or bi < 0 or ai >= n or bi >= n:
            continue
        nbrs[ai].add(bi)
        nbrs[bi].add(ai)

    clustering = np.zeros(n, dtype=np.float32)
    max_d = int(max_clust_degree)
    for i in range(n):
        d = len(nbrs[i])
        if d < 2 or d > max_d:
            continue
        neigh = nbrs[i]
        # Count triangles: edges among neighbors
        tri = 0
        for a in neigh:
            # only count a < b via iterating partners > a in set intersection size
            tri += sum(1 for b in nbrs[a] if b in neigh and b > a)
        clustering[i] = (2.0 * tri) / (d * (d - 1))

    parts = [deg_norm.reshape(-1, 1), log_deg.reshape(-1, 1), log_cc.reshape(-1, 1), hash_feats, clustering.reshape(-1, 1)]
    x = np.concatenate(parts, axis=1).astype(np.float32)
    names = (
        ("struct_degree_norm", "struct_log_degree", "struct_cc_log_size")
        + tuple(f"struct_cc_hash_{j}" for j in range(n_hash))
        + ("struct_clustering",)
    )
    assert_no_homology_features(names)
    meta = {
        "kind": "structural_topology",
        "n_cc_hash": n_hash,
        "seed": int(seed),
        "max_clust_degree": max_d,
        "n_features": int(x.shape[1]),
        "homology_leakage": False,
    }
    return x, names, meta


def append_structural_features_to_pack(
    pack_dir: Path,
    out_pack_dir: Path | None = None,
    *,
    n_cc_hash: int = 8,
    seed: int = 42,
) -> PackedGraph:
    """Copy / rewrite a pack with topology features concatenated (no re-MARKED)."""
    pack = load_packed_graph(Path(pack_dir))
    out = Path(out_pack_dir) if out_pack_dir is not None else Path(pack_dir)
    out.mkdir(parents=True, exist_ok=True)
    sx, snames, smeta = build_structural_features(
        pack.n_nodes,
        pack.edge_u,
        pack.edge_v,
        n_cc_hash=int(n_cc_hash),
        seed=int(seed),
    )
    # Drop any prior struct_* columns if re-appending
    keep_idx = [i for i, n in enumerate(pack.feature_names) if not str(n).startswith("struct_")]
    x0 = pack.x[:, keep_idx]
    names0 = tuple(pack.feature_names[i] for i in keep_idx)
    x = np.concatenate([x0, sx], axis=1)
    feature_names = names0 + snames
    assert_no_homology_features(feature_names)

    np.savez_compressed(
        out / "node_features.npz",
        x=x,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    if out.resolve() != Path(pack_dir).resolve():
        import shutil

        for name in ("edges_weighted.npz", "ids.txt"):
            src = Path(pack_dir) / name
            if src.is_file():
                shutil.copy2(src, out / name)
    meta = dict(pack.meta)
    meta["n_features"] = int(x.shape[1])
    meta["feature_names"] = list(feature_names)
    meta["structural_features"] = smeta
    meta["homology_in_encoder"] = False
    (out / "feature_meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return PackedGraph(
        ids=pack.ids,
        x=x,
        feature_names=feature_names,
        edge_u=pack.edge_u,
        edge_v=pack.edge_v,
        edge_w=pack.edge_w,
        edge_w_raw=pack.edge_w_raw,
        k=pack.k,
        meta=meta,
        pack_dir=out,
    )


def _features_chunk(seqs: list[str], k: int) -> np.ndarray:
    out = np.zeros((len(seqs), 1 + 4**int(k)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        out[i] = _features_for_one(seq, k)
    return out


def _normalize_edge_weights(raw: np.ndarray) -> np.ndarray:
    w = np.log1p(np.asarray(raw, dtype=np.float64))
    mx = float(w.max()) if w.size else 0.0
    if mx > 0.0:
        w = w / mx
    return w.astype(np.float32)


def pack_region_graph(
    graph_dir: Path,
    marked_dir: Path,
    pack_dir: Path,
    *,
    k: int | None = None,
    feature_k: int | None = None,
    feature_ks: Sequence[int] | None = None,
    per_k_project_dim: int = 256,
    project_dim: int | None = None,
    project_seed: int = 42,
    add_structural_features: bool = False,
    n_cc_hash: int = 8,
    max_ids: int | None = None,
    intersect_allow: bool = False,
) -> PackedGraph:
    """Load contingency graph, attach GC/k-mer features, persist under ``pack_dir``.

    ``k`` — graph k (from meta when omitted). ``feature_k`` — compositional
    spectrum k (defaults to ``k``). ``feature_ks`` — optional multi-k concat
    with light per-k projection (``per_k_project_dim``). ``project_dim`` —
    optional seeded projection of a single-k block. ``add_structural_features``
    appends topology-only (degree / CC hash / clustering) columns.
    """
    graph_dir = Path(graph_dir)
    marked_dir = Path(marked_dir)
    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_contingency_graph(graph_dir)
    ids = [str(x) for x in loaded["ids"]]
    meta_src = dict(loaded.get("meta") or {})
    k_use = int(k if k is not None else meta_src.get("k") or 5)
    if k_use < 1:
        raise ValueError(f"invalid k={k_use}")
    feat_k = int(feature_k) if feature_k is not None else k_use
    if feat_k < 1:
        raise ValueError(f"invalid feature_k={feat_k}")
    ks_use = tuple(int(x) for x in feature_ks) if feature_ks is not None else None
    if ks_use is not None and (not ks_use or any(x < 1 for x in ks_use)):
        raise ValueError(f"invalid feature_ks={feature_ks!r}")

    if max_ids is not None:
        max_ids = int(max_ids)
        if max_ids < 3:
            raise ValueError("max_ids must be >= 3")
        ids = ids[:max_ids]
        keep = set(range(len(ids)))
        edge_u_all = np.asarray(loaded["edge_u"], dtype=np.int64)
        edge_v_all = np.asarray(loaded["edge_v"], dtype=np.int64)
        edge_w_all = np.asarray(loaded["edge_w"], dtype=np.int32)
        mask = np.isin(edge_u_all, list(keep)) & np.isin(edge_v_all, list(keep))
        edge_u = edge_u_all[mask]
        edge_v = edge_v_all[mask]
        edge_w_raw = edge_w_all[mask]
    else:
        edge_u = np.asarray(loaded["edge_u"], dtype=np.int64)
        edge_v = np.asarray(loaded["edge_v"], dtype=np.int64)
        edge_w_raw = np.asarray(loaded["edge_w"], dtype=np.int32)

    if not marked_dir.is_dir():
        raise FileNotFoundError(f"MARKED directory missing: {marked_dir}")

    print(
        f"[vgae-pack] loading MARKED sequences n_ids={len(ids)} "
        f"from {marked_dir} (feature_k={feat_k}, feature_ks={ks_use}, "
        f"project_dim={project_dim})",
        flush=True,
    )
    try:
        sequences = load_fna_directory(marked_dir, ids=ids)
    except FileNotFoundError:
        if not intersect_allow:
            raise
        # Keep only IDs present on disk.
        available = {
            p.stem
            for p in marked_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fa", ".fna", ".fasta", ".fas", ".ext"}
        }
        keep_ids = [i for i in ids if i in available]
        if len(keep_ids) < 3:
            raise FileNotFoundError(
                f"after intersect_allow, only {len(keep_ids)} IDs remain under {marked_dir}"
            )
        id_to_new = {old: j for j, old in enumerate(keep_ids)}
        old_index = {rid: i for i, rid in enumerate(ids)}
        remap_u: list[int] = []
        remap_v: list[int] = []
        remap_w: list[int] = []
        for u, v, w in zip(edge_u.tolist(), edge_v.tolist(), edge_w_raw.tolist()):
            ru = ids[int(u)] if 0 <= int(u) < len(ids) else None
            rv = ids[int(v)] if 0 <= int(v) < len(ids) else None
            if ru in id_to_new and rv in id_to_new:
                remap_u.append(id_to_new[ru])
                remap_v.append(id_to_new[rv])
                remap_w.append(int(w))
        ids = keep_ids
        edge_u = np.asarray(remap_u, dtype=np.int64)
        edge_v = np.asarray(remap_v, dtype=np.int64)
        edge_w_raw = np.asarray(remap_w, dtype=np.int32)
        sequences = load_fna_directory(marked_dir, ids=ids)
        _ = old_index  # kept for clarity / future diagnostics

    print(
        f"[vgae-pack] loaded {len(sequences)} sequences; building features…",
        flush=True,
    )
    x: np.ndarray
    feature_names: tuple[str, ...]
    proj_meta: dict[str, Any]
    if ks_use is not None:
        x, feature_names, proj_meta = build_multik_projected_features(
            ids,
            sequences,
            ks=ks_use,
            per_k_project_dim=int(per_k_project_dim),
            seed=int(project_seed),
        )
    elif project_dim is not None and (1 + 4**feat_k) > int(project_dim):
        x, feature_names, proj_meta = build_compositional_features_projected(
            ids,
            sequences,
            k=feat_k,
            project_dim=int(project_dim),
            seed=int(project_seed),
        )
    else:
        x, feature_names = build_compositional_features(ids, sequences, k=feat_k)
        proj_meta = {"applied": False}
        if project_dim is not None:
            x, feature_names, proj_meta = project_features_fixed(
                x, feature_names, project_dim=int(project_dim), seed=int(project_seed)
            )

    struct_meta: dict[str, Any] | None = None
    if add_structural_features:
        sx, snames, struct_meta = build_structural_features(
            len(ids), edge_u, edge_v, n_cc_hash=int(n_cc_hash), seed=int(project_seed)
        )
        x = np.concatenate([x, sx], axis=1)
        feature_names = tuple(feature_names) + snames

    assert_no_homology_features(feature_names)
    edge_w = _normalize_edge_weights(edge_w_raw)
    print(
        f"[vgae-pack] features ready X={x.shape} edges={len(edge_u)} "
        f"projection={proj_meta.get('applied')} structural={bool(struct_meta)}",
        flush=True,
    )

    # Persist pack artifacts
    np.savez_compressed(
        pack_dir / "node_features.npz",
        x=x,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    np.savez_compressed(
        pack_dir / "edges_weighted.npz",
        edge_u=edge_u.astype(np.int32),
        edge_v=edge_v.astype(np.int32),
        edge_w=edge_w,
        edge_w_raw=edge_w_raw.astype(np.int32),
    )
    (pack_dir / "ids.txt").write_text(
        "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
    )
    meta = {
        "format": "gigamario_vgae_pack_v1",
        "grain": "region",
        "k": k_use,
        "feature_k": feat_k if ks_use is None else None,
        "feature_ks": list(ks_use) if ks_use is not None else None,
        "n_nodes": len(ids),
        "n_edges": int(len(edge_u)),
        "n_features": int(x.shape[1]),
        "feature_names": list(feature_names),
        "homology_in_encoder": False,
        "graph_dir": str(Path(graph_dir).resolve()),
        "marked_dir": str(marked_dir.resolve()),
        "source_meta": meta_src,
        "max_ids": max_ids,
        "edge_weight_transform": "log1p_then_maxnorm",
        "feature_projection": proj_meta,
        "structural_features": struct_meta,
    }
    (pack_dir / "feature_meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )

    return PackedGraph(
        ids=tuple(ids),
        x=x,
        feature_names=feature_names,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        edge_w_raw=edge_w_raw,
        k=k_use,
        meta=meta,
        pack_dir=pack_dir,
    )


def load_packed_graph(pack_dir: Path) -> PackedGraph:
    """Reload a pack written by :func:`pack_region_graph` (or hash pack)."""
    pack_dir = Path(pack_dir)
    meta_path = pack_dir / "feature_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing feature_meta.json under {pack_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = [
        ln.strip()
        for ln in (pack_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    with np.load(pack_dir / "node_features.npz", allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        feature_names = tuple(str(n) for n in data["feature_names"].tolist())
    assert_no_homology_features(feature_names)
    with np.load(pack_dir / "edges_weighted.npz") as data:
        edge_u = np.asarray(data["edge_u"], dtype=np.int64)
        edge_v = np.asarray(data["edge_v"], dtype=np.int64)
        edge_w = np.asarray(data["edge_w"], dtype=np.float32)
        edge_w_raw = np.asarray(data["edge_w_raw"], dtype=np.int32)
    if len(ids) != x.shape[0]:
        raise ValueError(f"ids ({len(ids)}) != X rows ({x.shape[0]})")
    return PackedGraph(
        ids=tuple(ids),
        x=x,
        feature_names=feature_names,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        edge_w_raw=edge_w_raw,
        k=int(meta.get("k") or 5),
        meta=meta,
        pack_dir=pack_dir,
    )
