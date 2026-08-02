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
    max_ids: int | None = None,
    intersect_allow: bool = False,
) -> PackedGraph:
    """Load contingency graph, attach GC/k-mer features, persist under ``pack_dir``."""
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

    x, feature_names = build_compositional_features(ids, sequences, k=k_use)
    assert_no_homology_features(feature_names)
    edge_w = _normalize_edge_weights(edge_w_raw)

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
