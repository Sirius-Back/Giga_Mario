"""Build / load the native pangenome contingency (repeat-graph) library."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "repeat_graph.cpp"
_HDR = _DIR / "repeat_graph.h"
_LIB_NAME = {
    "darwin": "libpangenome_repeat_graph.dylib",
    "win32": "pangenome_repeat_graph.dll",
}.get(sys.platform, "libpangenome_repeat_graph.so")
_LIB_PATH = _DIR / _LIB_NAME

METHOD_HASH_MAJORITY = "hash_majority"
METHOD_REGION_CONTINGENCY = "region_contingency"


def library_path() -> Path:
    return _LIB_PATH


def ensure_built(*, force: bool = False) -> Path:
    """Compile the shared library if missing (or force=True)."""
    if _LIB_PATH.is_file() and not force:
        if (
            _SRC.stat().st_mtime <= _LIB_PATH.stat().st_mtime
            and _HDR.is_file()
            and _HDR.stat().st_mtime <= _LIB_PATH.stat().st_mtime
        ):
            return _LIB_PATH
    if not _SRC.is_file():
        raise FileNotFoundError(f"native source missing: {_SRC}")
    cxx = os.environ.get("CXX", "g++")
    cmd = [
        cxx,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        str(_SRC),
        "-o",
        str(_LIB_PATH),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not _LIB_PATH.is_file():
        raise RuntimeError(
            "failed to build native pangenome repeat graph:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )
    return _LIB_PATH


@dataclass(frozen=True)
class ContingencyGraphResult:
    """Region fold labels + optional region–region edges for visualization."""

    cluster_ids: np.ndarray  # int32, shape (n,) — per-sequence fold labels
    n_clusters: int
    edge_u: np.ndarray  # int32
    edge_v: np.ndarray  # int32
    edge_w: np.ndarray  # int32 shared-kmer / shared-repeat weights
    method: str = METHOD_HASH_MAJORITY


class NativePangenomeGraph:
    """ctypes wrapper around native pangenome clustering."""

    def __init__(self, lib_path: Path | None = None) -> None:
        path = Path(lib_path) if lib_path is not None else ensure_built()
        self._lib = ctypes.CDLL(str(path))
        self._bind_fn("pangenome_contingency_clusters")
        self._bind_fn("pangenome_hash_majority_clusters")

    def _bind_fn(self, name: str) -> None:
        fn = getattr(self._lib, name)
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        fn.restype = ctypes.c_int

    def _prepare_blob(
        self, sequences: list[str]
    ) -> tuple[bytes, np.ndarray, int]:
        if not sequences:
            raise ValueError("sequences must be non-empty")
        n = len(sequences)
        cleaned = [("".join(ch for ch in s if not ch.isspace())) for s in sequences]
        parts = [s.encode("ascii", errors="ignore") for s in cleaned]
        blob = b"".join(parts)
        offsets = np.zeros(n + 1, dtype=np.int64)
        cursor = 0
        for i, p in enumerate(parts):
            offsets[i] = cursor
            cursor += len(p)
        offsets[n] = cursor
        return blob, offsets, n

    def _call(
        self,
        fn_name: str,
        sequences: list[str],
        *,
        k: int,
        param: int,
        max_edges: int,
        collect_edges: bool,
        method: str,
    ) -> ContingencyGraphResult:
        k = int(k)
        if k < 1 or k > 32:
            raise ValueError(f"k must be in 1..32; got {k}")
        blob, offsets, n = self._prepare_blob(sequences)

        cluster_out = np.zeros(n, dtype=np.int32)
        n_clusters = ctypes.c_int32(0)
        n_edges = ctypes.c_int32(0)

        if collect_edges and max_edges > 0:
            edge_u = np.zeros(max_edges, dtype=np.int32)
            edge_v = np.zeros(max_edges, dtype=np.int32)
            edge_w = np.zeros(max_edges, dtype=np.int32)
            eu = edge_u.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ev = edge_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ew = edge_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            max_e = int(max_edges)
        else:
            edge_u = np.zeros(0, dtype=np.int32)
            edge_v = np.zeros(0, dtype=np.int32)
            edge_w = np.zeros(0, dtype=np.int32)
            eu = ctypes.POINTER(ctypes.c_int32)()
            ev = ctypes.POINTER(ctypes.c_int32)()
            ew = ctypes.POINTER(ctypes.c_int32)()
            max_e = 0

        fn = getattr(self._lib, fn_name)
        rc = fn(
            blob,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int32(n),
            ctypes.c_int(k),
            ctypes.c_int32(int(param)),
            cluster_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(n_clusters),
            eu,
            ev,
            ew,
            ctypes.c_int32(max_e),
            ctypes.byref(n_edges),
        )
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed rc={rc}")
        ne = int(n_edges.value)
        return ContingencyGraphResult(
            cluster_ids=cluster_out.copy(),
            n_clusters=int(n_clusters.value),
            edge_u=edge_u[:ne].copy() if ne else edge_u[:0],
            edge_v=edge_v[:ne].copy() if ne else edge_v[:0],
            edge_w=edge_w[:ne].copy() if ne else edge_w[:0],
            method=method,
        )

    def hash_majority_clusters(
        self,
        sequences: list[str],
        *,
        k: int = 21,
        min_df: int = 2,
        max_edges: int = 100_000,
        collect_edges: bool = True,
    ) -> ContingencyGraphResult:
        """Repeat-hash UF + per-sequence majority fold labels."""
        return self._call(
            "pangenome_hash_majority_clusters",
            sequences,
            k=k,
            param=int(min_df),
            max_edges=max_edges,
            collect_edges=collect_edges,
            method=METHOD_HASH_MAJORITY,
        )

    def contingency_clusters(
        self,
        sequences: list[str],
        *,
        k: int = 21,
        min_shared: int = 1,
        max_edges: int = 100_000,
        collect_edges: bool = True,
        method: str = METHOD_HASH_MAJORITY,
        min_df: int = 2,
    ) -> ContingencyGraphResult:
        """Default: hash-majority. Pass method=region_contingency for legacy UF."""
        method_l = str(method).lower().strip()
        if method_l in {
            METHOD_HASH_MAJORITY,
            "hash_majority_cc",
            "majority",
            "default",
        }:
            return self.hash_majority_clusters(
                sequences,
                k=k,
                min_df=min_df,
                max_edges=max_edges,
                collect_edges=collect_edges,
            )
        if method_l in {
            METHOD_REGION_CONTINGENCY,
            "region",
            "legacy",
            "contingency_cc",
        }:
            return self._call(
                "pangenome_contingency_clusters",
                sequences,
                k=k,
                param=int(min_shared),
                max_edges=max_edges,
                collect_edges=collect_edges,
                method=METHOD_REGION_CONTINGENCY,
            )
        raise ValueError(
            f"unknown pangenome clustering method={method!r}; "
            f"expected {METHOD_HASH_MAJORITY!r} or {METHOD_REGION_CONTINGENCY!r}"
        )


_GRAPH: NativePangenomeGraph | None = None


def get_native_graph() -> NativePangenomeGraph:
    global _GRAPH
    if _GRAPH is None:
        # Force rebuild when sources newer than .so (ensure_built checks mtimes).
        ensure_built(force=False)
        _GRAPH = NativePangenomeGraph()
    return _GRAPH


def try_get_native_graph() -> NativePangenomeGraph | None:
    try:
        return get_native_graph()
    except (RuntimeError, OSError, FileNotFoundError):
        return None


def reset_native_graph() -> None:
    """Drop cached ctypes handle (after rebuild)."""
    global _GRAPH
    _GRAPH = None
