"""Build / load the native k-mer counter shared library."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "kmer_count.cpp"
_HDR = _DIR / "kmer_count.h"
_LIB_NAME = {
    "darwin": "libkmer_count.dylib",
    "win32": "kmer_count.dll",
}.get(sys.platform, "libkmer_count.so")
_LIB_PATH = _DIR / _LIB_NAME

_BASES = ("A", "C", "G", "T")


def library_path() -> Path:
    return _LIB_PATH


def index_to_kmer(index: int, k: int) -> str:
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    chars = ["A"] * k
    x = int(index)
    for pos in range(k - 1, -1, -1):
        chars[pos] = _BASES[x & 3]
        x >>= 2
    return "".join(chars)


def ensure_built(*, force: bool = False) -> Path:
    """Compile the shared library if missing (or force=True)."""
    if _LIB_PATH.is_file() and not force:
        if _SRC.stat().st_mtime <= _LIB_PATH.stat().st_mtime:
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
            "failed to build native k-mer counter:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )
    return _LIB_PATH


class NativeKmerCounter:
    """ctypes wrapper around ``kmer_count_dense`` / ``kmer_count_dimers``."""

    def __init__(self, lib_path: Path | None = None) -> None:
        path = Path(lib_path) if lib_path is not None else ensure_built()
        self._lib = ctypes.CDLL(str(path))
        self._lib.kmer_vocab_size.argtypes = [ctypes.c_int]
        self._lib.kmer_vocab_size.restype = ctypes.c_size_t
        self._lib.kmer_count_dense.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        self._lib.kmer_count_dense.restype = ctypes.c_int
        self._lib.kmer_count_dimers.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._lib.kmer_count_dimers.restype = ctypes.c_int

    def vocab_size(self, k: int) -> int:
        n = int(self._lib.kmer_vocab_size(int(k)))
        if n <= 0:
            raise ValueError(f"unsupported k for native counter: {k}")
        return n

    def count(self, sequence: str, k: int) -> dict[str, int]:
        """Return observed k-mer → abundance (zeros omitted)."""
        k = int(k)
        # Match Python local: drop whitespace; case handled in C++.
        cleaned = "".join(ch for ch in sequence if not ch.isspace())
        raw = cleaned.encode("ascii", errors="ignore")
        if k == 2:
            buf = (ctypes.c_uint32 * 16)()
            rc = self._lib.kmer_count_dimers(raw, len(raw), buf)
            if rc != 0:
                raise RuntimeError(f"kmer_count_dimers failed rc={rc}")
            out: dict[str, int] = {}
            for i in range(16):
                c = int(buf[i])
                if c:
                    out[index_to_kmer(i, 2)] = c
            return out

        n = self.vocab_size(k)
        buf = (ctypes.c_uint32 * n)()
        rc = self._lib.kmer_count_dense(raw, len(raw), k, buf, n)
        if rc != 0:
            raise RuntimeError(f"kmer_count_dense failed rc={rc} for k={k}")
        out = {}
        for i in range(n):
            c = int(buf[i])
            if c:
                out[index_to_kmer(i, k)] = c
        return out


_COUNTER: NativeKmerCounter | None = None


def get_native_counter() -> NativeKmerCounter:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = NativeKmerCounter()
    return _COUNTER


def try_get_native_counter() -> NativeKmerCounter | None:
    try:
        return get_native_counter()
    except (RuntimeError, OSError, FileNotFoundError):
        return None
