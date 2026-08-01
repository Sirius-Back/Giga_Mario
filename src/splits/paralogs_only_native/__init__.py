"""Build / load the native paralogs_only orthogroup-assignment library."""
from __future__ import annotations

import ctypes
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "paralogs_only.cpp"
_HDR = _DIR / "paralogs_only.h"
_LIB_NAME = {
    "darwin": "libparalogs_only.dylib",
    "win32": "paralogs_only.dll",
}.get(sys.platform, "libparalogs_only.so")
_LIB_PATH = _DIR / _LIB_NAME


def library_path() -> Path:
    return _LIB_PATH


def ensure_built(*, force: bool = False) -> Path:
    """Compile the shared library if missing (or force=True)."""
    if _LIB_PATH.is_file() and not force:
        if _SRC.stat().st_mtime <= _LIB_PATH.stat().st_mtime and _HDR.is_file():
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
            "failed to build native paralogs_only library:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )
    return _LIB_PATH


def _maybe_gunzip(path: Path, tmpdir: Path) -> Path:
    """Return a plain-TSV path; decompress ``.gz`` into tmpdir when needed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix != ".gz" and not str(path).endswith(".tsv.gz"):
        # also handle .tsv.gz where suffixes are ['.gz'] only — Path.suffix is .gz
        pass
    name = path.name
    if name.endswith(".gz"):
        out = tmpdir / name[:-3]
        with gzip.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        return out
    return path


class NativeParalogsOnly:
    """ctypes wrapper around ``paralogs_only_assign``."""

    def __init__(self, lib_path: Path | None = None) -> None:
        path = Path(lib_path) if lib_path is not None else ensure_built()
        self._lib = ctypes.CDLL(str(path))
        self._lib.paralogs_only_assign.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._lib.paralogs_only_assign.restype = ctypes.c_int

    def assign(
        self,
        *,
        edges_path: Path,
        nodes_path: Path,
        panel_ids: list[str],
        seed: int,
        out_assignment_path: Path,
        out_meta_json_path: Path | None = None,
    ) -> Path:
        if not panel_ids:
            raise ValueError("panel_ids must be non-empty")
        out_assignment_path = Path(out_assignment_path)
        out_assignment_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="paralogs_only_") as td:
            tmp = Path(td)
            edges = _maybe_gunzip(Path(edges_path), tmp)
            nodes = _maybe_gunzip(Path(nodes_path), tmp)
            ids_path = tmp / "panel_ids.txt"
            # preserve order, unique
            seen: set[str] = set()
            ordered: list[str] = []
            for i in panel_ids:
                if i in seen:
                    continue
                seen.add(i)
                ordered.append(i)
            ids_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")

            meta_arg = b""
            meta_path = out_meta_json_path
            if meta_path is not None:
                meta_path = Path(meta_path)
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_arg = str(meta_path).encode("utf-8")

            # Write assignment into tmp then copy — C++ needs writable path
            tmp_out = tmp / "assignment.tsv"
            rc = self._lib.paralogs_only_assign(
                str(edges).encode("utf-8"),
                str(nodes).encode("utf-8"),
                str(ids_path).encode("utf-8"),
                ctypes.c_uint64(int(seed)),
                str(tmp_out).encode("utf-8"),
                meta_arg if meta_arg else None,
            )
            if rc != 0:
                raise RuntimeError(f"paralogs_only_assign failed rc={rc}")
            if not tmp_out.is_file():
                raise RuntimeError("paralogs_only_assign did not write assignment")
            shutil.copyfile(tmp_out, out_assignment_path)
        return out_assignment_path


_NATIVE: NativeParalogsOnly | None = None


def get_native() -> NativeParalogsOnly:
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = NativeParalogsOnly()
    return _NATIVE


def try_get_native() -> NativeParalogsOnly | None:
    try:
        return get_native()
    except (RuntimeError, OSError, FileNotFoundError):
        return None
