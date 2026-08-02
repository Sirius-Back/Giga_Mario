"""Load / pack k-mer feature tables for MLP-VAE (no graph)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.splits.vgae.graph_data import assert_no_homology_features


@dataclass(frozen=True)
class PackedFeatures:
    """Encoder-safe feature pack (compositional / k-mer only)."""

    ids: tuple[str, ...]
    x: np.ndarray  # float32 (n, f)
    feature_names: tuple[str, ...]
    k: int
    meta: dict[str, Any]
    pack_dir: Path

    @property
    def n_nodes(self) -> int:
        return len(self.ids)


def expected_kmer_dim(k: int) -> int:
    return int(4 ** int(k))


def project_kmer_matrix(
    x: np.ndarray,
    *,
    project_dim: int,
    seed: int = 42,
    chunk_rows: int = 8192,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Seeded Gaussian projection of a pure k-mer matrix (no GC column).

    Additive helper for large ``4**k`` panels (e.g. k=7 → 16384 → 2048).
    Supports memmap / ndarray inputs; projects in row chunks.
    """
    d = int(project_dim)
    if d < 2:
        raise ValueError(f"project_dim must be >= 2; got {d}")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D; got {x.shape}")
    if x.shape[1] <= d:
        names = tuple(f"kmer_{i}" for i in range(x.shape[1]))
        return np.asarray(x, dtype=np.float32), names, {"applied": False, "reason": "already_small"}

    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((x.shape[1], d), dtype=np.float64)
    q, _ = np.linalg.qr(raw, mode="reduced")
    q = q[:, :d].astype(np.float32)
    n = int(x.shape[0])
    out = np.empty((n, d), dtype=np.float32)
    step = max(1, int(chunk_rows))
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        block = np.asarray(x[i0:i1], dtype=np.float32)
        out[i0:i1] = block @ q
    names = tuple(f"kmer_proj_{i}" for i in range(d))
    meta = {
        "applied": True,
        "from_dim": int(x.shape[1]),
        "to_dim": int(d),
        "seed": int(seed),
        "kept_gc": False,
        "chunk_rows": step,
    }
    return out, names, meta


def _extract_npz_member_to_npy(npz_path: Path, member: str, dest_npy: Path) -> Path:
    """Extract one array from a compressed npz to a standalone ``.npy`` (mmap-friendly)."""
    import zipfile

    dest_npy = Path(dest_npy)
    dest_npy.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path, "r") as zf:
        # npz stores arrays as <name>.npy
        name = member if member.endswith(".npy") else f"{member}.npy"
        if name not in zf.namelist():
            raise KeyError(f"{member} not in {npz_path}; have {zf.namelist()[:8]}")
        with zf.open(name) as src, dest_npy.open("wb") as dst:
            while True:
                chunk = src.read(16 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return dest_npy


def load_feature_table_mmap(
    path: Path,
    *,
    k: int | None = None,
    scratch_dir: Path | None = None,
) -> tuple[list[str], np.ndarray, list[str], Path | None]:
    """Load ids/names eagerly; return matrix as memmap when possible.

    Returns ``(ids, x, names, scratch_npy_or_None)``. Caller may delete scratch.
    """
    path = Path(path)
    if path.suffix != ".npz":
        ids, x, names = load_feature_table(path, k=k)
        return ids, x, names, None

    scratch_dir = Path(scratch_dir) if scratch_dir is not None else path.parent / ".vae_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=True, mmap_mode=None) as z:
        keys = set(z.files)
        id_key = "ids"
        name_key = "feature_names"
        mat_key = "matrix" if "matrix" in keys else ("x" if "x" in keys else None)
        if mat_key is None:
            raise ValueError(f"unexpected npz keys in {path}: {z.files}")
        ids = [str(x) for x in z[id_key].tolist()]
        names = [str(x) for x in z[name_key].tolist()]

    # Extract matrix to .npy for memmap (avoids holding full decompress twice)
    npy = scratch_dir / f"{path.stem}_{mat_key}.npy"
    if not npy.is_file():
        print(f"[vae] extracting {mat_key} from {path} → {npy} (disk)", flush=True)
        _extract_npz_member_to_npy(path, mat_key, npy)
    x = np.load(npy, mmap_mode="r")
    assert_no_homology_features(names)
    if k is not None and int(x.shape[1]) != expected_kmer_dim(k):
        raise ValueError(
            f"k={k} expects {expected_kmer_dim(k)} features; got shape {x.shape}"
        )
    return ids, x, names, npy


def load_feature_table(
    path: Path,
    *,
    k: int | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load ``feature_table.csv`` (``region|kmer_…``) or ``.npz`` → ids, X, names."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"feature table missing: {path}")

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            if "matrix" in z.files:
                ids = [str(x) for x in z["ids"].tolist()]
                names = [str(x) for x in z["feature_names"].tolist()]
                x = np.asarray(z["matrix"], dtype=np.float32)
            elif "x" in z.files:
                ids = [str(x) for x in z["ids"].tolist()]
                names = [str(x) for x in z["feature_names"].tolist()]
                x = np.asarray(z["x"], dtype=np.float32)
            else:
                raise ValueError(f"unexpected npz keys in {path}: {z.files}")
    else:
        import csv

        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="|")
            header = next(reader)
            if not header or header[0].lower() not in {"region", "id", "ids"}:
                raise ValueError(
                    f"expected first column region/id in {path}; got {header[:3]!r}"
                )
            names = [str(c) for c in header[1:]]
            ids_list: list[str] = []
            rows: list[list[float]] = []
            for row in reader:
                if not row:
                    continue
                ids_list.append(str(row[0]))
                if len(row) - 1 != len(names):
                    raise ValueError(
                        f"row width mismatch for id={row[0]!r}: "
                        f"{len(row) - 1} vs {len(names)} features"
                    )
                rows.append([float(v) for v in row[1:]])
        ids = ids_list
        x = np.asarray(rows, dtype=np.float32)

    assert_no_homology_features(names)
    if k is not None:
        exp = expected_kmer_dim(k)
        if x.shape[1] != exp:
            raise ValueError(
                f"k={k} expects {exp} features; got shape {x.shape} from {path}"
            )
    if x.shape[0] != len(ids):
        raise ValueError(f"n_ids {len(ids)} != matrix rows {x.shape[0]}")
    if len(ids) < 3:
        raise ValueError(f"need >=3 regions; got {len(ids)}")
    return ids, x, names


def pack_feature_table(
    features_path: Path,
    pack_dir: Path,
    *,
    k: int = 4,
    source_label: str | None = None,
    project_dim: int | None = None,
    project_seed: int = 42,
    keep_memmap: bool = False,
) -> PackedFeatures:
    """Convert CSV/NPZ → ``pack_dir`` NPZ + ``feature_meta.json``.

    For large k (e.g. 7), pass ``project_dim=2048`` to match VGAE k7 practice
    and avoid holding a full dense ``4**k`` panel in training RAM.

    ``keep_memmap=True`` (no projection): extract matrix to ``matrix.npy`` and
    mmap it — required for full k=7 16384-d GPU batch training under tight RAM.
    """
    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    meta_path = pack_dir / "feature_meta.json"
    x_path = pack_dir / "node_features.npz"
    ids_path = pack_dir / "ids.txt"
    matrix_npy = pack_dir / "matrix.npy"

    if meta_path.is_file() and ids_path.is_file():
        # Prefer memmap pack if present
        if (meta_path.is_file() and matrix_npy.is_file()) or x_path.is_file():
            return load_packed_features(pack_dir)

    scratch = None
    features_path = Path(features_path)
    proj_meta: dict[str, Any] = {"applied": False}

    if keep_memmap and project_dim is None and features_path.suffix == ".npz":
        ids, x_mm, names, scratch = load_feature_table_mmap(
            features_path, k=k, scratch_dir=pack_dir / ".scratch"
        )
        # Move/copy scratch into stable matrix.npy under pack_dir
        if scratch is not None and Path(scratch).resolve() != matrix_npy.resolve():
            import shutil

            print(f"[vae] placing full matrix memmap at {matrix_npy}", flush=True)
            if matrix_npy.is_file():
                matrix_npy.unlink()
            shutil.move(str(scratch), str(matrix_npy))
            scratch = None
        x = np.load(matrix_npy, mmap_mode="r")
        names_t = tuple(names)
        storage = "memmap"
    elif features_path.suffix == ".npz" and project_dim is not None:
        ids, x_raw, _names_raw, scratch = load_feature_table_mmap(
            features_path, k=k, scratch_dir=pack_dir / ".scratch"
        )
        print(
            f"[vae] projecting k={k} {tuple(x_raw.shape)} → project_dim={project_dim}",
            flush=True,
        )
        x, names_t, proj_meta = project_kmer_matrix(
            x_raw, project_dim=int(project_dim), seed=int(project_seed)
        )
        del x_raw
        storage = "npz"
        np.savez_compressed(
            x_path,
            x=x,
            feature_names=np.asarray(names_t, dtype=object),
            ids=np.asarray(ids, dtype=object),
        )
    else:
        ids, x, names = load_feature_table(features_path, k=k)
        names_t = tuple(names)
        if project_dim is not None and x.shape[1] > int(project_dim):
            x, names_t, proj_meta = project_kmer_matrix(
                x, project_dim=int(project_dim), seed=int(project_seed)
            )
        storage = "npz"
        np.savez_compressed(
            x_path,
            x=x,
            feature_names=np.asarray(names_t, dtype=object),
            ids=np.asarray(ids, dtype=object),
        )

    assert_no_homology_features(names_t)
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    # Always write a small sidecar npz for names/ids even in memmap mode
    if storage == "memmap":
        np.savez_compressed(
            x_path,
            feature_names=np.asarray(names_t, dtype=object),
            ids=np.asarray(ids, dtype=object),
            # tiny placeholder so loaders know matrix is external
            x_shape=np.asarray(x.shape, dtype=np.int64),
        )
    meta: dict[str, Any] = {
        "format": "gigamario_mlp_vae_pack_v1",
        "grain": "region",
        "model": "mlp_vae",
        "k": int(k),
        "n_nodes": int(len(ids)),
        "n_features": int(x.shape[1]),
        "feature_names": list(names_t),
        "homology_in_encoder": False,
        "source_features": str(Path(features_path).resolve()),
        "source_label": source_label,
        "expected_dim": expected_kmer_dim(k),
        "feature_projection": proj_meta,
        "project_dim": project_dim,
        "storage": storage,
        "matrix_npy": "matrix.npy" if storage == "memmap" else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if scratch is not None and Path(scratch).is_file():
        try:
            Path(scratch).unlink()
        except OSError:
            pass
    return PackedFeatures(
        ids=tuple(ids),
        x=x if storage == "memmap" else np.asarray(x, dtype=np.float32),
        feature_names=tuple(names_t),
        k=int(k),
        meta=meta,
        pack_dir=pack_dir,
    )


def load_packed_features(pack_dir: Path) -> PackedFeatures:
    pack_dir = Path(pack_dir)
    meta = json.loads((pack_dir / "feature_meta.json").read_text(encoding="utf-8"))
    if meta.get("homology_in_encoder"):
        raise ValueError("pack claims homology_in_encoder=True — refused")

    if meta.get("storage") == "memmap" or (
        meta.get("matrix_npy") and (pack_dir / str(meta["matrix_npy"])).is_file()
    ):
        npy = pack_dir / str(meta.get("matrix_npy") or "matrix.npy")
        x = np.load(npy, mmap_mode="r")
        with np.load(pack_dir / "node_features.npz", allow_pickle=True) as z:
            names = tuple(str(n) for n in z["feature_names"].tolist())
            if "ids" in z.files:
                ids = tuple(str(i) for i in z["ids"].tolist())
            else:
                ids = tuple(
                    ln.strip()
                    for ln in (pack_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                )
    else:
        with np.load(pack_dir / "node_features.npz", allow_pickle=True) as z:
            if "x" not in z.files:
                raise ValueError(f"node_features.npz missing x in {pack_dir}")
            x = np.asarray(z["x"], dtype=np.float32)
            names = tuple(str(n) for n in z["feature_names"].tolist())
            if "ids" in z.files:
                ids = tuple(str(i) for i in z["ids"].tolist())
            else:
                ids = tuple(
                    ln.strip()
                    for ln in (pack_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                )
    assert_no_homology_features(names)
    return PackedFeatures(
        ids=ids,
        x=x,
        feature_names=names,
        k=int(meta.get("k", 0)),
        meta=meta,
        pack_dir=pack_dir,
    )


def validate_k4_dims(feature_names: Sequence[str], *, k: int = 4) -> None:
    exp = expected_kmer_dim(k)
    if len(feature_names) != exp:
        raise ValueError(f"k={k} requires {exp} features; got {len(feature_names)}")
