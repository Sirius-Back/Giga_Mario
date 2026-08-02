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
) -> PackedFeatures:
    """Convert CSV/NPZ → ``pack_dir`` NPZ + ``feature_meta.json``."""
    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    meta_path = pack_dir / "feature_meta.json"
    x_path = pack_dir / "node_features.npz"
    ids_path = pack_dir / "ids.txt"

    if meta_path.is_file() and x_path.is_file() and ids_path.is_file():
        return load_packed_features(pack_dir)

    ids, x, names = load_feature_table(features_path, k=k)
    assert_no_homology_features(names)
    np.savez_compressed(
        x_path,
        x=x,
        feature_names=np.asarray(names, dtype=object),
        ids=np.asarray(ids, dtype=object),
    )
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    meta: dict[str, Any] = {
        "format": "gigamario_mlp_vae_pack_v1",
        "grain": "region",
        "model": "mlp_vae",
        "k": int(k),
        "n_nodes": int(len(ids)),
        "n_features": int(x.shape[1]),
        "feature_names": list(names),
        "homology_in_encoder": False,
        "source_features": str(Path(features_path).resolve()),
        "source_label": source_label,
        "expected_dim": expected_kmer_dim(k),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return PackedFeatures(
        ids=tuple(ids),
        x=x,
        feature_names=tuple(names),
        k=int(k),
        meta=meta,
        pack_dir=pack_dir,
    )


def load_packed_features(pack_dir: Path) -> PackedFeatures:
    pack_dir = Path(pack_dir)
    meta = json.loads((pack_dir / "feature_meta.json").read_text(encoding="utf-8"))
    with np.load(pack_dir / "node_features.npz", allow_pickle=True) as z:
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
    if meta.get("homology_in_encoder"):
        raise ValueError("pack claims homology_in_encoder=True — refused")
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
