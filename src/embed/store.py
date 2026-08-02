"""Persist embeddings as memmap-friendly ``.npy`` + manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.embed import LAYER_DIMS, ROLE_NAMES, ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from src.embed.discover import LegNetRun
from src.pipeline.mem_guard import ensure_allocation_fits


@dataclass
class EmbedStore:
    out_dir: Path
    ids: np.ndarray  # object/str
    roles: np.ndarray  # int8
    layers: dict[str, np.ndarray]  # float32 [N, D]

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])


def run_out_dir(base: Path, run: LegNetRun) -> Path:
    if run.fold is None:
        return Path(base) / run.run_name
    return Path(base) / run.run_name / f"fold{run.fold}"


def role_code(name: str) -> int:
    m = {"train": ROLE_TRAIN, "test": ROLE_TEST, "val": ROLE_VAL}
    if name not in m:
        raise ValueError(f"unknown role {name!r}")
    return m[name]


def allocate_layer_memmap(
    path: Path, n: int, d: int, *, label: str = "embed_layer"
) -> np.memmap:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbytes = n * d * 4
    ensure_allocation_fits(nbytes, label=label)
    mm = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n, d)
    )
    return mm


def write_ids_roles(out_dir: Path, ids: list[str], roles: list[int]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ids.npy", np.asarray(ids, dtype=object))
    np.save(out_dir / "roles.npy", np.asarray(roles, dtype=np.int8))


def write_manifest(
    out_dir: Path,
    *,
    run: LegNetRun,
    layers: Iterable[str],
    n_by_role: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    payload: dict[str, Any] = {
        "run_key": run.key,
        "run_name": run.run_name,
        "fold": run.fold,
        "train_dir": str(run.train_dir),
        "ckpt": str(run.ckpt_path),
        "config": str(run.config_json),
        "split_csv": str(run.split_csv),
        "legnet_tsv": str(run.legnet_tsv),
        "layers": {
            k: {"dim": LAYER_DIMS[k], "path": f"layer_{k}.npy"} for k in layers
        },
        "n_by_role": {r: int(n_by_role.get(r, 0)) for r in ROLE_NAMES},
        "role_codes": {
            "train": ROLE_TRAIN,
            "test": ROLE_TEST,
            "val": ROLE_VAL,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rc_averaged": True,
        "channel_order": "AGCT",
    }
    if extra:
        payload.update(extra)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_store(out_dir: Path, layers: Iterable[str] | None = None) -> EmbedStore:
    out_dir = Path(out_dir)
    ids = np.load(out_dir / "ids.npy", allow_pickle=True)
    roles = np.load(out_dir / "roles.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    layer_keys = list(layers) if layers is not None else list(manifest["layers"])
    mats: dict[str, np.ndarray] = {}
    for k in layer_keys:
        mats[k] = np.load(out_dir / f"layer_{k}.npy", mmap_mode="r")
    return EmbedStore(out_dir=out_dir, ids=ids, roles=roles, layers=mats)


def mask_role(roles: np.ndarray, role: int) -> np.ndarray:
    return roles == int(role)
