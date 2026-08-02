"""Reduce each model's last-Mamba output projection to a single vector, then
compare models pairwise and plot model-by-model divergence matrices.

Per model file (model_*):
  1. read out_proj.weight (256 x 512) from the safetensors checkpoint;
  2. average it into ONE vector (REDUCE_AXIS = 0 -> average over the 256 output
     channels => 512-d vector; REDUCE_AXIS = 1 -> average over the 512 hidden
     dims => 256-d vector);
  3. collect one vector per model.

Across models:
  * cosine divergence matrix  D = 1 - cos(v_i, v_j)   ("divergation")
  * Euclidean distance matrix
  * annotated heat-maps (model names on both axes, value in every cell)
  * CSVs: averaged vectors + both matrices.

Missing tensors are skipped with a warning; the run continues for other models.

Requires: numpy, matplotlib   (pip install numpy matplotlib)
Run:      python emb/compare_models.py
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR                       # where the model_* files live
OUTPUT_DIR = BASE_DIR / "model_comparison"

OUTPUT_PROJECTION_TENSOR = (
    "caduceus.backbone.layers.15.mixer.submodule.mamba_fwd.out_proj.weight"
)
OUT_PROJ_SUFFIXES = ("mamba_fwd.out_proj.weight", "out_proj.weight")

# 0 -> average over the 256 output channels  => vector of length 512
# 1 -> average over the 512 hidden dims      => vector of length 256
REDUCE_AXIS = 0

# dark theme (matches the earlier heat-maps)
BG, FG, GRID_EC = "#171717", "#e8e8e8", "#333333"
DARK_STEEL = LinearSegmentedColormap.from_list(
    "dark_steel", ["#101418", "#22303c", "#3f5d72", "#6f9bb8", "#9fc6e6"]
)


# ──────────────────────────────────────────────────────────────────────
# safetensors reading
# ──────────────────────────────────────────────────────────────────────

def _read_header(path: Path) -> dict:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def read_tensor(path: Path, header: dict, name: str):
    tensor = header[name]
    if tensor["dtype"] != "F32":
        raise ValueError(f"Unsupported dtype in {path.name}: {tensor['dtype']}")
    if len(tensor["shape"]) != 2:
        raise ValueError(f"Expected a 2-D matrix for '{name}', got {tensor['shape']}")
    start, end = tensor["data_offsets"]
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + header_size + start)
        data = f.read(end - start)
    return struct.unpack(f"<{len(data) // 4}f", data), tensor["shape"]


def find_out_proj_name(header: dict) -> str | None:
    if OUTPUT_PROJECTION_TENSOR in header:
        return OUTPUT_PROJECTION_TENSOR
    for suffix in OUT_PROJ_SUFFIXES:
        for key in header:
            if key.endswith(suffix):
                return key
    return None


# ──────────────────────────────────────────────────────────────────────
# Reduction & metrics
# ──────────────────────────────────────────────────────────────────────

def average_projection(values, shape: list[int], axis: int) -> np.ndarray:
    """Collapse the (channels x hidden) projection matrix into one vector."""
    mat = np.asarray(values, dtype=np.float64).reshape(shape)
    return np.mean(mat, axis=axis)


def cosine_divergence(vecs: list[np.ndarray]) -> np.ndarray:
    M = np.vstack(vecs)
    norms = np.linalg.norm(M, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cosine divergence is undefined for a zero vector")
    sim = (M @ M.T) / np.outer(norms, norms)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def euclidean_distance(vecs: list[np.ndarray]) -> np.ndarray:
    M = np.vstack(vecs)
    sq = np.sum(M * M, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (M @ M.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


# ──────────────────────────────────────────────────────────────────────
# CSV writers
# ──────────────────────────────────────────────────────────────────────

def write_vectors_csv(path: Path, names: list[str], vecs: list[np.ndarray],
                      source_shape: list[int], axis: int) -> None:
    dim = vecs[0].size
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "reduce_axis", "source_shape",
                    *[f"vec_{i}" for i in range(dim)]])
        for name, vec in zip(names, vecs):
            w.writerow([name, axis, f"{source_shape[0]}x{source_shape[1]}", *vec])


def write_matrix_csv(path: Path, names: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", *names])
        for i, name in enumerate(names):
            w.writerow([name, *matrix[i]])


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def _text_color(cmap, t: float) -> str:
    r, g, b, _ = cmap(max(0.0, min(1.0, t)))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#f2f2f2" if lum < 0.75 else "#101010"


def plot_model_heatmap(matrix: np.ndarray, names: list[str], title: str,
                       cbar_label: str, out_path: Path) -> None:
    n = len(names)
    vmax = max(float(np.max(matrix)) * 1.05, 1e-9)
    norm = Normalize(0.0, vmax, clip=True)
    cmap = DARK_STEEL

    size = max(7.0, n * 1.3)
    fig, ax = plt.subplots(figsize=(size + 3.0, size + 2.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor",
                       color=FG, fontsize=9)
    ax.set_yticklabels(names, color=FG, fontsize=9)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_EC)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    color=_text_color(cmap, norm(matrix[i, j])), fontsize=11)

    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    cb.outline.set_edgecolor(GRID_EC)

    fig.suptitle(title, fontsize=11, color=FG)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def print_table(names: list[str], matrix: np.ndarray, title: str) -> None:
    col = max([len(n) for n in names] + [10]) + 2
    print(title)
    print(" " * col + "".join(f"{n[:col-2]:>{col}}" for n in names))
    for i, name in enumerate(names):
        cells = "".join(f"{matrix[i, j]:>{col}.4f}" for j in range(len(names)))
        print(f"{name:>{col}}" + cells)
    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE_DIR.glob("model_*"))
    if not files:
        raise FileNotFoundError(f"No model_* files found in {SOURCE_DIR}")

    names: list[str] = []
    vecs: list[np.ndarray] = []
    source_shape: list[int] = []

    for path in files:
        try:
            header = _read_header(path)
            tensor_name = find_out_proj_name(header)
            if tensor_name is None:
                keys = [k for k in header if not k.startswith("__")]
                print(f"{path.name}: SKIP — out_proj tensor not found "
                      f"({len(keys)} tensors; first 10: {keys[:10]})")
                continue
            values, shape = read_tensor(path, header, tensor_name)
            vec = average_projection(values, shape, REDUCE_AXIS)
            names.append(path.name)
            vecs.append(vec)
            source_shape = list(shape)
            print(f"{path.name}: out_proj {shape[0]}x{shape[1]} "
                  f"--mean(axis={REDUCE_AXIS})--> vector[{vec.size}]  "
                  f"(tensor: {tensor_name})")
        except Exception as exc:
            print(f"{path.name}: ERROR -> {type(exc).__name__}: {exc}")

    if len(names) < 2:
        print("Need at least two models with a readable out_proj to compare.")
        return

    # identical-vector guard (mirrors the earlier frozen-embedding finding)
    if all(np.array_equal(vecs[0], v) for v in vecs[1:]):
        print("WARNING: averaged projection vectors are IDENTICAL across all "
              "models — the checkpoints likely share the same out_proj weights.")

    cos = cosine_divergence(vecs)
    euc = euclidean_distance(vecs)

    write_vectors_csv(OUTPUT_DIR / "averaged_projection_vectors.csv",
                      names, vecs, source_shape, REDUCE_AXIS)
    write_matrix_csv(OUTPUT_DIR / "model_cosine_divergence.csv", names, cos)
    write_matrix_csv(OUTPUT_DIR / "model_euclidean_distance.csv", names, euc)

    axis_note = (f"averaged over {source_shape[0]} channels -> "
                 f"{source_shape[1]}-d vector" if REDUCE_AXIS == 0
                 else f"averaged over {source_shape[1]} dims -> "
                      f"{source_shape[0]}-d vector")
    plot_model_heatmap(cos, names,
                       f"Model comparison  -  cosine divergence of averaged "
                       f"out_proj ({axis_note})",
                       "Cosine divergence  D = 1 - cos(theta)",
                       OUTPUT_DIR / "model_cosine_divergence_heatmap.png")
    plot_model_heatmap(euc, names,
                       f"Model comparison  -  Euclidean distance of averaged "
                       f"out_proj ({axis_note})",
                       "Euclidean distance",
                       OUTPUT_DIR / "model_euclidean_distance_heatmap.png")

    print()
    print_table(names, cos, f"Cosine divergence between models ({axis_note}):")
    print_table(names, euc, f"Euclidean distance between models ({axis_note}):")
    print(f"Done. Outputs -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()