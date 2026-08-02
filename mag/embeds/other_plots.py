"""Plot distance / correlation metrics for every Caduceus model.

Heat-maps are rendered "dark-theme, annotated" like the reference screenshot:
  * the numeric coefficient is printed inside every cell (when the matrix is
    small enough), with text colour chosen for contrast;
  * axes are labelled with the *decoded* element names (token names such as
    [CLS], A, C, G, T, ... for the embeddings; channel numbers for out_proj);
  * x tick labels are rotated 45 degrees;
  * the off-diagonal summary coefficients are printed below the figure.

Additionally, per model: histograms of pairwise values and pair-wise scatter
plots with linear-fit coefficients; and across models: comparison box-plots.
A summary table of all coefficients is written to plots/summary_coefficients.csv.

Missing files / metrics are skipped gracefully.

Requires: numpy, matplotlib   (pip install numpy matplotlib)

Run from anywhere:
    python emb/plot_metrics.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI required
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "embeddings"
PLOTS_DIR = DATA_DIR / "plots"


# suffix -> (layer, metric).  Longest first so that
# "..._out_proj_cosine_divergence.csv" is not mis-classified as embeddings.
SUFFIXES: list[tuple[str, str, str]] = [
    ("_last_mamba_fwd_out_proj_cosine_divergence.csv",   "out_proj", "cosine_divergence"),
    ("_last_mamba_fwd_out_proj_mahalanobis.csv",         "out_proj", "mahalanobis"),
    ("_last_mamba_fwd_out_proj_pearson_correlation.csv", "out_proj", "pearson_correlation"),
    ("_cosine_divergence.csv",       "embeddings", "cosine_divergence"),
    ("_pearson_correlation.csv",     "embeddings", "pearson_correlation"),
    ("_mahalanobis.csv",             "embeddings", "mahalanobis"),
]

METRIC_LABEL = {
    "cosine_divergence":   "Cosine divergence  D = 1 - cos(theta)",
    "pearson_correlation": "Pearson correlation  r",
    "mahalanobis":         "Mahalanobis distance  (pseudo-inverse Sigma+)",
}
LAYER_LABEL = {
    "embeddings": "Token embeddings (input layer)",
    "out_proj":   "Last Mamba output projection  (256 x 512)",
}

# ── rendering thresholds ──────────────────────────────────────────────
ANNOT_MAX = 24   # print numbers inside cells only if n <= this
LABEL_MAX = 40   # label every axis tick only if n <= this (else thin out)

# ── dark theme (matches the reference screenshot) ─────────────────────
BG       = "#171717"
FG       = "#e8e8e8"
GRID_EC  = "#333333"
DARK_STEEL = LinearSegmentedColormap.from_list(
    "dark_steel", ["#101418", "#22303c", "#3f5d72", "#6f9bb8", "#9fc6e6"]
)
METRIC_CMAP = {
    "cosine_divergence":   DARK_STEEL,
    "mahalanobis":         DARK_STEEL,
    "pearson_correlation": "RdBu_r",
}

# ── decoded token names, indexed by token id (order as in the screenshot) ──
# Edit this list if your vocabulary / special-token order differs; any id
# beyond the list length falls back to its numeric label.
TOKEN_NAMES: list[str] = [
    "[CLS]", "[SEP]", "[BOS]", "[MASK]", "[PAD]", "[RESERVED]", "[UNK]",
    "A", "C", "G", "T", "N", "R", "Y", "S", "W",
]


# ──────────────────────────────────────────────────────────────────────
# Loading & statistics
# ──────────────────────────────────────────────────────────────────────

def load_matrix_with_labels(path: Path):
    """Read a square metric CSV -> (matrix, row_labels, col_labels).

    row_labels come from the first column, col_labels from the header row
    (both as strings, e.g. token ids).
    """
    row_labels: list[str] = []
    col_labels: list[str] = []
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            col_labels = [str(c).strip() for c in header[1:]]
        for row in reader:
            if len(row) < 2:
                continue
            row_labels.append(str(row[0]).strip())
            rows.append([float(x) for x in row[1:]])
    return np.asarray(rows, dtype=np.float64), row_labels, col_labels


def decode_labels(labels: list[str], layer: str) -> list[str]:
    """Map numeric ids to human-readable names for the embeddings layer."""
    if layer != "embeddings":
        return labels  # out_proj keeps its channel numbers
    out: list[str] = []
    for lab in labels:
        try:
            idx = int(lab)
        except ValueError:
            out.append(lab)
            continue
        out.append(TOKEN_NAMES[idx] if 0 <= idx < len(TOKEN_NAMES) else lab)
    return out


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(matrix.shape[0], k=1)
    return matrix[i, j]


def stats(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan,
                "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def fmt(s: dict[str, float]) -> str:
    return (f"mean={s['mean']:.4f}  std={s['std']:.4f}  "
            f"median={s['median']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}  "
            f"(n_pairs={s['n']})")


# ──────────────────────────────────────────────────────────────────────
# Plotting primitives
# ──────────────────────────────────────────────────────────────────────

def _colorbar_range(metric: str, off: np.ndarray) -> dict:
    if metric == "pearson_correlation":
        return dict(vmin=-1.0, vmax=1.0, extend="neither")
    if metric == "cosine_divergence":
        return dict(vmin=0.0, vmax=2.0, extend="neither")
    finite = off[np.isfinite(off)]
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    return dict(vmin=0.0, vmax=max(vmax, 1e-9), extend="max")


def _text_color(cmap, norm_value: float) -> str:
    """White text on dark cells, dark text on light cells."""
    r, g, b, _ = cmap(max(0.0, min(1.0, norm_value)))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#f2f2f2" if lum < 0.75 else "#101010"


def _dark_axes(fig, ax) -> None:
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_EC)


def plot_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                 model: str, layer: str, metric: str, out_path: Path,
                 s: dict[str, float]) -> None:
    n = matrix.shape[0]
    off = off_diagonal(matrix)
    cr = _colorbar_range(metric, off)
    cmap = plt.get_cmap(METRIC_CMAP[metric])
    norm = Normalize(cr["vmin"], cr["vmax"], clip=True)

    # figure sizing: roomy when we annotate cells, compact otherwise
    if n <= ANNOT_MAX:
        cell = 0.62
    elif n <= LABEL_MAX:
        cell = 0.40
    else:
        cell = min(0.30, 14.0 / n)
    size = min(16.0, max(8.0, n * cell))
    fig, ax = plt.subplots(figsize=(size + 2.6, size + 2.6))
    _dark_axes(fig, ax)

    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=cr["vmin"], vmax=cr["vmax"])

    # axis ticks / decoded labels
    if n <= LABEL_MAX:
        xticks, xlabels = range(n), col_labels
        yticks, ylabels = range(n), row_labels
    else:
        step = max(1, math.ceil(n / 16))
        idx = list(range(0, n, step))
        xticks, xlabels = idx, [col_labels[i] for i in idx]
        yticks, ylabels = idx, [row_labels[i] for i in idx]
    ax.set_xticks(list(xticks))
    ax.set_yticks(list(yticks))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", rotation_mode="anchor",
                       color=FG, fontsize=9 if n <= LABEL_MAX else 8)
    ax.set_yticklabels(ylabels, color=FG, fontsize=9 if n <= LABEL_MAX else 8)
    ax.set_xlabel("vector index", color=FG)
    ax.set_ylabel("vector index", color=FG)

    # numeric coefficients inside the cells (only when readable)
    if n <= ANNOT_MAX:
        annot_fs = 9 if n <= 16 else 7
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        color=_text_color(cmap, norm(val)), fontsize=annot_fs)

    # colour bar
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, extend=cr["extend"])
    cb.set_label(METRIC_LABEL[metric], color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    cb.outline.set_edgecolor(GRID_EC)

    fig.suptitle(f"{model}  -  {LAYER_LABEL[layer]}", fontsize=11, color=FG)
    ax.set_title(METRIC_LABEL[metric], fontsize=10, color=FG)
    fig.subplots_adjust(bottom=0.22, top=0.90)
    fig.text(0.5, 0.06, "Off-diagonal coefficients:  " + fmt(s),
             ha="center", va="center", fontsize=8, color=FG,
             bbox=dict(boxstyle="round,pad=0.4", fc="#222222", ec="#555555", alpha=0.95))
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_histogram(off: np.ndarray, model: str, layer: str, metric: str,
                   out_path: Path, s: dict[str, float]) -> None:
    finite = off[np.isfinite(off)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(finite, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(s["mean"], color="crimson", lw=2, label=f"mean = {s['mean']:.4f}")
    ax.axvline(s["median"], color="seagreen", lw=2, ls="--",
               label=f"median = {s['median']:.4f}")
    ax.set_xlabel(METRIC_LABEL[metric])
    ax.set_ylabel("number of pairs")
    ax.set_title(f"{model}  -  {LAYER_LABEL[layer]}\n"
                 f"distribution of pairwise {metric}", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.97,
            f"std = {s['std']:.4f}\nmin = {s['min']:.4f}\nmax = {s['max']:.4f}\n"
            f"n_pairs = {s['n']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(off_a: np.ndarray, off_b: np.ndarray, model: str, layer: str,
                 metric_a: str, metric_b: str, out_path: Path) -> dict[str, float]:
    a = off_a.ravel()
    b = off_b.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    coeffs = {"n": int(a.size), "slope": math.nan, "intercept": math.nan, "R": math.nan}
    if a.size > 1 and np.std(a) > 0 and np.std(b) > 0:
        slope, intercept = np.polyfit(a, b, 1)
        R = float(np.corrcoef(a, b)[0, 1])
        coeffs.update(slope=float(slope), intercept=float(intercept), R=R)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, b, s=4, alpha=0.35, color="steelblue", edgecolors="none")
    if np.isfinite(coeffs["slope"]):
        xs = np.linspace(a.min(), a.max(), 100)
        ax.plot(xs, coeffs["slope"] * xs + coeffs["intercept"], color="crimson", lw=2,
                label=f"fit: y = {coeffs['slope']:.4f} x + {coeffs['intercept']:.4f}")
    ax.set_xlabel(METRIC_LABEL[metric_a])
    ax.set_ylabel(METRIC_LABEL[metric_b])
    ax.set_title(f"{model}  -  {LAYER_LABEL[layer]}\n{metric_a}  vs  {metric_b}", fontsize=10)
    ax.text(0.03, 0.97,
            f"Pearson R = {coeffs['R']:.4f}\nslope = {coeffs['slope']:.4f}\n"
            f"intercept = {coeffs['intercept']:.4f}\nn = {coeffs['n']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    if np.isfinite(coeffs["slope"]):
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return coeffs


def plot_compare_boxplot(per_model: dict[str, np.ndarray], layer: str, metric: str,
                         out_path: Path) -> None:
    names = list(per_model.keys())
    data = [off_diagonal(per_model[n])[np.isfinite(off_diagonal(per_model[n]))]
            for n in names]
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 5))
    bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True,
                    medianprops=dict(color="crimson", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.5)
    for i, d in enumerate(data):
        if d.size:
            ax.text(i + 1, float(np.median(d)), f"{np.median(d):.3f}",
                    ha="center", va="bottom", fontsize=8, color="black")
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_title(f"Comparison across models  -  {LAYER_LABEL[layer]}\n"
                 f"pairwise {metric}  (red = median)", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────

def classify(name: str) -> tuple[str, str, str] | None:
    stem = name[:-4] if name.endswith(".csv") else name
    for suffix, layer, metric in SUFFIXES:
        base = suffix[:-4]
        if stem.endswith(base):
            model = stem[: -len(base)]
            return model, layer, metric
    return None


def discover() -> dict[str, dict[tuple[str, str], Path]]:
    found: dict[str, dict[tuple[str, str], Path]] = {}
    for path in sorted(DATA_DIR.glob("*.csv")):
        c = classify(path.name)
        if c is None:
            continue
        model, layer, metric = c
        found.setdefault(model, {})[(layer, metric)] = path
    return found


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    catalog = discover()
    if not catalog:
        raise FileNotFoundError(f"No metric CSV files found in {DATA_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    compare: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for model in sorted(catalog):
        model_dir = PLOTS_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        matrices: dict[tuple[str, str], np.ndarray] = {}

        for (layer, metric), path in sorted(catalog[model].items()):
            matrix, row_labels, col_labels = load_matrix_with_labels(path)
            row_labels = decode_labels(row_labels, layer)
            col_labels = decode_labels(col_labels, layer)
            matrices[(layer, metric)] = matrix
            off = off_diagonal(matrix)
            s = stats(off)

            plot_heatmap(matrix, row_labels, col_labels, model, layer, metric,
                         model_dir / f"{layer}_{metric}_heatmap.png", s)
            plot_histogram(off, model, layer, metric,
                           model_dir / f"{layer}_{metric}_hist.png", s)

            compare.setdefault((layer, metric), {})[model] = matrix
            summary_rows.append({
                "model": model, "layer": layer, "metric": metric,
                "n_vectors": int(matrix.shape[0]), **s,
            })
            print(f"[{model}] {layer}/{metric}: {matrix.shape[0]}x{matrix.shape[1]}  "
                  f"-> {fmt(s)}")

        # pair-wise scatter of metrics sharing the same layer & size
        for layer in ("embeddings", "out_proj"):
            present = [m for (l, m) in matrices if l == layer]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    ma, mb = present[i], present[j]
                    A = matrices[(layer, ma)]
                    B = matrices[(layer, mb)]
                    if A.shape != B.shape:
                        continue
                    c = plot_scatter(off_diagonal(A), off_diagonal(B), model, layer,
                                     ma, mb,
                                     model_dir / f"{layer}_scatter_{ma}_vs_{mb}.png")
                    summary_rows.append({
                        "model": model, "layer": layer,
                        "metric": f"scatter:{ma}_vs_{mb}",
                        "n_vectors": int(A.shape[0]),
                        "n": c["n"], "mean": c["R"], "std": c["slope"],
                        "median": c["intercept"], "min": math.nan, "max": math.nan,
                    })
                    print(f"[{model}] {layer} scatter {ma} vs {mb}: "
                          f"R={c['R']:.4f}  slope={c['slope']:.4f}  "
                          f"intercept={c['intercept']:.4f}")
        print()

    for (layer, metric), per_model in sorted(compare.items()):
        if len(per_model) >= 2:
            plot_compare_boxplot(per_model, layer, metric,
                                 PLOTS_DIR / f"compare_{layer}_{metric}_boxplot.png")

    summary_path = PLOTS_DIR / "summary_coefficients.csv"
    fields = ["model", "layer", "metric", "n_vectors",
              "n", "mean", "std", "median", "min", "max"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"Done. Figures -> {PLOTS_DIR}")
    print(f"Coefficients table -> {summary_path}")


if __name__ == "__main__":
    main()