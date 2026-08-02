#!/usr/bin/env python3
"""Schematic: leaky vs orthogonal test loss — early stopping breaks under leakage.

Conceptual curves (not empirical). For presentation slides.

Usage:
  python -m src.run.presentation.plot_leak_breaks_early_stopping
  python -m src.run.presentation.plot_leak_breaks_early_stopping --lang en
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Okabe–Ito-adjacent + requested coral / teal accents
COLOR_TRAIN = "#6B6B6B"
COLOR_LEAKY = "#E07050"  # coral / red — reported metric
COLOR_ORTHO = "#2A9D8F"  # teal / green — true generalization
COLOR_DELTA = "#E07050"
COLOR_ANNOT = "#333333"


def _curves(n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthetic train / leaky-test / orthogonal-test losses vs epoch."""
    epochs = np.linspace(0.0, 100.0, n)

    # Monotone train decay
    train = 1.55 * np.exp(-epochs / 28.0) + 0.18

    # Leaky test tracks train closely and keeps falling (no U-turn)
    leaky = train + 0.055 + 0.02 * np.exp(-epochs / 40.0)

    # Orthogonal: asymmetric U with pinned minimum (true early-stop)
    e_star = 34.0
    y_star = 0.48
    left = (e_star - epochs) / e_star
    right = (epochs - e_star) / (100.0 - e_star)
    ortho = np.where(
        epochs <= e_star,
        y_star + 1.18 * np.maximum(left, 0.0) ** 1.25,
        y_star + 0.58 * np.maximum(right, 0.0) ** 1.35,
    )

    return epochs, train, leaky, ortho, e_star


def _labels(lang: str) -> dict[str, str]:
    if lang == "en":
        return {
            "xlabel": "Epoch",
            "ylabel": "Loss",
            "train": "train loss",
            "leaky": "test loss (leaky split)",
            "ortho": "test loss (orthogonal split)",
            "delta": r"$\Delta_{\mathrm{leak}}$ — what you treat as quality",
            "arrow_left": "red curve: keep training",
            "arrow_right": "model already overfit",
            "early_stop": "true early-stop",
            "takeaway": (
                "Leakage does not merely inflate the metric — it breaks early stopping\n"
                "and model selection: validation ceases to detect overfitting."
            ),
        }
    return {
        "xlabel": "Эпоха",
        "ylabel": "Loss",
        "train": "train loss",
        "leaky": "test loss (leaky split)",
        "ortho": "test loss (orthogonal split)",
        "delta": r"$\Delta_{\mathrm{leak}}$ — то, что вы принимаете за качество",
        "arrow_left": "здесь по красной кривой\nвы решаете учить ещё",
        "arrow_right": "здесь модель\nуже переобучилась",
        "early_stop": "истинный early-stop",
        "takeaway": (
            "Утечка не просто завышает метрику — она ломает early stopping и model selection,\n"
            "потому что валидация перестаёт быть детектором переобучения."
        ),
    }


def plot(
    out_dir: Path,
    *,
    lang: str = "ru",
    stem: str = "leak_breaks_early_stopping",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = _labels(lang)
    epochs, train, leaky, ortho, e_star = _curves()
    i_min = int(np.argmin(ortho))
    y_min = float(ortho[i_min])

    # Presentation-friendly size (widescreen-ish slide panel)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        epochs,
        train,
        color=COLOR_TRAIN,
        ls="--",
        lw=2.4,
        label=L["train"],
        zorder=3,
    )
    ax.plot(
        epochs,
        leaky,
        color=COLOR_LEAKY,
        ls="-",
        lw=2.8,
        label=L["leaky"],
        zorder=4,
    )
    ax.plot(
        epochs,
        ortho,
        color=COLOR_ORTHO,
        ls="-",
        lw=2.8,
        label=L["ortho"],
        zorder=5,
    )

    # Vertical dashed line at orthogonal minimum
    ax.axvline(
        e_star,
        color=COLOR_ORTHO,
        ls="--",
        lw=1.8,
        alpha=0.9,
        zorder=2,
    )
    ax.plot([e_star], [y_min], "o", color=COLOR_ORTHO, ms=9, zorder=6, clip_on=False)
    ax.annotate(
        L["early_stop"],
        xy=(e_star, y_min),
        xytext=(e_star - 2, 0.12),
        fontsize=10.5,
        color=COLOR_ORTHO,
        ha="center",
        va="bottom",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_ORTHO,
            lw=1.3,
            connectionstyle="arc3,rad=0.0",
        ),
    )

    # Δ_leak band at end of training (between leaky and ortho)
    e_lo, e_hi = 86.0, 100.0
    mask = (epochs >= e_lo) & (epochs <= e_hi)
    ax.fill_between(
        epochs[mask],
        leaky[mask],
        ortho[mask],
        color=COLOR_DELTA,
        alpha=0.28,
        zorder=1,
        linewidth=0,
    )
    # Bracket ticks on the Δ band
    y_l = float(leaky[-1])
    y_o = float(ortho[-1])
    ax.plot([99.2, 99.2], [y_l, y_o], color=COLOR_LEAKY, lw=2.0, solid_capstyle="butt", zorder=6)
    ax.plot([98.4, 99.2], [y_l, y_l], color=COLOR_LEAKY, lw=2.0, zorder=6)
    ax.plot([98.4, 99.2], [y_o, y_o], color=COLOR_LEAKY, lw=2.0, zorder=6)
    mid_e = 94.0
    mid_y = 0.5 * (y_l + y_o)
    ax.annotate(
        L["delta"],
        xy=(99.2, mid_y),
        xytext=(58, 1.42),
        fontsize=11.5,
        color=COLOR_LEAKY,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_LEAKY,
            lw=1.5,
            connectionstyle="arc3,rad=-0.25",
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=COLOR_LEAKY,
            alpha=0.95,
            linewidth=1.2,
        ),
        zorder=7,
    )

    # Dual narrative: keep training (by red) ↔ already overfit (by green)
    x_left, x_right = 62.0, 90.0
    y_box = 1.05
    ax.annotate(
        L["arrow_left"],
        xy=(x_left, float(np.interp(x_left, epochs, leaky))),
        xytext=(48, y_box),
        fontsize=10.5,
        color=COLOR_LEAKY,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_LEAKY,
            lw=1.4,
            connectionstyle="arc3,rad=0.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#FFF8F5",
            edgecolor=COLOR_LEAKY,
            alpha=0.95,
            linewidth=1.0,
        ),
        zorder=7,
    )
    ax.annotate(
        L["arrow_right"],
        xy=(x_right, float(np.interp(x_right, epochs, ortho))),
        xytext=(78, y_box),
        fontsize=10.5,
        color=COLOR_ORTHO,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_ORTHO,
            lw=1.4,
            connectionstyle="arc3,rad=-0.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F3FAF8",
            edgecolor=COLOR_ORTHO,
            alpha=0.95,
            linewidth=1.0,
        ),
        zorder=7,
    )
    # Explicit ↔ between the two decision narratives
    arr = FancyArrowPatch(
        (58.5, y_box),
        (67.5, y_box),
        arrowstyle="<->",
        mutation_scale=16,
        color=COLOR_ANNOT,
        lw=1.5,
        zorder=8,
        clip_on=False,
    )
    ax.add_patch(arr)

    ax.set_xlabel(L["xlabel"], fontsize=14)
    ax.set_ylabel(L["ylabel"], fontsize=14)
    ax.set_xlim(0, 102)
    ax.set_ylim(0.0, 1.78)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(True, which="major", color="#B0B0B0", alpha=0.28, lw=0.8)

    leg = ax.legend(
        loc="upper right",
        fontsize=11.5,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.97,
        borderpad=0.6,
    )
    for text, color in zip(
        leg.get_texts(),
        [COLOR_TRAIN, COLOR_LEAKY, COLOR_ORTHO],
    ):
        text.set_color(color)

    # Takeaway strip under the axes
    fig.text(
        0.5,
        0.015,
        L["takeaway"],
        ha="center",
        va="bottom",
        fontsize=11,
        color=COLOR_ANNOT,
        style="italic",
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.98))

    written: list[Path] = []
    for ext in ("svg", "pdf", "png"):
        path = out_dir / f"{stem}_{lang}.{ext}"
        kw: dict = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/presentation"),
        help="Output directory for SVG/PDF/PNG",
    )
    p.add_argument(
        "--lang",
        choices=("ru", "en", "both"),
        default="ru",
        help="Annotation language (default: ru)",
    )
    p.add_argument(
        "--stem",
        default="leak_breaks_early_stopping",
        help="Filename stem",
    )
    args = p.parse_args()
    langs = ("ru", "en") if args.lang == "both" else (args.lang,)
    for lang in langs:
        paths = plot(args.out_dir, lang=lang, stem=args.stem)
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
