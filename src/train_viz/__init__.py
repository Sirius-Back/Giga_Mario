"""Publication-quality training visualization (`python -m src.train_viz`).

Static figures: cnsplots. Interactive: Altair (HTML + Vega-Lite).

Split comparison (train/val/test/ZSV): ``python -m src.train_viz.split_compare``.
"""

from .viz import main

__all__ = ["main"]
